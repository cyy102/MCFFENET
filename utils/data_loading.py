"""Dataset with AFENet-style augmentations (RandomScale + SmartCropV1 + ColorJitter + Mosaic)."""

import logging
import random
import numbers
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── PIL-based transforms (from AFENet's transform.py) ──────────────────────

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomCrop(object):
    def __init__(self, size=512, ignore_index=12, nopad=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    def __call__(self, img, mask, centroid=None):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask

        if self.nopad:
            if th > h or tw > w:
                shorter_side = min(w, h)
                th, tw = shorter_side, shorter_side
        else:
            if th > h:
                pad_h = (th - h) // 2 + 1
            else:
                pad_h = 0
            if tw > w:
                pad_w = (tw - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                img = ImageOps.expand(img, border=border, fill=self.pad_color)
                mask = ImageOps.expand(mask, border=border, fill=self.ignore_index)
                w, h = img.size

        if centroid is not None:
            c_x, c_y = centroid
            max_x = w - tw
            max_y = h - th
            x1 = random.randint(c_x - tw, c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - th, c_y)
            y1 = min(max_y, max(0, y1))
        else:
            if w == tw:
                x1 = 0
            else:
                x1 = random.randint(0, w - tw)
            if h == th:
                y1 = 0
            else:
                y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomScale(object):
    def __init__(self, scale_list=[0.75, 1.0, 1.25], mode='value'):
        self.scale_list = scale_list
        self.mode = mode

    def __call__(self, img, mask):
        oh, ow = img.size
        scale_amt = 1.0
        if self.mode == 'value':
            scale_amt = np.random.choice(self.scale_list, 1)
        elif self.mode == 'range':
            scale_amt = random.uniform(self.scale_list[0], self.scale_list[-1])
        h = int(scale_amt * oh)
        w = int(scale_amt * ow)
        return img.resize((w, h), Image.BICUBIC), mask.resize((w, h), Image.NEAREST)


class SmartCropV1(object):
    """Smart crop: reject crops where a single class dominates > max_ratio."""

    def __init__(self, crop_size=512, max_ratio=0.75, ignore_index=12, nopad=False):
        self.crop_size = crop_size
        self.max_ratio = max_ratio
        self.ignore_index = ignore_index
        self.crop = RandomCrop(crop_size, ignore_index=ignore_index, nopad=nopad)

    def __call__(self, img, mask):
        assert img.size == mask.size
        count = 0
        while True:
            img_crop, mask_crop = self.crop(img.copy(), mask.copy())
            count += 1
            labels, cnt = np.unique(np.array(mask_crop), return_counts=True)
            cnt = cnt[labels != self.ignore_index]
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.max_ratio:
                break
            if count > 10:
                break
        return img_crop, mask_crop


# ── Dataset ────────────────────────────────────────────────────────────────

class BasicDataset(Dataset):
    """Dataset for multi-class semantic segmentation (Potsdam/Vaihingen).

    Training augmentations (AFENet-style):
      1. Mosaic (4-image stitch, probability = mosaic_ratio)
      2. RandomScale [0.5, 0.75, 1.0, 1.25, 1.5]
      3. SmartCropV1 (avoid single-class dominance)
      4. HorizontalFlip + VerticalFlip + RandomRotate90
      5. ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
      6. Normalize

    Validation/test: Normalize only (full image, no crop).
    """

    def __init__(self, images_dir: str, labels_dir: str, train: bool,
                 img_suffix: str = '.tif', mask_suffix: str = '.png',
                 crop_size=512, ignore_index=6,
                 mosaic_ratio=0.25,
                 scale_list=(0.5, 0.75, 1.0, 1.25, 1.5),
                 mask_offset=0):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.train = train
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.ignore_index = ignore_index
        self.mosaic_ratio = mosaic_ratio
        self.mask_offset = mask_offset

        # list ids from labels dir; if no labels dir, list from images dir
        if self.labels_dir.exists() and self.labels_dir.is_dir():
            self.ids = [splitext(file)[0] for file in listdir(labels_dir) if not file.startswith('.')]
        else:
            self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        self.ids.sort()

        if not self.ids:
            logging.warning(f'No input file found in {images_dir}, creating empty dataset')
            self.ids = []
        else:
            logging.info(f'Creating dataset with {len(self.ids)} examples')

        # PIL-based crop augmentation: RandomScale + SmartCropV1
        self.crop_aug = Compose([
            RandomScale(scale_list=list(scale_list), mode='value'),
            SmartCropV1(crop_size=crop_size, max_ratio=0.75,
                        ignore_index=ignore_index, nopad=False)
        ])

        # Albumentations post-crop (operates on numpy arrays)
        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.Normalize(),
        ])

        self.val_transforms = A.Compose([
            A.Normalize(),
        ])

        self.to_tensor = A.Compose([
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.ids)

    # ── Image loading ──

    def load_img_and_mask(self, idx):
        """Load image and mask as PIL Image (RGB / L).

        If no mask file exists (test set without masks), creates a dummy
        all-zero mask so the dataset can be used for inference.
        """
        img_id = self.ids[idx]
        img_file = list(self.images_dir.glob(img_id + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {img_id}: {img_file}'

        img = Image.open(img_file[0]).convert('RGB')

        label_file = list(self.labels_dir.glob(img_id + '.*')) if self.labels_dir.exists() else []
        if len(label_file) >= 1:
            mask = Image.open(label_file[0]).convert('L')
            if self.mask_offset:
                # LoveDA: raw mask 1-7 → 0-6 (subtract offset)
                mask = Image.fromarray(
                    np.clip(np.array(mask).astype(np.int32) - self.mask_offset,
                            0, self.ignore_index - 1).astype(np.uint8))
        else:
            # No mask (test set) — create dummy all-zero mask
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))
        return img_id, img, mask

    def load_mosaic_img_and_mask(self, idx):
        """Mosaic: stitch 4 random images into one, return as PIL Image."""
        indexes = [idx] + [random.randint(0, len(self.ids) - 1) for _ in range(3)]
        imgs, masks = [], []
        for i in indexes:
            _, img, mask = self.load_img_and_mask(i)
            imgs.append(np.array(img))
            masks.append(np.array(mask))

        h, w = imgs[0].shape[0], imgs[0].shape[1]
        start_x = w // 4
        start_y = h // 4
        offset_x = random.randint(start_x, w - start_x)
        offset_y = random.randint(start_y, h - start_y)

        crop_a = A.RandomCrop(width=offset_x, height=offset_y)
        crop_b = A.RandomCrop(width=w - offset_x, height=offset_y)
        crop_c = A.RandomCrop(width=offset_x, height=h - offset_y)
        crop_d = A.RandomCrop(width=w - offset_x, height=h - offset_y)

        a = crop_a(image=imgs[0], mask=masks[0])
        b = crop_b(image=imgs[1], mask=masks[1])
        c = crop_c(image=imgs[2], mask=masks[2])
        d = crop_d(image=imgs[3], mask=masks[3])

        top = np.concatenate((a['image'], b['image']), axis=1)
        bottom = np.concatenate((c['image'], d['image']), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((a['mask'], b['mask']), axis=1)
        bottom_mask = np.concatenate((c['mask'], d['mask']), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)

        img = Image.fromarray(np.ascontiguousarray(img))
        mask = Image.fromarray(np.ascontiguousarray(mask))
        return self.ids[idx], img, mask

    # ── __getitem__ ──

    def __getitem__(self, idx):
        if self.train and random.random() <= self.mosaic_ratio:
            img_id, img, mask = self.load_mosaic_img_and_mask(idx)
        else:
            img_id, img, mask = self.load_img_and_mask(idx)

        if self.train:
            # PIL: RandomScale + SmartCropV1
            img, mask = self.crop_aug(img, mask)
            # numpy: Flip + Rotate90 + ColorJitter + Normalize
            img_np, mask_np = np.array(img), np.array(mask)
            sample = self.train_transforms(image=img_np, mask=mask_np)
            img_np, mask_np = sample['image'], sample['mask']
        else:
            img_np, mask_np = np.array(img), np.array(mask)
            sample = self.val_transforms(image=img_np, mask=mask_np)
            img_np, mask_np = sample['image'], sample['mask']

        sample = self.to_tensor(image=img_np, mask=mask_np)
        tensor, mask_tensor = sample['image'].contiguous(), sample['mask'].contiguous()

        # keep label as long tensor with class indices for CrossEntropyLoss
        mask_tensor = mask_tensor.long()

        results = dict(img_id=img_id, img=tensor, gt_semantic_seg=mask_tensor)
        return results
