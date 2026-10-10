"""Test script for RS-Mamba (AFENet-style config).

Usage:
    python test.py -c config/vaihingen/rsmamba.py -o ./output
    python test.py -c config/vaihingen/rsmamba.py -o ./output -t d4 --rgb
"""

import argparse
import os
import time
import multiprocessing.pool as mpp
import multiprocessing as mp

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

try:
    import ttach as tta
except ImportError:
    tta = None

from tools.cfg import py2cfg
from tools.metric import Evaluator


CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')


def seed_everything(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb


def img_writer(inp):
    (mask, mask_id, rgb) = inp
    if rgb:
        mask_tif = label2rgb(mask)
        cv2.imwrite(mask_id + '.png', mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        cv2.imwrite(mask_id + '.png', mask_png)


def get_args():
    parser = argparse.ArgumentParser(description='Test RS-Mamba')
    arg = parser.add_argument
    arg("-c", "--config_path", type=str, required=True, help="Path to config file")
    arg("-o", "--output_path", type=str, required=True, help="Path to save resulting masks")
    arg("-t", "--tta", default=None, choices=[None, "d4", "lr"], help="Test time augmentation")
    arg("--rgb", action='store_true', help="Save output as RGB color masks")
    return parser.parse_args()


def main():
    seed_everything(42)
    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)

    config = py2cfg(args.config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO)
    print(f'Using device {device}')

    # ── Load model ──
    net = config.net.to(device)
    ckpt_path = os.path.join(config.weights_path, config.test_weights_name + '.pth')
    print(f'[DEBUG] Looking for: {ckpt_path}')
    if not os.path.exists(ckpt_path):
        import glob
        ckpt_files = glob.glob(os.path.join(config.weights_path, '*.pth'))
        if ckpt_files:
            ckpt_path = max(ckpt_files, key=os.path.getmtime)
    print(f'[DEBUG] Loading: {ckpt_path}, size={os.path.getsize(ckpt_path) if os.path.exists(ckpt_path) else "NOT FOUND"} bytes')
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['net'] if 'net' in ckpt else ckpt
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'[Warning] Missing keys (not in checkpoint): {missing}')
    if unexpected:
        print(f'[Warning] Unexpected keys (ignored, not in current model): {unexpected}')
    print(f'Loaded checkpoint: {ckpt_path}')
    net.eval()

    # ── TTA ──
    if args.tta == "lr" and tta is not None:
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        net = tta.SegmentationTTAWrapper(net, transforms)
    elif args.tta == "d4" and tta is not None:
        transforms = tta.Compose([
            tta.HorizontalFlip(), tta.VerticalFlip(),
            tta.Rotate90(angles=[90]),
            tta.Scale(scales=[0.5, 0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
        ])
        net = tta.SegmentationTTAWrapper(net, transforms)

    # ── Test dataset ──
    test_dataset = config.test_dataset
    test_loader = DataLoader(test_dataset, batch_size=2, num_workers=4,
                            pin_memory=True, drop_last=False)
    print(f'Test dataset: {len(test_dataset)} images')

    # ── Inference ──
    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()
    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            img = batch['img'].float().to(device)
            image_ids = batch['img_id']
            masks_true = batch['gt_semantic_seg']

            raw_predictions = net(img)
            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())
                mask_name = image_ids[i]
                results.append((mask, os.path.join(args.output_path, mask_name), args.rgb))

    # ── Metrics ──
    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    OA = evaluator.OA()
    for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class, f1_per_class):
        print('F1_{}:{:.4f}, IOU_{}:{:.4f}'.format(class_name, class_f1, class_name, class_iou))
    print('F1:{:.4f}, mIOU:{:.4f}, OA:{:.4f}'.format(
        np.nanmean(f1_per_class[:-1]), np.nanmean(iou_per_class[:-1]), OA))

    # ── Save predictions ──
    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    print('images writing spends: {} s'.format(t1 - t0))


if __name__ == "__main__":
    main()
