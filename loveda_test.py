"""LoveDA test script for semantic_segmentation_mamba project.

Same functionality as the root project's loveda_test.py, adapted for RSM project:
  - Checkpoint: .pth format (no Lightning), auto-find from config
  - Everything else identical: TTA, multi-process writing, LoveDA palette, all 7 classes mIoU

Usage:
    python loveda_test.py -c config/loveda/afenet_rsmssm_ada.py -o ./output
    python loveda_test.py -c config/loveda/afenet_rsmssm_ada.py -o ./output -t d4 --rgb
"""

import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train import seed_everything
import argparse
from pathlib import Path
import numpy as np
import glob
import os
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.cfg import py2cfg
from tools.metric import Evaluator


LOVEDA_PALETTE = np.array([
    [255, 255, 255],  # 0 background
    [255, 0, 0],      # 1 building
    [255, 255, 0],    # 2 road
    [0, 0, 255],      # 3 water
    [159, 129, 183],  # 4 barren
    [0, 255, 0],      # 5 forest
    [255, 195, 128],  # 6 agriculture
], dtype=np.uint8)


def label2rgb(mask):
    mask_rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(LOVEDA_PALETTE))
    mask_rgb[valid] = LOVEDA_PALETTE[mask[valid]]
    return mask_rgb


def img_writer(inp):
    (mask, mask_id, rgb) = inp
    if rgb:
        mask_name_png = mask_id + '.png'
        Image.fromarray(label2rgb(mask), mode='RGB').save(mask_name_png)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        Image.fromarray(mask_png, mode='L').save(mask_name_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("--rgb", help="whether output rgb images", action='store_true')
    return parser.parse_args()


def main():
    seed_everything(42)
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load model (auto-find .pth from config) ──
    net = config.net.to(device)
    ckpt_path = os.path.join(config.weights_path, config.test_weights_name + '.pth')
    if not os.path.exists(ckpt_path):
        ckpt_files = glob.glob(os.path.join(config.weights_path, '*.pth'))
        if ckpt_files:
            ckpt_path = max(ckpt_files, key=os.path.getmtime)
    print(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'net' in ckpt:
        net.load_state_dict(ckpt['net'])
    else:
        net.load_state_dict(ckpt)
    net.eval()

    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()

    # ── TTA ──
    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        net = tta.SegmentationTTAWrapper(net, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[90]),
                tta.Scale(scales=[0.5, 0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
            ]
        )
        net = tta.SegmentationTTAWrapper(net, transforms)

    test_dataset = config.test_dataset

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        results = []
        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            raw_predictions = net(input['img'].float().to(device))

            image_ids = input["img_id"]
            masks_true = input['gt_semantic_seg']

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())
                mask_name = image_ids[i]
                results.append((mask, str(args.output_path / mask_name), args.rgb))

    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    OA = evaluator.OA()
    for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class, f1_per_class):
        print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
    print('F1:{}, mIOU:{}, OA:{}'.format(np.nanmean(f1_per_class), np.nanmean(iou_per_class), OA))

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))


if __name__ == "__main__":
    main()
