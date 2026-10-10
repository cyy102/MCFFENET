"""RS-Mamba training script (AFENet-style config) with DDP support.

Usage (single GPU):
    python train.py -c config/potsdam/rsmamba.py

Usage (multi-GPU DDP):
    torchrun --nproc_per_node=2 train.py -c config/potsdam/rsmamba.py
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import logging

from tools.cfg import py2cfg
from tools.metric import Evaluator


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser(description='Train RS-Mamba')
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to config file")
    return parser.parse_args()


def setup_ddp():
    """Initialize DDP if running under torchrun."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


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


def main():
    args = get_args()
    seed_everything(42)

    # ── DDP setup ──
    rank, world_size, local_rank = setup_ddp()
    is_ddp = world_size > 1
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    is_main = rank == 0

    if is_main:
        logging.basicConfig(level=logging.INFO)

    config = py2cfg(args.config_path)

    # ── Model ──
    net = config.net.to(device)

    if is_ddp:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = nn.parallel.DistributedDataParallel(
            net, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)
        if is_main:
            print(f'Using DDP with {world_size} GPUs (batch_size={config.train_batch_size})')
    else:
        if torch.cuda.device_count() > 1:
            # Fallback to DataParallel if not launched with torchrun
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = nn.DataParallel(net)
            if is_main:
                print(f'Using DataParallel with {torch.cuda.device_count()} GPUs (tip: use torchrun for DDP)')
        else:
            if is_main:
                print(f'Using device {device}')

    # ── Optimizer ──
    optimizer = config.optimizer
    lr_scheduler = config.lr_scheduler
    torch.set_float32_matmul_precision('high')

    # ── Resume ──
    start_epoch = 0
    if config.resume_ckpt_path:
        ckpt = torch.load(config.resume_ckpt_path, map_location=device, weights_only=False)
        target_net = net.module if hasattr(net, 'module') else net
        if 'net' in ckpt:
            target_net.load_state_dict(ckpt['net'])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt.get('epoch', 0)
            if is_main:
                print(f'Resumed from {config.resume_ckpt_path}, starting at epoch {start_epoch}')
        else:
            target_net.load_state_dict(ckpt)
            if is_main:
                print(f'Model loaded from {config.resume_ckpt_path}')

    # ── Metrics ──
    metrics_train = Evaluator(num_class=config.num_classes)
    metrics_val = Evaluator(num_class=config.num_classes)

    # ── Data: use DistributedSampler for DDP ──
    train_sampler = None
    val_sampler = None
    if is_ddp:
        train_sampler = DistributedSampler(
            config.train_loader.dataset, num_replicas=world_size,
            rank=rank, shuffle=True, drop_last=True)
        if hasattr(config, 'val_loader') and len(config.val_loader.dataset) > 0:
            val_sampler = DistributedSampler(
                config.val_loader.dataset, num_replicas=world_size,
                rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        config.train_loader.dataset,
        batch_size=config.train_batch_size,
        shuffle=(train_sampler is None),
        num_workers=8, pin_memory=True, drop_last=True,
        sampler=train_sampler)

    val_loader = None
    if hasattr(config, 'val_loader') and len(config.val_loader.dataset) > 0:
        val_loader = DataLoader(
            config.val_loader.dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=8, pin_memory=True, drop_last=False,
            sampler=val_sampler)

    # ── Save dir ──
    if is_main:
        os.makedirs(config.weights_path, exist_ok=True)

    # ── Auto versioning ──
    import glob

    if start_epoch > 0:
        last_name = 'last'
        best_name = config.weights_name + '-best'
    else:
        last_path = os.path.join(config.weights_path, 'last.pth')
        if os.path.exists(last_path):
            existing_versions = []
            for f in glob.glob(os.path.join(config.weights_path, 'last-v*.pth')):
                try:
                    v = int(os.path.basename(f).split('-v')[1].split('.')[0])
                    existing_versions.append(v)
                except (ValueError, IndexError):
                    pass
            next_version = (max(existing_versions) + 1) if existing_versions else 1
            last_name = f'last-v{next_version}'
        else:
            last_name = 'last'

        best_base = config.weights_name + '-best'
        best_path = os.path.join(config.weights_path, best_base + '.pth')
        if os.path.exists(best_path):
            existing_best_versions = []
            for f in glob.glob(os.path.join(config.weights_path, best_base + '-v*.pth')):
                try:
                    v = int(os.path.basename(f).split('-v')[1].split('.')[0])
                    existing_best_versions.append(v)
                except (ValueError, IndexError):
                    pass
            next_best_version = (max(existing_best_versions) + 1) if existing_best_versions else 1
            best_name = f'{best_base}-v{next_best_version}'
        else:
            best_name = best_base

    last_save_path = os.path.join(config.weights_path, last_name + '.pth')
    best_save_path = os.path.join(config.weights_path, best_name + '.pth')

    if is_main:
        config.test_weights_name = best_name
        print(f'Saving last: {last_name}.pth, best: {best_name}.pth')

    best_miou = 0.0

    # ── Training loop ──
    for epoch in range(start_epoch, config.max_epoch):
        if is_main:
            print(f'\nEpoch {epoch}/{config.max_epoch - 1}')
        net.train()
        metrics_train.reset()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        tbar = tqdm(train_loader, desc='Train', disable=not is_main)

        for batch in tbar:
            if isinstance(batch, dict):
                img = batch['img'].float().to(device)
                mask = batch['gt_semantic_seg'].long().to(device)
            else:
                img = batch[0].float().to(device)
                mask = batch[1].long().to(device)

            optimizer.zero_grad()
            preds = net(img)
            loss = config.loss(preds, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1, norm_type=2)
            optimizer.step()

            pred_mask = nn.Softmax(dim=1)(preds).argmax(dim=1)
            for i in range(mask.shape[0]):
                metrics_train.add_batch(mask[i].cpu().numpy(), pred_mask[i].cpu().numpy())

            if is_main:
                tbar.set_postfix(loss=loss.item())
            del img, mask, preds, loss

        lr_scheduler.step()

        # ── Sync train metrics across GPUs ──
        if is_ddp and torch.distributed.is_initialized():
            cm = torch.from_numpy(metrics_train.confusion_matrix).to(device)
            torch.distributed.all_reduce(cm, op=torch.distributed.ReduceOp.SUM)
            metrics_train.confusion_matrix = cm.cpu().numpy()

        if is_main:
            iou_per_class = metrics_train.Intersection_over_Union()
            f1_per_class = metrics_train.F1()
            oa = metrics_train.OA()
            if 'loveda' in config.log_name:
                miou = np.nanmean(iou_per_class)
                f1 = np.nanmean(f1_per_class)
            else:
                miou = np.nanmean(iou_per_class[:-1])
                f1 = np.nanmean(f1_per_class[:-1])
            print(f'Epoch {epoch}: train mIoU={miou:.4f}, F1={f1:.4f}, OA={oa:.4f}')
            for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class, f1_per_class):
                print(f'  {class_name}: IoU={class_iou:.4f}, F1={class_f1:.4f}')

        # ── Validation ──
        if val_loader is not None and (epoch + 1) % config.check_val_every_n_epoch == 0:
            net.eval()
            metrics_val.reset()
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            tbar_val = tqdm(val_loader, desc='Val', disable=not is_main)
            with torch.no_grad():
                for batch in tbar_val:
                    img = batch['img'].float().to(device)
                    mask = batch['gt_semantic_seg'].long().to(device)
                    preds = net(img)
                    pred_mask = nn.Softmax(dim=1)(preds).argmax(dim=1)
                    for i in range(mask.shape[0]):
                        metrics_val.add_batch(mask[i].cpu().numpy(), pred_mask[i].cpu().numpy())
                    del img, mask, preds

            # ── Sync val metrics ──
            if is_ddp and torch.distributed.is_initialized():
                cm = torch.from_numpy(metrics_val.confusion_matrix).to(device)
                torch.distributed.all_reduce(cm, op=torch.distributed.ReduceOp.SUM)
                metrics_val.confusion_matrix = cm.cpu().numpy()

            if is_main:
                val_iou = metrics_val.Intersection_over_Union()
                val_f1 = metrics_val.F1()
                val_oa = metrics_val.OA()
                if 'loveda' in config.log_name:
                    val_miou = np.nanmean(val_iou)
                    val_f1_mean = np.nanmean(val_f1)
                else:
                    val_miou = np.nanmean(val_iou[:-1])
                    val_f1_mean = np.nanmean(val_f1[:-1])
                print(f'Epoch {epoch}: val mIoU={val_miou:.4f}, F1={val_f1_mean:.4f}, OA={val_oa:.4f}')

                if val_miou > best_miou:
                    best_miou = val_miou
                    target_net = net.module if hasattr(net, 'module') else net
                    torch.save(
                        {'net': target_net.state_dict(), 'epoch': epoch + 1, 'val_miou': val_miou},
                        best_save_path)
                    print(f'Best model saved! mIoU={val_miou:.4f} -> {best_save_path}')

        # ── Save last ──
        if is_main:
            target_net = net.module if hasattr(net, 'module') else net
            torch.save(
                {'net': target_net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1},
                last_save_path)

    if is_main:
        print(f'Training finished! Best mIoU={best_miou:.4f}')
    cleanup_ddp()


if __name__ == "__main__":
    main()
