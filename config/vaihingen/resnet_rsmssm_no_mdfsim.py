"""Vaihingen control: remove only MDFSIM from the original full decoder.

Encoder: ResNet18 + standard RSM SSM + DBFFB; EESSM disabled.
Each MSFEB retains SCModule, SFFModule, projections and its residual path.
Replace only MDFSIModule/AFSIM with a bypass returning spatial features.
Retain original TB stacks, skips, multi-scale aggregation and full head.
Compare with MCFFENet(block_type=None, envss_num_blocks=(2, 2, 9, 2)).
Training/data settings match the existing Vaihingen ablation configs.
Basic-decoder + TB control remains separately in resnet_rsmssm_basic_tb.py.

Usage (from semantic_segmentation_mamba):
    python train.py -c config/vaihingen/resnet_rsmssm_no_mdfsim.py
    python test.py -c config/vaihingen/resnet_rsmssm_no_mdfsim.py -o ./output/no_mdfsim -t d4 --rgb
"""

import torch
from torch.utils.data import DataLoader
from utils.losses import JointLoss, SoftBCEWithLogitsLoss, DiceLoss
from utils.data_loading import BasicDataset
from resnet_rsmssm_no_mdfsim_net import MCFFENet_NoMDFSIM

# Dataset
CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255],
           [0, 255, 0], [255, 204, 0], [255, 0, 0]]
num_classes = len(CLASSES)
ignore_index = len(CLASSES)
classes = CLASSES

# Data paths (same server paths as the original Vaihingen config)
data_root = '/root/rivermind-data/data/vaihingen'
img_dir = 'images_1024'
mask_dir = 'masks_1024'
train_split = 'train'
val_split = 'train'  # original config has no separate validation loader
test_split = 'eroded_test'

# Training parameters
max_epoch = 120
train_batch_size = 8
val_batch_size = 8
lr = 1e-3
weight_decay = 0.01
seed = 42

# Main ablation: PCSHE + original decoder without MDFSIM; EESSM disabled
net = MCFFENet_NoMDFSIM(
    num_classes=num_classes,
    backbone_name='swsl_resnet18',
    pretrained=True,
    block_type=None,
    block_stages=(True, True, False, False),
    envss_num_blocks=(2, 2, 9, 2),
)

# Loss
loss = JointLoss(
    SoftBCEWithLogitsLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0, 1.0,
)

# Datasets and loaders
train_dataset = BasicDataset(
    images_dir=f'{data_root}/{train_split}/{img_dir}/',
    labels_dir=f'{data_root}/{train_split}/{mask_dir}/',
    train=True,
    crop_size=512,
    ignore_index=ignore_index,
    mosaic_ratio=0.25,
)
test_dataset = BasicDataset(
    images_dir=f'{data_root}/{test_split}/{img_dir}/',
    labels_dir=f'{data_root}/{test_split}/{mask_dir}/',
    train=False,
)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True)

# Optimizer: backbone low learning rate, other parameters high learning rate
backbone_lr = 6e-5
backbone_params = []
other_params = []
for name, p in net.named_parameters():
    if not p.requires_grad:
        continue
    if name.startswith('backbone'):
        backbone_params.append(p)
    else:
        other_params.append(p)

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
    {'params': other_params,    'lr': lr,          'weight_decay': weight_decay},
])
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-5)

# Separate checkpoints and logs for this control
weights_name = "MCFFENet-PCSHE-NoMDFSIM-vaihingen-e{}".format(max_epoch)
weights_path = "model_weights/vaihingen/{}".format(weights_name)
test_weights_name = "last"
log_name = 'vaihingen/{}'.format(weights_name)

monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1

gpus = 'auto'
resume_ckpt_path = None
pretrained_ckpt_path = None
