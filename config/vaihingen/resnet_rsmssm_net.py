"""MCFFENet config for Vaihingen �?ResNet + MDFSI + FA-SSM.

FA-SSM (Frequency-Guided Adaptive SSM): FFT frequency decomposition guides
the local/global scan strategy assignment.
  - Conditional: only window+pool when H*W > 16384
  - Short sequences use standard full-resolution SSM
  - High-freq regions �?local window scan (detail preservation)
  - Low-freq regions �?global pool scan (context capture)
  - Learnable frequency cutoff (adaptive, not fixed threshold)
  - Connects with Decoder's MDFSI frequency module

Decoder: full MSFEB + TransformerBlock + multi-scale MDFSI aggregation.

Usage:
    python train.py -c config/vaihingen/resnet_rsmssm_net.py
    python test.py  -c config/vaihingen/resnet_rsmssm_net.py -o ./output -t d4 --rgb
"""

import torch
from torch.utils.data import DataLoader
from utils.losses import JointLoss, SoftBCEWithLogitsLoss, DiceLoss
from utils.data_loading import BasicDataset
from resnet_rsmssm import MCFFENet

# ── Dataset ──
CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255],
           [0, 255, 0], [255, 204, 0], [255, 0, 0]]
num_classes = len(CLASSES)
ignore_index = len(CLASSES)
classes = CLASSES

# ── Data paths ──
data_root = '/root/rivermind-data/data/vaihingen'
img_dir = 'images_1024'
mask_dir = 'masks_1024'
train_split = 'train'
val_split = 'train'  # no separate val set, use train dir
test_split = 'eroded_test'

# ── Training params ──
max_epoch = 100
train_batch_size = 8
val_batch_size = 8
lr = 1e-3
weight_decay = 0.01
seed = 42

# ── Model �?MCFFENet: Mamba-CNN Fusion and Frequency Enhancement ──
net = MCFFENet(
    num_classes=num_classes,
    backbone_name='swsl_resnet18',
    pretrained=True,
    block_type='ee',                                # EE-SSM: Sobel from feature map (verified best)
    block_stages=(True, True, False, False),        # on enc1 + enc2
    envss_num_blocks=(2, 2, 9, 2),                  # per-stage SSM block counts: (enc1, enc2, enc3, enc4)
)

# ── Loss ──
loss = JointLoss(
    SoftBCEWithLogitsLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0, 1.0,
)

# ── Datasets & Loaders ──
train_dataset = BasicDataset(
    images_dir=f'{data_root}/{train_split}/{img_dir}/',
    labels_dir=f'{data_root}/{train_split}/{mask_dir}/',
    train=True,
    crop_size=512,
    ignore_index=ignore_index,
    mosaic_ratio=0.25,
)
# val_dataset = BasicDataset(
#     images_dir=f'{data_root}/{val_split}/{img_dir}/',
#     labels_dir=f'{data_root}/{val_split}/{mask_dir}/',
#     train=False,
# )
test_dataset = BasicDataset(
    images_dir=f'{data_root}/{test_split}/{img_dir}/',
    labels_dir=f'{data_root}/{test_split}/{mask_dir}/',
    train=False,
)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True)
# val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False,
#                         num_workers=8, pin_memory=True, drop_last=False)

# ── Optimizer: two-group (backbone low-lr, rest high-lr) ──
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

# ── Checkpoint / logging ──
weights_name = "MCFFENet-vaihingen-e{}".format(max_epoch)
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
