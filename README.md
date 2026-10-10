# MCFFENet: Mamba-CNN Fusion and Frequency Enhancement Network

MCFFENet is a semantic segmentation network for high-resolution remote sensing imagery. It combines progressive CNN-SSM encoding, edge enhancement, and spatial-frequency interaction to capture local details, long-range context, and complementary frequency information.

The repository provides training and evaluation configurations for ISPRS Vaihingen, ISPRS Potsdam, and LoveDA.

<div align="center">
  <img src="resources/framework.png" alt="MCFFENet overall architecture" width="100%" />
</div>

[View the framework as a vector PDF](resources/framework.pdf)

## Model Overview

| Component | Full name | Function |
| --- | --- | --- |
| PCSHE | Progressive CNN-SSM Hybrid Encoder | Initializes the SSM pathway with shallow ResNet features and progressively propagates fused representations |
| OSSM | Omnidirectional Selective Scan Module | Models long-range dependencies through eight-directional selective scanning |
| DBFFB | Dual-Branch Feature Fusion Block | Fuses local CNN features and global SSM features using competitive spatial gating, channel recalibration, and residual learning |
| EE-SSM / EESSM | Edge-Enhanced State Space Module | Injects Sobel-derived edge features into shallow state-space representations through gated residual connections |
| MDFSIM | Multi-Directional Frequency-Spatial Interaction Module | Extracts directional frequency responses using image-conditioned Gaussian masks and interacts with spatial features |
| MSFEB | Multi-Scale Frequency Enhancement Block | Combines frequency-spatial interaction, spatial convolution, and feature fusion in the decoder |

The encoder produces four feature levels with 64, 128, 256, and 512 channels. The decoder enhances and aggregates multi-scale features, then predicts segmentation logits at the input image resolution.

The model is implemented in `MCFFENet/resnet_rsmssm.py`. Frequency enhancement components are implemented in `MCFFENet/MDFSIM.py`.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── resources/
│   ├── framework.png
│   └── framework.pdf
├── MCFFENet/
│   ├── resnet_rsmssm.py
│   ├── MDFSIM.py
│   ├── rs_mamba_ss.py
│   ├── train.py
│   ├── test.py
│   ├── loveda_test.py
│   ├── loveda_submit.py
│   ├── config/{vaihingen,potsdam,loveda}/
│   ├── tools/
│   │   ├── vaihingen_patch_split.py
│   │   └── potsdam_patch_split.py
│   └── utils/
├── semantic_segmentation_mamba_baseline/
└── change_detection_mamba/
```

Run environment installation commands from the repository root. Run preprocessing, training, and evaluation commands from `MCFFENet/`.

## Installation and Dependencies

### Create the Environment

The reference installation uses **Python 3.10, PyTorch 2.4.1, torchvision 0.19.1, and CUDA 12.1**. These versions were selected from the code dependencies; a complete training run in a freshly installed environment has not been validated.

```bash
conda create -n mcffenet python=3.10 -y
conda activate mcffenet
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

See the [official PyTorch installation instructions for previous versions](https://pytorch.org/get-started/previous-versions/) for supported PyTorch/torchvision combinations. If your hardware requires another combination, update both entries in `requirements.txt` and rebuild the CUDA extension.

The requirements cover the model, preprocessing, training, evaluation, and retained legacy utilities. Albumentations is pinned to 1.3.1 to match the patch splitting APIs. OpenCV uses the headless package for server environments.

### Install the CUDA Selective Scan Extension

The model uses `selective_scan_cuda_oflex` for its `forward_type="v2"` implementation. This extension must be compiled separately; its CUDA source is not included in this repository.

Training requires an NVIDIA GPU. Linux or WSL2 with GPU support is recommended for extension compilation. Install an NVIDIA driver, CUDA Toolkit with `nvcc`, Git, and a compatible C++ compiler. For the reference installation, use CUDA Toolkit 12.1 and set `CUDA_HOME` to its installation directory.

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

From the repository root, obtain and build the [VMamba selective scan extension](https://github.com/MzeroMiko/VMamba/tree/main/kernels/selective_scan):

```bash
git clone https://github.com/MzeroMiko/VMamba.git external/VMamba
python -m pip install --no-build-isolation ./external/VMamba/kernels/selective_scan
python -c "import selective_scan_cuda_oflex; print('selective_scan_cuda_oflex: OK')"
git -C external/VMamba rev-parse HEAD
```

Record the source revision printed by the last command for reproducibility. Installing `mamba-ssm` alone does not provide the oflex extension required by this model.

Verify the Python dependencies and model import:

```bash
python -m pip check
cd MCFFENet
python -c "import torch, timm, albumentations, cv2, einops, fvcore, ttach; from resnet_rsmssm import MCFFENet; print('Imports: OK')"
```

Configurations use `pretrained=True`. The first model construction may download SWSL-ResNet18 weights. Prepare the weight cache in advance for offline training.

## Datasets

- [ISPRS Vaihingen and Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)
- [LoveDA](https://github.com/Junjue-Wang/LoveDA)

Potsdam experiments use **IRRG imagery (infrared, red, and green)**. Use source image names such as `top_potsdam_2_10_IRRG.tif`.

Split the original scenes into training, validation, and testing sets before extracting patches. Keep patches from the same original scene in the same split.

### Directory Organization

```text
data/
├── vaihingen/
│   ├── train/
│   │   ├── images_1024/
│   │   └── masks_1024/
│   └── eroded_test/
│       ├── images_1024/
│       └── masks_1024/
├── potsdam/
│   ├── train/{images_1024,masks_1024}/
│   ├── val/{images_1024,masks_1024}/
│   └── test/{images_1024,masks_1024}/
└── loveda/
    ├── train/{images,masks}/
    ├── val/{images,masks}/
    └── test/images/
```

For Vaihingen and Potsdam, the splitting scripts produce `.tif` image patches and single-channel `.png` label patches with matching file stems. Label values 0-5 represent the six classes, and 6 is the ignore label.

## Data Preprocessing

After downloading the datasets, extract patches using the scripts below. All commands in this section run from `MCFFENet/`. If you are still at the repository root, run `cd MCFFENet` first.

The workflow follows the AFENet preprocessing instructions: generate training patches, generate testing patches, and optionally generate RGB ground truth for visualization.

### 1. Generate the Training Set

**Vaihingen:** use a patch size of 1024 and a stride of 512.

```bash
python tools/vaihingen_patch_split.py \
  --img-dir "data/vaihingen/train_images" \
  --mask-dir "data/vaihingen/train_masks" \
  --output-img-dir "data/vaihingen/train/images_1024" \
  --output-mask-dir "data/vaihingen/train/masks_1024" \
  --mode "train" --split-size 1024 --stride 512
```

**Potsdam:** use IRRG images, a patch size of 1024, and a stride of 1024.

```bash
python tools/potsdam_patch_split.py \
  --img-dir "data/potsdam/train_images" \
  --mask-dir "data/potsdam/train_masks" \
  --output-img-dir "data/potsdam/train/images_1024" \
  --output-mask-dir "data/potsdam/train/masks_1024" \
  --mode "train" --split-size 1024 --stride 1024
```

The Potsdam script reads IRRG images by default. All Potsdam preprocessing examples in this README use that default.

### 2. Generate the Testing Set

Use a patch size of 1024 and a stride of 1024. The following examples read existing eroded ground truth masks.

**Vaihingen:**

```bash
python tools/vaihingen_patch_split.py \
  --img-dir "data/vaihingen/test_images" \
  --mask-dir "data/vaihingen/test_masks_eroded" \
  --output-img-dir "data/vaihingen/eroded_test/images_1024" \
  --output-mask-dir "data/vaihingen/eroded_test/masks_1024" \
  --mode "val" --split-size 1024 --stride 1024 \
  --eroded
```

**Potsdam (IRRG):**

```bash
python tools/potsdam_patch_split.py \
  --img-dir "data/potsdam/test_images" \
  --mask-dir "data/potsdam/test_masks_eroded" \
  --output-img-dir "data/potsdam/test/images_1024" \
  --output-mask-dir "data/potsdam/test/masks_1024" \
  --mode "val" --split-size 1024 --stride 1024 \
  --eroded
```

The `--eroded` option selects existing boundary-eroded masks; it does not perform erosion. To use ordinary masks, change the input mask directory and omit this option.

The Potsdam configuration also expects a validation set. Extract its patches using:

```bash
python tools/potsdam_patch_split.py \
  --img-dir "data/potsdam/val_images" \
  --mask-dir "data/potsdam/val_masks" \
  --output-img-dir "data/potsdam/val/images_1024" \
  --output-mask-dir "data/potsdam/val/masks_1024" \
  --mode "val" --split-size 1024 --stride 1024
```

### 3. Generate RGB Ground Truth for Visualization

Add `--gt` to generate RGB label patches. Save them in a separate `masks_1024_rgb` directory so the single-channel labels used for training and evaluation remain available.

**Vaihingen:**

```bash
python tools/vaihingen_patch_split.py \
  --img-dir "data/vaihingen/test_images" \
  --mask-dir "data/vaihingen/test_masks_eroded" \
  --output-img-dir "data/vaihingen/eroded_test/images_1024_preview" \
  --output-mask-dir "data/vaihingen/eroded_test/masks_1024_rgb" \
  --mode "val" --split-size 1024 --stride 1024 \
  --eroded --gt
```

**Potsdam (IRRG images and RGB labels):**

```bash
python tools/potsdam_patch_split.py \
  --img-dir "data/potsdam/test_images" \
  --mask-dir "data/potsdam/test_masks_eroded" \
  --output-img-dir "data/potsdam/test/images_1024_preview" \
  --output-mask-dir "data/potsdam/test/masks_1024_rgb" \
  --mode "val" --split-size 1024 --stride 1024 \
  --eroded --gt
```

RGB refers to the visualization labels in this step; Potsdam source images remain IRRG.

### File Names and Split Arguments

Vaihingen images and ordinary masks must share the same name, for example `top_mosaic_09cm_area1.tif`. Eroded masks use `top_mosaic_09cm_area1_noBoundary.tif`.

Potsdam uses the following names:

```text
Image:         top_potsdam_2_10_IRRG.tif
Mask:          top_potsdam_2_10_label.tif
Eroded mask:   top_potsdam_2_10_label_noBoundary.tif
```

Keep only the corresponding imagery and masks in each input directory. The scripts read `*.tif` files and pair sorted image and mask lists.

| Argument | Description |
| --- | --- |
| `--img-dir` | Source image directory |
| `--mask-dir` | Source ground truth directory |
| `--output-img-dir` | Output image patch directory |
| `--output-mask-dir` | Output label patch directory |
| `--mode train` | Training preprocessing with augmentation |
| `--mode val` | Validation or testing preprocessing |
| `--split-size` | Patch side length |
| `--stride` | Sliding window stride |
| `--eroded` | Read existing eroded masks |
| `--gt` | Save RGB ground truth patches for visualization |

Commands use Bash line continuation. In PowerShell, put each complete command on one line or replace trailing backslashes with PowerShell backticks.

### LoveDA Preparation

LoveDA already provides image tiles, so the TIFF splitting scripts are unnecessary. Organize the Urban and Rural subsets into the `train`, `val`, and `test` directories shown above. If file names overlap, add matching `Urban_` or `Rural_` prefixes to both images and labels.

The configuration uses `mask_offset=1` to subtract one from raw label values. The current loader clips raw label value 0 to class 0, and the class-name order in the configuration differs from the evaluation palette. Check the label mapping and handling of invalid pixels before reporting LoveDA results.

## Training

### Configure the Dataset and Hyperparameters

Update `data_root` in the selected configuration to your actual dataset directory. The included configurations contain server-specific paths such as `/root/rivermind-data/data/vaihingen`.

For the example directory layout above, use:

```python
# Set this in the corresponding configuration.
data_root = 'data/vaihingen'
# data_root = 'data/potsdam'
# data_root = 'data/loveda'
```

| Configuration field | Description |
| --- | --- |
| `max_epoch` | Number of training epochs |
| `train_batch_size` / `val_batch_size` | Batch size; in DDP, this is the batch size per GPU |
| `lr` / `backbone_lr` | Learning rates for new modules and the pretrained backbone |
| `weight_decay` | Optimizer weight decay |
| `crop_size` in `BasicDataset(...)` | Online training crop size |
| `envss_num_blocks` | Number of SSM blocks at each encoder stage |
| `block_type` / `block_stages` | Edge-enhanced module type and enabled stages |
| `weights_path` / `weights_name` | Checkpoint directory and naming |
| `test_weights_name` | Evaluation checkpoint name without `.pth` |
| `resume_ckpt_path` | Checkpoint path for resuming training |

### Single-GPU Training

Choose the command for your dataset:

```bash
python train.py -c config/vaihingen/resnet_rsmssm_net.py
python train.py -c config/potsdam/resnet_rsmssm_net.py
python train.py -c config/loveda/resnet_rsmssm_net.py
```

The training script accepts `-c/--config_path`. Set epochs, learning rates, and batch sizes in the configuration file. On a machine with multiple GPUs, prefix a Bash command with `CUDA_VISIBLE_DEVICES=0` to select one GPU.

### Multi-GPU Training

```bash
torchrun --standalone --nproc_per_node=2 train.py -c config/vaihingen/resnet_rsmssm_net.py
```

Set `--nproc_per_node` to the number of GPUs. The current DDP implementation uses NCCL.

### Checkpoints and Resuming

Training saves `last.pth`. If validation is enabled and validation mIoU improves, it also saves `<weights_name>-best.pth`. Repeated runs may produce files with `-vN` suffixes.

The current Vaihingen configuration does not enable a validation loader and uses `last.pth` for testing. Potsdam and LoveDA configurations include validation loaders.

To resume training, set the following field and run the same training command:

```python
resume_ckpt_path = 'model_weights/vaihingen/MCFFENet-vaihingen-e120/last.pth'
```

## Testing and Prediction

### Vaihingen and Potsdam

```bash
python test.py -c config/vaihingen/resnet_rsmssm_net.py -o output/vaihingen --rgb
python test.py -c config/potsdam/resnet_rsmssm_net.py -o output/potsdam --rgb
```

### LoveDA

```bash
python loveda_test.py -c config/loveda/resnet_rsmssm_net.py -o output/loveda --rgb
```

| Argument | Description |
| --- | --- |
| `-c` | Configuration file |
| `-o` | Prediction output directory |
| `--rgb` | Save color predictions; omit it to save single-channel class indices |
| `-t lr` | Horizontal and vertical flip augmentation |
| `-t d4` | The implemented flip, rotation, and multi-scale test-time augmentation |

For example:

```bash
python test.py -c config/vaihingen/resnet_rsmssm_net.py -o output/vaihingen_tta -t d4 --rgb
```

Evaluation loads `weights_path/test_weights_name.pth`. If that file is absent, the scripts try the most recently modified `.pth` in the directory. Set the checkpoint name explicitly and inspect the loading log.

Vaihingen and Potsdam aggregate mIoU and F1 over the first five classes, excluding Clutter. The LoveDA evaluation script aggregates all seven classes. For LoveDA validation, set `test_split = 'val'`. Local metrics on the unlabeled test set are not meaningful because the loader supplies placeholder masks.

### LoveDA Submission

Generate single-channel predictions without `--rgb`, then package them:

```bash
python loveda_test.py -c config/loveda/resnet_rsmssm_net.py -o output/loveda_submission
python loveda_submit.py -i output/loveda_submission -o submission.zip
```

The packaging script keeps class indices 0-6, removes `Urban_` / `Rural_` prefixes, and creates a flat ZIP archive. Check class encoding and possible file name collisions before submission.

## Troubleshooting

- **Missing selective scan extension:** check that `selective_scan_cuda_oflex` is installed in the active environment and was compiled against the same PyTorch/CUDA installation.
- **Patch splitting file mismatch:** check input counts and naming conventions, including the IRRG suffix and eroded mask suffix.
- **Loading older AFEB checkpoints:** rename parameter keys from `AFEB` to `MSFEB` and from `afeb` to `msfeb` to match the current decoder attribute names.
- **Insufficient GPU memory:** reduce the training batch size or crop size, or disable multi-scale test-time augmentation.

## Acknowledgments

This project uses or builds on [RS-Mamba](https://github.com/walking-shadow/Official_Remote_Sensing_Mamba), [VMamba](https://github.com/MzeroMiko/VMamba), [timm](https://github.com/huggingface/pytorch-image-models), and related AFENet implementations. Please retain their source and copyright notices.

`README_Chinese.md` contains the original upstream RS-Mamba documentation. For MCFFENet, use this README and the current configuration files.

