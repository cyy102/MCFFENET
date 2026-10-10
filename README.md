# MCFFENet: Mamba-CNN Fusion and Frequency Enhancement Network

MCFFENet 面向高分辨率遥感影像语义分割，通过渐进式 CNN-SSM 协同编码、边缘增强和空间-频率交互，联合建模局部细节、长距离上下文与频率信息。本项目提供 Vaihingen、Potsdam 和 LoveDA 的训练与测试配置。

<div align="center">
  <img src="resources/framework.png" alt="MCFFENet overall architecture" width="100%" />
</div>

[查看矢量架构图 PDF](resources/framework.pdf)

## 模型架构

| 模块 | 全称 | 作用 |
| --- | --- | --- |
| PCSHE | Progressive CNN-SSM Hybrid Encoder | 以浅层 ResNet 特征初始化 SSM 路径，逐级传递融合特征；四级通道数为 64、128、256、512 |
| OSSM | Omnidirectional Selective Scan Module | 通过八方向选择性扫描建模全局上下文 |
| DBFFB | Dual-Branch Feature Fusion Block | 通过竞争式空间门控、通道重标定和残差连接融合 CNN 与 SSM 特征 |
| EE-SSM / EESSM | Edge-Enhanced State Space Module | 在浅层阶段通过门控残差引入 Sobel 边缘特征 |
| MDFSIM | Multi-Directional Frequency-Spatial Interaction Module | 使用图像条件化的方向性高斯掩码提取频率响应，并与空间特征交互 |
| MSFEB | Multi-Scale Frequency Enhancement Block | 整合 MDFSIM、空间卷积和特征融合，用于解码器的频率增强 |
| TB | Transformer Block | 在解码阶段细化特征 |

模型定义位于 `MCFFENet/resnet_rsmssm.py`，频率模块位于 `MCFFENet/MDFSIM.py`。编码器由 ResNet、`PCSHEStage` 和 DBFFB 共同组成；解码器包含 `MSFEB1/2/3`、TB、多尺度聚合和分割头，最终输出与输入影像同尺寸的类别 logits。

## 项目结构

~~~text
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
~~~

环境安装从项目根目录执行。数据处理、训练和测试命令统一在 `MCFFENet/` 下执行。

## 环境安装

### 1. Python 和 PyTorch

参考安装组合为 **Python 3.10、PyTorch 2.4.1、torchvision 0.19.1、CUDA 12.1**。这是根据代码依赖整理的安装方案，尚未在全新环境完成端到端训练验证。训练需要 NVIDIA GPU，CUDA 扩展建议在 Linux 或具备 GPU 支持的 WSL2 中编译；原生 Windows 的扩展编译需自行适配。

~~~bash
conda create -n mcffenet python=3.10 -y
conda activate mcffenet
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
~~~

PyTorch/torchvision 的版本配对和 CUDA wheel 安装方式见 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。若硬件需要其他版本，应同步调整 `requirements.txt` 中这两个包，并重新编译扫描扩展。

依赖清单覆盖模型、切图、训练、测试以及仓库保留的旧工具。Albumentations 固定为 1.3.1，以匹配切图脚本中的 `PadIfNeeded(value=...)` 和 `RandomSizedCrop(width=..., height=...)` 接口。OpenCV 使用 headless 版本，支持 `cv2.imread/imwrite`，适合服务器环境。

### 2. CUDA Selective Scan 扩展

当前模型的 `forward_type="v2"` 实际调用 **`selective_scan_cuda_oflex`**。该扩展需要单独编译，仓库未附带其 C++/CUDA 源码。

系统需要 NVIDIA 驱动、CUDA Toolkit（包含 `nvcc`）、Git 和兼容的 C++ 编译器。在上述参考环境中，Toolkit 使用 12.1，并设置 `CUDA_HOME` 指向其安装目录。PyTorch wheel 自带 CUDA 运行时，不等于已安装 `nvcc`。

~~~bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
~~~

从项目根目录获取 [VMamba 扫描扩展源码](https://github.com/MzeroMiko/VMamba/tree/main/kernels/selective_scan)，然后编译：

~~~bash
git clone https://github.com/MzeroMiko/VMamba.git external/VMamba
python -m pip install --no-build-isolation ./external/VMamba/kernels/selective_scan
python -c "import selective_scan_cuda_oflex; print('selective_scan_cuda_oflex: OK')"
git -C external/VMamba rev-parse HEAD
~~~

保留最后一条命令输出的源码版本，便于复现。该入口当前默认构建 oflex，以安装后的导入检查为准。单独安装 `mamba-ssm` 不能替代本模型要求的 oflex 扩展。

安装完成后检查依赖：

~~~bash
python -m pip check
cd MCFFENet
python -c "import torch, timm, albumentations, cv2, einops, fvcore, ttach; from resnet_rsmssm import MCFFENet; print('Imports: OK')"
~~~

默认 `pretrained=True`，首次构建 SWSL-ResNet18 时需要获取预训练权重。离线运行需预先准备权重缓存；仅检查模型结构时可使用 `pretrained=False`。

## 数据集准备

数据来源：[ISPRS Vaihingen / Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)、[LoveDA](https://github.com/Junjue-Wang/LoveDA)。

先按原始影像划分训练、验证和测试区域，再运行切图脚本，避免同一原图的相邻切块进入不同集合。下文命令均在 `MCFFENet/` 中执行；如果仍在项目根目录，先执行 `cd MCFFENet`。

### Vaihingen / Potsdam 目录

~~~text
data/
├── vaihingen/
│   ├── train/{images_1024,masks_1024}/
│   └── eroded_test/{images_1024,masks_1024}/
└── potsdam/
    ├── train/{images_1024,masks_1024}/
    ├── val/{images_1024,masks_1024}/
    └── test/{images_1024,masks_1024}/
~~~

切图脚本生成 `.tif` 影像和单通道 `.png` 标签，二者文件主名相同。标签值 0-5 为六个类别，6 为忽略区域。训练和评估使用类别索引标签，RGB 标签用于可视化。

### Split 参数

| 参数 | 含义 |
| --- | --- |
| `--img-dir` / `--mask-dir` | 原始影像 / RGB 标签目录，只读取目录下的 `*.tif` |
| `--output-img-dir` / `--output-mask-dir` | 切块影像 / 标签输出目录 |
| `--mode train` | 训练切图，包含原图和水平、垂直翻转；Vaihingen 还会增强部分车辆区域 |
| `--mode val` | 验证或测试切图，不执行训练翻转增强 |
| `--split-size` | 切块边长，例如 1024 |
| `--stride` | 滑窗步长；小于切块尺寸时产生重叠切块 |
| `--eroded` | 读取已存在的边界忽略标签；此选项本身不执行腐蚀运算 |
| `--gt` | 输出 RGB 标签，应使用独立输出目录 |
| `--rgb-image` | 仅 Potsdam：读取 `_RGB.tif`；未指定时读取 `_IRRG.tif` |
| `--val-scale` | 非训练模式的缩放比例，默认 1.0 |

**推荐切图设置：** Vaihingen 训练集使用 `split-size=1024, stride=512`；Potsdam 训练集使用 `split-size=1024, stride=1024`；验证和测试均使用 1024 步长。训练时加载器还会进行在线随机裁剪。

以下多行命令采用 Bash 续行格式。PowerShell 中可以将一个完整命令写成一行，或把行末的 `\` 改为反引号。

### Vaihingen 切图

影像和普通标签须使用相同文件主名，例如 `top_mosaic_09cm_area1.tif`。边界忽略标签须命名为 `top_mosaic_09cm_area1_noBoundary.tif`。输入目录只放本次对应的文件，检查影像和标签数量及排序后的配对关系。

训练集：

~~~bash
python tools/vaihingen_patch_split.py \
  --img-dir data/vaihingen/train_images \
  --mask-dir data/vaihingen/train_masks \
  --output-img-dir data/vaihingen/train/images_1024 \
  --output-mask-dir data/vaihingen/train/masks_1024 \
  --mode train --split-size 1024 --stride 512
~~~

测试集（读取现成 eroded 标签）：

~~~bash
python tools/vaihingen_patch_split.py \
  --img-dir data/vaihingen/test_images \
  --mask-dir data/vaihingen/test_masks_eroded \
  --output-img-dir data/vaihingen/eroded_test/images_1024 \
  --output-mask-dir data/vaihingen/eroded_test/masks_1024 \
  --mode val --split-size 1024 --stride 1024 --eroded
~~~

如使用普通标签，改用普通标签目录并移除 `--eroded`。独立验证集应先划分原图，再以 `--mode val` 切至 `val/images_1024` 和 `val/masks_1024`，并在配置中启用 `val_dataset`、`val_loader`。

### Potsdam 切图

文件示例：影像 `top_potsdam_2_10_RGB.tif`，普通标签 `top_potsdam_2_10_label.tif`，eroded 标签 `top_potsdam_2_10_label_noBoundary.tif`。使用 IRRG 时，影像后缀为 `_IRRG.tif` 并移除 `--rgb-image`。输入目录不要混放 RGB 和 IRRG 两套影像。

训练集：

~~~bash
python tools/potsdam_patch_split.py \
  --img-dir data/potsdam/train_images \
  --mask-dir data/potsdam/train_masks \
  --output-img-dir data/potsdam/train/images_1024 \
  --output-mask-dir data/potsdam/train/masks_1024 \
  --mode train --split-size 1024 --stride 1024 --rgb-image
~~~

验证集：

~~~bash
python tools/potsdam_patch_split.py \
  --img-dir data/potsdam/val_images \
  --mask-dir data/potsdam/val_masks \
  --output-img-dir data/potsdam/val/images_1024 \
  --output-mask-dir data/potsdam/val/masks_1024 \
  --mode val --split-size 1024 --stride 1024 --rgb-image
~~~

测试集（读取现成 eroded 标签）：

~~~bash
python tools/potsdam_patch_split.py \
  --img-dir data/potsdam/test_images \
  --mask-dir data/potsdam/test_masks_eroded \
  --output-img-dir data/potsdam/test/images_1024 \
  --output-mask-dir data/potsdam/test/masks_1024 \
  --mode val --split-size 1024 --stride 1024 --rgb-image --eroded
~~~

### 生成 RGB Ground Truth

添加 `--gt` 会把输出标签改为 RGB。请写入独立目录，保留训练和评估所需的单通道标签：

~~~bash
python tools/potsdam_patch_split.py \
  --img-dir data/potsdam/test_images \
  --mask-dir data/potsdam/test_masks_eroded \
  --output-img-dir data/potsdam/test/images_1024_rgb_preview \
  --output-mask-dir data/potsdam/test/masks_1024_rgb \
  --mode val --split-size 1024 --stride 1024 --rgb-image --eroded --gt
~~~

Vaihingen 使用对应的 `vaihingen_patch_split.py`、影像及标签目录，移除 `--rgb-image`，也将 RGB 标签写入单独目录。Vaihingen 脚本中的额外原图标签路径使用 Windows 反斜杠拼接，Linux 下该辅助目录名可能不符合预期；切块 RGB 标签仍写入 `--output-mask-dir`。

### LoveDA 数据组织

LoveDA 已提供切块，不需要运行上述 TIFF split。将 Urban、Rural 两部分按官方 train/val/test 划分整理为：

~~~text
data/loveda/
├── train/
│   ├── images/*.png
│   └── masks/*.png
├── val/
│   ├── images/*.png
│   └── masks/*.png
└── test/
    └── images/*.png
~~~

合并时如有同名文件，可给影像及对应标签同时添加 `Urban_` / `Rural_` 前缀。配置使用 `mask_offset=1`，加载有标签的数据时将像素值减 1。

当前加载器会把原始标签值 0 裁剪为类别 0；若原始标签含无效区域，需要核对忽略区域处理。LoveDA 配置的 `CLASSES` 顺序与测试脚本调色板顺序也存在差异，正式评估前应按所用标签编码核对。以上文档说明当前代码行为，未改动类别映射。

## 配置与训练

运行前修改对应配置中的 `data_root`，例如在 `MCFFENet/` 下使用 `data_root = 'data/vaihingen'`。现有配置里的 `/root/rivermind-data/data/...` 是服务器路径，需替换为自己的目录。

| 配置项 | 说明 |
| --- | --- |
| `data_root`、`img_dir`、`mask_dir` | 数据根目录和子目录 |
| `train_split`、`val_split`、`test_split` | 训练、验证及测试目录名 |
| `max_epoch` | 总训练轮数 |
| `train_batch_size`、`val_batch_size` | 批大小；DDP 时为每张 GPU 的批大小 |
| `lr`、`backbone_lr`、`weight_decay` | 模块及骨干学习率、权重衰减 |
| `crop_size` | 在 `BasicDataset(...)` 中设置的训练裁剪尺寸 |
| `envss_num_blocks` | SSM 阶段深度，可传入四元组 |
| `block_type`、`block_stages` | EE-SSM 类型及启用阶段 |
| `weights_path`、`weights_name` | 权重保存目录及命名 |
| `test_weights_name` | 测试权重主名，不包含 `.pth` |
| `resume_ckpt_path` | 恢复训练的权重路径，默认 `None` |

### 单 GPU 训练

~~~bash
python train.py -c config/vaihingen/resnet_rsmssm_net.py
python train.py -c config/potsdam/resnet_rsmssm_net.py
python train.py -c config/loveda/resnet_rsmssm_net.py
~~~

按需选择一条命令。`train.py` 只接受 `-c/--config_path`；轮数、学习率和批大小在配置中修改，没有 `--max_epoch`、`--lr`、`--batch_size` 命令行选项。多 GPU 机器如只使用一张卡，可在 Bash 命令前加 `CUDA_VISIBLE_DEVICES=0`。

### 多 GPU 训练

~~~bash
torchrun --standalone --nproc_per_node=2 train.py -c config/vaihingen/resnet_rsmssm_net.py
~~~

将 `2` 改为 GPU 数量，当前 DDP 使用 NCCL 后端。

### 权重和断点续训

训练保存 `last.pth`；启用验证且验证 mIoU 提升时，保存 `<weights_name>-best.pth`。已有同名文件时，新训练会生成带 `-vN` 后缀的权重，测试前应设置对应的 `test_weights_name`。

Vaihingen 当前配置未启用 `val_loader`，不会生成按验证指标选择的 best 权重，默认测试 `last.pth`。Potsdam、LoveDA 配置包含验证加载器，需准备验证数据。

在配置中设置恢复路径后，重新执行原训练命令：

~~~python
resume_ckpt_path = 'model_weights/vaihingen/MCFFENet-vaihingen-e120/last.pth'
~~~

## 测试与预测

Vaihingen / Potsdam：

~~~bash
python test.py -c config/vaihingen/resnet_rsmssm_net.py -o output/vaihingen --rgb
python test.py -c config/potsdam/resnet_rsmssm_net.py -o output/potsdam --rgb
~~~

LoveDA：

~~~bash
python loveda_test.py -c config/loveda/resnet_rsmssm_net.py -o output/loveda --rgb
~~~

| 参数 | 说明 |
| --- | --- |
| `-c` | 配置文件路径 |
| `-o` | 预测结果输出目录 |
| `--rgb` | 保存彩色图；不指定则保存单通道类别索引图 |
| `-t lr` | 水平、垂直翻转测试增强 |
| `-t d4` | 当前实现包含翻转、90 度旋转和多尺度测试增强 |

~~~bash
python test.py -c config/vaihingen/resnet_rsmssm_net.py -o output/vaihingen_tta -t d4 --rgb
~~~

Vaihingen/Potsdam 汇总 mIoU、F1 时排除最后的 Clutter 类；LoveDA 测试脚本汇总七类。对没有标签的 LoveDA test 集，加载器生成占位 mask，打印的本地指标没有评估意义。对带标签的 val 集评估时，将 LoveDA 配置的 `test_split` 改为 `'val'`。

测试优先读取 `weights_path/test_weights_name.pth`；不存在时尝试目录中最近修改的 `.pth`。应明确指定权重并核对加载日志。`test.py` 使用宽松权重加载，出现缺失或多余参数键时，需要处理后再报告结果。

### LoveDA 提交文件

生成提交预测时不加 `--rgb`，随后打包：

~~~bash
python loveda_test.py -c config/loveda/resnet_rsmssm_net.py -o output/loveda_submission
python loveda_submit.py -i output/loveda_submission -o submission.zip
~~~

打包脚本保留 0-6 类别索引，移除 `Urban_` / `Rural_` 前缀并生成平铺 ZIP。提交前核对类别映射、平台文件名要求，以及移除前缀后是否存在同名文件。

## 常见问题

- **扫描扩展未定义或导入失败：** 确认 `selective_scan_cuda_oflex` 安装在当前 conda 环境，且编译时的 PyTorch/CUDA 与运行环境一致。代码会捕获导入失败，错误可能直到第一次前向传播才出现。
- **切图文件名不匹配：** 检查后缀和 `--rgb-image`、`--eroded` 选项；脚本按排序后的位置配对，不会自动补齐遗漏影像。
- **加载旧 AFEB 权重：** 当前模块已改为 `MSFEB1/2/3` 和 `msfeb_squeeze1/2/3`。需把检查点参数键中的 `AFEB` 替换为 `MSFEB`、`afeb` 替换为 `msfeb`。
- **显存不足：** 降低训练批大小或裁剪尺寸；测试时先关闭多尺度 TTA。复现实验时应核对实际阶段深度和裁剪尺寸，配置文件是具体参数的依据。

## 致谢

本项目使用或参考了 [RS-Mamba](https://github.com/walking-shadow/Official_Remote_Sensing_Mamba)、[VMamba](https://github.com/MzeroMiko/VMamba)、[timm](https://github.com/huggingface/pytorch-image-models) 及 AFENet 的相关实现。请保留相应代码的来源和版权说明。`README_Chinese.md` 保留原始 RS-Mamba 项目说明，MCFFENet 的运行方法以本文件和当前代码为准。

