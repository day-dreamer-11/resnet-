# ResNet 论文复现 — Deep Residual Learning for Image Recognition

本仓库是对 **何恺明等人《Deep Residual Learning for Image Recognition》(CVPR 2016)** 的复现记录，包含两条主线：

1. **手搓 CNN 理解残差** —— 从零手写卷积网络，在 MNIST / CIFAR-10 上做小规模实验，直观对比"有无残差连接"、"不同学习率"的效果（根目录下的 Python 脚本）。
2. **CIFAR-10 完整复现** —— 复现论文第 4.2 节的 CIFAR-10 实验（20/32/44/56 层 plain net 与 ResNet 对比），代码位于 `resnet/` 目录。

> `resnet/` 目录整理自开源实现 [a-martyn/resnet](https://github.com/a-martyn/resnet)（MIT 协议），并在此基础上做了修改与扩展（见下文"我做了哪些工作"）。论文本身不随仓库分发，链接见下。

---

## 论文

- **标题**：Deep Residual Learning for Image Recognition
- **作者**：Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun（Microsoft Research）
- **会议**：CVPR 2016
- **链接**：[arXiv:1512.03385](https://arxiv.org/abs/1512.03385) ｜ [CVPR Open Access](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

**核心思想**：当网络变深时，普通网络会出现退化（degradation）——训练误差反而升高。作者提出**残差块**，让网络学习残差映射 `F(x) = H(x) - x`，并通过 **恒等捷径连接（identity shortcut）** 直接把输入 `x` 加到输出上：`y = F(x) + x`。这样梯度可以顺着捷径无损回传，使上百层的网络也能正常训练并取得更低误差。

---

## 目录结构

```
.
├── README.md                  # 本文件
├── .gitignore
│
├── 手搓.py                    # 手写 3 层卷积 + 残差连接（MNIST，50 epoch）
├── 残差对比.py                # 对比「有 / 无残差连接」的训练曲线
├── 学习率对比.py              # 对比不同学习率对收敛的影响
├── 单张训练.py                # 两层 CNN 在 MNIST 上训练 + 单张样本推理
├── 单张汽车.py                # 取 CIFAR-10 中一张汽车图，用小型自编码器重建
│
├── residual_compare.png       # 残差对比结果图
├── residual_compare.log       # 残差对比训练日志
├── lr_compare.png             # 学习率对比结果图
├── car_memorize.png           # 单张汽车重建结果图
│
├── data/
│   └── mnist/raw/*.gz         # MNIST 压缩包（解压后的大文件不纳入版本管理）
│
└── resnet/                    # CIFAR-10 完整复现（参考 a-martyn/resnet）
    ├── resnet.py              # 残差网络与 plain net 定义
    ├── train.py / train1epoch.py / train20epoch.py
    ├── reproduce.py           # 一键复现训练
    ├── compare50.py           # 50 层 plain vs resnet 对比
    ├── plot_loss.py / plot_compare50.py
    ├── analyze_gradflow.py    # 梯度回传分析
    ├── cifar10_npy.py         # 把 CIFAR-10 转成 .npy
    ├── data_loader.py / utils.py
    ├── main.ipynb             # 主训练 notebook
    ├── docs/                  # 原论文图表复现说明 notebook
    ├── assets/                # 论文 Figure 6 等图片
    ├── pretrained/            # 训练好的权重（.pt）
    ├── results/               # 训练结果 csv / 图片
    └── requirements.txt
```

---

## 环境依赖

- Python 3.12
- PyTorch + torchvision（建议按显卡 CUDA 版本单独安装，见 [pytorch.org](https://pytorch.org/get-started/locally/)）
- 其余依赖：

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

（也可直接 `pip install -r resnet/requirements.txt`，其中 torch/torchvision 特意未固定版本，避免覆盖已有的 GPU 版本。）

---

## 数据说明

本仓库**不包含**解压后的完整数据集（体积过大，GitHub 单文件上限 100MB）：

- **MNIST**：仓库保留了压缩包 `data/mnist/raw/*.gz`。运行脚本时 `torchvision.datasets.MNIST(root="./data", download=True)` 会自动识别并解压。
- **CIFAR-10**：仓库保留了原始压缩包 `resnet/data/cifar10/cifar-10-python.tar.gz`。`resnet/` 下的脚本会自动下载或从该压缩包解压，`cifar10_npy.py` 可将其转换为 `.npy` 加速读取。

首次运行若本地无数据，代码会自动从官方源下载。

---

## 复现结果

### 手搓 CNN（根目录脚本）

- `手搓.py`：3 层卷积各 64 通道，在第 2、3 层分别加入恒等残差连接，MNIST 训练 50 epoch，观察 loss 收敛。
- `残差对比.py`：同一网络分别开启 / 关闭残差连接，对比训练曲线（见 `residual_compare.png`）。

### CIFAR-10 完整复现（`resnet/` 目录）

对照论文 Figure 6 与 Table，分类错误率（% test error）如下：

| 架构 | 层数 | 论文报告 | 本实现 |
| --- | --- | --- | --- |
| Plain Net | 20 | 9.5\* | 9.5 |
| Plain Net | 32 | 10\* | 9.92 |
| Plain Net | 44 | 12\* | 11.35 |
| Plain Net | 56 | 13.5\* | 12.76 |
| ResNet | 20 | 8.75 | 8.0 |
| ResNet | 32 | 7.51 | 7.51 |
| ResNet | 44 | 7.17 | 7.38 |
| ResNet | 56 | 6.97 | 7.33 |

\* 论文 Figure 6 的近似读数。

**结论**：残差连接让最深的 56 层网络表现最好，而同深度 plain net 表现最差；对 20 层小网络，残差同样带来提升——与论文结论一致。

---

## 我做了哪些工作

在 `resnet/` 参考实现基础上（对比原仓库的差异）：

- 修改 `resnet.py`、`train.py`、`utils.py`、`main.ipynb`、`requirements.txt`；
- 新增训练 / 对比脚本：`reproduce.py`、`compare50.py`、`train1epoch.py`、`train20epoch.py`、`plot_loss.py`、`plot_compare50.py`、`analyze_gradflow.py`、`cifar10_npy.py`；
- 新增 50 层网络（plain / resnet）的训练权重与结果：`pretrained/*50*.pt`、`results/*50*.csv`、`results/compare50.png`、`results/loss50.png`；
- 根目录下从零手写的 CNN 与对比实验脚本。

---

## 参考与致谢

- 论文：K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- 参考实现：[a-martyn/resnet](https://github.com/a-martyn/resnet)（MIT License，原仓库 LICENSE 见 `resnet/LICENSE`）。
- 数据集：[MNIST](http://yann.lecun.com/exdb/mnist/)、[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)。
