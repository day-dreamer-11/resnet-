# 导入 PyTorch 主库，提供张量、自动求导、设备管理等基础能力
import torch
# 从 torch 中导入神经网络模块 nn，用于构建网络层
from torch import nn
# 导入函数式接口（本文件中其实没有用到 F，属于多余导入，可忽略）
import torch.nn.functional as F

"""
（以下是原作者关于“卷积层是否需要偏置”的说明）

Questions?

Should the convolutional layers have biases?
Paper reads: "biases are omitted for simplifying notation" (p.3) which is ambiguous.

Clarification from Kaiming He here:
>"Biases are in the BN layers that follow." (https://github.com/KaimingHe/deep-residual-networks/issues/10)

This can be accomplished in PyTorch BatchNorm2d by setting affine=True (default).
"""


# 定义残差块（Residual Block）类，它是 ResNet 的基本构建单元
class block(nn.Module):
    # 构造函数：filters 表示该块中所有卷积层的通道数；subsample 表示是否做下采样
    def __init__(self, filters, subsample=False):
        # 调用父类 nn.Module 的构造函数，完成模块初始化
        super().__init__()
        """
        （原作者的 docstring 说明）
        A 2-layer residual learning building block as illustrated by Fig.2
        in "Deep Residual Learning for Image Recognition"
        ...
        """

        # 根据是否下采样决定缩放系数 s：
        # 下采样时 s=0.5（输入通道数减半、步长翻倍）；否则 s=1.0（不变）
        s = 0.5 if subsample else 1.0

        # 第一个卷积层：输入通道 = filters*s（下采样时是上一块的一半），输出通道 = filters
        # 卷积核 3x3，步长 = 1/s（下采样时为 2，否则为 1），padding=1 保持尺寸，无偏置
        self.conv1 = nn.Conv2d(int(filters*s), filters, kernel_size=3,
                               stride=int(1/s), padding=1, bias=False)
        # 第一个批归一化层，track_running_stats=True 表示训练时统计并保存 running mean/std（推理时使用）
        self.bn1   = nn.BatchNorm2d(filters, track_running_stats=True)
        # 第一个 ReLU 激活函数
        self.relu1 = nn.ReLU()
        # 第二个卷积层：输入输出通道都是 filters，卷积核 3x3，步长 1，padding=1，无偏置
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        # 第二个批归一化层
        self.bn2   = nn.BatchNorm2d(filters, track_running_stats=True)
        # 第二个 ReLU 激活函数（放在残差相加之后）
        self.relu2 = nn.ReLU()

        # 捷径连接（shortcut）专用的下采样层：1x1 平均池化、步长 2，用于维度不匹配时缩小输入
        self.downsample = nn.AvgPool2d(kernel_size=1, stride=2)

        # 按 He 等人的方法初始化权重（参考 "Delving deep into rectifiers" 论文）
        # 遍历本模块内所有子模块
        for m in self.modules():
            # 如果子模块是卷积层
            if isinstance(m, nn.Conv2d):
                # 用 kaiming_normal_ 初始化卷积权重，fan_out 模式 + relu 非线性
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # 如果子模块是批归一化层
            elif isinstance(m, nn.BatchNorm2d):
                # 将 BN 的缩放系数 weight 初始化为 1
                nn.init.constant_(m.weight, 1)
                # 将 BN 的偏置 bias 初始化为 0
                nn.init.constant_(m.bias, 0)

    # 捷径连接方法：z 是块输出（残差函数 F(x)），x 是块的原始输入
    def shortcut(self, z, x):
        """（原作者的 docstring 说明）实现无参数的恒等映射捷径连接，维度不匹配时下采样 + 零填充"""
        # 如果输入 x 的形状和输出 z 的形状不一致（发生了下采样导致通道/尺寸变化）
        if x.shape != z.shape:
            # 先对输入 x 做 1x1/stride=2 的平均池化，使空间尺寸减半
            d = self.downsample(x)
            # 生成一个和 d 形状相同但全为 0 的张量 p
            p = torch.mul(d, 0)
            # 把下采样后的 d 和全零的 p 在通道维（dim=1）拼接，通道数翻倍，再与 z 相加
            return z + torch.cat((d, p), dim=1)
        # 如果形状一致，直接做恒等映射相加（z + x）
        else:
            return z + x

    # 前向传播：x 是输入张量，shortcuts 控制是否启用残差捷径连接
    def forward(self, x, shortcuts=False):
        # 第一层卷积
        z = self.conv1(x)
        # 第一个批归一化
        z = self.bn1(z)
        # 第一个 ReLU 激活
        z = self.relu1(z)

        # 第二层卷积
        z = self.conv2(z)
        # 第二个批归一化
        z = self.bn2(z)

        # 残差捷径连接
        # 下面这个 if 判断是“普通卷积网络”和“ResNet”之间的唯一区别！
        # 如果启用捷径连接，就把输入 x 通过 shortcut 加到输出 z 上
        if shortcuts:
            z = self.shortcut(z, x)

        # 第二个 ReLU 激活（对残差相加后的结果做非线性）
        z = self.relu2(z)

        # 返回块输出
        return z


# 定义 ResNet 类：由输入层 + 三个堆叠(stack) + 输出层组成
class ResNet(nn.Module):
    # 构造函数：n 决定每个 stack 的块数量（网络深度 = 6n+2 层）；shortcuts 控制是否启用残差
    def __init__(self, n, shortcuts=True):
        # 调用父类构造函数
        super().__init__()
        # 保存 shortcuts 标志，供 forward 使用
        self.shortcuts = shortcuts

        # ---- 输入层 ----
        # 输入卷积：3 通道(RGB) -> 16 通道，3x3 卷积核，步长 1，padding=1，无偏置
        self.convIn = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # 输入批归一化（16 通道）
        self.bnIn   = nn.BatchNorm2d(16, track_running_stats=True)
        # 输入 ReLU 激活
        self.relu   = nn.ReLU()

        # ---- Stack1（第一组，16 通道，不做下采样）----
        # n 个普通残差块，每个都是 16 通道、不下采样
        self.stack1 = nn.ModuleList([block(16, subsample=False) for _ in range(n)])

        # ---- Stack2（第二组，32 通道）----
        # 第一个块做下采样（通道 16->32，空间尺寸 32x32->16x16）
        self.stack2a = block(32, subsample=True)
        # 剩余 n-1 个块为普通块（32 通道，不下采样）
        self.stack2b = nn.ModuleList([block(32, subsample=False) for _ in range(n-1)])

        # ---- Stack3（第三组，64 通道）----
        # 第一个块做下采样（通道 32->64，空间尺寸 16x16->8x8）
        self.stack3a = block(64, subsample=True)
        # 剩余 n-1 个块为普通块（64 通道，不下采样）
        self.stack3b = nn.ModuleList([block(64, subsample=False) for _ in range(n-1)])

        # ---- 输出层 ----
        # 自适应平均池化：把任意大小的特征图统一池化成 1x1（得到 64 个标量）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层：64 维 -> 10 维（CIFAR-10 的 10 个类别），带偏置
        self.fcOut   = nn.Linear(64, 10, bias=True)
        # LogSoftmax：对输出取对数概率（配合 NLLLoss 使用）
        self.softmax = nn.LogSoftmax(dim=-1)

        # 初始化全连接层的权重
        # 遍历本模块内所有子模块
        for m in self.modules():
            # 如果子模块是全连接层
            if isinstance(m, nn.Linear):
                # 用 kaiming_normal 初始化全连接层权重
                nn.init.kaiming_normal(m.weight)
                # 将全连接层偏置全部置为 0
                m.bias.data.zero_()

    # 前向传播：x 是输入图像批次
    def forward(self, x):
        # 输入卷积
        z = self.convIn(x)
        # 输入批归一化
        z = self.bnIn(z)
        # 输入 ReLU 激活
        z = self.relu(z)

        # 依次经过 Stack1 的每个残差块
        for l in self.stack1: z = l(z, shortcuts=self.shortcuts)

        # 经过 Stack2 的第一个（下采样）块
        z = self.stack2a(z, shortcuts=self.shortcuts)
        # 依次经过 Stack2 的剩余块
        for l in self.stack2b:
            z = l(z, shortcuts=self.shortcuts)

        # 经过 Stack3 的第一个（下采样）块
        z = self.stack3a(z, shortcuts=self.shortcuts)
        # 依次经过 Stack3 的剩余块
        for l in self.stack3b:
            z = l(z, shortcuts=self.shortcuts)

        # 自适应平均池化，把特征图变成 1x1
        z = self.avgpool(z)
        # 展平：把 (batch, 64, 1, 1) 变成 (batch, 64)
        z = z.view(z.size(0), -1)
        # 全连接层：64 -> 10
        z = self.fcOut(z)
        # 返回 LogSoftmax 后的对数概率
        return self.softmax(z)
