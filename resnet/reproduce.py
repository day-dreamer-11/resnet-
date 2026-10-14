"""
ResNet 论文复现脚本：Deep Residual Learning for Image Recognition
在 CIFAR-10 上训练 PlainNet 和 ResNet（各 20/32/44/56 层），复现论文 Figure 6 / Table 6。

用法：
    python reproduce.py

训练参数与超参数均来自原论文（arXiv:1512.03385）及 main.ipynb。
"""
import torch
import torch.optim as optim
from torchvision import transforms

from resnet import ResNet
from data_loader import get_data_loaders
from train import train

# ---- 全局配置（与论文 / main.ipynb 一致）----
data_dir = 'data/cifar10'
batch_size = 128
epochs = 164                      # 论文的 64k 次迭代 ≈ 164 个 epoch
lr = 0.1                          # 论文初始学习率
momentum = 0.9
weight_decay = 0.0001
milestones = [82, 123]            # 对应论文在 32k / 48k 迭代处将学习率除以 10
gamma = 0.1

# 归一化参数（CIFAR-10 训练+测试集合并计算得到，见 main.ipynb）
means = [0.4918687901200927, 0.49185976472299225, 0.4918583862227116]
stds = [0.24697121702736, 0.24696766978537033, 0.2469719877121087]

normalize = transforms.Normalize(mean=means, std=stds)

# 训练集增强：四周各 pad 4 像素 → 随机水平翻转 → 随机裁剪回 32×32
train_transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    normalize,
])

# 测试集：不做增强，仅归一化
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


def run(shortcuts, n):
    """训练单个模型并保存结果 CSV 与最佳权重。
    shortcuts=True 为 ResNet，False 为 PlainNet。"""
    name = 'resnet' if shortcuts else 'plainnet'
    depth = 6 * n + 2
    print(f'\n===== 训练 {name}{depth} (n={n}, 共 {depth} 层) =====', flush=True)

    train_loader, test_loader = get_data_loaders(
        data_dir, batch_size, train_transform, test_transform,
        shuffle=True, num_workers=0, pin_memory=True)

    model = ResNet(n, shortcuts=shortcuts)
    criterion = torch.nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=milestones, gamma=gamma)

    results_file = f'results/{name}{depth}.csv'
    model_file = f'pretrained/{name}{depth}.pt'

    train(model, epochs, train_loader, test_loader, criterion,
          optimizer, results_file, scheduler=scheduler, MODEL_PATH=model_file)


if __name__ == '__main__':
    ns = [3, 5, 7, 9]  # 对应 20 / 32 / 44 / 56 层网络

    print('===== 第一步：训练 4 个 PlainNet（普通卷积网络）=====', flush=True)
    for n in ns:
        run(shortcuts=False, n=n)

    print('\n===== 第二步：训练 4 个 ResNet（残差网络）=====', flush=True)
    for n in ns:
        run(shortcuts=True, n=n)

    print('\n全部 8 个模型训练完成！', flush=True)
