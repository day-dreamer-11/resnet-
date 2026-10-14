"""训练并对比 50 层 ResNet（有残差）与 PlainNet（无残差）在 CIFAR-10 上的精度。

n=8 时网络深度 = 6n+2 = 50 层。
用法：python compare50.py
"""
import torch
import torch.optim as optim
from torchvision import transforms

from resnet import ResNet
from cifar10_npy import get_data_loaders
from train import train

data_dir = 'data/cifar10'
batch_size = 128
epochs = 164
lr = 0.1
momentum = 0.9
weight_decay = 0.0001
milestones = [82, 123]
gamma = 0.1

means = [0.4918687901200927, 0.49185976472299225, 0.4918583862227116]
stds = [0.24697121702736, 0.24696766978537033, 0.2469719877121087]

normalize = transforms.Normalize(mean=means, std=stds)

train_transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    normalize,
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


def run(shortcuts, n):
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
    n = 8  # 50 层
    print('===== 第一步：训练 PlainNet50（无残差连接）=====', flush=True)
    run(shortcuts=False, n=n)
    print('\n===== 第二步：训练 ResNet50（有残差连接）=====', flush=True)
    run(shortcuts=True, n=n)
    print('\n两个 50 层网络训练完成！', flush=True)
