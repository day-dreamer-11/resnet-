"""PlainNet50 与 ResNet50 各训练 20 个 epoch（GPU），保存结果 CSV 与最优权重。"""
import torch
import torch.optim as optim
from torchvision import transforms

from resnet import ResNet
from cifar10_npy import get_data_loaders
from train import train

data_dir = 'data/cifar10'
batch_size = 128
epochs = 20
lr = 0.1
momentum = 0.9
weight_decay = 0.0001

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
test_transform = transforms.Compose([transforms.ToTensor(), normalize])


def run(shortcuts):
    name = 'resnet' if shortcuts else 'plainnet'
    print(f'\n===== 训练 {name}50 (n=8, 共 50 层, {epochs} epochs) =====', flush=True)

    train_loader, test_loader = get_data_loaders(
        data_dir, batch_size, train_transform, test_transform,
        shuffle=True, num_workers=0, pin_memory=True)

    model = ResNet(8, shortcuts=shortcuts)
    criterion = torch.nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)

    results_file = f'results/{name}50_e20.csv'
    model_file = f'pretrained/{name}50_e20.pt'

    train(model, epochs, train_loader, test_loader, criterion,
          optimizer, results_file, MODEL_PATH=model_file)


if __name__ == '__main__':
    run(shortcuts=False)
    run(shortcuts=True)
    print('\n两个 50 层网络 20 epoch 训练完成！', flush=True)
