"""快速验证：PlainNet50 与 ResNet50 各训练 1 个 epoch（GPU），输出 train/test 误差。"""
import torch
import torch.optim as optim
from torchvision import transforms

from resnet import ResNet
from cifar10_npy import get_data_loaders
from train import evaluate

data_dir = 'data/cifar10'
batch_size = 128
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


def run_one_epoch(shortcuts):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    name = 'ResNet50' if shortcuts else 'PlainNet50'
    print(f'\n===== {name} (1 epoch, {device}) =====', flush=True)

    train_loader, test_loader = get_data_loaders(
        data_dir, batch_size, train_transform, test_transform,
        shuffle=True, num_workers=0, pin_memory=True)

    model = ResNet(8, shortcuts=shortcuts).to(device)
    criterion = torch.nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)

    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 50 == 0:
            print(f'  batch {i}/{len(train_loader)}  loss {running_loss / 50:.3f}',
                  flush=True)
            running_loss = 0.0

    model.eval()
    train_err = evaluate(model, train_loader, device)
    test_err = evaluate(model, test_loader, device)
    print(f'{name}: train_err={train_err * 100:.2f}%  test_err={test_err * 100:.2f}%',
          flush=True)
    return test_err


if __name__ == '__main__':
    run_one_epoch(shortcuts=False)
    run_one_epoch(shortcuts=True)
    print('\n1-epoch 快速验证完成。', flush=True)
