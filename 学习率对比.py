import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return self.fc(torch.flatten(x, 1))


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

transform = transforms.ToTensor()
train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256)


def train_one(lr, epochs=50):
    torch.manual_seed(0)                    # 三档同一个初始化,只让 lr 变
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        losses.append(total / len(train_loader))
        print(f"  lr={lr:g}  epoch {epoch + 1:2d}  train loss {losses[-1]:.4f}")

    # 测试集准确率
    model.eval()
    correct = n = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            n += labels.size(0)
    acc = correct / n * 100
    return losses, acc


lrs = [0.01, 0.001, 0.0001]
results = {}
for lr in lrs:
    print(f"\n=== lr = {lr:g} ===")
    losses, acc = train_one(lr)
    results[lr] = (losses, acc)
    print(f"  -> 最终 train loss {losses[-1]:.4f}   测试准确率 {acc:.2f}%")

print("\n=== 汇总 (50 epoch) ===")
print(f"{'lr':>10} {'train loss':>12} {'test acc':>10}")
for lr in lrs:
    losses, acc = results[lr]
    print(f"{lr:>10g} {losses[-1]:>12.4f} {acc:>9.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for lr in lrs:
    losses, _ = results[lr]
    axes[0].plot(range(1, 51), losses, label=f"lr={lr:g}")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss")
axes[0].set_title("train loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

for lr in lrs:
    losses, _ = results[lr]
    axes[1].plot(range(1, 51), losses, label=f"lr={lr:g}")
axes[1].set_yscale("log")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("train loss (log)")
axes[1].set_title("train loss (log scale)"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lr_compare.png", dpi=120)
print("已保存: lr_compare.png")
