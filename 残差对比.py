import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class SimpleCNN(nn.Module):
    def __init__(self, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc = nn.Linear(64 * 14 * 14, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        identity1 = x
        x = self.relu(self.conv2(x))
        if self.use_residual:
            x = x + identity1
        identity2 = x
        x = self.relu(self.conv3(x))
        if self.use_residual:
            x = x + identity2
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

transform = transforms.ToTensor()
train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256)


def train_one(use_residual, epochs=50, log=print):
    torch.manual_seed(0)
    model = SimpleCNN(use_residual).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses, accs = [], []
    for epoch in range(epochs):
        model.train()
        total, correct, n = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            n += labels.size(0)
        losses.append(total / len(train_loader))
        accs.append(correct / n * 100)
        log(f"  epoch {epoch + 1:2d}  train loss {losses[-1]:.4f}  train acc {accs[-1]:.2f}%")

    model.eval()
    correct = n = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            n += labels.size(0)
    return losses, accs, correct / n * 100


runs = {}
for name, use_residual in [("with_residual", True), ("no_residual", False)]:
    print(f"\n=== {name} ({'有残差' if use_residual else '去残差'}) ===")
    runs[name] = train_one(use_residual)

print("\n=== 汇总 (50 epoch) ===")
print(f"{'model':>16} {'train loss':>12} {'train acc':>10} {'test acc':>10}")
for name, (losses, accs, test_acc) in runs.items():
    print(f"{name:>16} {losses[-1]:>12.4f} {accs[-1]:>9.2f}% {test_acc:>9.2f}%")

if plt is None:
    print("未安装 matplotlib，跳过画图")
else:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    label = {"with_residual": "有残差", "no_residual": "去残差"}
    for name, (losses, accs, _) in runs.items():
        axes[0].plot(range(1, 51), losses, label=label[name])
        axes[1].plot(range(1, 51), accs, label=label[name])
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss")
    axes[0].set_title("train loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("train acc (%)")
    axes[1].set_title("train accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("residual_compare.png", dpi=120)
    print("已保存: residual_compare.png")
