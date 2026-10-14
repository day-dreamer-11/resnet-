import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)


transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# 找第一张标签为 7 的图片
idx = next(i for i, (_, label) in enumerate(train_dataset) if label == 7)
image, label = train_dataset[idx]
print(f"选中训练集第 {idx} 张图片, 标签 = {label}, 形状 = {tuple(image.shape)}")

x = image.unsqueeze(0)          # [1, 1, 28, 28]
y = torch.tensor([label])       # [1]

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    outputs = model(x)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1:2d}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    logits = model(x)
    pred = logits.argmax(1).item()
    print(f"\n这一张图: 真实标签 {label}, 预测 {pred}")
    print("10 类 logits:", [f"{v:.2f}" for v in logits[0].tolist()])

    # 在整个测试集上看泛化
    test_loader = DataLoader(test_dataset, batch_size=256)
    correct = total = 0
    for images, labels in test_loader:
        preds = model(images).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    print(f"测试集准确率: {correct / total * 100:.2f}%")
