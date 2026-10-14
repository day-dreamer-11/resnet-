import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                padding=1
            )         
            self.conv2 = nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        padding=1
            )
            self.conv3 = nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        padding=1
            )            
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(
                kernel_size=2,stride=2
            )
            self.fc = nn.Linear(64 * 14 * 14, 10)
        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            identity1 = x
            x = self.conv2(x)
            x = self.relu(x) 
            x = x + identity1
            identity2 = x
            x = self.conv3(x) 
            x = self.relu(x)
            x = x + identity2
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x 
model = SimpleCNN()
x = torch.randn(1, 1, 28, 28)
output = model(x)
print(output.shape)
print(output)
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
for epoch in range(50):
    total_loss = 0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(
        f"Epoch {epoch + 1}, "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )
