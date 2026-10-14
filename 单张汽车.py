import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ---------- 取一张汽车图并转灰度 ----------
images = np.load("resnet/data/cifar10/cifar10_npy/train_images.npy")
labels = np.load("resnet/data/cifar10/cifar10_npy/train_labels.npy")

idx = int(np.where(labels == 1)[0][0])          # CIFAR-10 里 1 = automobile
rgb = images[idx].astype(np.float32) / 255.0    # [32, 32, 3]
gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
target = torch.tensor(gray).unsqueeze(0).unsqueeze(0)   # [1, 1, 32, 32]

print(f"取到 CIFAR-10 第 {idx} 张图, 类别 1 = automobile, 灰度图 {tuple(target.shape)}")


# ---------- 小卷积自编码器 ----------
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 32->16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16->8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 8->4
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),            # 4->8
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(),            # 8->16
            nn.ConvTranspose2d(16, 1, 2, stride=2), nn.Sigmoid(),          # 16->32
        )

    def forward(self, x):
        return self.dec(self.enc(x))


torch.manual_seed(0)
model = AE()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []
for step in range(50):
    recon = model(target)               # 输入就是它自己要重建的那张图
    loss = criterion(recon, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Step {step + 1:2d}, MSE: {loss.item():.5f}")

model.eval()
with torch.no_grad():
    recon = model(target).clamp(0, 1)
    mse = criterion(recon, target).item()
    psnr = 10 * np.log10(1.0 / mse)

print(f"\n最终 MSE: {mse:.5f}   PSNR: {psnr:.2f} dB")

# ---------- 存一张 目标 vs 重建 对照图 ----------
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(target[0, 0], cmap="gray", vmin=0, vmax=1); axes[0].set_title("target (car, gray)")
axes[1].imshow(recon[0, 0], cmap="gray", vmin=0, vmax=1); axes[1].set_title(f"recon @50 steps")
axes[2].plot(range(1, 51), losses); axes[2].set_xlabel("step"); axes[2].set_ylabel("MSE"); axes[2].set_title("loss")
for ax in axes[:2]:
    ax.axis("off")
plt.tight_layout()
plt.savefig("car_memorize.png", dpi=120)
print("已保存对照图: car_memorize.png")
