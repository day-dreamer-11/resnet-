"""从 HuggingFace 的 parquet 格式加载 CIFAR-10，解码成 numpy 缓存。

uoft-cs/cifar10 的 parquet 里每张图为 PNG 字节（img['bytes']），标签为 0-9。
本模块把它们解码成 (N, 32, 32, 3) uint8 数组并缓存为 .npy，供训练使用。
"""
import io
import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


def decode_parquet(parquet_path):
    df = pd.read_parquet(parquet_path)
    n = len(df)
    images = np.empty((n, 32, 32, 3), dtype=np.uint8)
    for i in range(n):
        b = df['img'].iloc[i]['bytes']
        images[i] = np.array(Image.open(io.BytesIO(b)).convert('RGB'))
    labels = df['label'].to_numpy(dtype=np.int64)
    return images, labels


def prepare(data_dir):
    """解码 parquet 为 npy（若不存在）。返回 (train 路径, test 路径) 字典。"""
    npy_dir = os.path.join(data_dir, 'cifar10_npy')
    os.makedirs(npy_dir, exist_ok=True)
    paths = {}
    for split in ('train', 'test'):
        img_path = os.path.join(npy_dir, f'{split}_images.npy')
        lbl_path = os.path.join(npy_dir, f'{split}_labels.npy')
        if not (os.path.exists(img_path) and os.path.exists(lbl_path)):
            images, labels = decode_parquet(
                os.path.join(data_dir, 'parquet', f'{split}.parquet'))
            np.save(img_path, images)
            np.save(lbl_path, labels)
            print(f'decoded {split}: images {images.shape}, labels {labels.shape}',
                  flush=True)
        paths[split] = (img_path, lbl_path)
    return paths


class CIFAR10Npy(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_data_loaders(data_dir, batch_size, train_transform, test_transform,
                     shuffle=True, num_workers=0, pin_memory=False):
    paths = prepare(data_dir)

    train_images = np.load(paths['train'][0])
    train_labels = np.load(paths['train'][1])
    test_images = np.load(paths['test'][0])
    test_labels = np.load(paths['test'][1])

    train_dataset = CIFAR10Npy(train_images, train_labels, transform=train_transform)
    test_dataset = CIFAR10Npy(test_images, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=shuffle, num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, test_loader
