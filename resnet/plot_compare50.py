"""读取 plainnet50.csv 与 resnet50.csv，输出对比摘要并绘制 Figure 6 风格的误差曲线。"""
import os

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(name):
    path = f'results/{name}50_e20.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f'缺少 {path}，请先运行 train20epoch.py')
    return pd.read_csv(path)


def main():
    plain = load('plainnet')
    resnet = load('resnet')

    print('=' * 56)
    print(f"{'网络':<14}{'最佳测试误差':<14}{'最终测试误差':<14}{'最终训练误差'}")
    for name, df in [('PlainNet50', plain), ('ResNet50', resnet)]:
        best = df['test_err'].min() * 100
        final_test = df['test_err'].iloc[-1] * 100
        final_train = df['train_err'].iloc[-1] * 100
        print(f"{name:<14}{best:<14.2f}{final_test:<14.2f}{final_train:.2f}")
    print('=' * 56)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, (name, df) in zip(axes, [('PlainNet50', plain), ('ResNet50', resnet)]):
        ax.plot(df['epoch'], df['train_err'] * 100, '--', color='tab:blue',
                label='train error')
        ax.plot(df['epoch'], df['test_err'] * 100, '-', color='tab:red',
                linewidth=2, label='test error')
        ax.set_title(name)
        ax.set_xlabel('epoch')
        ax.set_ylabel('error (%)')
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle('CIFAR-10: 50-layer PlainNet vs ResNet (dashed=train, bold=test)')
    fig.tight_layout()
    out = 'results/compare50.png'
    fig.savefig(out, dpi=130)
    print(f'已保存曲线图: {out}')


if __name__ == '__main__':
    main()
