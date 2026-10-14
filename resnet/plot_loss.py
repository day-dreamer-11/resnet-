"""从 train20epoch.log 解析训练 loss，绘制 PlainNet50 与 ResNet50 的损失曲线。"""
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse(path):
    """返回 {网络名: (iterations, losses)}，loss 为每 50 batch 的运行平均。"""
    with open(path, encoding='utf-8') as f:
        text = f.read()

    # 按训练分段切分
    sections = {}
    for m in re.finditer(r'===== 训练 (plainnet50|resnet50)', text):
        name = m.group(1)
        start = m.end()
        nxt = text.find('=====', start)
        sections[name] = text[start:nxt]

    data = {}
    for name, sec in sections.items():
        iters, losses = [], []
        for line in sec.splitlines():
            mm = re.match(r'^(\d+)\s+(\d+)\s+([\d.]+)$', line.strip())
            if mm:
                epoch, batch, loss = int(mm.group(1)), int(mm.group(2)), float(mm.group(3))
                iters.append((epoch - 1) * 391 + batch)
                losses.append(loss)
        data[name] = (iters, losses)
    return data


def main():
    data = parse('train20epoch.log')

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {'plainnet50': 'tab:blue', 'resnet50': 'tab:red'}
    labels = {'plainnet50': 'PlainNet50 (no residual)',
              'resnet50': 'ResNet50 (residual)'}

    for name in ('plainnet50', 'resnet50'):
        iters, losses = data[name]
        ax.plot(iters, losses, '-', color=colors[name],
                linewidth=1.2, alpha=0.85, label=labels[name])

    ax.set_xlabel('iteration')
    ax.set_ylabel('training loss (NLL, running avg / 50 batches)')
    ax.set_title('CIFAR-10: 50-layer training loss, PlainNet vs ResNet (20 epochs)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = 'results/loss50.png'
    fig.savefig(out, dpi=130)
    print(f'已保存损失曲线: {out}')


if __name__ == '__main__':
    main()
