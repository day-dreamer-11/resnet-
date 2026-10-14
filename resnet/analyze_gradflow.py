"""测量 PlainNet vs ResNet 的梯度流：记录每个 stack 输出处的激活梯度范数。

梯度流 = 反向传播时 loss 对各层输出 z 的梯度 ||∂loss/∂z||。
比值越大，说明信号在向浅层传播时衰减越严重（梯度消失/爆炸），优化越困难。
"""
import torch
from resnet import ResNet
from cifar10_npy import get_data_loaders
from torchvision import transforms

means = [0.4918687901200927, 0.49185976472299225, 0.4918583862227116]
stds = [0.24697121702736, 0.24696766978537033, 0.2469719877121087]
norm = transforms.Normalize(mean=means, std=stds)
tt = transforms.Compose([transforms.ToTensor(), norm])

torch.manual_seed(0)
tl, _ = get_data_loaders('data/cifar10', 128, tt, tt,
                         shuffle=False, num_workers=0, pin_memory=False)
x, y = next(iter(tl))
criterion = torch.nn.NLLLoss()


def gradflow(shortcuts):
    m = ResNet(8, shortcuts=shortcuts)
    grads = {}

    def make_hook(name):
        def hook(g):
            grads[name] = g.norm().item()
        return hook

    # 记录各 stack 输出张量，注册 backward hook 捕获其梯度
    handles = []
    # 手动重写前向以挂 hook（复用模型结构，逐段计算）
    z = m.convIn(x); z = m.bnIn(z); z = m.relu(z)
    z.register_hook(make_hook('after_input'))
    for l in m.stack1:
        z = l(z, shortcuts=m.shortcuts)
    z.register_hook(make_hook('after_stack1'))
    z = m.stack2a(z, shortcuts=m.shortcuts)
    for l in m.stack2b:
        z = l(z, shortcuts=m.shortcuts)
    z.register_hook(make_hook('after_stack2'))
    z = m.stack3a(z, shortcuts=m.shortcuts)
    for l in m.stack3b:
        z = l(z, shortcuts=m.shortcuts)
    z.register_hook(make_hook('after_stack3'))

    z = m.avgpool(z); z = z.view(z.size(0), -1)
    z = m.fcOut(z); z = m.softmax(z)
    loss = criterion(z, y)
    loss.backward()

    names = ['after_input', 'after_stack1', 'after_stack2', 'after_stack3']
    print(f'  [{"ResNet" if shortcuts else "PlainNet"}] 各深度处 ||∂loss/∂z|| :')
    for n in names:
        print(f'    {n:16s}  {grads[n]:.4f}')
    # 输入层 vs 最深层 的衰减倍数
    ratio = grads['after_input'] / grads['after_stack3']
    print(f'    浅层/深层 梯度比 = {ratio:.1f}x')
    return ratio


print('=== 梯度流（激活梯度范数，同一 batch，初始化权重）===')
print()
r_plain = gradflow(False)
print()
r_res = gradflow(True)
print()
print(f'结论：PlainNet 浅/深层梯度比 {r_plain:.1f}x，ResNet 为 {r_res:.1f}x。'
      f'比值越小，梯度流越均衡。')
