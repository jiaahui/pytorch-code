import torch
from torch import nn

from evaluate import accuracy, evaluate_accuracy, evaluate_accuracy_gpu
from visualize import Animator
from utils import Accumulator, Timer

def sgd(params, LR, BATCH_SIZE):
    """
    小批量随机梯度下降
    """
    with torch.no_grad():
        for param in params:
            param -= LR * param.grad / BATCH_SIZE
            param.grad.zero_()

def train_epoch(net, train_iter, loss, updater):
    """
    训练模型一个迭代周期
    """
    if isinstance(net, torch.nn.Module):
        net.train()
    
    metric = Accumulator(3)  # 训练损失总和、训练准确度总和、样本数
    
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        
        if isinstance(updater, torch.optim.Optimizer):
            # 使用 PyTorch 内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用自定义的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_v1(net, train_iter, test_iter, loss, num_epochs, updater):
    """
    训练模型
    """
    animator = Animator(
        xlabel='epoch', 
        xlim=[1, num_epochs], 
        ylim=[0.3, 0.9],
        legend=['train loss', 'train acc', 'test acc']
    )
    
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    
    train_loss, train_acc = train_metrics
    
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def train_v2(net, train_iter, test_iter, num_epochs, lr, device):
    """
    用 GPU 训练模型
    """
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    
    net.apply(init_weights)
    
    net.to(device)
    print(f'training on {device}')
    
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)

    for epoch in range(num_epochs):
        metric = Accumulator(3)  # 训练损失之和，训练准确率之和，样本数
        net.train()

        for i, (X, y) in enumerate(train_iter):
            timer.start()

            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        
    print(f'loss:\t {train_l:.3f}\t train acc:\t {train_acc:.3f}\t test acc: {test_acc:.3f}', end="\t")
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')
