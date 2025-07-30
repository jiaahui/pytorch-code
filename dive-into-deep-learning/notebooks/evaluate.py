import torch
from torch import nn

from utils import Accumulator


def accuracy(y_hat, y):
    """
    计算预测正确的数量
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    
    cmp = y_hat.type(y.dtype) == y
    
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    """
    计算在指定数据集上模型的精度
    """
    if isinstance(net, torch.nn.Module):
        net.eval()
    
    metric = Accumulator(2)  # 正确预测数、预测总数
    
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    
    return metric[0] / metric[1]

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """
    使用 GPU 计算模型在数据集上的精度
    """
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    
    metric = Accumulator(2)  # 正确预测的数量，总预测的数量
    
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
