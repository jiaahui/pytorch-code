from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, Resize, ToTensor

def load_data_fashion_mnist(batch_size, resize=None):
    """
    下载 Fashion-MNIST 数据集，然后将其加载到内存中
    """
    trans = [ToTensor()]

    if resize:
        trans.insert(0, Resize(resize))
    
    trans = Compose(trans)

    mnist_train = FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    
    mnist_test = FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    
    return (
        DataLoader(mnist_train, batch_size, shuffle=True),
        DataLoader(mnist_test, batch_size, shuffle=False)
    )
