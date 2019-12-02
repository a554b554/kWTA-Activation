import torch
import torch.nn as nn
from kWTA import models
import math

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1) 

class MNIST_CNN(nn.Module):
    def __init__(self, num_channels, hidden_size=20000):
        super(MNIST_CNN, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, num_channels[0], 3, padding=1), nn.ReLU(),
                          nn.Conv2d(num_channels[0], num_channels[1], 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(num_channels[1], num_channels[2], 3, padding=1), nn.ReLU(),
                          nn.Conv2d(num_channels[2], num_channels[3], 3, padding=1, stride=2), nn.ReLU(),
                          Flatten(),
                          nn.Linear(7*7*num_channels[3], hidden_size), nn.ReLU(),
                          nn.Linear(hidden_size, 10))
    
    def forward(self, x):
        return self.cnn(x)


class SparseMNIST_CNN(nn.Module):
    def __init__(self, sp1, sp2, func, num_channels, hidden_size=20000, bias=True):
        super(SparseMNIST_CNN, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, num_channels[0], 3, padding=1, bias=bias), models.sparse_func_dict[func](sp1),
                          nn.Conv2d(num_channels[0], num_channels[1], 3, padding=1, stride=2, bias=bias), models.sparse_func_dict[func](sp1),
                          nn.Conv2d(num_channels[1], num_channels[2], 3, padding=1, bias=bias), models.sparse_func_dict[func](sp1),
                          nn.Conv2d(num_channels[2], num_channels[3], 3, padding=1, stride=2, bias=bias), models.sparse_func_dict[func](sp1),
                          Flatten(),
                          nn.Linear(7*7*num_channels[3], hidden_size), models.Sparsify1D(sp2),
                        #   nn.Linear(hidden_size, hidden_size), models.Sparsify1D(sp2),
                          nn.Linear(hidden_size, 10))
    
    def forward(self, x):
        return self.cnn(x)


class SparseMNIST_CNN_BN(nn.Module):
    def __init__(self, sp1, sp2, func, num_channels, hidden_size=20000, bias=True):
        super(SparseMNIST_CNN_BN, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, num_channels[0], 3, padding=1, bias=bias), nn.BatchNorm2D(channels[0]), models.sparse_func_dict[func](sp1),
                          nn.Conv2d(num_channels[0], num_channels[1], 3, padding=1, stride=2, bias=bias), nn.BatchNorm2D(channels[1]), models.sparse_func_dict[func](sp1),
                          nn.Conv2d(num_channels[1], num_channels[2], 3, padding=1, bias=bias), nn.BatchNorm2D(channels[2]), models.sparse_func_dict[func](sp1),
                          nn.Conv2d(num_channels[2], num_channels[3], 3, padding=1, stride=2, bias=bias), nn.BatchNorm2D(channels[3]), models.sparse_func_dict[func](sp1),
                          Flatten(),
                          nn.Linear(7*7*num_channels[3], hidden_size), models.Sparsify1D(sp2),
                        #   nn.Linear(hidden_size, hidden_size), models.Sparsify1D(sp),
                          nn.Linear(hidden_size, 10))
    
    def forward(self, x):
        return self.cnn(x)

class PartialMNIST_CNN(nn.Module):
    def __init__(self, sp, hidden_size=20000):
        super(PartialMNIST_CNN, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), models.sparse_func_dict[func](sp1),
                          nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                          Flatten(),
                          nn.Linear(7*7*64, hidden_size), nn.ReLU(),
                          nn.Linear(hidden_size, 10))
    
    def forward(self, x):
        return self.cnn(x)


class DNN(nn.Module):
    def __init__(self, hidden_size=20000):
        super(DNN, self).__init__()
        self.nn = nn.Sequential(Flatten(), nn.Linear(28*28, hidden_size), nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                nn.Linear(hidden_size, 10))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.data.shape[1]
                m.weight.data.normal_(0, math.sqrt(1. / n))
                m.bias.data.normal_(0, 1)


    def forward(self, x):
        return self.nn(x)


class SparseDNN(nn.Module):
    def __init__(self, hidden_size=20000, sp=0.5, bias=True, norm_factor=1, kact=False):
        super(SparseDNN, self).__init__()
        if kact:
            self.nn = nn.Sequential(Flatten(), nn.Linear(28*28, hidden_size, bias=bias), models.Sparsify1D_kactive(sp),
                nn.Linear(hidden_size, hidden_size, bias=bias), models.Sparsify1D_kactive(sp),
                nn.Linear(hidden_size, 10, bias=bias))
        else:
            self.nn = nn.Sequential(Flatten(), nn.Linear(28*28, hidden_size, bias=bias), models.Sparsify1D(sp),
                            nn.Linear(hidden_size, hidden_size, bias=bias), models.Sparsify1D(sp),
                            nn.Linear(hidden_size, 10, bias=bias))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.data.shape[1]
                m.weight.data.normal_(0, norm_factor*math.sqrt(1. / n))
                # m.bias.data.normal_(0, 1)
                if bias:
                    m.bias.data.zero_()

    def forward(self, x):
        return self.nn(x)













