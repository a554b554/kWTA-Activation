import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import torchvision
from kWTA import models


class SparseBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes,
        stride=1, sparsity=0.5, sparse_func='reg', norm='BN', ng=1, bias=True):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        elif norm == 'GN':
            self.bn1 = nn.GroupNorm(num_groups=ng, num_channels=planes)
            self.bn2 = nn.GroupNorm(num_groups=ng, num_channels=planes)
            self.bn3 = nn.GroupNorm(num_groups=ng, num_channels=planes*self.expansion)
        else:
            raise NotImplementedError

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)
        self.sparse1 = models.sparse_func_dict[sparse_func](sparsity)
        self.sparse2 = models.sparse_func_dict[sparse_func](sparsity)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias),
                self.bn3
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.sparse1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.sparse2(out)
        return out


class SparseBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes,
        stride=1, sparsity=0.5, sparse_func='reg', norm='BN', ng=1, bias=True):
        super(SparseBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=bias)
        
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(planes*self.expansion)
            self.bn4 = nn.BatchNorm2d(planes*self.expansion)
        elif norm == 'GN':
            self.bn1 = nn.GroupNorm(num_groups=ng, num_channels=planes)
            self.bn2 = nn.GroupNorm(num_groups=ng, num_channels=planes)
            self.bn3 = nn.GroupNorm(num_groups=ng, num_channels=planes*self.expansion)
            self.bn4 = nn.GroupNorm(num_groups=ng, num_channels=planes*self.expansion)
        else:
            raise NotImplementedError

        self.sparse1 = models.sparse_func_dict[sparse_func](sparsity)
        self.sparse2 = models.sparse_func_dict[sparse_func](sparsity)
        self.sparse3 = models.sparse_func_dict[sparse_func](sparsity)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias),
                self.bn4
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.sparse1(out)

        out = self.bn2(self.conv2(out))
        out = self.sparse2(out)

        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)

        out = self.sparse3(out)
        return out


class SparseResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channels,
    sparsities, num_classes=10, sparse_func='reg', 
    norm='BN', ng=1, linear_size=10000, in_channel=3, bias=True):
        super(SparseResNet, self).__init__()
        self.in_planes = num_channels[0]


        self.conv1 = nn.Conv2d(in_channel, num_channels[0], kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.layer1 = self._make_layer(block, num_channels[0], num_blocks[0], stride=1, sparsity=sparsities[0], sparse_func=sparse_func, norm=norm, ng=ng, bias=bias)
        self.layer2 = self._make_layer(block, num_channels[1], num_blocks[1], stride=2, sparsity=sparsities[1], sparse_func=sparse_func, norm=norm, ng=ng, bias=bias)
        self.layer3 = self._make_layer(block, num_channels[2], num_blocks[2], stride=2, sparsity=sparsities[2], sparse_func=sparse_func, norm=norm, ng=ng, bias=bias)
        self.layer4 = self._make_layer(block, num_channels[3], num_blocks[3], stride=2, sparsity=sparsities[3], sparse_func=sparse_func, norm=norm, ng=ng, bias=bias)
        self.linear1 = nn.Linear(num_channels[3]*block.expansion, linear_size)
        self.linear2 = nn.Linear(linear_size, num_classes)

        self.linear_sp = models.Sparsify1D(sparsities[4])

        self.sparse = models.sparse_func_dict[sparse_func](sparsities[0])


    def _make_layer(self, block, planes,
     num_blocks, stride, norm, ng, sparsity=0.5, sparse_func='reg', bias=True):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, sparsity, sparse_func=sparse_func, norm=norm, ng=ng, bias=bias))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.sparse(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear_sp(out)
        out = self.linear2(out)
        return out

def getSparseResNet_basic(num_channels, num_blocks, sparsities, sparse_func, norm='BN', ng=1, linear_size=10000, in_channel=3, bias=True):
    return SparseResNet(SparseBasicBlock, num_blocks, num_channels, sparsities, sparse_func=sparse_func, norm=norm, ng=ng, linear_size=linear_size, in_channel=in_channel, bias=bias)

def getSparseResNet_bottle(num_channels, num_blocks, sparsities, sparse_func, norm='BN', ng=1, linear_size=10000, in_channel=3, bias=True):
    return SparseResNet(SparseBottleneck, num_blocks, num_channels, sparsities, sparse_func=sparse_func, norm=norm, ng=ng, linear_size=linear_size, in_channel=in_channel, bias=bias)