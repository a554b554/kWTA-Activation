import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import torchvision
from kWTA import models


class SparseBasicBlockP(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes,  sparsities, funcs, stride=1, bias=False):
        super(SparseBasicBlockP, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.sparse1 = models.sparse_func_dict[funcs[0]](sparsities[0])
        self.sparse2 = models.sparse_func_dict[funcs[1]](sparsities[1])

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.sparse1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.sparse2(out)
        return out



class SparseBottleneckP(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes,  sparsities, funcs, stride=1, bias=False):
        super(SparseBottleneckP, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.sparse1 = models.sparse_func_dict[funcs[0]](sparsities[0])
        self.sparse2 = models.sparse_func_dict[funcs[1]](sparsities[1])
        self.sparse3 = models.sparse_func_dict[funcs[2]](sparsities[2])

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(self.expansion*planes)
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


class SparseResNetP(nn.Module):
    def __init__(self, block, num_blocks, acts, bias=False, num_classes=10):
        super(SparseResNetP, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, act=acts[0], bias=bias)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, act=acts[1], bias=bias)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, act=acts[2], bias=bias)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, act=acts[3], bias=bias)
        self.linear = nn.Linear(512*block.expansion, num_classes)


        self.activation = {}

    
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.cpu().detach()
        return hook

    def register_layer(self, layer, name):
        layer.register_forward_hook(self.get_activation(name))

    def _make_layer(self, block, planes, num_blocks, stride, act, bias):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, act['sp'], act['func'], stride, bias=bias))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# def TestSparseResNetP():
#     return SparseResNetP



# def ResNet18():
#     return ResNet(BasicBlock, [2,2,2,2])

# def ResNet34():
#     return ResNet(BasicBlock, [3,4,6,3])

# def ResNet50():
#     return ResNet(Bottleneck, [3,4,6,3])

# def ResNet101():
#     return ResNet(Bottleneck, [3,4,23,3])

# def ResNet152():
#     return ResNet(Bottleneck, [3,8,36,3])

# def SparseResNet18(relu=False, sparsities=[0.5,0.4,0.3,0.2], sparse_func='reg', bias=False):
#     return SparseResNet(SparseBasicBlock, [2,2,2,2], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)

# def SparseResNet34(relu=False, sparsities=[0.5,0.4,0.3,0.2], sparse_func='reg', bias=False):
#     return SparseResNet(SparseBasicBlock, [3,4,6,3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)

# def SparseResNet50(relu=False, sparsities=[0.5,0.4,0.3,0.2], sparse_func='reg', bias=False):
#     return SparseResNet(SparseBottleneck, [3,4,6,3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)

# def SparseResNet101(relu=False, sparsities=[0.5,0.4,0.3,0.2], sparse_func='reg', bias=False):
#     return SparseResNet(SparseBottleneck, [3,4,23,3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)

# def SparseResNet152(relu=False, sparsities=[0.5,0.4,0.3,0.2], sparse_func='reg', bias=False):
#     return SparseResNet(SparseBottleneck, [3,8,36,3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)

# def SparseResNet152_ImageNet(relu=False, sparsities=[0.5,0.4,0.3,0.2], sparse_func='reg', bias=False):
#     return SparseResNet_ImageNet(SparseBottleneck, [3,8,36,3], sparsities, sparse_func=sparse_func, bias=bias)

########### End resnet related ##################