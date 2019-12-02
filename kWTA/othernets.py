import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import torchvision
from kWTA import models

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16-sp': [64, 64, 128, 128, 256, 256, 128, 64, 32],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1) 


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, channels=[64,192,384,256,256]):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[3], channels[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

class SparseAlexNet(nn.Module):
    def __init__(self, num_classes=10, channels=[64,192,384,256,256], sp1=0.5, sp2=0.5, sp_func='reg'):
        super(SparseAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1),
            models.sparse_func_dict[sp_func](sp1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            models.sparse_func_dict[sp_func](sp1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            models.sparse_func_dict[sp_func](sp1),
            nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1),
            models.sparse_func_dict[sp_func](sp1),
            nn.Conv2d(channels[3], channels[4], kernel_size=3, padding=1),
            models.sparse_func_dict[sp_func](sp1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 4096),
            models.Sparsify1D(sp2),
            nn.Linear(4096, 4096),
            models.Sparsify1D(sp2),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


class VGG(nn.Module):
    def __init__(self, vgg_name, hidden_size=20000):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(nn.Linear(512, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, 10))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class SparseVGG(nn.Module):
    def __init__(self, vgg_name, sparsity=0.3, sparse_func='reg', sp_lin=0.02, hidden_size=20000):
        super(SparseVGG, self).__init__()

        self.sparsity = sparsity
        self.sparse_func = sparse_func


        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(nn.Linear(512, hidden_size), models.Sparsify1D(sp_lin),
        nn.Linear(hidden_size, 10))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           models.sparse_func_dict[self.sparse_func](self.sparsity)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class Small_CNN(nn.Module):
    def __init__(self, hidden_size=20000):
        super(Small_CNN, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                          Flatten(),
                          nn.Linear(4096, hidden_size), nn.ReLU(),
                          nn.Linear(hidden_size, 10))
    
    def forward(self, x):
        return self.cnn(x)

class SparseSmall_CNN(nn.Module):
    def __init__(self, sp1, sp2, func, hidden_size=20000):
        super(SparseSmall_CNN, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), models.sparse_func_dict[func](sp1),
                          nn.Conv2d(32, 32, 3, padding=1, stride=2), models.sparse_func_dict[func](sp1),
                          nn.Conv2d(32, 64, 3, padding=1), models.sparse_func_dict[func](sp1),
                          nn.Conv2d(64, 64, 3, padding=1, stride=2), models.sparse_func_dict[func](sp1),
                          Flatten(),
                          nn.Linear(4096, hidden_size), models.Sparsify1D(sp2),
                        #   nn.Linear(hidden_size, hidden_size), models.Sparsify1D(sp),
                          nn.Linear(hidden_size, 10))
    
    def forward(self, x):
        return self.cnn(x)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class SparseLeNet(nn.Module):
    def __init__(self, sparsities, sparse_func='reg'):
        super(SparseLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

        self.sparse1 = models.sparse_func_dict[sparse_func](sparsities[0])
        self.sparse2 = models.sparse_func_dict[sparse_func](sparsities[1])
        self.sparse3 = models.Sparsify1D(sparsities[2])
        self.sparse4 = models.Sparsify1D(sparsities[3])

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.sparse1(self.conv1(x))
        out = F.max_pool2d(out, 2)

        out = self.sparse2(self.conv2(out))
        out = F.max_pool2d(out, 2)


        # out = self.relu(self.conv1(x))
        # out = self.relu(self.conv2(out))

        out = out.view(out.size(0), -1)

        out = self.sparse3(self.fc1(out))
        out = self.sparse4(self.fc2(out))

        # out = self.relu(self.fc1(out))
        # out = self.relu(self.fc2(out))

        out = self.fc3(out)

        return out


class breakReLU(nn.Module):
    def __init__(self, h):
        super(breakReLU, self).__init__()
        self.h = h
        self.thre = nn.Threshold(0, -self.h)

    def forward(self, x):
        return self.thre(x)

class DNN(nn.Module):
    def __init__(self, sizes, activation='sp', bias=True, **kwargs):
        super(DNN, self).__init__()
        layers = []
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1], bias=bias))
            if activation == 'sp':
                act = models.Sparsify1D(**kwargs)
            elif activation == 'relu':
                act = nn.ReLU()
            elif activation == 'breakrelu':
                act = breakReLU(**kwargs)
            else:
                raise NotImplementedError
            
            if i == len(sizes)-2:
                break
            layers.append(act)
        self.nn = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                if bias:
                    # m.bias.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    def forward(self, x):
        return self.nn(x)