

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    def __init__(self, inplane, outplane):
        super().__init__()
        self.padding = (1, 1, 1, 1)
        self.conv = nn.Conv2d(inplane, outplane, kernel_size=3, padding=0, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = F.pad(x, self.padding, 'replicate')
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Residual block used ONLY inside conv3_extra
class ResidualExtraBlock(nn.Module):
    """
    Residual refinement block:
        y = x + res_scale * f(x)
    where f(x) = ConvBlock -> ConvBlock   (same style as baseline)
    """
    def __init__(self, channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = float(res_scale)
        self.cb1 = ConvBlock(channels, channels)
        self.cb2 = ConvBlock(channels, channels)

    def forward(self, x):
        y = self.cb1(x)
        y = self.cb2(y)
        return x + self.res_scale * y


class IFCNN(nn.Module):
    def __init__(self, resnet, fuse_scheme=0, extra_blocks=2, res_scale=1.0):
        super().__init__()
        self.fuse_scheme = fuse_scheme
        self.extra_blocks = int(extra_blocks)

        self.conv2 = ConvBlock(64, 64)

        self.conv3 = ConvBlock(64, 64)

        self.conv3_extra = nn.ModuleList(
            [ResidualExtraBlock(64, res_scale=res_scale) for _ in range(self.extra_blocks)]
        )

        self.conv4 = nn.Conv2d(64, 3, kernel_size=1, padding=0, stride=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

        for p in resnet.parameters():
            p.requires_grad = False

        self.conv1 = resnet.conv1
        self.conv1.stride = 1
        self.conv1.padding = (0, 0)

    def tensor_max(self, tensors):
        out = tensors[0]
        for t in tensors[1:]:
            out = torch.max(out, t)
        return out

    def tensor_sum(self, tensors):
        out = tensors[0]
        for t in tensors[1:]:
            out = out + t
        return out

    def tensor_mean(self, tensors):
        return self.tensor_sum(tensors) / len(tensors)

    def operate(self, operator, tensors):
        return [operator(t) for t in tensors]

    def tensor_padding(self, tensors, padding=(1, 1, 1, 1), mode='constant', value=0):
        return [F.pad(t, padding, mode=mode, value=value) for t in tensors]

    def forward(self, *tensors):
        outs = self.tensor_padding(tensors=tensors, padding=(3, 3, 3, 3), mode='replicate')
        outs = self.operate(self.conv1, outs)
        outs = self.operate(self.conv2, outs)

        if self.fuse_scheme == 0:
            out = self.tensor_max(outs)
        elif self.fuse_scheme == 1:
            out = self.tensor_sum(outs)
        elif self.fuse_scheme == 2:
            out = self.tensor_mean(outs)
        else:
            out = self.tensor_max(outs)

        out = self.conv3(out)
        for blk in self.conv3_extra:
            out = blk(out)

        out = self.conv4(out)
        return out


def myIFCNN(fuse_scheme=0, extra_blocks=2, res_scale=0.1):
    resnet = models.resnet101(weights=None)
    return IFCNN(resnet, fuse_scheme=fuse_scheme, extra_blocks=extra_blocks, res_scale=res_scale)
