import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models




class ConvBlock(nn.Module):
    def __init__(self, inplane, outplane):
        super(ConvBlock, self).__init__()
        assert inplane == outplane, "skip identity דורש inplane == outplane"
        self.padding = (1, 1, 1, 1)

        # ReLU
        self.relu1 = nn.ReLU(inplace=True)
        # "Linear"  = Conv + BN
        self.conv1 = nn.Conv2d(
            inplane, outplane,
            kernel_size=3, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(outplane)

        # ReLU
        self.relu2 = nn.ReLU(inplace=True)

        # "Linear"  = Conv + BN
        self.conv2 = nn.Conv2d(
            outplane, outplane,
            kernel_size=3, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(outplane)

    def forward(self, x):
        identity = x
        out = self.relu1(x)
        out = F.pad(out, self.padding, mode='replicate')
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu2(out)
        out = F.pad(out, self.padding, mode='replicate')
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity   # y = x + F(x)
        return out


class IFCNN(nn.Module):
    def __init__(self, resnet, fuse_scheme=0):
        super(IFCNN, self).__init__()
        self.fuse_scheme = fuse_scheme
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=1, padding=0, stride=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        for p in resnet.parameters():
            p.requires_grad = False

        self.conv1 = resnet.conv1
        self.conv1.stride = 1
        self.conv1.padding = (0, 0)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs
    ):
        remap = {}

        for k in list(state_dict.keys()):
            k2 = k
            if k2.startswith(prefix + "conv2.conv."):
                k2 = k2.replace(prefix + "conv2.conv.", prefix + "conv2.conv1.")
            elif k2.startswith(prefix + "conv2.bn."):
                k2 = k2.replace(prefix + "conv2.bn.", prefix + "conv2.bn1.")
            elif k2.startswith(prefix + "conv3.conv."):
                k2 = k2.replace(prefix + "conv3.conv.", prefix + "conv3.conv1.")
            elif k2.startswith(prefix + "conv3.bn."):
                k2 = k2.replace(prefix + "conv3.bn.", prefix + "conv3.bn1.")

            if k2 != k:
                remap[k2] = state_dict.pop(k)

        state_dict.update(remap)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

    def tensor_max(self, tensors):
        max_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                max_tensor = tensor
            else:
                max_tensor = torch.max(max_tensor, tensor)
        return max_tensor

    def tensor_sum(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        return sum_tensor

    def tensor_mean(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        mean_tensor = sum_tensor / len(tensors)
        return mean_tensor

    def operate(self, operator, tensors):
        out_tensors = []
        for tensor in tensors:
            out_tensor = operator(tensor)
            out_tensors.append(out_tensor)
        return out_tensors

    def tensor_padding(self, tensors, padding=(1, 1, 1, 1), mode='constant', value=0):
        out_tensors = []
        for tensor in tensors:
            out_tensor = F.pad(tensor, padding, mode=mode, value=value)
            out_tensors.append(out_tensor)
        return out_tensors

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
        out = self.conv4(out)
        return out


def myIFCNN(fuse_scheme=0):
    resnet = models.resnet101(weights=None)
    model = IFCNN(resnet, fuse_scheme=fuse_scheme)
    return model