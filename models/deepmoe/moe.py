import torch
import torch.nn as nn
from typing import Any, Callable, Optional, Type, Union, List
from utils import MoELayer, ShallowEmbeddingNetwork, MultiHeadedSparseGatingNetwork


# DeepMoE model
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MoEBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        wide: bool = False,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        if wide:
            in_channels = in_channels * 2
            out_channels = out_channels * 2

        self.conv1 = MoELayer(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MoELayer(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

        self.gate1 = MultiHeadedSparseGatingNetwork(emb_dim, out_channels)
        self.gate2 = MultiHeadedSparseGatingNetwork(emb_dim, out_channels)


    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        identity = x

        gate_values_1 = self.gate1(embedding)
        out = self.conv1(x, gate_values_1)
        out = self.bn1(out)
        out = self.relu(out)

        gate_values_2 = self.gate2(embedding)
        out = self.conv2(out, gate_values_2)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, [gate_values_1, gate_values_2]


class feature_extraction_moe(nn.Module):
    def __init__(self):
        super(feature_extraction_moe, self).__init__()

        a=1
        self.block1 = nn.Sequential(
            conv2d(3, 16, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(16, 16, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.block2 = nn.Sequential(
            conv2d(16, 32, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(32, 32, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.block3 = nn.Sequential(
            conv2d(32, 64, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(64, 64, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.block4 = nn.Sequential(
            conv2d(64, 96, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(96, 96, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.block5 = nn.Sequential(
            conv2d(96, 128, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(128, 128, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.block6 = nn.Sequential(
            conv2d(128, 192, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(192, 192, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x, mad=False):
        out1 = self.block1(x)
        out2 = self.block2(out1 if not mad else out1.detach())
        out3 = self.block3(out2 if not mad else out2.detach())
        out4 = self.block4(out3 if not mad else out3.detach())
        out5 = self.block5(out4 if not mad else out4.detach())
        out6 = self.block6(out5 if not mad else out5.detach())

        return x, out1, out2, out3, out4, out5, out6



