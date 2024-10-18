import torch
import torch.nn as nn
from typing import Any, Callable, Optional, Type, Union, List

from models.madnet2 import conv2d
from .utils import MoELayer, MultiHeadedSparseGatingNetwork


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


class MoECustomBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int = 128, #!DEBUG !c.f. DeepMoE github readme
        wide: bool = False,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
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

        # if wide:
        #     in_channels = in_channels * 2
        #     out_channels = out_channels * 2

        self.conv1 = MoELayer(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding)
        # self.bn1 = norm_layer(out_channels)

        # self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = MoELayer(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        # self.bn2 = norm_layer(out_channels)

        self.downsample = nn.Sequential(
                # conv1x1(in_channels, out_channels * block.expansion, stride=2),
                # norm_layer(out_channels * block.expansion),
                conv1x1(in_channels, out_channels, stride=2),
                norm_layer(out_channels),
            )
        # self.stride = stride

        self.gate1 = MultiHeadedSparseGatingNetwork(emb_dim, out_channels)
        self.gate2 = MultiHeadedSparseGatingNetwork(emb_dim, out_channels)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        gate_values_1 = self.gate1(embedding)
        out1 = self.conv1(x, gate_values_1)
        out2 = self.relu1(out1)

        gate_values_2 = self.gate2(embedding)
        out3 = self.conv2(out2, gate_values_2)
        # out = self.bn2(out)
        out4 = self.relu2(out3)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        # out += identity
        out5 = out4 + identity
        out6 = self.relu3(out5)

        return out6, [gate_values_1, gate_values_2]


class feature_extraction_moe(nn.Module):
    def __init__(self):
        super(feature_extraction_moe, self).__init__()

        a=1
        self.block1 = nn.Sequential(conv2d(3, 16, 3, 2, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    conv2d(16, 16, 3, 1, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.block2 = MoECustomBasicBlock(in_channels=16,
                                          out_channels=32,
                                        #   emb_dim=16,
                                          kernel_size=3,
                                        #   stride=2,
                                          padding=1,
                                          dilation=1)
        self.block3 = MoECustomBasicBlock(in_channels=32,
                                          out_channels=64,
                                        #   emb_dim=16,
                                          kernel_size=3,
                                        #   stride=2,
                                          padding=1,
                                          dilation=1)
        self.block4 = MoECustomBasicBlock(in_channels=64,
                                          out_channels=96,
                                        #   emb_dim=16,
                                          kernel_size=3,
                                        #   stride=2,
                                          padding=1,
                                          dilation=1)
        self.block5 = MoECustomBasicBlock(in_channels=96,
                                          out_channels=128,
                                        #   emb_dim=16,
                                          kernel_size=3,
                                        #   stride=2,
                                          padding=1,
                                          dilation=1)
        self.block6 = MoECustomBasicBlock(in_channels=128,
                                          out_channels=192,
                                        #   emb_dim=16,
                                          kernel_size=3,
                                        #   stride=2,
                                          padding=1,
                                          dilation=1)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #!DEBUG

        if False:
            self.block1 = nn.Sequential(

                # MADNet2 submodule
                # conv2d(3, 16, 3, 2, 1, 1)
                # (in_planes, out_planes, kernel_size, stride, pad, dilation)

                # MoELayer
                # (in_channels, out_channels, kernel_size, stride, padding, dilation)

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

    def forward(self, x, mad=False, embed=None):
        out1 = self.block1(x)
        out2, gate2 = self.block2(out1 if not mad else out1.detach(), embed)
        out3, gate3 = self.block3(out2 if not mad else out2.detach(), embed)
        out4, gate4 = self.block4(out3 if not mad else out3.detach(), embed)
        out5, gate5 = self.block5(out4 if not mad else out4.detach(), embed)
        out6, gate6 = self.block6(out5 if not mad else out5.detach(), embed)
        gates = [x for xs in [gate2, gate3, gate4, gate5, gate6] for x in xs]

        return x, out1, out2, out3, out4, out5, out6, gates #[gate2, gate3, gate4, gate5, gate6]


    # def forward(self, x, mad=False, embed=None):
    #     out1 = self.block1(x)
    #     _embed = self.avgpool(out1).squeeze() #!DEBUG
    #     out2, gate2 = self.block2(out1 if not mad else out1.detach(), _embed)
    #     out3, gate3 = self.block3(out2 if not mad else out2.detach(), _embed)
    #     out4, gate4 = self.block4(out3 if not mad else out3.detach(), _embed)
    #     out5, gate5 = self.block5(out4 if not mad else out4.detach(), _embed)
    #     out6, gate6 = self.block6(out5 if not mad else out5.detach(), _embed)
    #     gates = [x for xs in [gate2, gate3, gate4, gate5, gate6] for x in xs]

    #     return x, out1, out2, out3, out4, out5, out6, gates #[gate2, gate3, gate4, gate5, gate6]




