from abc import ABC, abstractmethod
from typing import cast

import torch as th
import torch.nn as nn
from .utils import Permute

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchsummary import summary


class CNNFtExtract(nn.Module, ABC):
    @property
    @abstractmethod
    def out_size(self) -> int:
        raise NotImplementedError()


############################
# Features extraction stuff
############################

# MNIST Stuff


class MNISTCnn(CNNFtExtract):
    """
    b_θ5 : R^f*f -> R^n
    """

    def __init__(self, f: int) -> None:
        super().__init__()

        self.__seq_conv = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Flatten(1, -1),
        )

        self.__out_size = 32 * (f // 4) ** 2

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        o_t = o_t[:, 0, None, :, :]  # grey scale
        return cast(th.Tensor, self.__seq_conv(o_t))

    @property
    def out_size(self) -> int:
        return self.__out_size


# RESISC-45 Stuff


class RESISC45Cnn(CNNFtExtract):
    def __init__(self, f: int) -> None:
        super().__init__()

        self.__seq_conv = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Flatten(1, -1),
        )

        self.__out_size = 64 * (f // 8) ** 2

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        return cast(th.Tensor, self.__seq_conv(o_t))

    @property
    def out_size(self) -> int:
        return self.__out_size


class AIDCnn(CNNFtExtract):
    def __init__(self, f: int) -> None:
        super().__init__()

        self.__seq_conv = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.Flatten(1, -1),
        )

        self.__out_size = 128 * (f // 16) ** 2

    @property
    def out_size(self) -> int:
        return self.__out_size

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        return cast(th.Tensor, self.__seq_conv(o_t))


class WorldStratCnn(CNNFtExtract):
    def __init__(self, f: int) -> None:
        super().__init__()

        self.__seq_conv = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.Flatten(1, -1),
        )

        self.__out_size = 256 * (f // 32) ** 2

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        return cast(th.Tensor, self.__seq_conv(o_t))

    @property
    def out_size(self) -> int:
        return self.__out_size


# Knee MRI stuff


class KneeMRICnn(CNNFtExtract):
    def __init__(self, f: int = 16):
        super().__init__()

        self.__seq_conv = nn.Sequential(
            nn.Conv3d(1, 8, (3, 3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool3d(2, 2),
            nn.BatchNorm2d(8),
            nn.Conv3d(8, 16, (3, 3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool3d(2, 2),
            nn.BatchNorm2d(16),
            nn.Conv3d(16, 32, (3, 3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool3d(2, 2),
            nn.BatchNorm2d(32),
            nn.Flatten(1, -1),
        )

        self.__out_size = 32 * (f // 8) ** 3

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        out = cast(th.Tensor, self.__seq_conv(o_t))
        return out

    @property
    def out_size(self) -> int:
        return self.__out_size


class SkinCancerCnn(CNNFtExtract):
    # https://github.com/Ipsedo/MARLClassification/issues/4
    # https://drive.google.com/drive/folders/17g6zFSbCNXTV3VaDKop73W7Cn-NJlTO7?usp=sharing
    def __init__(self, f: int) -> None:
        super().__init__()

        self.__seq_conv = nn.Sequential(
            nn.Conv2d(3, 8, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Flatten(1, -1),
        )

        self.__out_size = 32 * (f // 8) ** 2

    @property
    def out_size(self) -> int:
        return self.__out_size

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        out: th.Tensor = self.__seq_conv(o_t)
        return out


class CbisCnn(CNNFtExtract):
    def __init__(self, f: int) -> None:
        super().__init__()

        self.__out_size = 128 * (f // 16) ** 2

        # resnet18
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        inf = self.model.fc.in_features
        self.model.fc = nn.Linear(inf, self.__out_size)
        self.model.to(th.device('cuda'))

        # mobilenetV3 large
        # self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        # self.model.to(th.device('cuda'))

        # summary(self.model, (3, f, f))

        self.__seq_conv = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.Flatten(1, -1),
        )

    @property
    def out_size(self) -> int:
        return self.__out_size

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        out: th.Tensor = self.model(o_t)
        # return out
        max_val = out.max()
        min_val = out.min()
        out = (out - min_val) / (max_val - min_val) * 2 - 1
        return out * 5


############################
# State to features stuff
############################
class StateToFeatures(nn.Module):
    """
    λ_θ7 : R^d -> R^n
    """

    def __init__(self, d: int, n_d: int) -> None:
        super().__init__()

        self.__d = d
        self.__n_d = n_d

        self.__seq_lin = nn.Sequential(
            nn.Linear(self.__d, self.__n_d),
            nn.GELU(),
            Permute([1, 2, 0]),
            nn.BatchNorm1d(self.__n_d),
            Permute([2, 0, 1]),
        )

    def forward(self, p_t: th.Tensor) -> th.Tensor:
        return cast(th.Tensor, self.__seq_lin(p_t))
