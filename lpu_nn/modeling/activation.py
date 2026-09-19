#!/usr/bin/env python3

# system
import math
from collections.abc import Sequence

# 3rd
import torch
from torch import nn

# local
from lpu_nn import modeling

def gelu(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    #return swish(x, 1.702, inplace)
    y = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    if inplace:
        x.data = y
        return x
    else:
        return y

def mish(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    y = x * torch.tanh(torch.log(1 + x.exp()))
    if inplace:
        x.data = y
        return x
    else:
        return y

def swish(x: torch.Tensor, beta: "float | torch.Tensor" = 1,
          inplace: bool = False) -> torch.Tensor:
    s = torch.sigmoid(beta * x)
    if inplace:
        return x.mul_(s)
    else:
        return x.mul(s)

class GELU(modeling.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gelu(x, self.inplace)

class Mish(modeling.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mish(x, self.inplace)

class Swish(modeling.Module):
    def __init__(self, shape: "int | Sequence[int]", init_beta: float = 1.0,
                 inplace: bool = False) -> None:
        super().__init__()
        if isinstance(shape, int):
            shape = [shape]
        self.beta = nn.Parameter(torch.full(shape, init_beta))
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish(x, self.beta, self.inplace)

    def extra_repr(self) -> str:
        if self.inplace:
            return f'{self.beta.shape}, inplace'
        else:
            return f'{self.beta.shape}'

class Swish1(modeling.Module):
    """Swish with a fixed beta of 1, i.e. SiLU

    beta を 1 に固定した Swish (すなわち SiLU)。
    """

    def __init__(self, inplace: bool = False) -> None:
        # 以前は super(Swish, self) と別のクラスを渡しており、Swish1 は
        # Swish の派生ではないため get_activator('swish1') は必ず
        # TypeError で落ちていた
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish(x, 1.0, self.inplace)

def get_activator(name: str, size: "int | Sequence[int]") -> nn.Module:
    name = name.lower()
    if name == 'gelu':
        return GELU()
    if name == 'mish':
        return Mish()
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'swish':
        return Swish(size, init_beta=1.0)
    if name == 'swish1':
        return Swish1()
    if name == 'relu':
        return nn.ReLU()
    if name == 'tanh':
        return nn.Tanh()
    raise ValueError(f"unknown name for activation function: {name}")

def get_gain(name: str) -> float:
    if name == 'none':
        return 1.0
    if name == 'gelu':
        #return 1
        #return math.pi / 2
        #return 2 ** 0.5 # RE2
        #return 1.5
        #return math.pi ** (1/3.0) # ~ 1.4656
        return 1.48
    if name == 'mish':
        return 1.45
    if name == 'sigmoid':
        return 1 # pytorch
        #return 2
    if name == 'swish':
        #return 1
        #return 5.0 / 3
        return math.pi / 2
    if name == 'swish1':
        #return 1
        #return 5.0 / 3
        return math.pi / 2
    if name == 'relu':
        return 2 ** 0.5
    if name == 'tanh':
        return 5.0 / 3
    raise ValueError(f"unknown name for activation function: {name}")
