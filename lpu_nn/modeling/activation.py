#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# system
import math

# 3rd
import torch
from torch import nn

# local
from lpu_nn.common.utils import parameterize
from lpu_nn import modeling

def gelu(x, inplace=False):
    #return swish(x, 1.702, inplace)
    y = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    if inplace:
        x.data = y
        return x
    else:
        return y

def mish(x, inplace=False):
    y = x * torch.tanh(torch.log(1 + x.exp()))
    if inplace:
        x.data = y
        return x
    else:
        return y

def swish(x, beta=1, inplace=False):
    """
    :param torch.Tensor x:
    :param float or torch.Tensor beta:
    :param bool inplace:
    :rtype: torch.Tensor
    """
    s = torch.sigmoid(beta * x)
    if inplace:
        return x.mul_(s)
    else:
        return x.mul(s)

class GELU(modeling.Module):
    def __init__(self, inplace=False):
        """
        :param bool inplace:
        """
        super(GELU,self).__init__()
        self.inplace = inplace

    def forward(self, x):
        """
        :param torch.Tensor x:
        :rtype: torch.Tensor
        """
        return gelu(x, self.inplace)

class Mish(modeling.Module):
    def __init__(self, inplace=False):
        """
        :param bool inplace:
        """
        super(Mish,self).__init__()
        self.inplace = inplace

    def forward(self, x):
        """
        :param torch.Tensor x:
        :rtype: torch.Tensor
        """
        return mish(x, self.inplace)

class Swish(modeling.Module):
    def __init__(self, shape, init_beta=1.0, inplace=False):
        """
        :param float or torch.Tensor init_beta:
        :param bool inplace:
        """
        super(Swish,self).__init__()
        if isinstance(shape, int):
            shape = [shape]
        self.beta = nn.Parameter(torch.full(shape, init_beta))
        self.inplace = inplace

    def forward(self, x):
        """
        :param torch.Tensor x:
        :rtype: torch.Tensor
        """
        return swish(x, self.beta, self.inplace)

    def extra_repr(self):
        if self.inplace:
            return '{}, inplace'.format(self.beta.shape)
        else:
            return '{}'.format(self.beta.shape)

class Swish1(modeling.Module):
    def __init__(self, inplace=False):
        """
        :param float or torch.Tensor init_beta:
        :param bool inplace:
        """
        super(Swish,self).__init__()
        self.inplace = inplace

    def forward(self, x):
        """
        :param torch.Tensor x:
        :rtype: torch.Tensor
        """
        return swish(x, 1.0, self.inplace)

def get_activator(name, size):
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
    raise ValueError("unknown name for activation function: {}".format(name))

def get_gain(name):
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
    raise ValueError("unknown name for activation function: {}".format(name))
