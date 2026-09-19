#!/usr/bin/env python3

# system
from typing import Any

# 3rd
import torch
from torch import nn

# local
from lpu.common import logging
from lpu_nn import modeling
from lpu_nn.modeling.activation import get_activator
from lpu_nn.modeling.activation import get_gain

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 activation: "str | None" = None, **kwargs: Any) -> None:
        self.activation = activation
        self.hparams = kwargs
        super().__init__(in_features, out_features, bias)
        if activation not in ['none', None]:
            self.mod_activate = get_activator(activation, out_features)

    def init_weights(self, name: "str | None" = None) -> None:
        name = self.__class__.__name__
        initializer = self.hparams.get('initializer')
        if self.activation in [None, "none"]:
            gain = 1.0
        else:
            gain = get_gain(self.activation)
        if initializer in ['orthogonal']:
            logger.debug(f"initializing {name} weight orthogonally")
            nn.init.orthogonal_(self.weight, gain=gain)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        elif initializer in ['he-normal']:
            logger.debug(f"initializing {name} weight with Kaiming He's Normal")
            std = gain / (self.in_features ** 0.5)
            nn.init.normal_(self.weight, std=std)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        else: # if initilizer in ['pytorch', None]:
            logger.debug(f"initializing {name} weight with PyTorch's default method")
            pass # pytorch default initializer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        if self.activation not in ['none', None]:
            output = self.mod_activate(output)
        return output

    def to(self, *args: Any, **kwargs: Any) -> "Linear":
        return modeling.apply_to(self, *args, **kwargs)
