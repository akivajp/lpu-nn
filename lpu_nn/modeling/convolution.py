#!/usr/bin/env python3

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

class SequenceConvolution1d(nn.Conv1d):
    def __init__(self, input_size, output_size, ngram_order, activation=None, **kwargs):
        self.hparams = kwargs
        self.input_size  = input_size
        self.output_size = output_size
        self.ngram_order = ngram_order
        self.activation  = activation
        super().__init__(
            in_channels  = self.input_size,
            out_channels = self.output_size,
            kernel_size  = self.ngram_order,
            bias         = True,
        )
        if self.activation not in [None, "none"]:
            self.mod_activate = get_activator(self.activation, self.output_size)

    def init_weights(self, name=None):
        name = self.__class__.__name__
        initializer = self.hparams.get('initializer')
        #dprint(initializer)
        if self.activation in [None, "none"]:
            gain = 1.0
        else:
            gain = get_gain(self.activation)
        if initializer in ['orthogonal']:
            logger.debug(f"initializing {name} weight orthogonally")
            #nn.init.orthogonal_(self.mod_conv.weight, gain=gain)
            nn.init.orthogonal_(self.weight, gain=gain)
            #nn.init.zeros_(self.mod_conv.bias)
            nn.init.zeros_(self.bias)
        elif initializer in ['he-normal']:
            logger.debug(f"initializing {name} weight with Kaiming He's Normal")
            actual_input_size = self.input_size * self.ngram_order
            std = gain / (actual_input_size ** 0.5)
            #nn.init.normal_(self.mod_conv.weight, std=std)
            nn.init.normal_(self.weight, std=std)
            #nn.init.zeros_(self.mod_conv.bias)
            nn.init.zeros_(self.bias)
        else: # if initilizer in ['pytorch', None]:
            logger.debug(f"initializing {name} weight with PyTorch's default method")
            pass # pytorch default initializer

    def forward(self, seq):
        batch_size, _length, input_size = seq.shape
        n = self.ngram_order
        seq = seq.transpose(1,2) # (B, I, L)
        in_seq = seq
        if n == 2:
            pad_right = torch.zeros([batch_size, input_size, 1]).to(self.device)
            in_seq = torch.cat([seq,pad_right], dim=2) # (B, I, L+1)
        elif n >= 3:
            pad_left  = torch.zeros([batch_size, input_size, (n-1)//2]).to(self.device)
            pad_right = torch.zeros([batch_size, input_size, n//2]).to(self.device)
            in_seq = torch.cat([pad_left,seq,pad_right], dim=2) # (B, I, L+K-1)
        #out_seq = self.mod_conv(in_seq) # (B, O, L)
        out_seq = super().forward(in_seq) # (B, O, L)
        out_seq = out_seq.transpose(1, 2) # (B, L, O)
        if self.activation not in [None, "none"]:
            out_seq = self.mod_activate(out_seq)
        return out_seq

    def to(self, *args, **kwargs):
        return modeling.Module.to(self, *args, **kwargs)
