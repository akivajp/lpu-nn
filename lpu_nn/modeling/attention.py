#!/usr/bin/env python3

# system
import math
from typing import Any, cast

# 3rd
import torch
from torch import nn

# local
from lpu.common import logging
from lpu_nn.common.utils import purge_tensor
from lpu_nn import modeling
from lpu_nn.modeling.activation import get_activator
from lpu_nn.modeling.activation import get_gain

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

def get_attention(name: str, **params: Any) -> modeling.Module:
    name = name.lower()
    if name == 'dot':
        return DotAttention(**params)
    if name == 'concat':
        return ConcatAttention(**params)
    if name == 'general':
        return GeneralAttention(**params)
    if name == 'mlp':
        return MLPAttention(**params)
    if name == 'none':
        return NoAttention(**params)
    raise ValueError(f"unknown name for attention layer: {name}")

class NoAttention(modeling.Module):
    def __init__(self, **params: Any) -> None:
        super().__init__()
    #def forward(self, memory, h_dec, **features):
    def forward(self, h_dec: torch.Tensor, memory: torch.Tensor,
                **features: Any) -> torch.Tensor:
        return memory[:,-1]

class AttentionBase(modeling.Module):
    def __init__(self, **params: Any) -> None:
        super().__init__()
        # 注意機構は hidden_size を前提に線形層を組むため、欠けていれば
        # ここで弾く (以前は None * 3 として nn.Linear に渡っていた)
        hidden_size = params.get('hidden_size')
        if hidden_size is None:
            raise ValueError("attention layers require a hidden_size")
        self.hidden_size: int = int(hidden_size)
        self.local_attention = params.get('local_attention')
        self.dropout_ratio = params.get('dropout_ratio', 0.1)
        if self.local_attention:
            self.window_size = params.get('window_size', 5)
            self.mod_focus: nn.Sequential = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
                nn.Dropout(self.dropout_ratio),
                nn.Linear(self.hidden_size, 1),
                nn.Sigmoid(),
            )

    def init_weights(self) -> None:
        if self.local_attention:
            first = cast(nn.Linear, self.mod_focus[0])
            last = cast(nn.Linear, self.mod_focus[3])
            nn.init.orthogonal_(first.weight, gain=get_gain('tanh'))
            first.bias.data.zero_()
            nn.init.orthogonal_(last.weight, gain=get_gain('sigmoid'))
            last.bias.data.zero_()

    def score(self, memory: torch.Tensor, h_dec: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_local_attention(self, attention: torch.Tensor, memory_mask: torch.Tensor,
                            h_dec: torch.Tensor) -> torch.Tensor:
        device = attention.device
        #window_size = 1
        #window_size = 2
        #window_size = 5
        window_size = self.window_size
        eps = 1e-7
        sigma = self.window_size / 2.0
        _batch_size, len_mem = memory_mask.shape
        batch_len = memory_mask.sum(dim=1)[:,None].float() # (B,1)
        #pos = torch.arange(len_mem)[None,:].repeat(batch_size, axis=0) # (B, L)
        pos = torch.arange(len_mem)[None,:].float().to(device) # (1, L)
        focus = self.mod_focus(h_dec) # (B,H) -> (B,1)
        focus = batch_len * focus
        local_weight = torch.exp( - (pos - focus)**2 / (2 * sigma ** 2) ) # (B, L)
        local_mask = (focus - window_size <= pos) & (pos <= focus + window_size)
        #local_weight = purge_variables(local_weight, local_mask, 0)
        attention = attention * local_weight[:,:,None] # (B, L, 1)
        attention = purge_tensor(attention, local_mask, 0)
        # rescaling
        attention = attention / (torch.sum(attention, dim=1)[:,None,:] + eps)
        return attention

    #def forward(self, memory, h_dec, **features):
    def forward(self, h_dec: torch.Tensor, memory: torch.Tensor,
                **features: Any) -> torch.Tensor:
        # memory : (B, L, H) or (B, L, H*2)
        # h_dec : (B, H)
        mask_mem = features['mask_mem'] # mandatory
        scores = self.score(memory, h_dec)
        if mask_mem is not None:
            scores = purge_tensor(scores, mask_mem, -math.inf)
        attention = torch.softmax(scores, dim=1) # (B, L, 1)
        attention = purge_tensor(attention, mask_mem, 0.0)
        if self.local_attention:
            attention = self.get_local_attention(attention, mask_mem, h_dec)
        context = torch.matmul(memory.transpose(1,2), attention) # (B, H, L) * (B, L, 1) -> (B,H,1)
        return context.squeeze(2) # (B,H)

class DotAttention(AttentionBase):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.bidirectional = params.get('bidirectional_encoder')

    def score(self, memory: torch.Tensor, h_dec: torch.Tensor) -> torch.Tensor:
        # (B, L, H) * (B, H, 1) => (B, L, 1)
        return torch.matmul(memory, h_dec[:,:,None]) # (B, L, 1)

    #def forward(self, memory, h_dec, **features):
    def forward(self, h_dec: torch.Tensor, memory: torch.Tensor,
                **features: Any) -> torch.Tensor:
        #memory_mask = features['mask_x']
        if self.bidirectional:
            # (B, L, H*2) -> (B, L, H)
            # The encoder concatenates the two directions in blocks,
            # torch.cat([forward, backward], dim=2), so the halves have to be
            # paired dimension by dimension. split(2, dim=2) instead took
            # adjacent pairs, averaging f0 with f1 and b0 with b1: a mixture
            # of one direction with itself, never the two directions together.
            # (エンコーダは torch.cat([前向き, 後向き], dim=2) とブロックで
            #  連結するため、前半と後半を次元ごとに対応させる必要がある。
            #  split(2, dim=2) は隣り合う 2 要素を取っており、f0 と f1、
            #  b0 と b1 を平均していた。同じ向き同士を混ぜているだけで、
            #  2 つの向きが対応することは無かった)
            forward, backward = memory.chunk(2, dim=2)
            memory = (forward + backward) / 2
        #return super(DotAttention,self).forward(memory, h_dec, **features)
        return super().forward(h_dec, memory, **features)

class ConcatAttention(AttentionBase):
    def __init__(self, **params: Any) -> None:
        self.bidirectional = params.get('bidirectional_encoder')
        super().__init__(**params)
        if self.bidirectional:
            self.mod_linear: nn.Linear = nn.Linear(self.hidden_size*3, 1)
        else:
            self.mod_linear = nn.Linear(self.hidden_size*2, 1)

    def init_weights(self) -> None:
        super().init_weights()
        nn.init.orthogonal_(self.mod_linear.weight)
        self.mod_linear.bias.data.zero_()

    def score(self, memory: torch.Tensor, h_dec: torch.Tensor) -> torch.Tensor:
        # memory : (B, L, H1)
        # h_dec  : (B, H2)
        _batch_size, len_mem, _hidden_size1 = memory.shape
        h_dec = h_dec[:,None,:].repeat(1, len_mem, 1) # (B, L, H2)
        concat = torch.cat([memory, h_dec], dim=2) # (B, L, H1+H2)
        return cast(torch.Tensor, self.mod_linear(concat)) # (B, L, 1)

class GeneralAttention(AttentionBase):
    def __init__(self, **params: Any) -> None:
        self.bidirectional = params.get('bidirectional_encoder')
        super().__init__(**params)
        if self.bidirectional:
            self.mod_linear: nn.Linear = nn.Linear(self.hidden_size*2, self.hidden_size)
        else:
            self.mod_linear = nn.Linear(self.hidden_size, self.hidden_size)

    def init_weights(self) -> None:
        super().init_weights()
        nn.init.orthogonal_(self.mod_linear.weight)
        self.mod_linear.bias.data.zero_()

    def score(self, memory: torch.Tensor, h_dec: torch.Tensor) -> torch.Tensor:
        # (B, nL, H) -> (B, L, H)
        prod = cast(torch.Tensor, self.mod_linear(memory))
        # (B, L, H) * (B, H, 1) -> (B, L, 1)
        return torch.matmul(prod, h_dec[:,:,None])

class MLPAttention(AttentionBase):
    def __init__(self, **params: Any) -> None:
        # MLP 注意は中間層に活性化関数を挟むため、種類の指定が要る
        activation = params.get('activation')
        if activation is None:
            raise ValueError("the mlp attention requires an activation")
        self.activation: str = str(activation)
        self.bidirectional = params.get('bidirectional_encoder')
        super().__init__(**params)
        if self.bidirectional:
            linear1 = nn.Linear(self.hidden_size*3, self.hidden_size)
        else:
            linear1 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.mod_score: nn.Sequential = nn.Sequential(
            linear1,
            get_activator(self.activation, self.hidden_size),
            nn.Linear(self.hidden_size, 1),
        )

    def init_weights(self) -> None:
        super().init_weights()
        first = cast(nn.Linear, self.mod_score[0])
        last = cast(nn.Linear, self.mod_score[2])
        nn.init.orthogonal_(first.weight, gain=get_gain('relu'))
        first.bias.data.zero_()
        nn.init.orthogonal_(last.weight)
        last.bias.data.zero_()

    def score(self, memory: torch.Tensor, h_dec: torch.Tensor) -> torch.Tensor:
        # memory : (B, L, H)
        # h_dec  : (B, H)
        #h_dec = h_dec[:,None,:].expand_as(memory) # (B, L, H)
        _batch_size, len_mem, _hidden_size1 = memory.shape
        h_dec = h_dec[:,None,:].repeat(1, len_mem, 1) # (B, L, H2)
        concat = torch.cat([memory, h_dec], dim=2) # (B, L, H1+H2)
        return cast(torch.Tensor, self.mod_score(concat)) # (B, L, 1)

