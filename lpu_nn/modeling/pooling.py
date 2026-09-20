#!/usr/bin/env python3

# system
from collections.abc import Callable
import math
from typing import Any

# 3rd
import torch
from torch import nn

# local
from lpu.common import logging
from lpu_nn import modeling

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

def sequence_max_pooling(c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # c.shape : (B, Len, H)
    # mask.shape : (B, Len)
    return c.masked_fill(~mask[:,:,None], -math.inf).max(dim=1)[0] # (B,H)

def sequence_average_pooling(c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # c.shape : (B, Len, H)
    # mask.shape : (B, Len)
    dtype = c.dtype
    length = mask.to(dtype).sum(1, keepdim=True) # (B, 1)
    total = c.masked_fill(~mask[:,:,None], 0).sum(dim=1) # (B,H)
    return total / length # (B,H)

def sequence_attention_pooling(
    query: torch.Tensor, memory: torch.Tensor, mask_mem: torch.Tensor,
    map: "Callable[[torch.Tensor], torch.Tensor] | None" = None,
) -> torch.Tensor:
    ## query.shape : (H)
    # query.shape : (B, H) or (1, H)
    # memory.shape : (B, Len, H)
    # mask_mem.shape : (B, Len)
    seq_value = memory
    if map is None:
        seq_key = memory
    else:
        seq_key = map(memory)
    scores = torch.matmul(seq_key, query[:,:,None]) # (B,Len,H) * (B,H,1) -> (B,Len,1)
    #scores = scores.squeeze(2).masked_fill(~mask_mem, -math.inf) # (B,Len)
    scores = scores.squeeze(2).masked_fill(~mask_mem, -10**7) # (B,Len)
    weight = scores.softmax(1)
    weight = weight.masked_fill(~mask_mem, 0)
    pooled = torch.matmul(weight.unsqueeze(1), seq_value) # (B,1,Len) * (B,Len,H) -> (B,1,H)
    return pooled.squeeze(1) # (B,H)

class SequenceMaxPooling(modeling.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return sequence_max_pooling(c, mask)

class SequenceAveragePooling(modeling.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return sequence_average_pooling(c, mask)

class SequenceAttentionPooling(modeling.Module):
    def __init__(self, vector_size: int, map: str = 'linear',
                 activation: "str | None" = None, **kwargs: Any) -> None:
        super().__init__()
        # torch.Tensor(n) は未初期化メモリを返し、NaN を含みうる。
        # 1 次元パラメータは apply_init_weights の汎用処理でも初期化されない
        # (weight.dim() > 1 の枝に入らない) ため、ここで確実に初期化する。
        self.weight = nn.Parameter(torch.empty(vector_size))
        self.scale_dot = vector_size ** -0.5
        if map == 'linear':
            #self.mod_map = nn.Linear(vector_size, vector_size, bias=False)
            self.mod_map = modeling.Linear(
                vector_size, vector_size, bias=True, activation=activation, **kwargs
            )
        self.init_weights()

    def init_weights(self) -> None:
        # 全要素 1.0 の問い合わせベクトルから学習を始める
        # (scale_dot と合わせて score = sum(key) / sqrt(H) となる)
        nn.init.constant_(self.weight, 1.0)
        #dprint(self.weight)
        # mod_map は modeling.Linear 自身が初期化を持つため、ここでは触らない

    def forward(self, c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        #dprint(self.weight.tolist())
        w = self.weight * self.scale_dot
        map = getattr(self, 'mod_map', None)
        #return sequence_attention_pooling(c, mask, w)
        return sequence_attention_pooling(w[None,:], c, mask, map)

    def extra_repr(self) -> str:
        return f'vector_size={self.weight.shape}'

def get_sequence_pooler(name: str, vector_size: int, **kwargs: Any) -> modeling.Module:
    name = name.lower()
    if name == 'max':
        return SequenceMaxPooling()
    if name == 'average':
        return SequenceAveragePooling()
    if name == 'attention':
        return SequenceAttentionPooling(vector_size, **kwargs)
    raise ValueError(f"unknown name for pooling function: {name}")
