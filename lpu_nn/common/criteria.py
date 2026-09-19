#!/usr/bin/env python3

# system
import math
from typing import Any

# 3rd party
import torch
from torch import nn

# local
from lpu.common import logging
from lpu_nn.common.utils import purge_tensor

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

def cross_entropy(h: torch.Tensor, t: torch.Tensor, ignore_index: "int | list[int]" = -1,
                  reduction: str = 'mean', **kwargs: Any) -> torch.Tensor:
    """Cross entropy over (B, V, L) logits and (B, L) targets

    (B, V, L) のロジットと (B, L) の正解に対する交差エントロピー。

    `reduction='hmean'` averages within each sample and returns one value
    per sample; the other reductions are passed through to
    `nn.functional.cross_entropy`. A list of ignored indices is supported
    only by `'hmean'`, because that is the only path that builds the mask
    itself.

    `reduction='hmean'` はサンプル内で平均し、サンプルごとに 1 値を返す。
    それ以外の reduction は `nn.functional.cross_entropy` へそのまま渡す。
    無視する添字のリストを受け付けるのは `'hmean'` のみで、マスクを自前で
    構築するのがこの経路だけであるため。

    Extra positional arguments are not accepted: they would be forwarded
    positionally to `nn.functional.cross_entropy`, where they land on
    `weight` / `size_average` / `ignore_index` rather than being passed
    through, and collide with the keywords set here.

    追加の位置引数は受け付けない。`nn.functional.cross_entropy` へ位置の
    まま渡ると `weight` / `size_average` / `ignore_index` に割り当てられて
    しまい、ここで指定しているキーワードと衝突するため。
    """
    dtype = h.dtype
    h = h.float()
    if reduction == 'hmean':
        if isinstance(ignore_index, int):
            t_valid = (t != ignore_index) # (B, L)
            element_wise_entropy = nn.functional.cross_entropy(h, t, ignore_index=ignore_index, reduction='none', **kwargs)
        elif isinstance(ignore_index, list):
            t_valid = (t == t) # (B, L)
            for ignore in ignore_index:
                t_valid = t_valid & (t != ignore)
            element_wise_entropy = nn.functional.cross_entropy(h, t, reduction='none', **kwargs)
            # torch 側に無視指定を渡せないため、マスクした位置を自前で 0 にする
            element_wise_entropy = purge_tensor(element_wise_entropy, t_valid, 0.0)
        else:
            raise TypeError(
                f"ignore_index must be an int or a list of ints, "
                f"given: {type(ignore_index).__name__}"
            )
        normalizer = t_valid.sum(1).float() # (B,)
        xent = element_wise_entropy.sum(1) / normalizer # (B,)
    else:
        if not isinstance(ignore_index, int):
            raise TypeError(
                f"a list of ignored indices is only supported by reduction='hmean', "
                f"given reduction={reduction!r}"
            )
        xent = nn.functional.cross_entropy(h, t, ignore_index=ignore_index, reduction=reduction, **kwargs)
    return xent.to(dtype)

def perplexity(h: torch.Tensor, t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return torch.exp(cross_entropy(h, t, **kwargs))

def accuracy(h: torch.Tensor, t: torch.Tensor, ignore_index: "int | list[int]" = -1,
             reduction: str = 'mean') -> torch.Tensor:
    if h.dim() > t.dim():
        pred = h.argmax(dim=1) # (B, L)
    else:
        pred = h
    if isinstance(ignore_index, int):
        t_valid = (t != ignore_index) # (B, L)
    elif isinstance(ignore_index, list):
        t_valid = (t == t)
        for ignore in ignore_index:
            # `t_valid != ignore` compared the boolean mask against the index,
            # which is True for every ordinary index, so nothing was ever
            # masked out. cross_entropy above has this right.
            # (`t_valid != ignore` は真偽マスクと添字を比較しており、通常の
            #  添字では常に True になるため、何も除外されていなかった。
            #  上の cross_entropy は正しく書かれている)
            t_valid = t_valid & (t != ignore)
    else:
        raise TypeError(f"ignore_index must be an int or a list of ints, given: {type(ignore_index).__name__}")
    correct = (pred == t) * t_valid # (B, L)
    if reduction == 'mean':
        return correct.sum().float() / t_valid.sum().float()
    elif reduction == 'hmean':
        return correct.sum(1).float() / t_valid.sum(1).float()
    else:
        raise ValueError(f"unknown reduction method: {reduction}")

def sequence_accuracy(h: torch.Tensor, t: torch.Tensor,
                      ignore_index: "int | list[int]" = -1) -> torch.Tensor:
    acc = accuracy(h, t, ignore_index, reduction='hmean') # (B)
    return (acc == 1.0).sum().float() / len(acc)

def smoothed_cross_entropy(h: torch.Tensor, t: torch.Tensor, smooth: float = 0.1,
                           ignore_index: int = -1, reduction: str = 'mean') -> torch.Tensor:
    dtype  = h.dtype
    h = h.float()
    _batch_size, vocab_size, _seq_len = h.shape
    t_valid = (t != ignore_index) # (B, L)
    zeros = torch.zeros_like(h)
    t_dummy = purge_tensor(t, t_valid, 0)
    true_dist = zeros.scatter(1, t_dummy[:,None], 1.0)
    uniform_dist = torch.ones_like(h) / (vocab_size - 1) # vocab without <pad>
    smooth_dist = true_dist * (1 - smooth) + uniform_dist * smooth
    neglog_pred = -purge_tensor(h, t_valid[:, None], -math.inf).log_softmax(1) # (B, V, L)
    neglog_pred =  purge_tensor(neglog_pred, t_valid[:, None], 0.0) # (B, V, L)
    element_wise_entropy = (neglog_pred * smooth_dist).sum(1) # (B, L)
    if reduction == 'none':
        return element_wise_entropy.to(dtype) # (B, L)
    normalizer = t_valid.sum(1).float() # (B,)
    batch_loss = element_wise_entropy.sum(1) / normalizer # (B,)
    if reduction == 'hmean':
        return batch_loss.to(dtype)
    elif reduction == 'mean':
        return batch_loss.mean().to(dtype)
    else:
        raise ValueError(f"unknown reduction method: {reduction}")

