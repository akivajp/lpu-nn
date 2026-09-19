#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# system
import math

# 3rd party
import torch
from torch import nn

# local
from lpu.common import logging
from lpu_nn.common.utils import purge_tensor

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

def cross_entropy(h, t, ignore_index=-1, reduction='mean', *args, **kwargs):
    dtype = h.dtype
    h = h.float()
    if reduction == 'hmean':
        if isinstance(ignore_index, int):
            t_valid = (t != ignore_index) # (B, L)
            element_wise_entropy = nn.functional.cross_entropy(h, t, ignore_index=ignore_index, reduction='none', *args, **kwargs)
        elif isinstance(ignore_index, list):
            t_valid = (t == t) # (B, L)
            for ignore in ignore_index:
                t_valid = t_valid & (t != ignore)
            element_wise_entropy = nn.functional.cross_entropy(h, t, reduction='none', *args, **kwargs)
            element_wise_entropy = purge_tensor(element_wise_entropy, t_valid, 0.0)
        normalizer = t_valid.sum(1).float() # (B,)
        xent = element_wise_entropy.sum(1) / normalizer # (B,)
    else:
        xent = nn.functional.cross_entropy(h, t, ignore_index=ignore_index, reduction=reduction, *args, **kwargs)
    return xent.to(dtype)

def perplexity(h, t, *args, **kwargs):
    return torch.exp(cross_entropy(h, t, *args, **kwargs))

def accuracy(h, t, ignore_index=-1, reduction='mean'):
    if h.dim() > t.dim():
        pred = h.argmax(dim=1) # (B, L)
    else:
        pred = h
    if isinstance(ignore_index, int):
        t_valid = (t != ignore_index) # (B, L)
    elif isinstance(ignore_index, list):
        t_valid = (t == t)
        for ignore in ignore_index:
            t_valid = t_valid & (t_valid != ignore)
    correct = (pred == t) * t_valid # (B, L)
    if reduction == 'mean':
        return correct.sum().float() / t_valid.sum().float()
    elif reduction == 'hmean':
        return correct.sum(1).float() / t_valid.sum(1).float()
    else:
        raise ValueError("unknown reduction method: {}".format(reduction))

def sequence_accuracy(h, t, ignore_index=-1):
    acc = accuracy(h, t, ignore_index, reduction='hmean') # (B)
    return (acc == 1.0).sum().float() / len(acc)

def smoothed_cross_entropy(h, t, smooth=0.1, ignore_index=-1, reduction='mean'):
    dtype  = h.dtype
    h = h.float()
    batch_size, vocab_size, seq_len = h.shape
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
        raise ValueError("unknown reduction method: {}".format(reduction))

