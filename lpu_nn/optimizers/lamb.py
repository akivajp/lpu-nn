#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Lamb optimizer."""

# system
import math

# 3rd
import torch
from torch.optim import Optimizer

# local
from lpu.common import logging

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

# Derived from cybertronai/pytorch-lamb (commit ff2245e).
# Copyright (c) 2019 cybertronai. Licensed under the MIT License; see
# licenses/pytorch-lamb-LICENSE.txt and licenses/NOTICE.md.
# https://github.com/cybertronai/pytorch-lamb/blob/ff2245eaa458278b096e682a66c29a2d73f690d7/pytorch_lamb/lamb.py
#
# cybertronai/pytorch-lamb (コミット ff2245e) に由来。MIT ライセンスで提供される。
# 全文と変更点は licenses/pytorch-lamb-LICENSE.txt と licenses/NOTICE.md を参照。
# (2019/05/02)
# modified to work with float16

class Lamb(Optimizer):
    r"""Implements Lamb algorithm.

    It has been proposed in `Reducing BERT Pre-Training Time from 3 Days to 76 Minutes`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.

    .. _Reducing BERT Pre-Training Time from 3 Days to 76 Minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, adam=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    #state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg'] = torch.zeros_like(p.data).float()
                    #state['exp_avg'] = torch.zeros_like(p.data).to(device, dtype)
                    # Exponential moving average of squared gradient values
                    #state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data).float()
                    #state['exp_avg_sq'] = torch.zeros_like(p.data).to(device, dtype)
                #dprint(state['step'])

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                #exp_avg.data    = exp_avg.to(device)
                #exp_avg_sq.data = exp_avg_sq.to(device)
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    #grad.add_(group['weight_decay'], p.data)
                    grad.add_(p.data.float(), alpha=group['weight_decay'])
                    #grad.add_(group['weight_decay'], p.data.to(device, dtype))

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                adam_step = exp_avg / denom
                # L2 norm uses sum, but here since we're dividing, use mean to avoid overflow.
                r1 = p.data.pow(2).mean().sqrt()
                r2 = adam_step.pow(2).mean().sqrt()
                r = 1 if r1 == 0 or r2 == 0 else min(r1/r2, 10)
                #state['r1'] = r1
                #state['r2'] = r2
                #state['r'] = r
                if self.adam:
                    r = 1

                #p.data.add_(-step_size * r, adam_step)
                p.data.add_(adam_step.to(p.dtype), alpha=-step_size * r)
                #p.data.add_(-step_size * r, adam_step.to(p.device, p.dtype))
                #exp_avg.data = exp_avg.cpu()
                #exp_avg_sq.data = exp_avg_sq.cpu()

        return loss

