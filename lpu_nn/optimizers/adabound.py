#!/usr/bin/env python3

# system
import math
from collections.abc import Iterable
from typing import Any

# 3rd
import torch
from torch.optim import Optimizer

# local
from lpu.common import logging
logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

# Derived from Luolc/AdaBound (commit d5bb5ff), by the AdaBound authors.
# Licensed under the Apache License 2.0; see licenses/AdaBound-LICENSE.txt
# and licenses/NOTICE.md for the full terms and the list of changes.
# https://github.com/Luolc/AdaBound/blob/d5bb5ffcea39f5733d642567e7fc55cb35ef6c82/adabound/adabound.py
#
# Luolc/AdaBound (コミット d5bb5ff) に由来。Apache License 2.0 で提供される。
# 全文と変更点は licenses/AdaBound-LICENSE.txt と licenses/NOTICE.md を参照。
# (2019/03/07)
# modified to work with float16

class AdaBound(Optimizer):
    """Implements AdaBound algorithm.
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(self, params: Iterable[Any], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 final_lr: float = 0.1, gamma: float = 1e-3,
                 eps: float = 1e-8, weight_decay: float = 0,
                 amsbound: bool = False) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= final_lr:
            raise ValueError(f"Invalid final learning rate: {final_lr}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        defaults = {"lr": lr, "betas": betas, "final_lr": final_lr, "gamma": gamma, "eps": eps,
                        "weight_decay": weight_decay, "amsbound": amsbound}
        super().__init__(params, defaults)

        self.base_lrs = [group['lr'] for group in self.param_groups]

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure: "Any | None" = None) -> Any:
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs, strict=False):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                # base_lr で割るのは、学習率スケジューラによる減衰を
                # final_lr にも反映させるための仕掛け。ただし base_lr が 0
                # だと ZeroDivisionError になる。ウォームアップを 0 から
                # 始めた場合がこれに当たる。base_lr が 0 なら実効学習率も 0
                # なので、上下限を 0 にしてパラメータを動かさない
                final_lr = 0.0 if base_lr == 0 else group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                p.data.add_(-step_size)

        return loss

class AdaBoundW(Optimizer):
    """Implements AdaBound algorithm with Decoupled Weight Decay (arxiv.org/abs/1711.05101)
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(self, params: Iterable[Any], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 final_lr: "float | None" = 0.1, gamma: float = 1e-3,
                 eps: float = 1e-8, weight_decay: float = 0,
                 amsbound: bool = False) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        #if not 0.0 <= final_lr:
        #    raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        defaults = {"lr": lr, "betas": betas, "final_lr": final_lr, "gamma": gamma, "eps": eps,
                        "weight_decay": weight_decay, "amsbound": amsbound}
        super().__init__(params, defaults)

        self.base_lrs = [group['lr'] for group in self.param_groups]

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure: "Any | None" = None) -> Any:
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs, strict=False):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float() # !
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data).float() # !
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data).float() # !
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data).float() # !

                # The moments have to sit on the parameter's device, and in
                # float32 even when the parameter is float16. `.to()` returns a
                # copy whenever it actually moves, so the result must go back
                # into the state: otherwise the in-place updates below land on
                # a copy that is dropped at the end of the step, and the
                # optimizer silently loses its history every step. That is what
                # happens after resuming from a checkpoint saved on the CPU.
                # (モーメントはパラメータと同じデバイス、かつパラメータが
                #  float16 でも float32 に保つ必要がある。`.to()` は実際に
                #  移動する場合にコピーを返すため、結果を state へ書き戻さ
                #  なければならない。さもないと以下のその場更新が、ステップ
                #  終了時に捨てられるコピーに対して行われ、最適化器は毎
                #  ステップ履歴を失う。CPU で保存したチェックポイントから
                #  再開した場合がこれに当たる)
                state['exp_avg'] = exp_avg = state['exp_avg'].to(p.device, torch.float32)
                state['exp_avg_sq'] = exp_avg_sq = state['exp_avg_sq'].to(p.device, torch.float32)
                if amsbound:
                    state['max_exp_avg_sq'] = max_exp_avg_sq = \
                        state['max_exp_avg_sq'].to(p.device, torch.float32)
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                step_size = torch.full_like(denom, step_size)
                final_lr = group.get('final_lr')
                if final_lr is None:
                    step_size.div_(denom).mul_(exp_avg)
                else:
                    # base_lr が 0 のときは ZeroDivisionError になるため、
                    # 実効学習率 0 として上下限も 0 にする (非 W 版と同じ)
                    final_lr = 0.0 if base_lr == 0 else group['final_lr'] * group['lr'] / base_lr
                    lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                    upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                    step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                if group['weight_decay'] != 0:
                    decayed_weights = torch.mul(p.data, group['weight_decay'])
                    p.data.add_(-step_size.to(p.dtype))
                    p.data.sub_(decayed_weights)
                else:
                    p.data.add_(-step_size.to(p.dtype))
        return loss

