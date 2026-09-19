'''Tests for lpu_nn.optimizers

lpu_nn.optimizers のテスト。

AdaBound と LAMB は上流の実装に由来し、float16 のパラメータを扱えるよう
内部状態を float32 に保つ改変が入っている (詳細は licenses/NOTICE.md)。
ここでは最適化アルゴリズムの不変量 (勾配ゼロなら動かない、学習率 0 なら
動かない、状態が蓄積される) と、その改変が意図どおり働くことを確認する。
'''

import pytest
import torch
from torch import nn

from lpu_nn import optimizers

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available')

FACTORIES = {
    'AdaBound': lambda params, **kw: optimizers.AdaBound(params, **kw),
    'AdaBoundW': lambda params, **kw: optimizers.AdaBoundW(params, **kw),
    'Lamb': lambda params, **kw: optimizers.Lamb(params, **kw),
}


def make_model(dtype=torch.float32, device='cpu'):
    '''A tiny model with a reproducible initialization

    再現可能な初期化を持つ小さなモデル。
    '''
    torch.manual_seed(0)
    return nn.Linear(4, 2).to(device, dtype)


def take_step(model, optimizer, dtype=torch.float32, device='cpu'):
    '''Run one backward pass and one optimizer step

    逆伝播と最適化を 1 ステップ実行する。
    '''
    torch.manual_seed(1)
    model(torch.randn(3, 4, device=device, dtype=dtype)).pow(2).mean().backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)


class TestEveryOptimizer:
    @pytest.mark.parametrize('name', list(FACTORIES))
    def test_a_step_moves_the_parameters(self, name):
        model = make_model()
        optimizer = FACTORIES[name](model.parameters(), lr=1e-2)
        before = model.weight.detach().clone()
        take_step(model, optimizer)
        assert not torch.equal(before, model.weight.detach())

    @pytest.mark.parametrize('name', list(FACTORIES))
    def test_a_zero_gradient_leaves_the_parameters_alone(self, name):
        model = make_model()
        optimizer = FACTORIES[name](model.parameters(), lr=1e-2)
        for param in model.parameters():
            param.grad = torch.zeros_like(param)
        before = model.weight.detach().clone()
        optimizer.step()
        assert torch.equal(before, model.weight.detach())

    @pytest.mark.parametrize('name', list(FACTORIES))
    def test_a_zero_learning_rate_leaves_the_parameters_alone(self, name):
        model = make_model()
        optimizer = FACTORIES[name](model.parameters(), lr=0.0)
        before = model.weight.detach().clone()
        take_step(model, optimizer)
        assert torch.equal(before, model.weight.detach())

    @pytest.mark.parametrize('name', list(FACTORIES))
    def test_the_step_counter_advances(self, name):
        model = make_model()
        optimizer = FACTORIES[name](model.parameters(), lr=1e-2)
        take_step(model, optimizer)
        take_step(model, optimizer)
        assert optimizer.state[next(model.parameters())]['step'] == 2

    @pytest.mark.parametrize('name', ['AdaBoundW', 'Lamb'])
    def test_the_moments_stay_float32_for_a_float16_parameter(self, name):
        # 上流への改変の目的そのもの: パラメータが float16 でも内部状態は
        # float32 に保つ。AdaBound (非 W) は上流のままでこの改変を含まない
        model = make_model(torch.float16)
        optimizer = FACTORIES[name](model.parameters(), lr=1e-2)
        take_step(model, optimizer, torch.float16)
        state = optimizer.state[next(model.parameters())]
        assert state['exp_avg'].dtype == torch.float32
        assert model.weight.dtype == torch.float16

    @pytest.mark.parametrize('name', list(FACTORIES))
    def test_a_sparse_gradient_is_rejected(self, name):
        model = make_model()
        optimizer = FACTORIES[name](model.parameters(), lr=1e-2)
        weight = model.weight
        weight.grad = torch.sparse_coo_tensor(
            torch.zeros(2, 1, dtype=torch.long), torch.ones(1), weight.shape)
        with pytest.raises(RuntimeError):
            optimizer.step()

    @pytest.mark.parametrize('name', list(FACTORIES))
    def test_a_parameter_without_a_gradient_is_skipped(self, name):
        model = make_model()
        optimizer = FACTORIES[name](model.parameters(), lr=1e-2)
        before = model.bias.detach().clone()
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        assert torch.equal(before, model.bias.detach())

    @pytest.mark.parametrize('name', list(FACTORIES))
    def test_weight_decay_shrinks_a_parameter_with_no_gradient(self, name):
        model = make_model()
        optimizer = FACTORIES[name](model.parameters(), lr=1e-2, weight_decay=0.5)
        for param in model.parameters():
            param.grad = torch.zeros_like(param)
        before = model.weight.detach().abs().sum().item()
        optimizer.step()
        assert model.weight.detach().abs().sum().item() < before


class TestAdaBoundValidation:
    @pytest.mark.parametrize('kwargs', [
        {'lr': -1.0},
        {'eps': -1.0},
        {'betas': (-0.1, 0.999)},
        {'betas': (0.9, 1.0)},
        {'gamma': 1.0},
    ])
    def test_an_invalid_hyper_parameter_is_rejected(self, kwargs):
        model = make_model()
        with pytest.raises(ValueError):
            optimizers.AdaBoundW(model.parameters(), **kwargs)


class TestAdaBoundW:
    def test_final_lr_none_skips_the_dynamic_bound(self):
        # 上流は final_lr を必須とし、常に学習率をクランプする。この移植では
        # None を許し、その場合クランプしない (実質 Adam として動く)
        model = make_model()
        optimizer = optimizers.AdaBoundW(model.parameters(), lr=1e-2, final_lr=None)
        take_step(model, optimizer)
        assert torch.isfinite(model.weight).all()

    def test_the_amsbound_variant_keeps_a_running_maximum(self):
        model = make_model()
        optimizer = optimizers.AdaBoundW(
            model.parameters(), lr=1e-2, final_lr=None, amsbound=True)
        take_step(model, optimizer)
        state = optimizer.state[next(model.parameters())]
        assert 'max_exp_avg_sq' in state
        assert (state['max_exp_avg_sq'] >= 0).all()

    @pytest.mark.parametrize('amsbound', [False, True])
    @CUDA
    def test_the_state_follows_the_parameters_across_devices(self, amsbound):
        # 状態がパラメータと別のデバイスにあると (CPU で保存した
        # チェックポイントからの再開など)、`.to()` が返すコピーに対して
        # その場更新が行われ、書き戻されないまま捨てられていた
        model = make_model(device='cuda')
        optimizer = optimizers.AdaBoundW(
            model.parameters(), lr=1e-2, final_lr=None, amsbound=amsbound)
        take_step(model, optimizer, device='cuda')
        param = next(model.parameters())
        state = optimizer.state[param]
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.cpu()
        before = state['exp_avg'].norm().item()
        for _ in range(3):
            take_step(model, optimizer, device='cuda')
        after = optimizer.state[param]
        assert after['exp_avg'].device == param.device
        assert after['exp_avg'].norm().item() != pytest.approx(before)


class TestLamb:
    def test_the_adam_mode_drops_the_trust_ratio(self):
        # adam=True は信頼比を 1 に固定する
        model = make_model()
        optimizer = optimizers.Lamb(model.parameters(), lr=1e-2, adam=True)
        take_step(model, optimizer)
        assert torch.isfinite(model.weight).all()

    def test_the_trust_ratio_changes_the_step(self):
        adam_model = make_model()
        lamb_model = make_model()
        take_step(adam_model, optimizers.Lamb(adam_model.parameters(), lr=1e-2, adam=True))
        take_step(lamb_model, optimizers.Lamb(lamb_model.parameters(), lr=1e-2, adam=False))
        assert not torch.allclose(adam_model.weight, lamb_model.weight)

    @pytest.mark.parametrize('kwargs', [{'lr': -1.0}, {'eps': -1.0}, {'betas': (0.9, 1.0)}])
    def test_an_invalid_hyper_parameter_is_rejected(self, kwargs):
        model = make_model()
        with pytest.raises(ValueError):
            optimizers.Lamb(model.parameters(), **kwargs)


class TestOptimizerExports:
    @pytest.mark.parametrize('name', ['AdaBound', 'AdaBoundW', 'Lamb', 'Adamax', 'SGD'])
    def test_every_optimizer_the_trainer_selects_is_exported(self, name):
        assert hasattr(optimizers, name)
        assert name in optimizers.__all__
