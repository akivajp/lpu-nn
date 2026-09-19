'''Tests for the parameter update path of lpu_nn.common.training.Trainer

lpu_nn.common.training.Trainer のパラメータ更新経路のテスト。

`update_parameters` は逆伝播・勾配クリッピング・最適化の 1 ステップを担う。
非有限な損失や勾配を見送る判断がここにあり、これが効かないとモデルが
NaN に落ちたまま学習が続く。
'''

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lpu_nn.common import training


class StubTrainer:
    '''A stand-in self with just what update_parameters touches

    update_parameters が触れるものだけを持つ代用の self。
    '''

    def __init__(self, model, optimizer, gradient_clipping=1.0, move_optimizer=False):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device('cpu')
        self.args = SimpleNamespace(move_optimizer=move_optimizer)
        self.config = SimpleNamespace(data=SimpleNamespace(
            train=SimpleNamespace(gradient_clipping=gradient_clipping)))


@pytest.fixture
def setup():
    torch.manual_seed(0)
    # update_parameters は model.device を見る。実モデルでは
    # modeling.Module がこれを持つ
    model = modeling_module_with(nn.Linear(4, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer


def modeling_module_with(inner):
    """Wrap a layer in the package's Module so it carries a device

    パッケージの Module で包み、device 属性を持たせる。
    """
    from lpu_nn import modeling

    class Wrapped(modeling.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)

    return Wrapped()


def make_loss(model, scale=1.0):
    '''Build a loss whose gradient is well behaved

    勾配が素直に出る損失を作る。
    '''
    torch.manual_seed(1)
    return model(torch.randn(3, 4)).pow(2).mean() * scale


class TestUpdateParameters:
    def test_a_step_changes_the_parameters(self, setup):
        model, optimizer = setup
        trainer = StubTrainer(model, optimizer)
        before = model.inner.weight.detach().clone()
        training.Trainer.update_parameters(trainer, make_loss(model), report={})
        assert not torch.equal(before, model.inner.weight.detach())

    def test_a_non_finite_loss_is_skipped(self, setup):
        # NaN のまま更新すると、以降のパラメータが全て NaN になる
        model, optimizer = setup
        trainer = StubTrainer(model, optimizer)
        before = model.inner.weight.detach().clone()
        loss = make_loss(model) * float('nan')
        training.Trainer.update_parameters(trainer, loss, report={})
        assert torch.equal(before, model.inner.weight.detach())

    def test_an_infinite_loss_is_skipped(self, setup):
        model, optimizer = setup
        trainer = StubTrainer(model, optimizer)
        before = model.inner.weight.detach().clone()
        training.Trainer.update_parameters(trainer, make_loss(model) * float('inf'), report={})
        assert torch.equal(before, model.inner.weight.detach())

    def test_the_gradient_norm_is_reported(self, setup):
        model, optimizer = setup
        report = {}
        training.Trainer.update_parameters(
            StubTrainer(model, optimizer), make_loss(model), report=report)
        assert float(report['gnorm']) >= 0

    def test_a_clipped_step_is_flagged(self, setup):
        # 大きな勾配はクリップされ、その旨が報告されること
        model, optimizer = setup
        report = {}
        trainer = StubTrainer(model, optimizer, gradient_clipping=1e-6)
        training.Trainer.update_parameters(trainer, make_loss(model, 100.0), report=report)
        assert report['clip'] == 1

    def test_a_small_gradient_is_not_flagged(self, setup):
        model, optimizer = setup
        report = {}
        trainer = StubTrainer(model, optimizer, gradient_clipping=1e6)
        training.Trainer.update_parameters(trainer, make_loss(model), report=report)
        assert report['clip'] == 0

    def test_it_accepts_no_report(self, setup):
        # 署名は report=None を許すが、勾配ノルムの代入だけ確認していなかった
        model, optimizer = setup
        training.Trainer.update_parameters(
            StubTrainer(model, optimizer), make_loss(model), report=None)
        assert torch.isfinite(model.inner.weight).all()

    def test_the_parameters_stay_finite_after_a_step(self, setup):
        model, optimizer = setup
        training.Trainer.update_parameters(
            StubTrainer(model, optimizer), make_loss(model), report={})
        assert all(torch.isfinite(p).all() for p in model.parameters())


class TestOptimizerAttributes:
    def test_the_weight_decay_lives_in_the_param_groups(self):
        # Chainer 時代の名前 weight_decay_rate は torch の最適化器に無い
        from lpu_nn import optimizers
        optimizer = optimizers.AdaBoundW(
            [torch.zeros(2, requires_grad=True)], lr=1e-3, weight_decay=0.02)
        assert not hasattr(optimizer, 'weight_decay_rate')
        assert optimizer.param_groups[0]['weight_decay'] == 0.02
