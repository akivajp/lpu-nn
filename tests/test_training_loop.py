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


class TestOutOfMemoryRecovery:
    '''The batch-shrinking path that a CUDA out-of-memory error triggers

    CUDA のメモリ不足で働く、バッチ縮小の経路。

    実際に OOM を起こさずに確かめられるよう、`feed_one_batch` に例外を
    注入する。この経路が働かないと、メモリが足りない設定で学習がそのまま
    落ちる。
    '''

    def build_trainer(self, failures, batch_type='samples', batch_size=8):
        '''A trainer whose feed_one_batch fails the first `failures` times

        最初の `failures` 回だけ `feed_one_batch` が失敗する訓練器。
        '''
        import pandas as pd

        calls = []

        class Recorder(training.Trainer):
            default = training.default

            def __init__(self):
                self.model = SimpleNamespace(
                    zero_grad=lambda: None, reset_state=lambda: None,
                    train=lambda: None, eval=lambda: None,
                    device=torch.device('cpu'))
                self.optimizer = SimpleNamespace(zero_grad=lambda: None)
                self.train_df = pd.DataFrame({'criterion': [0.0] * 8,
                                              'feed_count': [0] * 8,
                                              'last_step': [0] * 8,
                                              'last_epoch': [0] * 8})
                self.config = SimpleNamespace(data=_config_data(batch_type, batch_size))
                self.progress = None

            def update_train_step(self, increment=True):
                pass

            def set_max_steps(self, max_steps=None, min_steps=1):
                pass

            def show_progress_report(self):
                pass

            def feed_one_batch(self, batch, fallback=False, df=None):
                calls.append(len(batch))
                if len(calls) <= failures:
                    raise torch.cuda.OutOfMemoryError('CUDA out of memory. Tried ...')
                # 実装は pd.Series を返し、呼び出し側はそれを足し合わせる
                return pd.Series({'loss': 1.0})

        return Recorder(), calls

    def test_an_out_of_memory_error_shrinks_the_batch(self):
        # 縮小が働かなければ、同じ大きさで落ち続ける
        import pandas as pd
        trainer, _calls = self.build_trainer(failures=1)
        batches = [pd.DataFrame({'len': [1] * 4}, index=range(4)) for _ in range(2)]
        before = trainer.config.data.train.batch_size
        training.Trainer.feed_batches(trainer, batches, train=True)
        assert trainer.config.data.train.batch_size < before

    def test_the_error_is_counted(self):
        import pandas as pd
        trainer, _calls = self.build_trainer(failures=1)
        batches = [pd.DataFrame({'len': [1] * 4}, index=range(4)) for _ in range(2)]
        training.Trainer.feed_batches(trainer, batches, train=True)
        assert trainer.progress['total_errors'] == 1

    def test_the_batch_size_never_falls_below_the_minimum(self):
        import pandas as pd
        trainer, _calls = self.build_trainer(failures=5)
        batches = [pd.DataFrame({'len': [1] * 4}, index=range(4)) for _ in range(5)]
        training.Trainer.feed_batches(trainer, batches, train=True)
        cdata = trainer.config.data
        assert cdata.train.batch_size >= cdata.train.min_batch_size

    def test_another_runtime_error_is_not_swallowed(self):
        # メモリ不足以外は縮小せずに送出すること
        import pandas as pd
        trainer, _calls = self.build_trainer(failures=0)

        def always_fail(batch, fallback=False, df=None):
            raise RuntimeError('something else went wrong')

        trainer.feed_one_batch = always_fail
        batches = [pd.DataFrame({'len': [1] * 4}, index=range(4))]
        with pytest.raises(RuntimeError, match='something else'):
            training.Trainer.feed_batches(trainer, batches, train=True)


def _config_data(batch_type, batch_size):
    '''The configuration feed_batches reads

    feed_batches が参照する設定。
    '''
    from lpu.common.config import ConfigData

    data = ConfigData()
    data.train = ConfigData()
    data.train.batch_size = batch_size
    data.train.min_batch_size = 1
    data.train.batch_type = batch_type
    data.train.optimizer = 'lamb'
    data.train.schedule_num_steps = False
    data.train.weight_decay_rate = 0.0
    data.log = ConfigData()
    data.log.epoch = 1
    data.log.elapsed = 0.0
    data.log.fed_tokens = 0
    data.log.train_step = 0
    data.log.interval = 0
    data.model = ConfigData()
    data.model.max_length = 16
    return data
