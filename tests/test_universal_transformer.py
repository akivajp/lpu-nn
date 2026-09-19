'''Tests for lpu_nn.modeling.universal_transformer and lpu_nn.common.initialization

lpu_nn.modeling.universal_transformer と lpu_nn.common.initialization のテスト。

Universal Transformer は同じ層を繰り返し適用し、ACT (Adaptive Computation
Time) では位置ごとに繰り返し回数を変える。ここでは全ての繰り返し方式が
動作すること、歩数の集計が形として妥当であること、初期化が全モジュールに
行き渡ることを確認する。
'''

import pytest
import torch
from torch import nn

from lpu_nn.common.initialization import apply_init_weights, init_weights
from lpu_nn.modeling.linear import Linear
from lpu_nn.modeling.universal_transformer import UniversalTransformer

RECURRENCES = ['basic', 'act', 'act-prob', 'act-accum']
BATCH, LENGTH, HIDDEN = 2, 4, 8


def build(recurrence='act', max_steps=3):
    '''Build a small Universal Transformer

    小さな Universal Transformer を構築する。
    '''
    return UniversalTransformer(
        embed_size=HIDDEN, hidden_size=HIDDEN, num_heads=2,
        max_steps=max_steps, recurrence=recurrence,
    )


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    seq = torch.randn(BATCH, LENGTH, HIDDEN)
    mask = torch.ones(BATCH, LENGTH, LENGTH, dtype=torch.bool)
    return seq, mask


class TestUniversalTransformer:
    @pytest.mark.parametrize('recurrence', RECURRENCES)
    def test_every_recurrence_preserves_the_shape(self, recurrence, inputs):
        # 'basic' は self.transform という存在しない属性を呼んでおり、
        # 必ず AttributeError になっていた
        seq, mask = inputs
        assert build(recurrence)(seq, mask_self=mask).shape == (BATCH, LENGTH, HIDDEN)

    def test_it_runs_without_an_explicit_reset(self, inputs):
        # last_state は reset_state() でしか作られず、構築直後の forward は
        # AttributeError になっていた
        seq, mask = inputs
        assert build()(seq, mask_self=mask).shape == (BATCH, LENGTH, HIDDEN)

    def test_the_act_recurrence_needs_a_mask(self, inputs):
        seq, _mask = inputs
        with pytest.raises(ValueError, match='mask_self'):
            build('act')(seq)

    def test_the_ponder_cost_is_one_value_per_position(self, inputs):
        seq, mask = inputs
        module = build()
        module(seq, mask_self=mask)
        assert module.get_state()['ponder_cost'].shape == (BATCH, LENGTH)

    def test_the_max_ponder_is_one_value_per_sample(self, inputs):
        # torch.max(..., dim=) は (values, indices) の組を返すため、
        # そのままでは歩数のテンソルにならなかった
        seq, mask = inputs
        module = build()
        module(seq, mask_self=mask)
        max_ponder = module.get_state()['max_ponder']
        assert isinstance(max_ponder, torch.Tensor)
        assert max_ponder.shape == (BATCH,)

    def test_the_ponder_cost_does_not_exceed_the_step_limit(self, inputs):
        seq, mask = inputs
        module = build(max_steps=5)
        module(seq, mask_self=mask)
        assert (module.get_state()['ponder_cost'] <= 5 + 1).all()

    def test_more_steps_are_allowed_to_cost_more(self, inputs):
        seq, mask = inputs
        torch.manual_seed(0)
        few = build(max_steps=2)
        few(seq, mask_self=mask)
        assert few.get_state()['max_ponder'].max() <= 2

    def test_the_step_limit_can_be_overridden_per_call(self, inputs):
        seq, mask = inputs
        module = build(max_steps=8)
        module(seq, mask_self=mask, max_steps=2)
        assert module.get_state()['max_ponder'].max() <= 2

    def test_reset_state_clears_the_steps(self, inputs):
        seq, mask = inputs
        module = build()
        module(seq, mask_self=mask)
        assert module.get_state() != {}
        module.reset_state()
        assert module.get_state() == {}

    def test_set_state_of_none_resets_and_returns_self(self):
        module = build()
        assert module.set_state(None) is module
        assert module.get_state() == {}

    def test_set_state_returns_self(self):
        module = build()
        assert module.set_state({'a': 1}) is module
        assert module.get_state() == {'a': 1}

    def test_init_weights_sets_the_halting_bias_to_one(self):
        # 以前は新しいテンソルを割り当てており、配置先と dtype を失っていた
        module = build()
        module.init_weights()
        assert module.mod_halt[0].bias.tolist() == pytest.approx([1.0])

    def test_the_module_can_be_imported_on_its_own(self):
        # transformer との循環 import で、このモジュールを先に読み込むと
        # ImportError になっていた
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, '-c', 'import lpu_nn.modeling.universal_transformer'],
            capture_output=True)
        assert result.returncode == 0, result.stderr.decode()


class TestInitialization:
    def test_a_module_initializes_its_own_weights(self):
        # 自前の init_weights を持つモジュールにはそれを任せること
        called = []

        class Recorder(nn.Module):
            def init_weights(self, name=None):
                called.append(name)

        init_weights(Recorder(), 'some.path')
        assert called == ['some.path']

    def test_an_initializer_without_a_name_is_retried(self):
        # name を取らない init_weights は引数無しで呼び直されること
        called = []

        class Recorder(nn.Module):
            def init_weights(self):
                called.append(True)

        init_weights(Recorder(), 'some.path')
        assert called == [True]

    def test_an_embedding_is_initialized_orthogonally(self):
        layer = nn.Embedding(8, 4)
        with torch.no_grad():
            layer.weight.zero_()
        init_weights(layer)
        assert layer.weight.abs().sum() > 0

    def test_an_lstm_bias_is_zeroed(self):
        layer = nn.LSTM(4, 4)
        with torch.no_grad():
            for name, param in layer.named_parameters():
                if 'bias' in name:
                    param.fill_(1.0)
        init_weights(layer)
        for name, param in layer.named_parameters():
            if 'bias' in name:
                assert param.abs().sum() == 0

    def test_a_primitive_layer_is_left_to_its_parent(self):
        layer = nn.Linear(4, 4)
        before = layer.weight.clone()
        init_weights(layer)
        assert torch.equal(layer.weight, before)

    def test_a_module_without_weights_is_accepted(self):
        init_weights(nn.Identity())

    def test_apply_reaches_every_nested_module(self):
        called = []

        class Recorder(nn.Module):
            def init_weights(self, name=None):
                called.append(name)

        top = nn.Sequential(Recorder(), nn.Sequential(Recorder()))
        apply_init_weights(top)
        assert len(called) == 2

    def test_apply_runs_over_a_real_model(self):
        model = nn.Sequential(Linear(4, 4, activation='relu', initializer='orthogonal'))
        apply_init_weights(model)
        assert torch.isfinite(model[0].weight).all()
