'''Tests for the module-level helpers of lpu_nn.common.training

lpu_nn.common.training のモジュール関数のテスト。

`Trainer` 本体から切り離して確かめられる部分を対象とする。
`setup_optimizer` は、bias と 1 次元パラメータ (LayerNorm 等) に
weight decay を掛けないという方針をここで固定する。
'''

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from lpu_nn.common import training


def make_config(**overrides):
    '''Build the configuration object setup_optimizer reads

    setup_optimizer が参照する設定オブジェクトを組み立てる。
    '''
    train = SimpleNamespace(
        optimizer='sgd', weight_decay_rate=0.01,
        adam_alpha=1e-3, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
        adabound_gamma=1e-3, sgd_learning_rate=0.1,
    )
    for key, value in overrides.items():
        setattr(train, key, value)
    return SimpleNamespace(data=SimpleNamespace(train=train))


class TestFormatTime:
    @pytest.mark.parametrize('seconds,expected', [
        (0.0, '0.00S'),
        (1.5, '1.50S'),
        (59.5, '59.50S'),
        (61, '1M1S'),
        (3601, '1H1S'),
    ])
    def test_renders_the_expected_string(self, seconds, expected):
        assert training.format_time(seconds) == expected

    def test_a_duration_over_a_day_carries_no_seconds(self):
        # 1 日を超える場合は秒を落とす
        assert training.format_time(100000) == '1D3H46M'

    def test_exactly_a_minute_is_shown_in_seconds(self):
        # 表示のみの都合で "1M" ではなく "60S" になる
        assert training.format_time(60) == '60S'


class TestGetRecordName:
    def test_extracts_the_name_after_the_record_prefix(self):
        assert training.get_record_name('w/record.best_dev_loss') == 'best_dev_loss'

    def test_returns_none_without_a_record_part(self):
        assert training.get_record_name('workdir/model.pt') is None

    def test_the_latest_record_is_recognized(self):
        assert training.get_record_name('w/record.latest') == 'latest'


@pytest.fixture
def frame():
    '''Three rows whose token counts are 3, 4 and 5

    トークン数が 3, 4, 5 の 3 行。
    '''
    return pd.DataFrame({'len': [3, 4, 5], 'x': ['a', 'b', 'c']})


class TestBuildBatches:
    def test_samples_are_grouped_by_count(self, frame):
        batches = list(training.build_batches(frame, 2, 'samples'))
        assert [len(b) for b in batches] == [2, 1]

    def test_tokens_are_grouped_until_the_budget_is_reached(self, frame):
        # 3 + 4 >= 7 で切れ、残りが次のバッチになる
        batches = list(training.build_batches(frame, 7, 'tokens'))
        assert [b.len.sum() for b in batches] == [7, 5]

    def test_every_row_appears_exactly_once(self, frame):
        for batch_type, size in [('samples', 2), ('tokens', 7)]:
            batches = list(training.build_batches(frame, size, batch_type))
            assert sum(len(b) for b in batches) == len(frame)

    def test_a_row_larger_than_the_budget_stands_alone(self, frame):
        batches = list(training.build_batches(frame, 1, 'tokens'))
        assert [len(b) for b in batches] == [1, 1, 1]

    def test_an_unknown_batch_type_is_rejected(self, frame):
        # 以前は末尾に抜けて None を返し、呼び出し側が反復しようとしていた
        with pytest.raises(ValueError, match='batch type'):
            list(training.build_batches(frame, 2, 'chars'))


class TestReduceBatchSize:
    def test_samples_are_truncated(self, frame):
        assert len(training.reduce_batch_size(frame, 2, 'samples')) == 2

    def test_tokens_drop_the_longest_row_first(self, frame):
        reduced = training.reduce_batch_size(frame.copy(), 7, 'tokens')
        assert reduced.len.sum() <= 7
        assert 5 not in reduced.len.tolist()

    def test_one_row_always_survives(self, frame):
        reduced = training.reduce_batch_size(frame.copy(), 1, 'tokens')
        assert len(reduced) == 1

    def test_an_unknown_batch_type_is_rejected(self, frame):
        with pytest.raises(ValueError, match='batch type'):
            training.reduce_batch_size(frame, 2, 'chars')


class TestSetupOptimizer:
    @pytest.fixture
    def model(self):
        # Linear の weight は 2 次元、bias と LayerNorm は 1 次元
        return nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))

    def test_weight_decay_skips_the_biases_and_one_dimensional_parameters(self, model):
        # bias と LayerNorm に weight decay を掛けないこと
        optimizer = training.setup_optimizer(make_config(optimizer='adam'), model)
        decayed, undecayed = optimizer.param_groups
        assert decayed['weight_decay'] == 0.01
        assert undecayed['weight_decay'] == 0
        assert all(param.dim() > 1 for param in decayed['params'])
        assert all(param.dim() == 1 for param in undecayed['params'])

    def test_every_parameter_lands_in_exactly_one_group(self, model):
        optimizer = training.setup_optimizer(make_config(), model)
        grouped = sum(len(group['params']) for group in optimizer.param_groups)
        assert grouped == len(list(model.parameters()))

    @pytest.mark.parametrize('name,expected', [
        ('adam', 'AdaBoundW'),
        ('amsgrad', 'AdaBoundW'),
        ('adamax', 'Adamax'),
        ('adabound', 'AdaBoundW'),
        ('amsbound', 'AdaBoundW'),
        ('lamb', 'Lamb'),
        ('sgd', 'SGD'),
    ])
    def test_every_name_builds_its_optimizer(self, model, name, expected):
        optimizer = training.setup_optimizer(make_config(optimizer=name), model)
        assert type(optimizer).__name__ == expected

    def test_an_unknown_name_falls_back_to_sgd(self, model):
        config = make_config(optimizer='rmsprop')
        optimizer = training.setup_optimizer(config, model)
        assert type(optimizer).__name__ == 'SGD'
        # 選ばれた名前は設定へ書き戻される
        assert config.data.train.optimizer == 'sgd'

    def test_the_name_is_normalized_to_lower_case(self, model):
        config = make_config(optimizer='LAMB')
        training.setup_optimizer(config, model)
        assert config.data.train.optimizer == 'lamb'

    def test_the_built_optimizer_can_take_a_step(self, model):
        optimizer = training.setup_optimizer(make_config(optimizer='adabound'), model)
        model(torch.randn(2, 4)).pow(2).mean().backward()
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
