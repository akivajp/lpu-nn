'''Tests for the progress report of lpu_nn.common.training.Trainer

lpu_nn.common.training.Trainer の進捗表示のテスト。

訓練中に一定間隔で出る一行の指標。計算を誤っても実行は成功し、表示される
数字が静かに間違うだけなので、各項目の値と、欠損時に例外で消えないことを
固定する。
'''

import logging as stdlib_logging
import re
from types import SimpleNamespace

import pandas as pd
import pytest

from lpu.common.config import ConfigData
from lpu_nn.common import training


class StubTrainer:
    '''A trainer holding just what show_progress_report reads

    show_progress_report が読むものだけを持つ訓練器。
    '''

    def __init__(self, report_fields, mean_values=None, optimizer_name='lamb',
                 training_mode=True):
        self.specific = SimpleNamespace(
            data=SimpleNamespace(report=report_fields))
        self.model = SimpleNamespace(training=training_mode)
        self.optimizer = SimpleNamespace(param_groups=[{
            'lr': 0.001, 'final_lr': 0.1, 'gamma': 1e-3}])
        self.config = SimpleNamespace(data=_config_data(optimizer_name))
        values = mean_values if mean_values is not None else {}
        self.progress = ConfigData()
        self.progress['accum_batches'] = 1 if values else 0
        self.progress['accum_report'] = pd.Series(values) if values else 0
        self.progress['accum_errors'] = 3
        self.progress['elapsed'] = 0.0
        self.progress['last_time'] = 0.0
        self.progress['last_tokens'] = 0
        self.progress.num_batches = 10
        self.progress.batch_i = 4


def _config_data(optimizer_name):
    data = ConfigData()
    data.train = ConfigData()
    data.train.optimizer = optimizer_name
    data.log = ConfigData()
    data.log.epoch = 7
    data.log.train_step = 100
    data.log.fed_samples = 1234
    data.log.fed_tokens = 5678
    data.log.elapsed = 0.0
    return data


def render(trainer, caplog):
    '''Run the report and return the line it logged

    進捗表示を実行し、出力された一行を返す。
    '''
    with caplog.at_level(stdlib_logging.INFO, logger='lpu_nn.common.training'):
        training.Trainer.show_progress_report(trainer)
    return caplog.records[-1].getMessage()


class TestProgressFields:
    def test_the_epoch_is_shown(self, caplog):
        assert 'epoch: 7' in render(StubTrainer(['epoch']), caplog)

    def test_the_batch_position_is_padded_to_the_total(self, caplog):
        line = render(StubTrainer(['proc']), caplog)
        assert 'proc:  5/10' in line

    def test_the_error_count_is_shown(self, caplog):
        assert 'err: 3' in render(StubTrainer(['err']), caplog)

    def test_the_fed_counts_are_grouped(self, caplog):
        line = render(StubTrainer(['samples', 'tokens']), caplog)
        assert 'samples: 1,234' in line
        assert 'tokens: 5,678' in line

    def test_the_step_count_is_shown(self, caplog):
        assert 'steps: 100' in render(StubTrainer(['steps']), caplog)

    def test_the_counts_are_hidden_outside_training(self, caplog):
        # 評価中は、訓練の歩数やサンプル数は意味を持たない
        line = render(StubTrainer(['samples', 'steps'], training_mode=False), caplog)
        assert 'samples' not in line
        assert 'steps' not in line

    def test_the_averages_come_from_the_accumulated_report(self, caplog):
        trainer = StubTrainer(['loss', 'acc', 'ppl'],
                              mean_values={'loss': 2.0, 'acc': 0.5, 'ppl': 7.0})
        line = render(trainer, caplog)
        assert 'loss: 2.0000' in line
        assert 'acc: 0.5000' in line
        assert 'ppl: 7.000' in line

    def test_a_missing_metric_does_not_remove_the_others(self, caplog):
        # 欠損した項目は nan として出し、後続の項目を巻き添えにしないこと
        trainer = StubTrainer(['loss', 'acc'], mean_values={'loss': 2.0})
        line = render(trainer, caplog)
        assert 'loss: 2.0000' in line
        assert 'acc: nan' in line

    @pytest.mark.parametrize('field', ['rest_acc', 'cont_acc'])
    def test_a_missing_restore_metric_is_formatted_as_a_number(self, caplog, field):
        # float を通さないと、欠損時の 'nan' が数値書式に渡って
        # ValueError になり、項目ごと表示から消えていた
        trainer = StubTrainer([field], mean_values={'loss': 1.0})
        assert f'{field}: nan' in render(trainer, caplog)

    def test_the_learning_rate_is_shown(self, caplog):
        assert 'lr: 0.00100000' in render(StubTrainer(['lr']), caplog)

    def test_the_adabound_lower_bound_raises_the_shown_rate(self, caplog):
        # 動的な下限のほうが大きければ、そちらが実効学習率になる
        line = render(StubTrainer(['lr'], optimizer_name='adabound'), caplog)
        assert 'lr: 0.00100000' not in line

    def test_the_amsbound_variant_is_recognized(self, caplog):
        # 'amdbound' と綴られており、この指定では下限が反映されなかった
        adabound = render(StubTrainer(['lr'], optimizer_name='adabound'), caplog)
        amsbound = render(StubTrainer(['lr'], optimizer_name='amsbound'), caplog)
        assert amsbound == adabound

    def test_an_unknown_field_is_skipped_without_failing(self, caplog):
        line = render(StubTrainer(['epoch', 'no_such_field', 'err']), caplog)
        assert 'epoch: 7' in line
        assert 'err: 3' in line


class TestProgressReset:
    def test_the_accumulators_are_cleared(self, caplog):
        trainer = StubTrainer(['epoch'], mean_values={'loss': 1.0})
        render(trainer, caplog)
        assert trainer.progress['accum_batches'] == 0
        assert trainer.progress['accum_errors'] == 0
        assert trainer.progress['accum_report'] == 0

    def test_the_token_mark_moves_to_the_current_count(self, caplog):
        trainer = StubTrainer(['epoch'])
        render(trainer, caplog)
        assert trainer.progress['last_tokens'] == 5678

    def test_the_elapsed_time_accumulates(self, caplog):
        trainer = StubTrainer(['epoch'])
        render(trainer, caplog)
        assert trainer.progress['elapsed'] > 0

    def test_the_token_rate_survives_a_zero_interval(self, caplog):
        # 同じ時刻に 2 度呼ばれても 0 除算で落ちないこと
        trainer = StubTrainer(['tokens/s'])
        trainer.progress['last_time'] = None
        import time as time_module
        trainer.progress['last_time'] = time_module.time()
        line = render(trainer, caplog)
        assert isinstance(line, str)


class TestReportFormat:
    def test_the_fields_are_separated_by_commas(self, caplog):
        line = render(StubTrainer(['epoch', 'err', 'steps']), caplog)
        assert re.match(r'epoch: 7, err: 3, steps: 100$', line)
