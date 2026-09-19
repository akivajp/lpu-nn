'''Tests for the best-model bookkeeping of lpu_nn.common.training.Trainer

lpu_nn.common.training.Trainer のベストモデル判定のテスト。

どのチェックポイントが残るかを決める部分であり、比較の向きを取り違えても
訓練は最後まで走り、最悪のモデルが `record.best_*` として残るだけになる。
指標ごとの向きをここで固定する。
'''

from types import SimpleNamespace

import pandas as pd
import pytest

from lpu.common.config import Config
from lpu_nn.common import training

# 小さいほど良い指標と、大きいほど良い指標
LOWER_IS_BETTER = ['loss', 'ppl', 'ntppl', 'mr']
HIGHER_IS_BETTER = ['acc', 'acc_seq', 'ntacc', 'bleu', 'mrr', 'map',
                    'precision', 'recall', 'f1']


class StubTrainer:
    '''A trainer that records which records it was asked to link

    どの記録へリンクするよう求められたかを記録する訓練器。
    '''

    def __init__(self):
        self.config = Config(training.default)
        self.linked = []
        self.saved_configs = []

    def save_config(self, path, record=None, log=True):
        self.saved_configs.append(record)

    def link_status(self, path, src_record, dist_record, log=True):
        self.linked.append(dist_record)

    def save_best_status(self, *args, **kwargs):
        return training.Trainer.save_best_status(self, *args, **kwargs)

    def save_last_score(self, *args, **kwargs):
        return training.Trainer.save_last_score(self, *args, **kwargs)


def call(trainer, report, name, display, ascend, best='best'):
    '''Invoke save_best_status with a stand-in self

    代用の self を渡して save_best_status を呼ぶ。
    '''
    return training.Trainer.save_best_status(
        trainer, 'workdir', report, name, display, ascend=ascend, best=best)


class TestSaveBestStatus:
    def test_the_first_score_is_always_kept(self):
        trainer = StubTrainer()
        assert call(trainer, {'loss': 5.0}, 'loss', 'dev loss', ascend=False)
        assert trainer.linked == ['best_dev_loss']

    def test_a_lower_score_wins_when_descending(self):
        trainer = StubTrainer()
        call(trainer, {'loss': 5.0}, 'loss', 'dev loss', ascend=False)
        assert call(trainer, {'loss': 4.0}, 'loss', 'dev loss', ascend=False)

    def test_a_higher_score_loses_when_descending(self):
        trainer = StubTrainer()
        call(trainer, {'loss': 5.0}, 'loss', 'dev loss', ascend=False)
        assert not call(trainer, {'loss': 6.0}, 'loss', 'dev loss', ascend=False)

    def test_a_higher_score_wins_when_ascending(self):
        trainer = StubTrainer()
        call(trainer, {'bleu': 0.1}, 'bleu', 'dev bleu', ascend=True)
        assert call(trainer, {'bleu': 0.2}, 'bleu', 'dev bleu', ascend=True)

    def test_a_lower_score_loses_when_ascending(self):
        trainer = StubTrainer()
        call(trainer, {'bleu': 0.2}, 'bleu', 'dev bleu', ascend=True)
        assert not call(trainer, {'bleu': 0.1}, 'bleu', 'dev bleu', ascend=True)

    def test_an_equal_score_does_not_update(self):
        trainer = StubTrainer()
        call(trainer, {'loss': 5.0}, 'loss', 'dev loss', ascend=False)
        assert not call(trainer, {'loss': 5.0}, 'loss', 'dev loss', ascend=False)

    @pytest.mark.parametrize('score', [float('nan'), float('inf'), float('-inf')])
    def test_a_non_finite_score_is_ignored(self, score):
        # NaN を最良として残すと、以降どの値でも更新されなくなる
        trainer = StubTrainer()
        assert not call(trainer, {'loss': score}, 'loss', 'dev loss', ascend=False)
        assert trainer.linked == []

    def test_a_missing_score_is_ignored(self):
        trainer = StubTrainer()
        assert not call(trainer, {}, 'loss', 'dev loss', ascend=False)

    def test_a_non_numeric_score_is_ignored(self):
        trainer = StubTrainer()
        assert not call(trainer, {'loss': 'n/a'}, 'loss', 'dev loss', ascend=False)

    def test_the_record_name_follows_the_display_name(self):
        trainer = StubTrainer()
        call(trainer, {'ppl': 1.0}, 'ppl', 'worst dev ppl', ascend=False, best='min')
        assert trainer.linked == ['min_worst_dev_ppl']

    def test_no_model_is_linked_when_not_asked(self):
        trainer = StubTrainer()
        training.Trainer.save_best_status(
            trainer, 'workdir', {'mrr': 0.5}, 'mrr', 'dev mrr',
            ascend=True, save_model=False)
        assert trainer.linked == []
        # 設定そのものは残る
        assert trainer.saved_configs == ['tmp']


class TestMetricDirections:
    '''The direction each metric is compared in, read off update_best_scores

    update_best_scores が各指標に与えている比較の向き。
    '''

    @pytest.fixture
    def directions(self):
        import inspect
        import re
        source = inspect.getsource(training.Trainer.update_best_scores)
        found = {}
        for match in re.finditer(
                r"save_best_status\([^,]+, [a-z_]+_report, '([a-z_0-9]+)',\s*'([^']+)'"
                r"(?:,\s*ascend=(True|False))?", source):
            _name, display, ascend = match.groups()
            found.setdefault(display, ascend)
        return found

    @pytest.mark.parametrize('metric', LOWER_IS_BETTER)
    def test_a_lower_is_better_metric_descends(self, directions, metric):
        for display, ascend in directions.items():
            if display.endswith(' ' + metric.replace('_', ' ')) or display.endswith(metric):
                assert ascend == 'False', f'{display} should compare descending'

    @pytest.mark.parametrize('metric', HIGHER_IS_BETTER)
    def test_a_higher_is_better_metric_ascends(self, directions, metric):
        for display, ascend in directions.items():
            if display.endswith(metric.replace('_', ' ')):
                assert ascend == 'True', f'{display} should compare ascending'

    def test_the_worst_perplexity_is_tracked_as_a_minimum(self, directions):
        # 最悪 perplexity は小さいほど良く、record.min_* として残る
        assert directions['worst dev ppl'] == 'False'
        assert directions['worst train ppl'] == 'False'


class TestUpdateBestScores:
    def test_the_worst_perplexity_comes_from_the_samples(self):
        # サンプルごとの criterion の最大値が worst_ppl になる
        trainer = StubTrainer()
        trainer.dev_df = pd.DataFrame({'criterion': [1.0, 9.0, -1.0]})
        trainer.train_df = pd.DataFrame({'criterion': [-1.0]})
        report = {'ppl': 3.0}
        training.Trainer.update_best_scores(
            trainer, 'workdir', dev_report=report)
        assert report['worst_ppl'] == 9.0

    def test_no_worst_perplexity_without_computed_criteria(self):
        trainer = StubTrainer()
        trainer.dev_df = pd.DataFrame({'criterion': [-1.0, -1.0]})
        report = {'ppl': 3.0}
        training.Trainer.update_best_scores(trainer, 'workdir', dev_report=report)
        assert 'worst_ppl' not in report

    def test_a_missing_report_is_accepted(self):
        trainer = StubTrainer()
        training.Trainer.update_best_scores(trainer, 'workdir')
        assert trainer.linked == []


class TestHostname:
    def test_the_hostname_is_read_portably(self):
        # os.uname は Unix 専用で、Windows では AttributeError になる
        import inspect
        source = inspect.getsource(training.Trainer.train_epoch)
        # 実際の呼び出しのみを見る (コメント中の言及に当たらないよう括弧付き)
        assert 'os.uname()' not in source
        assert 'platform.node()' in source


def _unused(*_args, **_kwargs):
    return SimpleNamespace()
