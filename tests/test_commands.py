'''Tests for lpu_nn.commands

lpu_nn.commands のテスト。

`Seq2SeqTrainer` の損失・評価指標の算出は、訓練ループから切り離しても
確かめられる。ここでは代用の self を渡して、それらの計算と報告内容だけを
確認する。ロガー名の対象についても、ログファイルに何が入るかを決めるため
併せて固定する。
'''

import math

import pytest
import torch

from lpu_nn.commands import run_seq2seq, train_seq2seq
from lpu_nn.common import training

VOCAB_SIZE, LENGTH = 6, 4


class StubModel:
    '''The attributes the metric helpers read from the model

    指標の算出が参照するモデルの属性だけを持つ代用品。
    '''

    padding = 0
    dtype = torch.float32


class StubConfigData:
    class train:
        loss = 'xent'
        time_penalty = 0.0


class StubConfig:
    data = StubConfigData


class StubTrainer:
    '''A stand-in self for the metric helpers

    指標算出に渡す代用の self。
    '''

    model = StubModel()
    config = StubConfig()


@pytest.fixture
def batch():
    '''Uniform logits and a target that starts with a BOS-like token

    一様なロジットと、先頭に開始記号相当を置いた正解。
    '''
    torch.manual_seed(0)
    logits = torch.zeros(2, VOCAB_SIZE, LENGTH - 1)
    target = torch.tensor([[1, 4, 5, 2], [1, 4, 5, 2]])
    return logits, target


class TestCalcLoss:
    def test_uniform_logits_give_the_log_of_the_vocabulary(self, batch):
        logits, target = batch
        report = {}
        loss = train_seq2seq.Seq2SeqTrainer.calc_loss(
            StubTrainer(), report, logits, target, {})
        assert loss.item() == pytest.approx(math.log(VOCAB_SIZE), abs=1e-5)

    def test_the_cross_entropy_is_reported(self, batch):
        logits, target = batch
        report = {}
        train_seq2seq.Seq2SeqTrainer.calc_loss(StubTrainer(), report, logits, target, {})
        assert report['loss_xent'] == pytest.approx(report['loss'])

    def test_the_ponder_cost_is_added_with_its_penalty(self, batch):
        logits, target = batch

        class Penalized(StubTrainer):
            class config:
                class data:
                    class train:
                        loss = 'xent'
                        time_penalty = 2.0

        report = {}
        plain = train_seq2seq.Seq2SeqTrainer.calc_loss(
            StubTrainer(), {}, logits, target, {})
        with_cost = train_seq2seq.Seq2SeqTrainer.calc_loss(
            Penalized(), report, logits, target, {'ponder_cost': torch.tensor(3.0)})
        assert with_cost.item() == pytest.approx(plain.item() + 6.0, abs=1e-5)

    def test_the_smoothed_loss_is_reported_under_its_own_key(self, batch):
        logits, target = batch

        class Smoothed(StubTrainer):
            class config:
                class data:
                    class train:
                        loss = 'smooth'
                        time_penalty = 0.0

        report = {}
        train_seq2seq.Seq2SeqTrainer.calc_loss(Smoothed(), report, logits, target, {})
        assert 'loss_smooth' in report
        assert 'loss_xent' not in report

    def test_the_reported_values_carry_no_gradient(self, batch):
        # report は float。detach しないと torch が警告する
        logits, target = batch
        logits = logits.requires_grad_(True)
        report = {}
        train_seq2seq.Seq2SeqTrainer.calc_loss(StubTrainer(), report, logits, target, {})
        assert isinstance(report['loss'], float)


class TestCalcPerplexity:
    def test_uniform_logits_give_the_vocabulary_size(self, batch):
        logits, target = batch
        report = {}
        train_seq2seq.Seq2SeqTrainer.calc_perplexity(
            StubTrainer(), report, logits, target)
        assert report['ppl'] == pytest.approx(VOCAB_SIZE, abs=1e-3)

    def test_it_returns_one_value_per_sample(self, batch):
        logits, target = batch
        values = train_seq2seq.Seq2SeqTrainer.calc_perplexity(
            StubTrainer(), {}, logits, target)
        assert len(values) == 2

    def test_a_non_finite_sample_is_reported_as_minus_one(self, batch):
        # 無限大になったサンプルは -1 として返し、平均からは除く
        logits, target = batch
        logits = logits.clone()
        logits[0] = -math.inf
        logits[0, 4] = 0.0
        values = train_seq2seq.Seq2SeqTrainer.calc_perplexity(
            StubTrainer(), {}, logits, target)
        assert len(values) == 2


class TestCalcPonderCost:
    def test_no_transformer_state_gives_no_cost(self):
        state = {}
        cost = train_seq2seq.Seq2SeqTrainer.calc_ponder_cost(
            StubTrainer(), {}, torch.tensor([[4, 5]]), torch.tensor([[4, 5]]), state)
        assert cost == 0
        assert state['ponder_cost'] == 0

    def test_the_encoder_cost_is_averaged_over_the_valid_positions(self):
        # 2 位置のうち 1 つがパディングなら、合計を 1 で割る
        report = {}
        state = {'encoder_state': {'transformer_state': {
            'ponder_cost': torch.tensor([[3.0, 5.0]])}}}
        cost = train_seq2seq.Seq2SeqTrainer.calc_ponder_cost(
            StubTrainer(), report, torch.tensor([[4, 0]]), torch.tensor([[4, 5]]), state)
        assert cost.item() == pytest.approx(8.0)
        assert report['encoder_ponder_cost'] == pytest.approx(8.0)

    def test_the_two_sides_are_added(self):
        report = {}
        state = {
            'encoder_state': {'transformer_state': {'ponder_cost': torch.tensor([[2.0]])}},
            'decoder_state': {'transformer_state': {'ponder_cost': torch.tensor([[3.0]])}},
        }
        cost = train_seq2seq.Seq2SeqTrainer.calc_ponder_cost(
            StubTrainer(), report, torch.tensor([[4]]), torch.tensor([[4]]), state)
        assert cost.item() == pytest.approx(5.0)
        assert report['ponder_cost'] == pytest.approx(5.0)

    def test_a_non_dict_state_is_tolerated(self):
        state = {'encoder_state': {'transformer_state': []}}
        cost = train_seq2seq.Seq2SeqTrainer.calc_ponder_cost(
            StubTrainer(), {}, torch.tensor([[4]]), torch.tensor([[4]]), state)
        assert cost == 0


class TestLoggerTargets:
    def test_the_targets_cover_the_packages_own_loggers(self):
        # ロガー名はドット区切りで、その配下にしか効かない。移植前の
        # 'common' / 'modeling' のままでは lpu_nn.* に届かず、--logging で
        # 指定したファイルに訓練の出力が入らなかった
        assert 'lpu_nn' in training.target_loggers
        assert 'lpu' in training.target_loggers
        assert 'common' not in training.target_loggers

    def test_every_target_is_an_ancestor_of_a_real_logger(self):
        import importlib
        import logging as stdlib_logging

        for name in ['lpu_nn.common.training', 'lpu_nn.modeling.transformer']:
            # ロガーはモジュールの読み込み時に作られるため、先に読み込む
            importlib.import_module(name)
            logger = stdlib_logging.getLogger(name)
            ancestors = set()
            while logger is not None:
                ancestors.add(logger.name)
                logger = logger.parent
            assert ancestors & set(training.target_loggers)


class TestCommandEntryPoints:
    @pytest.mark.parametrize('module', [train_seq2seq, run_seq2seq])
    def test_main_is_callable(self, module):
        assert callable(module.main)

    def test_the_decoder_help_exits_successfully(self, monkeypatch, capsys):
        # --help は要求どおりの動作なので正常終了すること
        monkeypatch.setattr('sys.argv', ['lpu-nn-run-seq2seq', '--help'])
        with pytest.raises(SystemExit) as exc_info:
            run_seq2seq.main()
        assert exc_info.value.code == 0
        assert 'usage' in capsys.readouterr().out.lower()
