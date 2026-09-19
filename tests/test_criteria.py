'''Tests for lpu_nn.common.criteria

lpu_nn.common.criteria のテスト。
期待値は定義から手計算するか、独立した不変量 (平滑化を 0 にすると通常の
交差エントロピーに一致する等) として検証しており、実装そのものを
期待値の出どころにしていない。

Shapes follow the trainer's convention: logits are (B, V, L) and targets
are (B, L).
(形状は訓練側の規約に従う。ロジットは (B, V, L)、正解は (B, L))
'''

import math

import pytest
import torch

from lpu_nn.common import criteria


@pytest.fixture
def uniform_logits():
    '''Two positions over a two-word vocabulary, all logits equal

    語彙 2 語・位置 2 の一様なロジット。各位置の -log p は log(2) になる。
    '''
    return torch.zeros(1, 2, 2)


class TestCrossEntropy:
    def test_uniform_logits_give_log_of_the_vocabulary_size(self, uniform_logits):
        target = torch.tensor([[0, 1]])
        xent = criteria.cross_entropy(uniform_logits, target, reduction='mean')
        assert xent.item() == pytest.approx(math.log(2), abs=1e-6)

    def test_hmean_averages_within_each_sample(self):
        # 位置ごとに損失が異なる場合、hmean はサンプル内平均を返すこと
        logits = torch.zeros(2, 2, 2)
        target = torch.tensor([[0, 1], [1, 0]])
        xent = criteria.cross_entropy(logits, target, reduction='hmean')
        assert xent.shape == (2,)
        assert xent[0].item() == pytest.approx(math.log(2), abs=1e-6)

    def test_ignored_positions_do_not_contribute(self):
        # 無視する位置を増やしても、有効な位置だけの平均は変わらないこと
        logits = torch.zeros(1, 2, 3)
        without_ignored = criteria.cross_entropy(
            logits[:, :, :2], torch.tensor([[0, 1]]), ignore_index=-1, reduction='hmean')
        with_ignored = criteria.cross_entropy(
            logits, torch.tensor([[0, 1, -1]]), ignore_index=-1, reduction='hmean')
        assert with_ignored.item() == pytest.approx(without_ignored.item(), abs=1e-6)

    def test_a_list_of_ignored_indices_is_accepted(self):
        logits = torch.zeros(1, 3, 3)
        xent = criteria.cross_entropy(
            logits, torch.tensor([[0, 1, 2]]), ignore_index=[2], reduction='hmean')
        # 3 位置のうち 1 つを除外しても、一様なので値は log(3) のまま
        assert xent.item() == pytest.approx(math.log(3), abs=1e-6)

    def test_the_input_dtype_is_preserved(self):
        # 内部で float32 に上げても、戻り値は入力の dtype に戻すこと
        logits = torch.zeros(1, 2, 2, dtype=torch.float16)
        xent = criteria.cross_entropy(logits, torch.tensor([[0, 1]]), reduction='mean')
        assert xent.dtype == torch.float16


class TestPerplexity:
    def test_is_the_exponential_of_the_cross_entropy(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 5, 3)
        target = torch.tensor([[1, 2, 0], [3, 4, 1]])
        xent = criteria.cross_entropy(logits, target, reduction='hmean')
        ppl = criteria.perplexity(logits, target, reduction='hmean')
        assert torch.allclose(ppl, torch.exp(xent), atol=1e-4)

    def test_uniform_logits_give_the_vocabulary_size(self, uniform_logits):
        ppl = criteria.perplexity(uniform_logits, torch.tensor([[0, 1]]), reduction='mean')
        assert ppl.item() == pytest.approx(2.0, abs=1e-5)


class TestAccuracy:
    def test_a_perfect_prediction_is_one(self):
        target = torch.tensor([[1, 2, 3]])
        assert criteria.accuracy(target.clone(), target).item() == 1.0

    def test_counts_only_the_valid_positions(self):
        target = torch.tensor([[1, 2, 3, 0]])
        prediction = torch.tensor([[1, 9, 3, 0]])
        # 0 を無視 => 有効 3 件中 2 件正解
        assert criteria.accuracy(prediction, target, ignore_index=0).item() == pytest.approx(2 / 3)

    def test_a_list_of_ignored_indices_masks_every_one_of_them(self):
        # `t_valid != ignore` と書かれていたため、リスト指定では何も無視
        # されず 0.75 が返っていた
        target = torch.tensor([[1, 2, 3, 0]])
        prediction = torch.tensor([[1, 9, 3, 0]])
        # 0 と 3 を無視 => 有効 2 件中 1 件正解
        accuracy = criteria.accuracy(prediction, target, ignore_index=[0, 3])
        assert accuracy.item() == pytest.approx(0.5)

    def test_a_single_element_list_matches_the_integer_form(self):
        target = torch.tensor([[1, 2, 3, 0]])
        prediction = torch.tensor([[1, 9, 3, 0]])
        as_int = criteria.accuracy(prediction, target, ignore_index=0)
        as_list = criteria.accuracy(prediction, target, ignore_index=[0])
        assert as_list.item() == pytest.approx(as_int.item())

    def test_takes_the_argmax_of_logits(self):
        # ロジット (B, V, L) を渡した場合は語彙軸で argmax を取ること
        logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0]]])   # 予測は [0, 1]
        assert criteria.accuracy(logits, torch.tensor([[0, 1]])).item() == 1.0
        assert criteria.accuracy(logits, torch.tensor([[1, 0]])).item() == 0.0

    def test_hmean_reports_one_value_per_sample(self):
        target = torch.tensor([[1, 2], [3, 4]])
        prediction = torch.tensor([[1, 9], [3, 4]])
        accuracy = criteria.accuracy(prediction, target, reduction='hmean')
        assert accuracy.shape == (2,)
        assert accuracy.tolist() == pytest.approx([0.5, 1.0])

    def test_an_unknown_reduction_is_rejected(self):
        target = torch.tensor([[1, 2]])
        with pytest.raises(ValueError):
            criteria.accuracy(target.clone(), target, reduction='sum')


class TestSequenceAccuracy:
    def test_counts_only_the_fully_correct_sequences(self):
        target = torch.tensor([[1, 2], [3, 4]])
        prediction = torch.tensor([[1, 9], [3, 4]])
        # 2 系列中 1 系列が完全一致
        assert criteria.sequence_accuracy(prediction, target).item() == pytest.approx(0.5)

    def test_is_one_when_every_sequence_matches(self):
        target = torch.tensor([[1, 2], [3, 4]])
        assert criteria.sequence_accuracy(target.clone(), target).item() == 1.0

    def test_is_zero_when_no_sequence_matches(self):
        target = torch.tensor([[1, 2], [3, 4]])
        prediction = torch.tensor([[9, 2], [9, 4]])
        assert criteria.sequence_accuracy(prediction, target).item() == 0.0


class TestSmoothedCrossEntropy:
    def test_without_smoothing_it_matches_the_plain_cross_entropy(self):
        # 平滑化を 0 にすると、one-hot 分布との交差エントロピーに一致すること
        torch.manual_seed(0)
        logits = torch.randn(2, 5, 3)
        target = torch.tensor([[1, 2, 0], [3, 4, 1]])
        smoothed = criteria.smoothed_cross_entropy(logits, target, smooth=0.0, reduction='hmean')
        plain = criteria.cross_entropy(logits, target, reduction='hmean')
        assert torch.allclose(smoothed, plain, atol=1e-5)

    def test_reduction_none_keeps_the_position_axis(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 5, 3)
        target = torch.tensor([[1, 2, 0], [3, 4, 1]])
        loss = criteria.smoothed_cross_entropy(logits, target, reduction='none')
        assert loss.shape == (2, 3)

    def test_reduction_mean_is_the_mean_of_the_per_sample_losses(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 5, 3)
        target = torch.tensor([[1, 2, 0], [3, 4, 1]])
        per_sample = criteria.smoothed_cross_entropy(logits, target, reduction='hmean')
        overall = criteria.smoothed_cross_entropy(logits, target, reduction='mean')
        assert overall.item() == pytest.approx(per_sample.mean().item(), abs=1e-5)

    def test_smoothing_raises_the_loss_of_a_confident_correct_prediction(self):
        # 自信のある正解に対しては、平滑化は損失を増やす方向に働くこと
        logits = torch.zeros(1, 4, 1)
        logits[0, 2, 0] = 10.0
        target = torch.tensor([[2]])
        plain = criteria.smoothed_cross_entropy(logits, target, smooth=0.0, reduction='mean')
        smoothed = criteria.smoothed_cross_entropy(logits, target, smooth=0.1, reduction='mean')
        assert smoothed.item() > plain.item()

    def test_an_unknown_reduction_is_rejected(self):
        logits = torch.zeros(1, 2, 2)
        with pytest.raises(ValueError):
            criteria.smoothed_cross_entropy(logits, torch.tensor([[0, 1]]), reduction='sum')
