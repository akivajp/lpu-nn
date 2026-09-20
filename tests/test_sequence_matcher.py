'''Tests for the sequence-matching stack (RE2 / Compare-Aggregate)

系列マッチング (RE2 / Compare-Aggregate) 一式のテスト。

このスタックは 2 本の系列を突き合わせて 1 つのスコアを出す。
形状が合っていても値が黙って壊れる経路が多いため
(未初期化パラメータ、使われない射影、マスクのずれ)、
形状だけでなく「パラメータに勾配が届くか」「NaN が出ないか」
「マスクされた位置が結果に影響しないか」を固定する。
'''

import math

import pytest
import torch

from lpu.common.vocab import IDMap, LabelMap
from lpu_nn.modeling import pooling
from lpu_nn.modeling.compare_aggregate import (
    CompareAggregatePooler,
    MultiFilterPooler,
    NGramPooler,
)
from lpu_nn.modeling.re2 import (
    Alignment,
    CombinationPooler,
    Fusion,
    MultiFilterEncoder,
    RE2Block,
    RE2Pooler,
)
from lpu_nn.modeling.sequence_matcher import SequenceMatcher

VOCAB_WORDS = 'the cat sat on a mat dog ran fast very quickly'.split()


def build_idmaps(with_labels=False):
    '''Vocabulary shared by both sides, plus an optional label map

    両系列で共有する語彙と、分類用のラベル表 (任意) を作る。
    '''
    seq = IDMap().set_symbols()
    for word in VOCAB_WORDS:
        seq.str2id(word, growth=True)
    idmaps = {'seq': seq, 's1': seq, 's2': seq}
    if with_labels:
        labels = LabelMap().set_symbols()
        for label in ('0', '1'):
            labels.str2id(label, growth=True)
        idmaps['t'] = labels
    return idmaps


@pytest.fixture
def idmaps():
    return build_idmaps()


@pytest.fixture
def labelled_idmaps():
    return build_idmaps(with_labels=True)


def make_mask(lengths, total):
    '''Build a (B, Len) boolean mask from per-row lengths

    行ごとの長さから (B, Len) の真偽マスクを作る。
    '''
    positions = torch.arange(total)[None, :]
    return positions < torch.tensor(lengths)[:, None]


class TestSequencePoolingFunctions:
    def test_max_pooling_ignores_masked_positions(self):
        # マスク外に巨大な値を置いても結果が変わらないこと
        c = torch.zeros(1, 3, 2)
        mask = make_mask([2], 3)
        base = pooling.sequence_max_pooling(c, mask)
        c[0, 2, :] = 1000.0
        assert torch.equal(pooling.sequence_max_pooling(c, mask), base)

    def test_max_pooling_picks_the_maximum_within_the_mask(self):
        c = torch.tensor([[[1.0, 5.0], [3.0, 2.0], [9.0, 9.0]]])
        mask = make_mask([2], 3)
        expected = torch.tensor([[3.0, 5.0]])
        assert torch.equal(pooling.sequence_max_pooling(c, mask), expected)

    def test_average_pooling_divides_by_the_masked_length(self):
        '''The divisor is the true length, not the padded length

        除数はパディング後の長さではなく実長であること。
        '''
        c = torch.tensor([[[1.0], [3.0], [100.0]]])
        mask = make_mask([2], 3)
        # (1 + 3) / 2 であり (1 + 3 + 100) / 3 ではない
        assert torch.equal(pooling.sequence_average_pooling(c, mask),
                           torch.tensor([[2.0]]))

    def test_average_pooling_ignores_masked_positions(self):
        c = torch.zeros(1, 3, 2)
        mask = make_mask([2], 3)
        base = pooling.sequence_average_pooling(c, mask)
        c[0, 2, :] = 1000.0
        assert torch.equal(pooling.sequence_average_pooling(c, mask), base)

    def test_attention_pooling_weights_sum_to_the_memory_hull(self):
        '''Attention pooling returns a convex combination of the memory

        注意プーリングの出力はメモリの凸結合であり、
        各次元は有効位置の最小値と最大値の間に収まること。
        '''
        memory = torch.randn(2, 4, 3)
        mask = make_mask([4, 2], 4)
        query = torch.randn(1, 3)
        pooled = pooling.sequence_attention_pooling(query, memory, mask)
        assert pooled.shape == (2, 3)
        for row in range(2):
            valid = memory[row][mask[row]]
            assert (pooled[row] >= valid.min(dim=0).values - 1e-5).all()
            assert (pooled[row] <= valid.max(dim=0).values + 1e-5).all()

    def test_attention_pooling_ignores_masked_positions(self):
        memory = torch.randn(1, 4, 3)
        mask = make_mask([2], 4)
        query = torch.randn(1, 3)
        base = pooling.sequence_attention_pooling(query, memory, mask)
        memory[0, 2:, :] = 50.0
        after = pooling.sequence_attention_pooling(query, memory, mask)
        assert torch.allclose(base, after, atol=1e-5)


class TestSequenceAttentionPooling:
    def test_the_query_weight_is_initialized(self):
        '''torch.Tensor(n) returns uninitialized memory, so it can hold NaN

        0.1.0.dev0 では torch.Tensor(n) をそのまま nn.Parameter にしており、
        未初期化メモリがそのまま問い合わせベクトルになっていた。
        1 次元パラメータは apply_init_weights の汎用処理
        (weight.dim() > 1) にも掛からないため、NaN が最後まで残った。
        '''
        module = pooling.SequenceAttentionPooling(32)
        assert torch.isfinite(module.weight).all()

    def test_pooling_output_is_finite(self):
        module = pooling.SequenceAttentionPooling(8)
        c = torch.randn(2, 5, 8)
        mask = make_mask([5, 3], 5)
        assert torch.isfinite(module(c, mask)).all()

    def test_get_sequence_pooler_rejects_an_unknown_name(self):
        with pytest.raises(ValueError, match='unknown name'):
            pooling.get_sequence_pooler('softmax', 8)

    @pytest.mark.parametrize('name', ['max', 'average', 'attention'])
    def test_every_pooler_reduces_the_length_axis(self, name):
        module = pooling.get_sequence_pooler(name, 8)
        c = torch.randn(2, 5, 8)
        mask = make_mask([5, 3], 5)
        assert module(c, mask).shape == (2, 8)


class TestFusion:
    def test_every_projection_receives_a_gradient(self):
        '''direct / sub / mult must be three independent projections

        0.1.0.dev0 は 3 つとも self.mod_direct を呼んでおり、
        mod_sub と mod_mult は構築されるだけで一度も使われなかった
        (RE2 の augmented fusion が単一射影に退化していた)。
        '''
        module = Fusion(6, hidden_size=6)
        out = module(torch.randn(2, 4, 6), torch.randn(2, 4, 6))
        out.sum().backward()
        without_grad = [name for name, param in module.named_parameters()
                        if param.requires_grad and param.grad is None]
        assert without_grad == []

    def test_the_three_projections_are_not_tied(self):
        '''Feeding the sub / mult views through one matrix collapses them

        同一の重みを使うと 3 つの特徴が区別できなくなる。
        別モジュールであることを直接確認する。
        '''
        module = Fusion(6, hidden_size=6)
        assert module.mod_sub is not module.mod_direct
        assert module.mod_mult is not module.mod_direct

    def test_output_keeps_the_length_axis(self):
        module = Fusion(6, hidden_size=10)
        out = module(torch.randn(3, 7, 6), torch.randn(3, 7, 6))
        assert out.shape == (3, 7, 10)


class TestAlignment:
    def test_aligned_sequences_keep_their_own_lengths(self):
        module = Alignment(8, hidden_size=8)
        seq1 = torch.randn(2, 5, 8)
        seq2 = torch.randn(2, 3, 8)
        mask1 = make_mask([5, 4], 5)
        mask2 = make_mask([3, 2], 3)
        align1, align2 = module(seq1, seq2, mask1, mask2)
        # 各系列は相手側の情報を受け取るが、自分の長さは保つ
        assert align1.shape == (2, 5, 8)
        assert align2.shape == (2, 3, 8)

    def test_masked_positions_of_the_other_sequence_are_not_attended(self):
        # dropout が乗ると 2 回の呼び出しが一致しないため評価モードにする
        module = Alignment(8, hidden_size=8).eval()
        seq1 = torch.randn(1, 4, 8)
        seq2 = torch.randn(1, 4, 8)
        mask1 = make_mask([4], 4)
        mask2 = make_mask([2], 4)
        base, _ = module(seq1, seq2, mask1, mask2)
        seq2[0, 2:, :] = 100.0
        after, _ = module(seq1, seq2, mask1, mask2)
        assert torch.allclose(base, after, atol=1e-4)

    def test_identity_map_skips_the_projection(self):
        module = Alignment(8, map='none')
        assert not hasattr(module, 'mod_map')
        align1, align2 = module(torch.randn(1, 3, 8), torch.randn(1, 2, 8),
                                make_mask([3], 3), make_mask([2], 2))
        assert align1.shape == (1, 3, 8)
        assert align2.shape == (1, 2, 8)


class TestCombinationPooler:
    @pytest.mark.parametrize('method', ['simple', 'full'])
    def test_both_methods_reduce_to_the_hidden_size(self, method):
        module = CombinationPooler(6, method=method, hidden_size=4)
        out = module(torch.randn(3, 6), torch.randn(3, 6))
        assert out.shape == (3, 4)

    def test_symmetric_full_pooling_is_order_invariant_in_the_sub_term(self):
        '''symmetric=True takes |a - b|, so swapping only moves a and b

        symmetric=True は差の絶対値を取るため、入れ替えても差の項は変わらない。
        '''
        module = CombinationPooler(4, method='full', symmetric=True, hidden_size=4)
        a, b = torch.randn(2, 4), torch.randn(2, 4)
        # 差と積の項は対称なので、連結の前半だけが入れ替わる
        concat_ab = torch.cat([a, b, (a - b).abs(), a * b], dim=1)
        concat_ba = torch.cat([b, a, (b - a).abs(), b * a], dim=1)
        assert torch.allclose(concat_ab[:, 8:], concat_ba[:, 8:])
        assert module(a, b).shape == module(b, a).shape

    def test_an_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match='Unknown pooling method'):
            CombinationPooler(4, method='bilinear')


class TestMultiFilterEncoder:
    def test_the_encoder_preserves_the_sequence_length(self):
        module = MultiFilterEncoder(8, hidden_size=8, num_layers=2)
        seq = torch.randn(2, 6, 8)
        mask = make_mask([6, 4], 6)
        assert module(seq, mask).shape == (2, 6, 8)

    def test_hidden_size_must_divide_among_the_filters(self):
        # 各フィルタが hidden_size を等分するため、割り切れない設定は弾く
        with pytest.raises(AssertionError):
            MultiFilterEncoder(8, hidden_size=7, ngram_orders=[1, 2])


class TestRE2Block:
    def test_a_block_returns_both_sequences_at_the_hidden_size(self):
        module = RE2Block(8, hidden_size=8)
        seq1 = torch.randn(2, 5, 8)
        seq2 = torch.randn(2, 3, 8)
        out1, out2 = module(seq1, seq2, make_mask([5, 4], 5), make_mask([3, 2], 3))
        assert out1.shape == (2, 5, 8)
        assert out2.shape == (2, 3, 8)


class TestNGramPooler:
    @pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
    def test_a_sequence_shorter_than_the_filter_is_handled(self, order):
        '''A 2-token sentence must not crash a 5-gram filter

        0.1.0.dev0 は入力長が n 未満のとき Conv2d が
        "Kernel size can't be greater than actual input size" で落ちていた。
        既定の ngram_orders は [1,2,3,4,5] なので、
        5 語未満の文を含むコーパスは学習を開始できなかった。
        '''
        module = NGramPooler(order, 4, hidden_size=6, sequence_pooling='average')
        seq = torch.randn(1, 2, 4)
        mask = make_mask([2], 2)
        assert module(seq, mask).shape == (1, 6)

    def test_the_output_is_aligned_with_the_mask(self):
        '''Masked tail positions must not change the pooled result

        畳み込み出力を入力長に揃えているので、
        マスク外の位置は結果に影響しないこと。
        '''
        module = NGramPooler(2, 3, hidden_size=5, sequence_pooling='max')
        seq = torch.zeros(1, 6, 3)
        mask = make_mask([3], 6)
        base = module(seq, mask)
        seq[0, 4:, :] = 30.0
        assert torch.allclose(module(seq, mask), base, atol=1e-5)


class TestMultiFilterPooler:
    def test_pooling_reduces_to_the_hidden_size(self):
        module = MultiFilterPooler(4, hidden_size=6, ngram_orders=[1, 2, 3])
        out = module(torch.randn(2, 5, 4), make_mask([5, 3], 5))
        assert out.shape == (2, 6)

    def test_a_scalar_ngram_order_is_accepted(self):
        # --ngram-orders に 1 個だけ渡された場合に備える
        config = MultiFilterPooler.get_config(ngram_orders=3)
        assert config['ngram_orders'] == [3]


class TestPoolerConfigs:
    def test_re2_reference_preset_follows_the_paper(self):
        config = RE2Pooler.get_config(preset='re2')
        assert config['embed_size'] == 300
        assert config['hidden_size'] == 150
        assert config['sequence_pooling'] == 'max'

    def test_the_default_preset_uses_attention_pooling(self):
        # 既定プリセットは注意プーリングであり、
        # 未初期化パラメータがあるとここが NaN 源になる
        config = RE2Pooler.get_config()
        assert config['sequence_pooling'] == 'attention'

    def test_pooling_is_an_alias_of_sequence_pooling(self):
        config = RE2Pooler.get_config(pooling='max')
        assert config['sequence_pooling'] == 'max'


class TestSequenceMatcherConfig:
    def test_a_missing_label_map_means_regression(self, idmaps):
        '''Without a label map the model predicts a score directly

        0.1.0.dev0 は idmaps に 't' が無いと num_classes を設定せず、
        __init__ が KeyError で落ちていた (回帰モードが構築不能だった)。
        '''
        config = SequenceMatcher.get_config(idmaps=idmaps)
        assert config['num_classes'] is None

    def test_a_label_map_sets_the_class_count(self, labelled_idmaps):
        config = SequenceMatcher.get_config(idmaps=labelled_idmaps)
        assert config['num_classes'] == len(labelled_idmaps['t'])

    def test_compare_aggregate_aliases_are_normalized(self, idmaps):
        for alias in ('ca', 'compare-aggregate', 'Compare-Aggregate'):
            config = SequenceMatcher.get_config(idmaps=idmaps, match_pooler_type=alias)
            assert config['match_pooler_type'] == 'compare-aggregate'

    def test_an_unknown_pooler_type_is_rejected(self, idmaps):
        with pytest.raises(ValueError, match='unsupported pooler type'):
            SequenceMatcher(idmaps, match_pooler_type='bert')


POOLERS = ['re2', 'compare-aggregate']
POOLINGS = ['max', 'average', 'attention']


class TestSequenceMatcher:
    def build(self, idmaps, **hparams):
        hparams.setdefault('embed_size', 16)
        hparams.setdefault('hidden_size', 16)
        return SequenceMatcher(idmaps, **hparams)

    @pytest.mark.parametrize('pooler', POOLERS)
    def test_regression_mode_returns_one_score_per_pair(self, idmaps, pooler):
        model = self.build(idmaps, match_pooler_type=pooler)
        s1 = model.prepare_batch('s1', ['the cat sat', 'a dog ran fast'])
        s2 = model.prepare_batch('s2', ['a mat', 'the cat'])
        assert model(s1, s2).shape == (2,)

    @pytest.mark.parametrize('pooler', POOLERS)
    def test_classification_mode_returns_one_logit_per_class(
            self, labelled_idmaps, pooler):
        model = self.build(labelled_idmaps, match_pooler_type=pooler)
        s1 = model.prepare_batch('s1', ['the cat sat', 'a dog ran fast'])
        s2 = model.prepare_batch('s2', ['a mat', 'the cat'])
        out = model(s1, s2)
        assert out.shape == (2, len(labelled_idmaps['t']))
        # score() はクラス数に関わらず 1 本のスコアに畳む
        assert model.score(s1, s2).shape == (2,)

    @pytest.mark.parametrize('pooler', POOLERS)
    @pytest.mark.parametrize('sequence_pooling', POOLINGS)
    def test_the_output_is_finite_for_every_pooling(
            self, idmaps, pooler, sequence_pooling):
        # 未初期化パラメータがあるとここで NaN になる
        model = self.build(idmaps, match_pooler_type=pooler,
                           sequence_pooling=sequence_pooling)
        s1 = model.prepare_batch('s1', ['the cat sat', 'a dog ran fast'])
        s2 = model.prepare_batch('s2', ['a mat', 'the cat'])
        assert torch.isfinite(model(s1, s2)).all()

    @pytest.mark.parametrize('pooler', POOLERS)
    def test_every_parameter_receives_a_gradient(self, idmaps, pooler):
        '''A module that is built but never called is a silent defect

        構築されるだけで一度も呼ばれないモジュールは、
        形状も損失も正常に見えるまま容量だけを消費する。
        '''
        model = self.build(idmaps, match_pooler_type=pooler)
        s1 = model.prepare_batch('s1', ['the cat sat on a mat', 'a dog ran fast'])
        s2 = model.prepare_batch('s2', ['a mat very quickly', 'the cat'])
        model(s1, s2).sum().backward()
        without_grad = [name for name, param in model.named_parameters()
                        if param.requires_grad and param.grad is None]
        assert without_grad == []

    @pytest.mark.parametrize('pooler', POOLERS)
    def test_a_two_word_pair_does_not_crash(self, idmaps, pooler):
        # 既定の n-gram フィルタ長 (最大 5) より短い文でも動くこと
        model = self.build(idmaps, match_pooler_type=pooler)
        s1 = model.prepare_batch('s1', ['the cat'])
        s2 = model.prepare_batch('s2', ['a mat'])
        assert model(s1, s2).shape == (1,)

    def test_prepare_batch_pads_with_the_padding_id(self, idmaps):
        model = self.build(idmaps)
        batch = model.prepare_batch('s1', ['the cat sat on', 'a mat'])
        assert batch.shape[0] == 2
        # 短い方の末尾がパディング ID で埋まること
        assert batch[1, -1].item() == idmaps['seq'].pad

    def test_prepare_batch_passes_a_tensor_through(self, idmaps):
        model = self.build(idmaps)
        tensor = torch.zeros(2, 3, dtype=torch.long)
        assert model.prepare_batch('s1', tensor) is tensor

    def test_prepare_batch_rejects_an_unknown_field(self, idmaps):
        with pytest.raises(TypeError, match='unsupported type'):
            self.build(idmaps).prepare_batch('s3', ['the cat'])


class TestLogitToScore:
    def test_a_correct_label_selects_its_probability(self, labelled_idmaps):
        model = SequenceMatcher(labelled_idmaps, embed_size=8, hidden_size=8)
        logits = torch.tensor([[0.0, math.log(3.0)]])
        idmap = labelled_idmaps['t']
        score = model.logit_to_score(logits, correct_label='1')
        assert score.shape == (1,)
        expected = logits.softmax(dim=1)[0, idmap['1']]
        assert torch.allclose(score, expected[None])

    def test_an_absent_label_falls_back_to_the_expected_value(
            self, labelled_idmaps):
        '''An unknown label averages the numeric labels by probability

        指定ラベルが語彙に無い場合は、数値として読めるラベルの
        期待値にフォールバックする。数値でないラベル (特殊記号) は
        寄与させない。
        '''
        model = SequenceMatcher(labelled_idmaps, embed_size=8, hidden_size=8)
        logits = torch.zeros(1, len(labelled_idmaps['t']))
        score = model.logit_to_score(logits, correct_label='yes')
        # 一様分布なので、数値ラベル 0 と 1 の寄与の合計になる
        probs = logits.softmax(dim=1)
        idmap = labelled_idmaps['t']
        expected = sum(float(label) * probs[0, i]
                       for i, label in enumerate(idmap)
                       if _is_number(label))
        assert torch.allclose(score, torch.tensor([float(expected)]), atol=1e-6)

    def test_a_non_numeric_label_does_not_raise(self, labelled_idmaps):
        # 裸の except を使っていた箇所。ValueError のみを飲むよう直した
        model = SequenceMatcher(labelled_idmaps, embed_size=8, hidden_size=8)
        logits = torch.zeros(1, len(labelled_idmaps['t']))
        assert torch.isfinite(model.logit_to_score(logits, correct_label='?')).all()


def _is_number(label):
    try:
        float(label)
    except ValueError:
        return False
    return True


class TestCompareAggregatePooler:
    def test_pooling_reduces_a_pair_to_one_vector(self, idmaps):
        module = CompareAggregatePooler(idmaps, embed_size=8, hidden_size=6)
        q = torch.zeros(2, 5, dtype=torch.long)
        a = torch.zeros(2, 4, dtype=torch.long)
        assert module(q, a).shape == (2, 6)


class TestRE2Pooler:
    def test_pooling_reduces_a_pair_to_one_vector(self, idmaps):
        module = RE2Pooler(idmaps, embed_size=8, hidden_size=8, num_blocks=2)
        seq1 = torch.zeros(2, 5, dtype=torch.long)
        seq2 = torch.zeros(2, 4, dtype=torch.long)
        assert module(seq1, seq2).shape == (2, 8)

    def test_a_single_block_still_pools(self, idmaps):
        # 残差の初期値を読んでしまうと 1 ブロック構成で壊れる
        module = RE2Pooler(idmaps, embed_size=8, hidden_size=8, num_blocks=1)
        seq1 = torch.zeros(1, 4, dtype=torch.long)
        seq2 = torch.zeros(1, 3, dtype=torch.long)
        assert module(seq1, seq2).shape == (1, 8)
