'''Tests for lpu_nn.modeling.transformer

lpu_nn.modeling.transformer のテスト。

Transformer は状態を持ち、呼び出しをまたいで入力と出力を溜め込む
(逐次復号のため)。形状の不変量に加えて、その状態の扱いを確認する。
'''

import itertools

import pytest
import torch
from torch import nn

from lpu_nn.modeling.transformer import (
    EmbedPosition,
    EmbedRelativePosition,
    FeedForward,
    ModuleConnection,
    MultiHeadAttention,
    MultiStepTransformer,
    PositionalEncoder,
    Transformer,
    resolve_num_heads,
)

BATCH, LENGTH, HIDDEN, HEADS = 2, 4, 8, 2
BASE = {'embed_size': HIDDEN, 'hidden_size': HIDDEN, 'num_heads': HEADS}


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    seq = torch.randn(BATCH, LENGTH, HIDDEN)
    mask = torch.ones(BATCH, LENGTH, LENGTH, dtype=torch.bool)
    return seq, mask


class TestResolveNumHeads:
    def test_derives_the_key_size_from_the_head_count(self):
        assert resolve_num_heads({'hidden_size': 64, 'num_heads': 8})['key_size'] == 8

    def test_derives_the_head_count_from_the_default_key_size(self):
        assert resolve_num_heads({'hidden_size': 512})['num_heads'] == 8

    def test_a_hidden_size_below_the_key_size_is_reported(self):
        # 以前はヘッド数 0 になり、ずっと後で ZeroDivisionError になっていた
        with pytest.raises(ValueError, match='num-heads'):
            resolve_num_heads({'hidden_size': 32})


class TestEmbedPosition:
    def test_derives_the_hidden_size_from_the_embed_size(self):
        # setdefault が 'embed_size' と 2 度書かれており、hidden_size が
        # 導出されず KeyError になっていた
        assert EmbedPosition.get_config(embed_size=16)['hidden_size'] == 16

    def test_returns_one_vector_per_position(self):
        module = EmbedPosition(**BASE, max_length=32)
        assert module(LENGTH).shape == (LENGTH, HIDDEN)

    def test_the_start_offset_shifts_the_positions(self):
        module = EmbedPosition(**BASE, max_length=32)
        assert not torch.equal(module(LENGTH), module(LENGTH, start=1))


class TestPositionalEncoder:
    def test_returns_one_encoding_per_position(self):
        assert PositionalEncoder()((BATCH, LENGTH, HIDDEN)).shape == (1, LENGTH, HIDDEN)

    def test_the_encoding_is_deterministic(self):
        encoder = PositionalEncoder()
        first = encoder((BATCH, LENGTH, HIDDEN))
        assert torch.equal(first, encoder((BATCH, LENGTH, HIDDEN)))

    def test_a_shorter_request_reuses_the_cached_encoding(self):
        encoder = PositionalEncoder()
        long = encoder((BATCH, 8, HIDDEN))
        short = encoder((BATCH, 4, HIDDEN))
        assert torch.equal(short, long[:, :4])

    def test_the_step_changes_the_encoding(self):
        encoder = PositionalEncoder()
        assert not torch.equal(
            encoder((BATCH, LENGTH, HIDDEN), step=0),
            encoder((BATCH, LENGTH, HIDDEN), step=1))


class TestFeedForward:
    def test_preserves_the_shape(self, inputs):
        seq, _mask = inputs
        assert FeedForward(**BASE)(seq).shape == (BATCH, LENGTH, HIDDEN)

    def test_init_weights_runs(self):
        FeedForward(**BASE).init_weights()


class TestMultiHeadAttention:
    def test_preserves_the_query_shape(self, inputs):
        seq, mask = inputs
        module = MultiHeadAttention(**BASE)
        assert module(seq, seq, seq, mask).shape == (BATCH, LENGTH, HIDDEN)

    def test_a_masked_key_does_not_contribute(self, inputs):
        seq, mask = inputs
        mask = mask.clone()
        mask[:, :, -1] = False
        module = MultiHeadAttention(**BASE)
        module.eval()
        altered = seq.clone()
        altered[:, -1] = 1000.0
        assert torch.allclose(
            module(seq, seq, seq, mask), module(seq, altered, altered, mask), atol=1e-4)

    def test_the_attention_weights_are_recorded_per_head(self, inputs):
        seq, mask = inputs
        module = MultiHeadAttention(**BASE)
        module(seq, seq, seq, mask)
        weights = module.get_state()['attention_weight']
        assert weights.shape == (BATCH, HEADS, LENGTH, LENGTH)

    def test_the_relative_attention_variant_runs(self, inputs):
        seq, mask = inputs
        module = MultiHeadAttention(**BASE, relative_attention=True, clip_distance=2)
        assert module(seq, seq, seq, mask).shape == (BATCH, LENGTH, HIDDEN)

    def test_a_missing_mask_is_accepted(self, inputs):
        seq, _mask = inputs
        assert MultiHeadAttention(**BASE)(seq, seq, seq, None).shape == (BATCH, LENGTH, HIDDEN)

    def test_init_weights_runs(self):
        MultiHeadAttention(**BASE).init_weights()


class TestEmbedRelativePosition:
    def test_maps_a_clipped_distance_to_a_vector(self):
        module = EmbedRelativePosition(**BASE, clip_distance=2)
        positions = torch.zeros(3, 3, dtype=torch.long)
        assert module(positions).shape == (3, 3, HIDDEN // HEADS)


class TestModuleConnection:
    @pytest.mark.parametrize('pre,post', list(itertools.product(
        ['', 'n', 'd', 'dn'], ['dan', 'da', 'a', 'n', ''])))
    def test_every_pre_and_post_combination_runs(self, inputs, pre, post):
        seq, _mask = inputs
        module = ModuleConnection(
            HIDDEN, sublayer_preprocess=pre, sublayer_postprocess=post)
        assert module(seq, nn.Identity()).shape == (BATCH, LENGTH, HIDDEN)

    def test_the_residual_adds_the_original_input(self, inputs):
        seq, _mask = inputs
        module = ModuleConnection(HIDDEN, sublayer_preprocess='', sublayer_postprocess='a')
        module.eval()
        assert torch.allclose(module(seq, nn.Identity()), seq * 2, atol=1e-6)


class TestTransformer:
    def test_preserves_the_shape(self, inputs):
        seq, mask = inputs
        assert Transformer(**BASE)(seq, mask_self=mask).shape == (BATCH, LENGTH, HIDDEN)

    def test_the_conditioned_variant_attends_to_the_memory(self, inputs):
        seq, mask = inputs
        memory = torch.randn(BATCH, LENGTH, HIDDEN)
        module = Transformer(conditioned=True, **BASE)
        out = module(seq, memory=memory, mask_self=mask, mask_combine=mask)
        assert out.shape == (BATCH, LENGTH, HIDDEN)

    def test_the_state_accumulates_across_calls(self, inputs):
        # 逐次復号のため、呼び出しをまたいで入力を溜め込むこと
        seq, mask = inputs
        module = Transformer(**BASE)
        module(seq[:, :2], mask_self=mask[:, :2, :2])
        assert module.get_state()['input'].shape[1] == 2
        module(seq[:, 2:], mask_self=mask[:, 2:, :])
        assert module.get_state()['input'].shape[1] == LENGTH

    def test_reset_state_clears_the_history(self, inputs):
        seq, mask = inputs
        module = Transformer(**BASE)
        module(seq, mask_self=mask)
        assert module.reset_state() is module
        assert module.get_state().get('input') is None

    def test_set_state_of_none_resets(self, inputs):
        module = Transformer(**BASE)
        assert module.set_state(None) is module


class TestMultiStepTransformer:
    @pytest.mark.parametrize('num_layers', [1, 2, 3])
    def test_the_depth_does_not_change_the_shape(self, inputs, num_layers):
        seq, mask = inputs
        module = MultiStepTransformer(num_layers=num_layers, **BASE)
        assert module(seq, mask_self=mask).shape == (BATCH, LENGTH, HIDDEN)

    def test_the_output_normalization_is_applied(self, inputs):
        # 判定が `normalize_output` という、どこにも代入されない名前を見て
        # いたため、作られた LayerNorm は一度も適用されていなかった
        seq, mask = inputs
        module = MultiStepTransformer(
            sublayer_postprocess='da', num_layers=1, **BASE)
        assert hasattr(module, 'mod_norm_output')
        module.eval()
        out = module(seq, mask_self=mask)
        # LayerNorm を通れば各位置の分散が 1 に揃う
        assert out.var(dim=-1, unbiased=False).mean().item() == pytest.approx(1.0, abs=0.05)

    def test_the_input_normalization_is_applied(self, inputs):
        seq, mask = inputs
        module = MultiStepTransformer(
            sublayer_preprocess='d', num_layers=1, **BASE)
        assert hasattr(module, 'mod_norm_input')
        assert module(seq, mask_self=mask).shape == (BATCH, LENGTH, HIDDEN)

    def test_the_default_setting_builds_no_extra_normalization(self):
        # 既定の 'dan' は 'n' を含むため、余分な LayerNorm は作られない
        module = MultiStepTransformer(num_layers=1, **BASE)
        assert not hasattr(module, 'mod_norm_output')

    def test_the_state_has_one_entry_per_layer(self, inputs):
        seq, mask = inputs
        module = MultiStepTransformer(num_layers=2, **BASE)
        module(seq, mask_self=mask)
        assert len(module.get_state()) == 2

    def test_set_state_returns_self(self, inputs):
        seq, mask = inputs
        module = MultiStepTransformer(num_layers=2, **BASE)
        module(seq, mask_self=mask)
        assert module.set_state(module.get_state()) is module

    def test_set_state_of_none_resets(self):
        module = MultiStepTransformer(num_layers=2, **BASE)
        assert module.set_state(None) is module

    @pytest.mark.parametrize('embed_positions', [False, True])
    def test_both_position_representations_run(self, inputs, embed_positions):
        seq, mask = inputs
        module = MultiStepTransformer(
            embed_positions=embed_positions, max_length=32, num_layers=1, **BASE)
        assert module(seq, mask_self=mask).shape == (BATCH, LENGTH, HIDDEN)
