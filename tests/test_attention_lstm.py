'''Tests for lpu_nn.modeling.attention and lpu_nn.modeling.lstm

lpu_nn.modeling.attention と lpu_nn.modeling.lstm のテスト。

注意機構は (B, L, H) の記憶と (B, H) の復号器状態から (B, H) の文脈を返す。
LSTM は状態を跨いで保持するため、マスクされた位置で状態が進まないことと、
状態の往復が保たれることを確認する。
'''

import pytest
import torch

from lpu_nn.modeling.attention import get_attention
from lpu_nn.modeling.lstm import MultiLayerLSTM, StatefulLSTM

ATTENTION_NAMES = ['dot', 'concat', 'general', 'mlp', 'none']

BATCH, LENGTH, HIDDEN = 2, 4, 6


def build(name, bidirectional=False, local=False):
    '''Build one attention module with the parameters it needs

    必要な引数を与えて注意機構を 1 つ構築する。
    '''
    return get_attention(
        name,
        hidden_size=HIDDEN,
        bidirectional_encoder=bidirectional,
        local_attention=local,
        activation='relu',
        dropout_ratio=0.1,
    )


@pytest.fixture
def inputs():
    '''A decoder state, a memory and a full mask

    復号器の状態、記憶、および全て有効なマスク。
    '''
    torch.manual_seed(0)
    h_dec = torch.randn(BATCH, HIDDEN)
    memory = torch.randn(BATCH, LENGTH, HIDDEN)
    mask = torch.ones(BATCH, LENGTH, dtype=torch.bool)
    return h_dec, memory, mask


class TestAttention:
    @pytest.mark.parametrize('name', ATTENTION_NAMES)
    def test_returns_one_context_vector_per_sample(self, name, inputs):
        h_dec, memory, mask = inputs
        context = build(name)(h_dec, memory, mask_mem=mask)
        assert context.shape == (BATCH, HIDDEN)

    @pytest.mark.parametrize('name', ['dot', 'concat', 'general', 'mlp'])
    def test_a_masked_position_does_not_contribute(self, name, inputs):
        # マスクした位置の記憶を差し替えても、文脈が変わらないこと
        h_dec, memory, mask = inputs
        mask = mask.clone()
        mask[:, -1] = False
        before = build(name)(h_dec, memory, mask_mem=mask)
        altered = memory.clone()
        altered[:, -1] = 1000.0
        module = build(name)
        torch.manual_seed(0)
        after = module(h_dec, altered, mask_mem=mask)
        torch.manual_seed(0)
        baseline = module(h_dec, memory, mask_mem=mask)
        assert torch.allclose(after, baseline, atol=1e-5)
        assert before.shape == after.shape

    @pytest.mark.parametrize('name', ['concat', 'general', 'mlp'])
    def test_a_bidirectional_memory_is_accepted(self, name, inputs):
        # 双方向エンコーダでは記憶の幅が 2 倍になる
        h_dec, _memory, mask = inputs
        memory = torch.randn(BATCH, LENGTH, HIDDEN * 2)
        context = build(name, bidirectional=True)(h_dec, memory, mask_mem=mask)
        assert context.shape[0] == BATCH

    def test_dot_attention_pairs_the_two_directions(self):
        # エンコーダは torch.cat([前向き, 後向き], dim=2) と連結するため、
        # 前半と後半を次元ごとに対応させる必要がある。以前は隣り合う
        # 2 要素を平均しており、同じ向き同士を混ぜていた
        forward = torch.tensor([[[10.0, 20.0, 30.0]]])
        backward = torch.tensor([[[1.0, 2.0, 3.0]]])
        memory = torch.cat([forward, backward], dim=2)
        module = get_attention(
            'dot', hidden_size=3, bidirectional_encoder=True, dropout_ratio=0.1)
        context = module(
            torch.tensor([[1.0, 0.0, 0.0]]),
            memory,
            mask_mem=torch.ones(1, 1, dtype=torch.bool),
        )
        # 記憶が 1 位置しかないので、注意は 1 に定まり文脈は平均そのもの
        assert context.flatten().tolist() == pytest.approx([5.5, 11.0, 16.5])

    @pytest.mark.parametrize('name', ['dot', 'concat', 'general', 'mlp'])
    def test_local_attention_runs(self, name, inputs):
        # 局所注意は h_dec を受け取っておらず、一度も動作しなかった
        h_dec, memory, mask = inputs
        context = build(name, local=True)(h_dec, memory, mask_mem=mask)
        assert context.shape == (BATCH, HIDDEN)

    @pytest.mark.parametrize('name', ['concat', 'general', 'mlp'])
    def test_init_weights_runs(self, name):
        module = build(name, local=True)
        module.init_weights()

    def test_an_unknown_attention_is_rejected(self):
        with pytest.raises(ValueError):
            build('unknown')

    def test_a_missing_hidden_size_is_rejected(self):
        # 以前は None のまま nn.Linear に渡っていた
        with pytest.raises(ValueError):
            get_attention('mlp', activation='relu', dropout_ratio=0.1)

    def test_the_mlp_attention_needs_an_activation(self):
        with pytest.raises(ValueError):
            get_attention('mlp', hidden_size=HIDDEN, dropout_ratio=0.1)


class TestStatefulLSTM:
    def test_maps_a_sequence_to_the_output_size(self):
        lstm = StatefulLSTM(5, 3)
        assert lstm(torch.randn(2, 4, 5)).shape == (2, 4, 3)

    def test_a_single_step_input_is_expanded(self):
        # (B, E) を渡すと (B, 1, E) として扱われること
        lstm = StatefulLSTM(5, 3)
        assert lstm(torch.randn(2, 5)).shape == (2, 1, 3)

    def test_the_state_advances_across_calls(self):
        lstm = StatefulLSTM(5, 3)
        assert lstm.get_state() is None
        lstm(torch.randn(2, 4, 5))
        state = lstm.get_state()
        assert state is not None
        assert state['h'].shape == (2, 3)
        assert state['c'].shape == (2, 3)

    def test_reset_state_clears_it(self):
        lstm = StatefulLSTM(5, 3)
        lstm(torch.randn(2, 4, 5))
        lstm.reset_state()
        assert lstm.get_state() is None

    def test_the_state_round_trips(self):
        lstm = StatefulLSTM(5, 3)
        x = torch.randn(2, 4, 5)
        lstm(x)
        saved = lstm.get_state()
        expected = lstm(x)
        lstm.set_state(saved)
        assert torch.allclose(lstm(x), expected, atol=1e-6)

    def test_a_masked_step_does_not_advance_the_state(self):
        # マスクが偽の位置では状態が据え置かれること
        lstm = StatefulLSTM(5, 3)
        x = torch.randn(2, 3, 5)
        mask = torch.ones(2, 3, dtype=torch.bool)
        mask[:, -1] = False
        lstm(x, mask)
        masked_state = lstm.get_state()
        lstm.reset_state()
        lstm(x[:, :-1], mask[:, :-1])
        assert torch.allclose(masked_state['h'], lstm.get_state()['h'], atol=1e-6)


class TestMultiLayerLSTM:
    @pytest.mark.parametrize('num_layers', [1, 2, 3])
    def test_the_output_shape_does_not_depend_on_the_depth(self, num_layers):
        lstm = MultiLayerLSTM(5, 3, num_layers=num_layers)
        assert lstm(torch.randn(2, 4, 5)).shape == (2, 4, 3)

    def test_more_than_one_layer_turns_on_the_residual_path(self):
        assert MultiLayerLSTM.get_config(num_layers=2)['residual_connection'] is True
        assert MultiLayerLSTM.get_config(num_layers=1)['residual_connection'] is False

    def test_the_normalization_layers_match_the_depth(self):
        lstm = MultiLayerLSTM(5, 3, num_layers=3)
        assert len(lstm.mods_norm) == 2

    def test_the_state_has_one_entry_per_layer(self):
        lstm = MultiLayerLSTM(5, 3, num_layers=2)
        lstm(torch.randn(2, 4, 5))
        assert len(lstm.get_state()) == 2

    def test_the_state_round_trips(self):
        lstm = MultiLayerLSTM(5, 3, num_layers=2)
        x = torch.randn(2, 4, 5)
        lstm.eval()
        lstm(x)
        saved = lstm.get_state()
        expected = lstm(x)
        lstm.set_state(saved)
        assert torch.allclose(lstm(x), expected, atol=1e-6)

    def test_reset_state_clears_every_layer(self):
        lstm = MultiLayerLSTM(5, 3, num_layers=2)
        lstm(torch.randn(2, 4, 5))
        lstm.reset_state()
        assert lstm.get_state() == [None, None]

    def test_a_mask_is_accepted(self):
        lstm = MultiLayerLSTM(5, 3, num_layers=2)
        mask = torch.ones(2, 4, dtype=torch.bool)
        mask[:, -1] = False
        assert lstm(torch.randn(2, 4, 5), mask).shape == (2, 4, 3)
