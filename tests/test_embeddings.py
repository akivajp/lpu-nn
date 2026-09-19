'''Tests for lpu_nn.modeling.embeddings

lpu_nn.modeling.embeddings のテスト。

`Embedding` は ID をベクトルへ写すだけでなく、パディング位置を 0 に落とし、
埋め込み次元の平方根で拡大する。この 2 つは後段の Transformer が前提に
しているため、形状と併せて固定する。
'''

import pandas as pd
import pytest
import torch

from lpu_nn.modeling.embeddings import (
    CharacterEmbedding,
    ContextualStringEmbedding,
    Embedding,
    extract_vector,
)

NUM_IDS, EMBED_SIZE = 10, 4


class TestExtractVector:
    def test_finds_an_exact_match(self):
        vectors = {'cat': torch.ones(3)}
        assert torch.equal(extract_vector('cat', vectors), torch.ones(3))

    def test_falls_back_to_the_lowercase_form(self):
        vectors = {'cat': torch.ones(3)}
        assert torch.equal(extract_vector('Cat', vectors), torch.ones(3))

    def test_strips_the_sentencepiece_word_mark(self):
        # SentencePiece の語頭記号を外した形でも探すこと
        vectors = {'cat': torch.ones(3)}
        assert torch.equal(extract_vector('▁cat', vectors), torch.ones(3))

    def test_normalizes_full_width_characters(self):
        vectors = {'abc': torch.ones(3)}
        # 全角のラテン文字は、NFKC 正規化を検証するための入力そのもの
        assert torch.equal(extract_vector('ａｂｃ', vectors), torch.ones(3))  # noqa: RUF001

    def test_returns_none_when_nothing_matches(self):
        assert extract_vector('dog', {'cat': torch.ones(3)}) is None

    def test_an_empty_token_does_not_match(self):
        assert extract_vector('▁', {'': torch.ones(3)}) is None


class TestEmbedding:
    def test_appends_the_embedding_axis(self):
        out = Embedding(NUM_IDS, EMBED_SIZE)(torch.zeros(2, 5, dtype=torch.long))
        assert out.shape == (2, 5, EMBED_SIZE)

    def test_scales_by_the_square_root_of_the_size(self):
        # 後段の Transformer がこの拡大を前提にしている
        layer = Embedding(NUM_IDS, EMBED_SIZE)
        with torch.no_grad():
            layer.weight.fill_(1.0)
        out = layer(torch.zeros(1, 1, dtype=torch.long))
        assert out.flatten().tolist() == pytest.approx([EMBED_SIZE ** 0.5] * EMBED_SIZE)

    def test_a_padded_position_becomes_zero(self):
        layer = Embedding(NUM_IDS, EMBED_SIZE, padding=-1)
        with torch.no_grad():
            layer.weight.fill_(1.0)
        ids = torch.tensor([[0, -1]])
        out = layer(ids)
        assert out[0, 0].abs().sum() > 0
        assert out[0, 1].abs().sum() == 0

    def test_a_padding_id_outside_the_table_is_accepted(self):
        # パディング ID は語彙の範囲外でもよく、内部で 0 に置き換えられる
        layer = Embedding(NUM_IDS, EMBED_SIZE, padding=-1)
        assert layer(torch.tensor([[-1, -1]])).shape == (1, 2, EMBED_SIZE)

    def test_without_padding_nothing_is_masked(self):
        layer = Embedding(NUM_IDS, EMBED_SIZE)
        with torch.no_grad():
            layer.weight.fill_(1.0)
        assert (layer(torch.zeros(1, 3, dtype=torch.long)) != 0).all()

    @pytest.mark.parametrize('initializer', ['orthogonal', 'he-normal', None])
    def test_every_initializer_leaves_a_finite_weight(self, initializer):
        layer = Embedding(NUM_IDS, EMBED_SIZE, initializer=initializer)
        layer.init_weights()
        assert torch.isfinite(layer.weight).all()

    def test_extra_repr_mentions_the_padding(self):
        assert 'padding' in Embedding(NUM_IDS, EMBED_SIZE, padding=0).extra_repr()
        assert 'padding' not in Embedding(NUM_IDS, EMBED_SIZE).extra_repr()

    def test_to_returns_the_layer_itself(self):
        layer = Embedding(NUM_IDS, EMBED_SIZE)
        assert layer.to('cpu') is layer


class TestCharacterEmbedding:
    def test_round_trips_a_string_through_the_ids(self):
        layer = CharacterEmbedding(EMBED_SIZE)
        assert layer.tensor2str(layer.str2tensor('abc')) == 'abc'

    def test_round_trips_non_ascii_text(self):
        # UTF-8 のバイト列として扱うため、多バイト文字も往復すること
        layer = CharacterEmbedding(EMBED_SIZE)
        assert layer.tensor2str(layer.str2tensor('日本語')) == '日本語'

    def test_the_ids_are_surrounded_by_the_symbols(self):
        layer = CharacterEmbedding(EMBED_SIZE)
        ids = layer.str2tensor('a').tolist()
        assert ids[0] == layer.bos
        assert ids[-1] == layer.eos

    def test_tensor2bytes_rejects_a_batched_tensor(self):
        # 元実装は tensor.ndim() と書いており、プロパティを呼んで TypeError
        # になっていた
        layer = CharacterEmbedding(EMBED_SIZE)
        with pytest.raises(AssertionError):
            layer.tensor2bytes(torch.zeros(2, 3, dtype=torch.long))

    def test_prepare_batch_pads_to_the_longest(self):
        # この実装は ContextualStringEmbedding 側に置かれており、
        # CharacterEmbedding では None を返していた
        layer = CharacterEmbedding(EMBED_SIZE)
        batch = layer.prepare_batch(['a', 'abc'])
        assert batch.shape == (2, len('abc') + 2)

    def test_prepare_batch_accepts_a_series(self):
        layer = CharacterEmbedding(EMBED_SIZE)
        assert layer.prepare_batch(pd.Series(['a', 'bb'])).shape[0] == 2

    def test_prepare_batch_passes_a_tensor_through(self):
        layer = CharacterEmbedding(EMBED_SIZE)
        tensor = torch.zeros(2, 3, dtype=torch.long)
        assert layer.prepare_batch(tensor) is tensor

    def test_prepare_batch_rejects_an_unsupported_type(self):
        # 元実装は batch が未定義のまま先へ進んでいた
        layer = CharacterEmbedding(EMBED_SIZE)
        with pytest.raises(TypeError):
            layer.prepare_batch(3.14)


class TestContextualStringEmbedding:
    @pytest.fixture
    def layer(self):
        # 元実装は nn.Module.__init__ を呼んでおらず、構築すらできなかった
        return ContextualStringEmbedding(
            char_embed_size=8, hidden_size=4, num_layers=1)

    def test_it_can_be_constructed(self, layer):
        assert layer.hidden_size == 4

    def test_one_vector_per_token_from_both_directions(self, layer):
        # 連結軸が dim=3 と書かれており、3 次元テンソルには範囲外だった
        out = layer(torch.zeros(2, 3, 5, dtype=torch.long))
        assert out.shape == (2, 3, 4 * 2)

    def test_the_output_width_follows_the_hidden_size(self):
        layer = ContextualStringEmbedding(
            char_embed_size=8, hidden_size=6, num_layers=1)
        assert layer(torch.zeros(1, 2, 4, dtype=torch.long)).shape == (1, 2, 12)

    def test_reset_state_clears_both_directions(self, layer):
        layer(torch.zeros(1, 2, 3, dtype=torch.long))
        layer.reset_state()
        assert layer.mod_rnn_forward.get_state() == [None]
        assert layer.mod_rnn_backward.get_state() == [None]

    def test_prepare_batch_delegates_to_the_character_embedding(self, layer):
        # 委譲先を持たないまま self.weight / self.str2tensor を参照していた
        assert layer.prepare_batch(['ab']).shape[0] == 1
