'''Tests for lpu_nn.modeling.encoder_decoder

lpu_nn.modeling.encoder_decoder のテスト。

`EncoderDecoder` は LSTM と Transformer を自由に組み合わせられる。
訓練経路 (forward) は 4 通りすべてを、復号経路 (generate / beam_search) は
`run_seq2seq` が直接叩くため、形と終了条件を確認する。
'''

import itertools

import pytest
import torch

from lpu_nn.modeling.encoder_decoder import (
    EncoderDecoder,
    LSTMDecoder,
    LSTMEncoder,
)

VOCAB_SIZE, EMBED, HIDDEN = 20, 8, 8
BATCH, SRC_LEN, TRG_LEN = 2, 5, 4


class StubVocabulary:
    '''The parts of a Vocabulary the models actually read

    モデルが実際に参照する Vocabulary の部分だけを持つ代用品。
    '''

    pad, bos, eos, unk = 0, 1, 2, 3

    def __len__(self):
        return VOCAB_SIZE

    def encode(self, sent, add_symbols=False):
        ids = [4 + (len(word) % 10) for word in sent.split()]
        return self.safe_add_symbols(ids) if add_symbols else ids

    def safe_add_symbols(self, ids, add_bos=True, add_eos=True):
        return [self.bos, *list(ids), self.eos]

    def decode(self, ids, remove_symbols=True, as_tokens=False):
        return ' '.join(str(int(i)) for i in ids)

    def clean_ids(self, ids):
        return [i for i in list(ids) if i not in (self.bos, self.eos, self.pad)]


@pytest.fixture
def idmaps():
    vocab = StubVocabulary()
    return {'x': vocab, 'seq': vocab}


def build(idmaps, encoder_type='transformer', decoder_type='transformer', **extra):
    '''Build a small encoder-decoder of the requested kinds

    指定した種類の小さな encoder-decoder を構築する。
    '''
    params = {
        'embed_size': EMBED, 'hidden_size': HIDDEN, 'num_heads': 2,
        'vocab_size': VOCAB_SIZE, 'padding': 0, 'max_length': 32,
        'encoder_type': encoder_type, 'decoder_type': decoder_type,
    }
    params.update(extra)
    return EncoderDecoder(idmaps, **params)


@pytest.fixture
def batch():
    torch.manual_seed(0)
    source = torch.randint(4, VOCAB_SIZE, (BATCH, SRC_LEN))
    target = torch.randint(4, VOCAB_SIZE, (BATCH, TRG_LEN))
    return source, target


class TestEncoderDecoder:
    @pytest.mark.parametrize('encoder_type,decoder_type', list(
        itertools.product(['lstm', 'transformer'], ['lstm', 'transformer'])))
    def test_every_combination_produces_logits(
            self, idmaps, batch, encoder_type, decoder_type):
        source, target = batch
        model = build(idmaps, encoder_type, decoder_type)
        logits = model(source, target)
        assert logits.shape == (BATCH, VOCAB_SIZE, TRG_LEN)

    def test_it_runs_without_an_explicit_reset(self, idmaps, batch):
        # LSTMDecoder は last_state を __init__ で用意しておらず、
        # 構築直後の forward が AttributeError になっていた
        source, target = batch
        model = build(idmaps, 'lstm', 'lstm')
        assert model(source, target).shape == (BATCH, VOCAB_SIZE, TRG_LEN)

    @pytest.mark.parametrize('attention', ['dot', 'concat', 'general', 'mlp', 'none'])
    def test_every_attention_type_runs_on_the_lstm_path(self, idmaps, batch, attention):
        source, target = batch
        model = build(idmaps, 'lstm', 'lstm', attention_type=attention)
        assert model(source, target).shape == (BATCH, VOCAB_SIZE, TRG_LEN)

    @pytest.mark.parametrize('bidirectional', [False, True])
    def test_both_encoder_directions_run(self, idmaps, batch, bidirectional):
        source, target = batch
        model = build(idmaps, 'lstm', 'lstm', bidirectional_encoder=bidirectional)
        assert model(source, target).shape == (BATCH, VOCAB_SIZE, TRG_LEN)

    @pytest.mark.parametrize('input_feeding', [False, True])
    def test_input_feeding_runs_either_way(self, idmaps, batch, input_feeding):
        source, target = batch
        model = build(idmaps, 'lstm', 'lstm', input_feeding=input_feeding)
        assert model(source, target).shape == (BATCH, VOCAB_SIZE, TRG_LEN)

    @pytest.mark.parametrize('key', ['encoder_type', 'decoder_type'])
    def test_an_unknown_component_name_is_rejected(self, idmaps, key):
        # 名前の正規化 (fix_component_name) が先に弾くため、encoder /
        # decoder のどちらでも同じメッセージになる
        with pytest.raises(ValueError, match='Unsupported component name'):
            build(idmaps, **{key: 'gru'})

    def test_reset_state_clears_the_history(self, idmaps, batch):
        source, target = batch
        model = build(idmaps, 'lstm', 'lstm')
        model(source, target)
        model.reset_state()
        assert model.mod_decode.get_state().get('input_feed') is None


class TestLSTMDecoderState:
    def test_it_can_be_built_on_its_own(self, idmaps):
        # memory_size と share_embedding が EncoderDecoder 側でしか設定
        # されておらず、単独で構築すると KeyError になっていた
        decoder = LSTMDecoder(
            idmaps, embed_size=EMBED, hidden_size=HIDDEN, vocab_size=VOCAB_SIZE,
            padding=0)
        assert decoder.vocab_size == VOCAB_SIZE

    def test_reset_state_returns_self(self, idmaps):
        decoder = LSTMDecoder(
            idmaps, embed_size=EMBED, hidden_size=HIDDEN, vocab_size=VOCAB_SIZE,
            padding=0)
        assert decoder.reset_state() is decoder

    def test_set_state_of_none_resets(self, idmaps):
        decoder = LSTMDecoder(
            idmaps, embed_size=EMBED, hidden_size=HIDDEN, vocab_size=VOCAB_SIZE,
            padding=0)
        assert decoder.set_state(None) is decoder

    def test_set_state_tolerates_a_missing_rnn_entry(self, idmaps):
        # 以前は 'rnn_state' が無いと KeyError になっていた
        decoder = LSTMDecoder(
            idmaps, embed_size=EMBED, hidden_size=HIDDEN, vocab_size=VOCAB_SIZE,
            padding=0)
        decoder.set_state({})
        # get_state は配下の LSTM から状態を取り直す (層ごとに 1 要素)
        assert decoder.get_state()['rnn_state'] == [None]

    def test_prepare_features_records_the_token_ids(self, idmaps):
        # decode_one は seq_enc= で呼んでおり、引数 seq は None のままだった
        decoder = LSTMDecoder(
            idmaps, embed_size=EMBED, hidden_size=HIDDEN, vocab_size=VOCAB_SIZE,
            padding=0)
        features = decoder.prepare_features(seq=torch.zeros(2, 3, dtype=torch.long))
        assert features['id_seq'].shape == (2, 3)

    def test_a_single_step_input_is_expanded(self, idmaps):
        decoder = LSTMDecoder(
            idmaps, embed_size=EMBED, hidden_size=HIDDEN, vocab_size=VOCAB_SIZE,
            padding=0)
        features = decoder.prepare_features(seq=torch.zeros(2, dtype=torch.long))
        assert features['id_seq'].shape == (2, 1)


class TestLSTMEncoder:
    def test_records_the_memory_mask(self, idmaps):
        encoder = LSTMEncoder(
            idmaps, embed_size=EMBED, hidden_size=HIDDEN, vocab_size=VOCAB_SIZE,
            padding=0)
        seq = torch.tensor([[4, 5, 0]])
        features = encoder.prepare_features(seq=seq)
        assert features['mask_mem'].tolist() == [[True, True, False]]


class TestDecoding:
    @pytest.mark.parametrize('encoder_type,decoder_type', list(
        itertools.product(['lstm', 'transformer'], ['lstm', 'transformer'])))
    def test_generate_returns_a_sequence(self, idmaps, encoder_type, decoder_type):
        model = build(idmaps, encoder_type, decoder_type)
        model.eval()
        result = model.generate('one two three', max_length=6)
        assert result is not None

    def test_generate_respects_the_length_limit(self, idmaps):
        model = build(idmaps, 'lstm', 'lstm')
        model.eval()
        result = model.generate('one two', max_length=3)
        assert len(result) <= 3

    @pytest.mark.parametrize('encoder_type,decoder_type', list(
        itertools.product(['lstm', 'transformer'], ['lstm', 'transformer'])))
    def test_beam_search_returns_ranked_candidates(
            self, idmaps, encoder_type, decoder_type):
        model = build(idmaps, encoder_type, decoder_type)
        model.eval()
        results = model.beam_search('one two', beam_width=3, max_length=5)
        assert len(results) > 0
        # スコアは累積コストで、小さいほど良い。run_seq2seq は先頭を採る
        scores = [score for _seq, score in results]
        assert scores == sorted(scores)

    def test_a_wider_beam_does_not_return_fewer_candidates(self, idmaps):
        model = build(idmaps, 'lstm', 'lstm')
        model.eval()
        narrow = model.beam_search('one two', beam_width=1, max_length=5)
        wide = model.beam_search('one two', beam_width=4, max_length=5)
        assert len(wide) >= len(narrow)

    def test_beam_search_stops_at_the_length_limit(self, idmaps):
        model = build(idmaps, 'lstm', 'lstm')
        model.eval()
        for seq, _score in model.beam_search('one two', beam_width=2, max_length=4):
            assert len(seq) <= 4
