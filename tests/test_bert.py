'''Tests for the BERT stack

BERT 一式 (事前学習・分類・ランキング) のテスト。

このスタックは 2019-2020 年当時のまま取り残されており、
共有側の改名 (modules -> modeling, self.vocab -> idmaps) や
署名の変更に追従していなかった。ここでは形状の不変量に加えて、
「特殊記号が語彙に入るか」「パディングが埋め込みを壊さないか」
「セグメント情報がモデルに届くか」を固定する。
'''

import pytest
import torch

from lpu_nn.modeling import bert
from lpu_nn.modeling.bert_classifier import BertClassifier
from lpu_nn.modeling.bert_ranker import BertRanker


class StubVocab:
    '''The subset of Vocabulary that the BERT modules touch

    BERT が参照する語彙の機能だけを備えた代用品。

    SentencePiece の pad_id は既定で -1 (パディング記号を持たない) で
    あり、その値がそのままモデルへ渡る。ここでもそれを再現する。
    '''

    def __init__(self, size=32, pad=-1):
        self.size = size
        self.pad = pad
        self.cls = 1
        self.sep = 2
        self.mask = 3
        self.unk = 0

    def __len__(self):
        return self.size

    def encode(self, sent, to='ids', add_symbols=False):
        # 語を決定的に ID へ写す (記号 4 件分を避けて 4 以上に割り当てる)
        return [4 + (abs(hash(tok)) % (self.size - 4)) for tok in str(sent).split()]

    def convert(self, sent, to='ids'):
        if isinstance(sent, str):
            return self.encode(sent)
        return list(sent)

    def safe_add_symbols(self, ids):
        return list(ids)

    def clean_ids(self, ids):
        return [i for i in ids if i >= 4]


@pytest.fixture
def idmaps():
    return {'seq': StubVocab()}


HPARAMS = {'embed_size': 16, 'hidden_size': 16, 'num_heads': 2, 'num_layers': 2}


class TestBertConfig:
    def test_the_default_is_a_plain_multi_step_transformer(self):
        config = bert.Bert.get_config()
        assert config['universal'] is False
        assert config['num_token_types'] == 0

    def test_the_universal_preset_reaches_its_own_config(self):
        # UniversalTransformer は別モジュールへ分割されており、
        # transformer 側を参照していると AttributeError になる
        config = bert.Bert.get_config(universal=True)
        assert config['universal'] is True
        assert 'max_steps' in config

    def test_the_language_model_shares_its_embedding_by_default(self):
        config = bert.BertLanguageModel.get_config()
        assert config['share_embedding'] is True

    def test_relative_attention_turns_off_position_embedding(self):
        # 相対位置注意を使うときは絶対位置の埋め込みを重ねない
        config = bert.BertLanguageModel.get_config(relative_attention=True)
        assert config['embed_positions'] is False


class TestBert:
    def build(self, idmaps, **overrides):
        hparams = dict(HPARAMS)
        hparams.update(overrides)
        return bert.Bert(idmaps, **hparams)

    def test_the_encoder_keeps_the_sequence_length(self, idmaps):
        model = self.build(idmaps)
        model.reset_state()
        id_seq = torch.tensor([[1, 5, 6, 7], [1, 8, 9, 10]])
        assert model(id_seq).shape == (2, 4, 16)

    def test_a_negative_padding_id_does_not_break_the_embedding(self, idmaps):
        '''SentencePiece has no pad token, so its pad_id is -1

        0.1.0.dev0 は生の nn.Embedding に padding_idx=-1 を渡し、
        バッチ内のパディング位置 (-1) を参照して
        "index out of range in self" で落ちていた。
        modeling.embeddings.Embedding は参照前に 0 へ差し替える。
        '''
        model = self.build(idmaps)
        model.reset_state()
        # 2 行目が短く、末尾が -1 で埋まっている状況
        id_seq = torch.tensor([[1, 5, 6, 7], [1, 8, -1, -1]])
        out = model(id_seq)
        assert out.shape == (2, 4, 16)
        assert torch.isfinite(out).all()

    def test_the_pooled_state_is_published(self, idmaps):
        model = self.build(idmaps)
        model.reset_state()
        model(torch.tensor([[1, 5, 6]]))
        pooled = model.get_pooled()
        assert pooled is not None
        assert pooled.shape == (1, 16)

    def test_resetting_the_state_clears_the_pooled_output(self, idmaps):
        model = self.build(idmaps)
        model.reset_state()
        model(torch.tensor([[1, 5, 6]]))
        model.reset_state()
        assert model.get_pooled() is None

    def test_segment_embedding_changes_the_output(self, idmaps):
        '''Two token types must actually reach the model

        num_token_types を 2 以上にしたとき、セグメント情報が
        出力に効いていること。
        '''
        model = self.build(idmaps, num_token_types=2).eval()
        id_seq = torch.tensor([[1, 5, 6, 7]])
        model.reset_state()
        first = model(id_seq, segment_info=torch.tensor([[1, 1, 1, 1]]))
        model.reset_state()
        second = model(id_seq, segment_info=torch.tensor([[1, 1, 2, 2]]))
        assert not torch.allclose(first, second)

    def test_segment_embedding_tolerates_the_padding_id(self, idmaps):
        # セグメント列も -1 で埋められる
        model = self.build(idmaps, num_token_types=2)
        model.reset_state()
        out = model(torch.tensor([[1, 5, -1]]),
                    segment_info=torch.tensor([[1, 1, -1]]))
        assert torch.isfinite(out).all()

    def test_prepare_input_marks_the_two_segments(self, idmaps):
        model = self.build(idmaps)
        prepared = model.prepare_input('the cat', 'a mat')
        # 先頭は <cls>、以降は各系列の長さだけセグメント 1 / 2 が並ぶ
        assert prepared['x'][0] == idmaps['seq'].cls
        assert prepared['segment_info'][0] == 1
        assert set(prepared['segment_info']) == {1, 2}
        assert len(prepared['x']) == len(prepared['segment_info'])

    def test_prepare_input_of_one_sequence_uses_one_segment(self, idmaps):
        model = self.build(idmaps)
        prepared = model.prepare_input('the cat sat')
        assert set(prepared['segment_info']) == {1}
        assert 's2' not in prepared

    def test_prepare_batch_pads_to_the_longest(self, idmaps):
        model = self.build(idmaps)
        batch = model.prepare_batch([[1, 5, 6, 7], [1, 8]])
        assert batch.shape == (2, 4)
        assert batch[1, -1].item() == idmaps['seq'].pad

    def test_prepare_batch_passes_a_tensor_through(self, idmaps):
        model = self.build(idmaps)
        tensor = torch.zeros(2, 3, dtype=torch.long)
        assert model.prepare_batch(tensor) is tensor

    def test_the_universal_variant_runs(self, idmaps):
        # UniversalTransformer の参照先を誤っていると構築で落ちる
        model = self.build(idmaps, universal=True)
        model.reset_state()
        out = model(torch.tensor([[1, 5, 6]]))
        assert out.shape == (1, 3, 16)

    def test_the_universal_variant_reports_a_ponder_cost(self, idmaps):
        model = self.build(idmaps, universal=True)
        model.reset_state()
        model(torch.tensor([[1, 5, 6]]))
        state = model.get_state()
        assert 'ponder_cost' in state['transformer_state']


class TestBertLanguageModel:
    def build(self, idmaps, **overrides):
        hparams = dict(HPARAMS)
        hparams.update(overrides)
        return bert.BertLanguageModel(idmaps, **hparams)

    def test_decoding_produces_one_distribution_per_token_after_the_cls(
            self, idmaps):
        model = self.build(idmaps)
        model.reset_state()
        h = model(torch.tensor([[1, 5, 6, 7]]))
        logits = model.decode(h)
        # <cls> の分を除いた長さ、語彙軸が第 2 次元に来る
        assert logits.shape == (1, len(idmaps['seq']), 3)

    def test_next_sentence_prediction_is_binary(self, idmaps):
        model = self.build(idmaps)
        model.reset_state()
        h = model(torch.tensor([[1, 5, 6]]))
        assert model.predict_next_sentence(h).shape == (1, 2)

    def test_the_generator_is_tied_to_the_embedding(self, idmaps):
        model = self.build(idmaps, share_embedding=True)
        assert model.mod_generate.weight is model.mod_bert.mod_embed_tok.weight

    def test_the_generator_can_be_untied(self, idmaps):
        model = self.build(idmaps, share_embedding=False)
        assert model.mod_generate.weight is not model.mod_bert.mod_embed_tok.weight

    def test_every_parameter_receives_a_gradient(self, idmaps):
        model = self.build(idmaps)
        model.reset_state()
        h = model(torch.tensor([[1, 5, 6, 7]]))
        (model.decode(h).sum() + model.predict_next_sentence(h).sum()).backward()
        without_grad = [name for name, param in model.named_parameters()
                        if param.requires_grad and param.grad is None]
        assert without_grad == []


class TestBertClassifier:
    def build(self, idmaps, **overrides):
        hparams = dict(HPARAMS)
        hparams.update(overrides)
        return BertClassifier(idmaps, **hparams)

    def test_the_constructor_takes_a_field_map(self, idmaps):
        '''The argument is the field map, not a vocabulary

        0.1.0.dev0 は引数名が vocab のままで vocab.pad を引いており、
        辞書が渡るため AttributeError で構築できなかった。
        '''
        model = self.build(idmaps)
        assert model.padding == idmaps['seq'].pad
        assert model.vocab is idmaps['seq']

    def test_one_logit_per_class(self, idmaps):
        model = self.build(idmaps, num_classes=3)
        out = model(torch.tensor([[1, 5, 6], [1, 7, 8]]))
        assert out.shape == (2, 3)

    def test_the_default_is_binary_classification(self):
        assert BertClassifier.get_config()['num_classes'] == 2

    def test_every_parameter_receives_a_gradient(self, idmaps):
        model = self.build(idmaps, num_classes=3)
        model(torch.tensor([[1, 5, 6, 7]])).sum().backward()
        without_grad = [name for name, param in model.named_parameters()
                        if param.requires_grad and param.grad is None]
        assert without_grad == []


class TestBertRanker:
    def build(self, idmaps, **overrides):
        hparams = dict(HPARAMS)
        hparams.update(overrides)
        return BertRanker(idmaps, **hparams)

    def test_one_score_per_pair(self, idmaps):
        model = self.build(idmaps)
        assert model(torch.tensor([[1, 5, 6], [1, 7, 8]])).shape == (2,)

    def test_the_segment_information_reaches_the_model(self, idmaps):
        '''The keyword was segment_id_seq, which Bert never reads

        0.1.0.dev0 は segment_id_seq という名前で渡しており、
        Bert.forward の引数名 (segment_info) と一致しないため
        **features に吸い込まれて黙って捨てられていた。
        2 系列を区別できないまま学習していたことになる。
        '''
        model = self.build(idmaps, num_token_types=2).eval()
        seq = torch.tensor([[1, 5, 6, 7]])
        first = model(seq, segment_info=torch.tensor([[1, 1, 1, 1]]))
        second = model(seq, segment_info=torch.tensor([[1, 1, 2, 2]]))
        assert not torch.allclose(first, second)

    def test_the_constructor_takes_a_field_map(self, idmaps):
        model = self.build(idmaps)
        assert model.padding == idmaps['seq'].pad

    def test_prepare_input_covers_both_sides(self, idmaps):
        model = self.build(idmaps)
        prepared = model.prepare_input('the cat', 'a mat')
        assert set(prepared['segment_info']) == {1, 2}

    def test_every_parameter_receives_a_gradient(self, idmaps):
        model = self.build(idmaps)
        model(torch.tensor([[1, 5, 6, 7]])).sum().backward()
        without_grad = [name for name, param in model.named_parameters()
                        if param.requires_grad and param.grad is None]
        assert without_grad == []
