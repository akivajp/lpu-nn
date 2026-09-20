'''Tests for the sequence tagging stack

系列タギング一式 (CRF / SequenceTagger / タガーのコマンド) のテスト。

このまとまりは符号化器ごとに系列の長さとマスクの対応が変わる
(BERT だけ先頭に <cls> が付いて 1 つ長い) ため、位置のずれが
静かに成績を壊す。ここでは長さの不変量と、パディングが損失・
復号・タグ列に影響しないことを固定する。
'''

import inspect

import pytest
import torch

from lpu.common.vocab import IDMap
from lpu_nn.commands import train_tagger
from lpu_nn.common import criteria, training
from lpu_nn.common.vocab import FieldMap
from lpu_nn.modeling.sequence_tagger import CRF, SequenceTagger

# SentencePiece はパディング記号を持たず pad_id に -1 を返す。
# タグ表 (IDMap) の pad は 0 であり、両者は一致しない
SEQ_PAD = -1


class StubVocab:
    '''The subset of Vocabulary that the tagger touches

    タガーが参照する語彙の機能だけを備えた代用品。
    '''

    def __init__(self, size=24):
        self.size = size
        self.pad = SEQ_PAD
        self.bos = 1
        self.eos = 2
        self.unk = 0
        self.cls = 3

    def __len__(self):
        return self.size

    def encode(self, sent, to='ids', add_symbols=False):
        return [4 + (abs(hash(tok)) % (self.size - 4)) for tok in str(sent).split()]

    def convert(self, sent, to='ids'):
        return self.encode(sent) if isinstance(sent, str) else list(sent)

    def decode(self, ids, remove_symbols=True, as_tokens=False):
        tokens = [f'w{int(i)}' for i in ids]
        return tokens if as_tokens else ' '.join(tokens)


def build_tagmap():
    '''A BIO tag map, as the trainer builds it

    トレーナーが作るのと同じ BIO のタグ表。
    '''
    tagmap = IDMap().set_symbols()
    for tag in ('O', 'B-EVEN', 'I-EVEN'):
        tagmap.str2id(tag, growth=True)
    return tagmap


@pytest.fixture
def idmaps():
    # FieldMap は系列語彙をフィールド名と 'seq' の両方の鍵で保持する
    # (BERT 符号化器は 'seq' を引く)
    vocab = StubVocab()
    return {'x': vocab, 'seq': vocab, 't': build_tagmap()}


# transformer.Encoder は語彙数を設定から読む (訓練時はトレーナーが
# cdata.model.vocab_size に入れる)
HPARAMS = {'embed_size': 16, 'hidden_size': 16, 'num_heads': 2,
           'num_layers': 1, 'vocab_size': 24}


class TestCRF:
    def build(self, idmaps):
        return CRF(idmaps['t'])

    def make(self, batch, length, num_tags):
        torch.manual_seed(0)
        return torch.randn(batch, length, num_tags)

    def test_a_padded_tag_does_not_crash_the_loss(self, idmaps):
        '''Tags are padded with the sequence vocabulary's pad, which is -1

        タグ列は系列語彙のパディング ID (-1) で埋められる。CRF が見て
        いたのはタグ表のパディング (0) だったため、gather が
        "index -1 is out of bounds" で落ちていた。
        '''
        crf = self.build(idmaps)
        num_tags = len(idmaps['t'])
        logits = self.make(2, 4, num_tags)
        t = torch.tensor([[4, 5, 6, 4], [4, 5, SEQ_PAD, SEQ_PAD]])
        mask = (t != SEQ_PAD)
        loss = crf.loss(logits, t, mask_x=mask)
        assert torch.isfinite(loss).all()

    def test_padding_does_not_change_the_loss_of_a_shorter_row(self, idmaps):
        '''Padding a row further must not move its loss

        同じ内容の行をさらにパディングしても損失が変わらないこと。
        パディング位置で前向きスコアを更新していると、分配関数が
        パディング分まで積み上がって値が動く。
        '''
        crf = self.build(idmaps)
        num_tags = len(idmaps['t'])
        torch.manual_seed(0)
        base_logits = torch.randn(1, 2, num_tags)
        short_t = torch.tensor([[4, 5]])
        short = crf.loss(base_logits, short_t, mask_x=(short_t != SEQ_PAD),
                         reduction='none')
        # 同じ 2 語に、パディングを 3 つ足した行
        long_logits = torch.cat([base_logits, torch.randn(1, 3, num_tags)], dim=1)
        long_t = torch.tensor([[4, 5, SEQ_PAD, SEQ_PAD, SEQ_PAD]])
        long = crf.loss(long_logits, long_t, mask_x=(long_t != SEQ_PAD),
                        reduction='none')
        assert torch.allclose(short, long, atol=1e-5)

    def test_the_loss_is_normalized_by_the_real_length(self, idmaps):
        '''A short row must not be penalized for the batch's padding

        正規化は実長で行うこと。固定長で割ると、バッチ内の短い系列が
        パディングの分だけ過小評価される。
        '''
        crf = self.build(idmaps)
        source = inspect.getsource(crf.loss)
        assert 'mask_x.sum(1)' in source

    def test_decoding_returns_one_tag_per_position(self, idmaps):
        crf = self.build(idmaps)
        num_tags = len(idmaps['t'])
        logits = self.make(2, 5, num_tags)
        mask = torch.ones(2, 5, dtype=torch.bool)
        decoded = crf.decode(logits, mask_x=mask, mask_x_bos=~mask,
                             mask_x_eos=~mask)
        assert decoded.shape == (2, 5)
        assert decoded.min() >= 0
        assert decoded.max() < num_tags


ENCODERS = ['lstm', 'transformer', 'bert']
DECODERS = ['linear', 'crf']


class TestSequenceTagger:
    def build(self, idmaps, **overrides):
        hparams = dict(HPARAMS)
        hparams.update(overrides)
        return SequenceTagger(idmaps, **hparams)

    @pytest.mark.parametrize('encoder_type', ENCODERS)
    def test_the_tagger_is_built_for_every_encoder(self, idmaps, encoder_type):
        # BERT 符号化器は微調整用の Transformer も併せて作る
        model = self.build(idmaps, encoder_type=encoder_type)
        assert model.num_tags == len(idmaps['t'])
        assert hasattr(model, 'mod_encode')

    def test_the_fine_tuning_layer_is_not_conditioned(self, idmaps):
        '''Transformer's first parameter is `conditioned`, not the field map

        0.1.0.dev0 は transformer.Transformer(idmaps, ...) と呼んでおり、
        第 1 引数 conditioned (bool) に辞書が渡っていた。真値として
        解釈され、BERT の上に載せる微調整層が条件付き
        (cross-attention) モードで構築されていた。
        '''
        model = self.build(idmaps, encoder_type='bert')
        assert model.mod_tune.conditioned is False

    @pytest.mark.parametrize('decoder_type', DECODERS)
    def test_the_crf_is_built_only_when_asked(self, idmaps, decoder_type):
        model = self.build(idmaps, decoder_type=decoder_type)
        assert hasattr(model, 'mod_crf') == (decoder_type == 'crf')

    def test_prepare_batch_keeps_the_sequence_aligned_with_the_tags(self, idmaps):
        '''Adding symbols here would break the 1:1 alignment

        encode_pair が系列とタグを 1:1 に対応付けて ID 化しているため、
        ここで特殊記号を足すと対応が崩れる。0.1.0.dev0 は batch を作る
        行が両方ともコメントアウトされており、BERT 以外の符号化器では
        UnboundLocalError になっていた。
        '''
        import pandas as pd

        model = self.build(idmaps)
        series = pd.Series([[4, 5, 6], [7, 8]])
        batch = model.prepare_batch('x', series)
        assert batch.shape == (2, 3)
        assert batch[1, -1].item() == SEQ_PAD

    def test_prepare_batch_skips_segments_without_bert(self, idmaps):
        import pandas as pd

        model = self.build(idmaps, encoder_type='lstm')
        assert model.prepare_batch('segment', pd.Series([None, None])) is None

    @pytest.mark.parametrize('encoder_type', ENCODERS)
    def test_prepare_features_supplies_what_the_encoder_needs(
            self, idmaps, encoder_type):
        '''Each encoder declares its own mandatory features

        LSTM 符号化器は mask_mem を必須とするなど、必要な特徴は
        符号化器ごとに異なる。タガーが自前の特徴しか用意しないと
        forward が KeyError になる。
        '''
        model = self.build(idmaps, encoder_type=encoder_type)
        x = torch.tensor([[4, 5, 6]])
        features = model.prepare_features(x)
        assert 'mask_x' in features
        # 系列は位置引数で渡すため、同じものを指す id_seq は残さない
        assert 'id_seq' not in features

    @pytest.mark.parametrize('encoder_type', ENCODERS)
    @pytest.mark.parametrize('decoder_type', DECODERS)
    def test_the_loss_is_finite_for_every_combination(
            self, idmaps, encoder_type, decoder_type):
        model = self.build(idmaps, encoder_type=encoder_type,
                           decoder_type=decoder_type)
        model.reset_state()
        x, t = self.make_batch(model, encoder_type)
        loss = model.loss(x, t)
        assert torch.isfinite(loss).all()

    @pytest.mark.parametrize('encoder_type', ENCODERS)
    @pytest.mark.parametrize('decoder_type', DECODERS)
    def test_decoding_returns_one_tag_per_tag_position(
            self, idmaps, encoder_type, decoder_type):
        '''The BERT encoder prefixes <cls>, so x is one longer than t

        BERT 符号化器は先頭に <cls> を付けるため x は t より 1 つ長い。
        復号結果はタグ側の長さに一致すること。
        '''
        model = self.build(idmaps, encoder_type=encoder_type,
                           decoder_type=decoder_type)
        model.reset_state()
        x, t = self.make_batch(model, encoder_type)
        decoded = model.decode(x)
        assert decoded.shape == t.shape

    def make_batch(self, model, encoder_type):
        '''A padded batch whose tags are one shorter under BERT

        パディング済みのバッチ。BERT 符号化器では x だけ <cls> の分
        長くなる。
        '''
        t = torch.tensor([[4, 5, 6], [4, 5, SEQ_PAD]])
        if encoder_type == 'bert':
            x = torch.tensor([[3, 7, 8, 9], [3, 7, 8, SEQ_PAD]])
        else:
            x = torch.tensor([[7, 8, 9], [7, 8, SEQ_PAD]])
        return x, t


class TestTagNormalization:
    '''The model may predict tags the BIO scheme has no place for

    モデルはタグ表の特殊記号も予測しうる。
    '''

    class StubTagger:
        def __init__(self, tagmap):
            self.idmaps = {'t': tagmap}

    def make_trainer(self, idmaps):
        trainer = train_tagger.TaggerTrainer(None)
        trainer.model = self.StubTagger(idmaps['t'])
        return trainer

    @pytest.mark.parametrize('symbol', ['<pad>', '<s>', '</s>', '<unk>'])
    def test_a_symbol_tag_becomes_outside(self, idmaps, symbol):
        '''head = tag[0:1] turns '<pad>' into '<' and 'ad>'

        0.1.0.dev0 は <unk> だけを潰しており、他の記号は
        head='<' / label='ad>' として扱われ、出力に "<ad>>" のような
        壊れたタグが現れていた。
        '''
        trainer = self.make_trainer(idmaps)
        assert trainer.normalize_tag(symbol) == 'O'

    @pytest.mark.parametrize('tag', ['O', 'B-EVEN', 'I-EVEN'])
    def test_a_real_tag_is_left_alone(self, idmaps, tag):
        trainer = self.make_trainer(idmaps)
        assert trainer.normalize_tag(tag) == tag


class TestCriteriaWithNegativePadding:
    '''A list of ignored indices must tolerate a negative padding id

    無視する添字をリストで渡す経路が、負のパディング ID を扱えること。
    '''

    def test_a_negative_target_is_not_rejected(self):
        '''torch validates the target before we can mask it

        torch は範囲外の添字を計算前に検査するため、マスクより先に
        有効な添字へ差し替える必要がある。0.1.0.dev0 はそのまま渡して
        おり "Target -1 is out of bounds" で落ちていた。
        '''
        logits = torch.randn(1, 3, 4)
        t = torch.tensor([[0, 1, -1, -1]])
        loss = criteria.cross_entropy(logits, t, ignore_index=[-1],
                                      reduction='hmean')
        assert torch.isfinite(loss).all()

    def test_the_ignored_positions_do_not_contribute(self):
        logits = torch.randn(1, 3, 4)
        t = torch.tensor([[0, 1, 2, 2]])
        # 同じ入力で、後ろ 2 つを無視した場合と実際に切り詰めた場合
        masked = criteria.cross_entropy(logits, t, ignore_index=[2],
                                        reduction='hmean')
        kept = criteria.cross_entropy(logits[:, :, :2], t[:, :2],
                                      ignore_index=[2], reduction='hmean')
        assert torch.allclose(masked, kept, atol=1e-5)


class TestTaggerTrainerSignatures:
    def test_feed_one_batch_accepts_the_data_frame(self):
        params = inspect.signature(train_tagger.TaggerTrainer.feed_one_batch).parameters
        assert 'df' in params

    def test_evaluate_matches_the_base_parameters(self):
        base = set(inspect.signature(training.Trainer.evaluate).parameters)
        assert base <= set(inspect.signature(train_tagger.TaggerTrainer.evaluate).parameters)

    def test_the_parser_is_built_without_a_conflict(self):
        '''--pre and -P already belong to --preset on the base parser

        別名が衝突すると argparse はパーサ構築時に落ち、--help すら
        出せなくなる。
        '''
        parser = train_tagger.TaggerTrainer.create_parser('Tagger')
        options = {}
        for group in parser.values():
            actions = getattr(group, '_group_actions', None) or group._actions
            for action in actions:
                options[action.dest] = action.option_strings
        assert '--pre-trained-model' in options['pre_trained_model']
        assert '-P' not in options['pre_trained_model']
        assert '-P' in options['preset']

    def test_the_parser_accepts_model_specific_defaults(self):
        parser = train_tagger.TaggerTrainer.create_parser(
            'Tagger', train_tagger.TaggerTrainer.default)
        assert parser['main'].format_help()

    def test_the_tagger_reads_a_sequence_and_its_tags(self):
        fields = train_tagger.specific.data.format
        assert fields.input.x == 'seq'
        assert fields.output.t == 'tags'


class TestFieldMapSymbols:
    '''The extra symbols belong to the sequence vocabulary only

    追加記号は系列語彙のためのものであること。
    '''

    def test_a_tag_map_keeps_its_separator(self, tmp_path):
        '''Registering 'sep' on an IDMap clobbers its field separator

        IDMap は 'sep' をフィールド分割用の文字列として持つ。BERT の
        <sep> をそこへ登録すると属性が整数で上書きされ、その語彙の
        encode / decode が TypeError になる。タグ表に無関係な記号が
        増えて分類クラスが水増しされる問題もある。
        '''
        fieldmap = FieldMap()
        tagmap = build_tagmap()
        fieldmap.dict_maps['t'] = tagmap
        before = len(tagmap)
        fieldmap.set_symbols({'cls': '<cls>', 'sep': '<sep>', 'mask': '<mask>'})
        assert tagmap.sep == ' '
        assert len(tagmap) == before
        # 区切り子が壊れていなければ符号化・復号が通る
        assert tagmap.decode(tagmap.encode('O B-EVEN')) == 'O B-EVEN'
