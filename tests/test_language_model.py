'''Tests for the language model and the two small commands around it

言語モデルと、その周辺の小さなコマンドのテスト。

`LanguageModel` は復号器を差し替えられるが、CLI から選べるのは LSTM
だけなので、Transformer 側はここで通しておく。サーバとトークナイザの
コマンドは外部との接点を持つため、既定値と後片付けを固定する。
'''

import datetime
import inspect

import pandas as pd
import pytest
import torch

from lpu_nn.commands import run_tokenizer, serve_seq2seq
from lpu_nn.modeling.language_models import LanguageModel

# SentencePiece はパディング記号を持たず、pad_id に -1 を返す
SEQ_PAD = -1


class StubVocab:
    '''The subset of Vocabulary that the language model touches

    言語モデルが参照する語彙の機能だけを備えた代用品。
    '''

    def __init__(self, size=24):
        self.size = size
        self.pad = SEQ_PAD
        self.bos = 1
        self.eos = 2
        self.unk = 0

    def __len__(self):
        return self.size

    def encode(self, sent, to='ids', add_symbols=False):
        ids = [4 + (abs(hash(tok)) % (self.size - 4)) for tok in str(sent).split()]
        return self.safe_add_symbols(ids) if add_symbols else ids

    def safe_add_symbols(self, ids):
        ids = list(ids)
        if not ids or ids[0] != self.bos:
            ids = [self.bos, *ids, self.eos]
        return ids

    def clean_ids(self, ids):
        return [i for i in ids if i >= 4]

    def decode(self, ids, remove_symbols=True, as_tokens=False):
        tokens = [f'w{int(i)}' for i in ids]
        return tokens if as_tokens else ' '.join(tokens)


@pytest.fixture
def idmaps():
    return {'x': StubVocab()}


HPARAMS = {'embed_size': 16, 'hidden_size': 16, 'num_heads': 2,
           'num_layers': 1, 'vocab_size': 24, 'padding': SEQ_PAD}


class TestLanguageModelConfig:
    def test_the_decoder_follows_the_main_component(self):
        assert LanguageModel.get_config()['decoder_type'] == 'lstm'

    def test_the_decoder_can_be_chosen_on_its_own(self):
        config = LanguageModel.get_config(decoder_type='transformer')
        assert config['decoder_type'] == 'transformer'

    def test_an_lstm_decoder_defaults_to_no_attention(self):
        # 言語モデルには参照先の系列が無いので注意機構を持たない
        assert LanguageModel.get_config()['attention_type'] == 'none'


class TestLanguageModel:
    def build(self, idmaps, **overrides):
        hparams = dict(HPARAMS)
        hparams.update(overrides)
        return LanguageModel(idmaps, **hparams)

    @pytest.mark.parametrize('decoder_type', ['lstm', 'transformer'])
    def test_one_distribution_per_position(self, idmaps, decoder_type):
        '''The transformer decoder is reachable from the model, not the CLI

        Transformer 復号器はモデル側にはあるが CLI からは選べないため、
        ここで経路を通しておく。
        '''
        model = self.build(idmaps, decoder_type=decoder_type)
        model.reset_state()
        logits = model(torch.tensor([[1, 5, 6, 2], [1, 7, 8, 2]]))
        # (B, V, L) の並びで返る
        assert logits.shape[0] == 2
        assert logits.shape[1] == len(idmaps['x'])
        assert torch.isfinite(logits).all()

    def test_an_unknown_decoder_is_rejected(self, idmaps):
        with pytest.raises(ValueError, match='unknown decoder type'):
            self.build(idmaps, decoder_type='rnn')

    def test_prepare_batch_wraps_a_single_string(self, idmaps):
        '''pad_sequence takes a sequence of tensors, not one 2-D tensor

        0.1.0.dev0 は文字列 1 件のときだけ 2 次元テンソルを渡しており、
        他の枝と形が揃っていなかった。
        '''
        model = self.build(idmaps)
        batch = model.prepare_batch('the cat sat')
        assert batch.dim() == 2
        assert batch.shape[0] == 1

    def test_prepare_batch_pads_a_series(self, idmaps):
        model = self.build(idmaps)
        batch = model.prepare_batch(pd.Series(['a b c', 'a']))
        assert batch.shape[0] == 2
        assert batch[1, -1].item() == SEQ_PAD

    def test_prepare_batch_passes_a_tensor_through(self, idmaps):
        model = self.build(idmaps)
        tensor = torch.zeros(2, 3, dtype=torch.long)
        assert model.prepare_batch(tensor) is tensor

    def test_prepare_batch_rejects_an_unsupported_type(self, idmaps):
        '''An unhandled type used to fall through to an unbound name

        未対応の型は、未代入の変数を参照して UnboundLocalError になって
        いた。型を名指しして断る。
        '''
        model = self.build(idmaps)
        with pytest.raises(TypeError, match='unsupported type'):
            model.prepare_batch(42)

    def test_restore_batch_drops_the_symbols(self, idmaps):
        model = self.build(idmaps)
        restored = model.restore_batch(torch.tensor([[1, 5, 6, 2]]), to=list)
        assert restored == [[5, 6]]

    def test_restore_batch_squeezes_a_single_row(self, idmaps):
        model = self.build(idmaps)
        assert isinstance(model.restore_batch(torch.tensor([[1, 5, 2]])), str)

    def test_restore_batch_rejects_an_unknown_target(self, idmaps):
        model = self.build(idmaps)
        with pytest.raises(TypeError, match='unknown conversion type'):
            model.restore_batch(torch.tensor([[1, 5, 2]]), to=set)

    def test_resetting_the_state_returns_the_model(self, idmaps):
        model = self.build(idmaps)
        assert model.reset_state() is model
        assert model.last_state == {}


class TestRunTokenizer:
    def test_the_output_format_does_not_shadow_the_builtin(self):
        # 組み込みの format を隠していた引数名を改めた
        params = inspect.signature(run_tokenizer.run_tokenizer).parameters
        assert 'output_format' in params
        assert 'format' not in params

    def test_the_logger_targets_the_packages(self):
        '''Passing the logger object limits the configuration to it

        ロガーオブジェクトを渡すとその 1 つにしか効かない。他のコマンドと
        同じくパッケージ名を対象にする。
        '''
        source = inspect.getsource(run_tokenizer.main)
        assert "'lpu_nn'" in source
        assert "'lpu'" in source

    def test_the_token_format_is_the_default(self):
        assert run_tokenizer.DEFAULT_FORMAT in run_tokenizer.FORMAT_CHOICES


class TestServeSeq2Seq:
    def test_no_environment_variable_is_set_on_import(self):
        '''Importing a module must not turn debugging on for the process

        0.1.0.dev0 は読み込むだけでデバッグ表示を強制する環境変数を
        立てていた。名前も移植前のものが残っていた。
        '''
        source = inspect.getsource(serve_seq2seq)
        assert 'os.environ[' not in source

    def test_the_timestamp_carries_its_zone(self):
        '''The zone is applied when the time is taken, not afterwards

        以前は素の現在時刻 (実行環境の地方時) に日本時間の札を貼って
        いたため、日本時間以外の環境では誤った時刻を報告していた。
        '''
        stamp = serve_seq2seq.timestamp()
        assert stamp.endswith('JST')
        parsed = datetime.datetime.strptime(stamp[:19], '%Y-%m-%d %H:%M:%S')
        expected = datetime.datetime.now(serve_seq2seq.TIMEZONE)
        assert abs((parsed - expected.replace(tzinfo=None)).total_seconds()) < 60

    def test_the_server_listens_on_localhost_by_default(self):
        '''Binding every interface is not a default a research tool needs

        既定で全インターフェースに公開しない。
        '''
        assert serve_seq2seq.DEFAULT_HOST == '127.0.0.1'

    def test_the_debug_mode_is_not_forced(self):
        '''bottle's debug mode returns tracebacks to the caller

        要求元に例外の内容を返してしまうため、既定では有効にしない。
        '''
        live = '\n'.join(
            line for line in inspect.getsource(serve_seq2seq.main).splitlines()
            if not line.lstrip().startswith('#'))
        assert 'debug=True' not in live

    def test_the_template_is_shipped_with_the_package(self):
        import os

        path = os.path.join(serve_seq2seq.TEMPLATE_DIR, 'seq2seq.html')
        assert os.path.isfile(path)

    def test_bottle_is_optional(self):
        '''The module has to import even without the optional dependency

        任意依存が無くても読み込めること。起動時に導線を示して終了する。
        '''
        assert hasattr(serve_seq2seq, 'BOTTLE_IMPORT_ERROR')
        source = inspect.getsource(serve_seq2seq.main)
        assert 'lpu-nn[serve]' in source
