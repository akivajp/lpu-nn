'''Tests for the BERT commands and the vocabulary pieces they need

BERT のコマンドと、それが必要とする語彙まわりのテスト。

このまとまりは共有側の変更 (基底 Trainer の署名、self.vocab の廃止、
モジュール分割) に追従しておらず、引数パーサの構築すらできなかった。
ここでは「基底と署名が一致していること」「特殊記号が語彙に入ること」
を固定する。実際の学習が通ることは CI のスモークテストが担保する。
'''

import inspect

import pytest

from lpu_nn.commands import (
    run_bert_classifier,
    run_bert_ranker,
    train_bert,
    train_bert_classifier,
    train_bert_ranker,
)
from lpu_nn.common import training
from lpu_nn.common.tokenizer import train_tokenizer


def make_trainer():
    '''A trainer that can actually be constructed

    基底の Trainer は specific = None のため単体では構築できない。
    ラベルを扱う具象トレーナーを使う。
    '''
    return train_bert_classifier.BertClassifierTrainer(None)

TRAINERS = [
    train_bert.BertTrainer,
    train_bert_classifier.BertClassifierTrainer,
    train_bert_ranker.BertRankerTrainer,
]
TRAINER_IDS = [cls.__name__ for cls in TRAINERS]


class TestTrainerSignatures:
    '''Overrides must accept everything the base trainer passes

    基底の訓練ループが渡す引数を、上書き側が受け取れること。
    署名がずれると実行時に TypeError となり、その例外は
    ループ側の except に飲まれて「静かに何も学習しない」状態になる。
    '''

    @pytest.mark.parametrize('trainer', TRAINERS, ids=TRAINER_IDS)
    def test_feed_one_batch_accepts_the_data_frame(self, trainer):
        # 基底は df= を渡す。受け取れないと 1 バッチも学習できない
        params = inspect.signature(trainer.feed_one_batch).parameters
        assert 'df' in params

    @pytest.mark.parametrize('trainer', TRAINERS, ids=TRAINER_IDS)
    def test_feed_one_batch_matches_the_base_parameters(self, trainer):
        base = set(inspect.signature(training.Trainer.feed_one_batch).parameters)
        assert base <= set(inspect.signature(trainer.feed_one_batch).parameters)

    @pytest.mark.parametrize('trainer', TRAINERS, ids=TRAINER_IDS)
    def test_evaluate_matches_the_base_parameters(self, trainer):
        # 基底は feed_batches / report も渡す
        base = set(inspect.signature(training.Trainer.evaluate).parameters)
        assert base <= set(inspect.signature(trainer.evaluate).parameters)

    @pytest.mark.parametrize('trainer', TRAINERS, ids=TRAINER_IDS)
    def test_the_trainer_can_be_constructed_with_arguments(self, trainer):
        '''main() builds the trainer as Trainer(args)

        0.1.0.dev0 は args を取らない __init__ を上書きしており、
        main() からの構築が TypeError になっていた。
        '''
        assert trainer(None) is not None


class TestParsers:
    '''Every command must be able to build its argument parser

    引数パーサが構築できること。別名が衝突すると、コマンドは
    --help すら出せずに argparse.ArgumentError で落ちる。
    '''

    @pytest.mark.parametrize('trainer, name', [
        (train_bert.BertTrainer, 'BERT'),
        (train_bert_classifier.BertClassifierTrainer, 'BERT Classifier'),
        (train_bert_ranker.BertRankerTrainer, 'BERT Ranker'),
    ], ids=TRAINER_IDS)
    def test_the_parser_is_built_without_a_conflict(self, trainer, name):
        assert trainer.create_parser(name) is not None

    @pytest.mark.parametrize('trainer, name', [
        (train_bert.BertTrainer, 'BERT'),
        (train_bert_classifier.BertClassifierTrainer, 'BERT Classifier'),
        (train_bert_ranker.BertRankerTrainer, 'BERT Ranker'),
    ], ids=TRAINER_IDS)
    def test_the_parser_accepts_model_specific_defaults(self, trainer, name):
        '''main() rebuilds the parser with the defaults to print --help

        0.1.0.dev0 はファインチューニングの 2 コマンドだけ
        staticmethod のままで default を受け取らず、--help が
        TypeError で落ちていた (通常の学習は 1 引数の呼び出しなので
        影響を受けず、失敗が見えにくかった)。
        '''
        parser = trainer.create_parser(name, trainer.default)
        assert parser['main'].format_help()

    @pytest.mark.parametrize('trainer', [
        train_bert_classifier.BertClassifierTrainer,
        train_bert_ranker.BertRankerTrainer,
    ], ids=['classifier', 'ranker'])
    def test_the_fine_tuning_flag_does_not_shadow_the_preset_flag(self, trainer):
        '''--pre and -P already belong to --preset on the base parser

        0.1.0.dev0 は --pre / -P を --pre-trained-model の別名にしており、
        基底の --preset と衝突してパーサ構築時に落ちていた。
        '''
        parser = trainer.create_parser('x')
        # dest -> 別名一覧。同じ別名を 2 つの引数が持つと argparse が落ちる
        # parser['main'] は ArgumentParser 本体、他は引数グループ
        options = {}
        for group in parser.values():
            group_actions = getattr(group, '_group_actions', None)
            if group_actions is None:
                group_actions = group._actions
            for action in group_actions:
                options[action.dest] = action.option_strings
        assert '--pre-trained-model' in options['pre_trained_model']
        # --pre と -P は基底の --preset のもの。取り合ってはならない
        assert '--pre' not in options['pre_trained_model']
        assert '-P' not in options['pre_trained_model']
        assert '-P' in options['preset']


class TestLoggerTargets:
    '''The scorers must target the packages they actually log from

    採点コマンドが、実際にログを出すパッケージを対象にしていること。
    '''

    @pytest.mark.parametrize('module', [run_bert_classifier, run_bert_ranker],
                             ids=['classifier', 'ranker'])
    def test_the_scorer_targets_the_ported_package_names(self, module):
        source = inspect.getsource(module.main)
        assert "'lpu_nn'" in source
        assert "'lpu'" in source

    @pytest.mark.parametrize('module', [run_bert_classifier, run_bert_ranker],
                             ids=['classifier', 'ranker'])
    def test_the_logging_config_is_used_as_a_context_manager(self, module):
        '''Assigning using_config() never applies it

        0.1.0.dev0 は using_config() の戻り値を変数に代入するだけで、
        設定が一度も適用されなかった (--debug が効かなかった)。
        '''
        source = inspect.getsource(module.main)
        assert 'with logging.using_config(' in source

    @pytest.mark.parametrize('module', [run_bert_classifier, run_bert_ranker],
                             ids=['classifier', 'ranker'])
    def test_no_chainer_leftover_remains_in_the_scoring_path(self, module):
        '''xp and F were chainer names that no longer exist

        xp (chainer の配列モジュール) と F (chainer.functions) は
        移植後も残っており、採点経路は必ず NameError になっていた。
        '''
        # コメントとして残された旧実装は移植の経緯を示すため残してある。
        # ここで確かめたいのは、実行されるコードに残っていないこと
        live = '\n'.join(line for line in inspect.getsource(module).splitlines()
                         if not line.lstrip().startswith('#'))
        assert 'xp = model.xp' not in live
        assert 'F.argmax' not in live
        assert 'F.pad_sequence' not in live
        assert 'sent2idvec' not in live


class TestSaveLabels:
    def test_saved_labels_round_trip_through_load(self, tmp_path):
        '''save_labels must write what load_labels reads

        load_labels はラベルを文字列として保持するよう変更されたのに、
        save_labels だけが対の vocab.decode を呼んでおり、
        しかも self.vocab 自体が廃止されていた。
        '''
        trainer = make_trainer()
        source = tmp_path / 'labels-in.txt'
        source.write_text('long\nshort\n', encoding='utf-8')
        trainer.load_labels(str(source))
        saved = tmp_path / 'labels-out.txt'
        trainer.save_labels(str(saved))
        written = saved.read_text(encoding='utf-8').split()
        # <unk> が先頭に入り、以降は読み込んだ順で並ぶ
        assert written == ['<unk>', 'long', 'short']

    def test_reloading_the_saved_file_gives_the_same_mapping(self, tmp_path):
        trainer = make_trainer()
        source = tmp_path / 'labels.txt'
        source.write_text('long\nshort\n', encoding='utf-8')
        trainer.load_labels(str(source))
        before = dict(trainer.label2id)
        saved = tmp_path / 'saved.txt'
        trainer.save_labels(str(saved))
        again = make_trainer()
        again.load_labels(str(saved))
        # <unk> は load_labels 自身が先頭に足すため重複しないこと
        assert again.labels == trainer.labels
        assert dict(again.label2id) == before


class TestUserDefinedSymbols:
    '''BERT needs <cls> / <sep> / <mask> in the SentencePiece vocabulary

    BERT はコーパスに現れない記号を語彙に必要とする。
    SentencePiece に登録しないと piece_to_id() が unk を返し、
    Vocabulary.set_symbols が ValueError を送出して学習を開始できない。
    '''

    @pytest.fixture
    def corpus(self, tmp_path):
        path = tmp_path / 'corpus.txt'
        with open(path, 'w', encoding='utf-8', newline='\n') as fobj:
            for i in range(200):
                fobj.write(f'the cat sat on the mat number {i % 20}\n')
        return str(path)

    def test_the_symbols_are_registered_in_the_model(self, corpus, tmp_path):
        import sentencepiece as spm

        prefix = str(tmp_path / 'sp')
        train_tokenizer(prefix, corpus, vocab_size=60,
                        user_defined_symbols=['<cls>', '<sep>', '<mask>'])
        sp = spm.SentencePieceProcessor()
        sp.load(prefix + '.model')
        for symbol in ('<cls>', '<sep>', '<mask>'):
            # unk に落ちず、固有の ID を持つこと
            assert sp.piece_to_id(symbol) != sp.unk_id()

    def test_without_the_symbols_they_fall_back_to_unk(self, corpus, tmp_path):
        import sentencepiece as spm

        prefix = str(tmp_path / 'sp')
        train_tokenizer(prefix, corpus, vocab_size=60)
        sp = spm.SentencePieceProcessor()
        sp.load(prefix + '.model')
        # 指定しなければ未知語になる (これが移植時の失敗原因だった)
        assert sp.piece_to_id('<cls>') == sp.unk_id()

    def test_a_symbol_containing_a_comma_is_rejected(self, corpus, tmp_path):
        # SentencePiece の引数はカンマ区切りのため分割できない
        prefix = str(tmp_path / 'sp')
        with pytest.raises(ValueError, match='comma'):
            train_tokenizer(prefix, corpus, vocab_size=60,
                            user_defined_symbols=['<a,b>'])


class TestPreTrainedVocabularyGuard:
    """Fine-tuning must refuse a pre-trained model with another vocabulary

    ファインチューニングは、語彙の異なる事前学習モデルを拒むこと。

    mod_bert をまるごと差し替えるため、語彙が違うと埋め込みの形と
    設定上の語彙サイズが食い違い、書き出したチェックポイントを
    読み直せなくなる。作業ディレクトリごとにトークナイザを学習するので、
    --sentencepiece を指定しない限りまず一致しない。
    """

    @pytest.mark.parametrize('module, attr', [
        ('train_bert_classifier', 'BertClassifierTrainer'),
        ('train_bert_ranker', 'BertRankerTrainer'),
    ])
    def test_the_guard_is_present_and_names_the_remedy(self, module, attr):
        import importlib

        trainer = getattr(importlib.import_module(f'lpu_nn.commands.{module}'), attr)
        source = inspect.getsource(trainer.setup_model)
        assert 'vocab_size' in source
        # 対処方法を利用者に示すこと
        assert '--sentencepiece' in source


class TestLoadLabelsForScorer:
    def test_the_label_file_is_found_beside_the_checkpoint(self, tmp_path):
        '''labels.txt lives in the work directory, not in the record

        labels.txt はチェックポイントの中ではなく作業ディレクトリ直下に
        書かれるため、チェックポイントの親も探索すること。
        '''
        workdir = tmp_path / 'work'
        record = workdir / 'record.best'
        record.mkdir(parents=True)
        (workdir / 'labels.txt').write_text('long\nshort\n', encoding='utf-8')
        trainer = make_trainer()
        run_bert_classifier.load_labels_for(trainer, str(record))
        assert trainer.labels == ['<unk>', 'long', 'short']

    def test_a_missing_label_file_is_reported(self, tmp_path):
        trainer = make_trainer()
        with pytest.raises(RuntimeError, match='not found labels file'):
            run_bert_classifier.load_labels_for(trainer, str(tmp_path))
