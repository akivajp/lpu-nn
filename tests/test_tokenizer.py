'''Tests for lpu_nn.common.tokenizer

lpu_nn.common.tokenizer のテスト。

前処理はコーパスをセグメント単位に分解して作業用ファイルへ書き出し、
SentencePiece にはそのファイルの**パス**を渡す。つまりバッファに残った
内容は SentencePiece からは存在しないため、書き出しの確実性がそのまま
学習データの量になる。
'''

import os

import pytest

from lpu_nn.common import tokenizer


@pytest.fixture
def corpus(tmp_path):
    '''A 20-line parallel corpus where every word repeats

    全ての語が繰り返し現れる 20 行の対訳コーパス。
    '''
    import random
    rng = random.Random(0)
    words = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    path = tmp_path / 'corpus.tsv'
    with open(path, 'w', encoding='utf-8', newline='\n') as fobj:
        for _ in range(20):
            left = ' '.join(rng.choice(words) for _ in range(3))
            right = ' '.join(rng.choice(words) for _ in range(3))
            fobj.write(f'{left}\t{right}\n')
    return str(path)


def run_preprocess(paths, **kwargs):
    '''Call preprocess with the arguments train_tokenizer uses

    train_tokenizer が渡すのと同じ引数で preprocess を呼ぶ。
    '''
    kwargs.setdefault('min_count', 2)
    kwargs.setdefault('max_count', 5000)
    kwargs.setdefault('max_bytes', 2 ** 31 - 1)
    return tokenizer.preprocess(paths, **kwargs)


class TestSplitDigitChars:
    def test_separates_each_digit(self):
        assert tokenizer.split_digit_chars(b'a1b') == [b'a', b'1', b'b']

    def test_splits_a_run_of_digits_one_by_one(self):
        assert tokenizer.split_digit_chars(b'12') == [b'1', b'2']

    def test_leaves_a_word_untouched(self):
        assert tokenizer.split_digit_chars(b'word') == [b'word']

    def test_an_empty_input_yields_nothing(self):
        assert tokenizer.split_digit_chars(b'') == []

    def test_the_pieces_reassemble_into_the_input(self):
        source = b'ab12cd3'
        assert b''.join(tokenizer.split_digit_chars(source)) == source


class TestLoadCoverage:
    def test_collects_every_written_form(self, tmp_path):
        # 原形・小文字・NFKC 正規化のいずれでも引けるようにすること。
        # 全角のラテン文字は、正規化を検証するための入力そのもの
        path = tmp_path / 'cover.txt'
        path.write_text('Ｃａｔ\n', encoding='utf-8')  # noqa: RUF001
        coverage = tokenizer.load_coverage(str(path))
        assert 'Ｃａｔ'.encode() in coverage  # noqa: RUF001
        assert b'Cat' in coverage
        assert b'cat' in coverage

    def test_splits_a_line_on_spaces(self, tmp_path):
        path = tmp_path / 'cover.txt'
        path.write_text('one two\n', encoding='utf-8')
        coverage = tokenizer.load_coverage(str(path))
        assert b'one' in coverage and b'two' in coverage


class TestPreprocess:
    def test_the_scratch_file_is_on_disk_when_it_returns(self, corpus):
        # SentencePiece はパスを渡されてディスクから読むため、バッファに
        # 残った内容は存在しないのと同じ。flush が無いと小さなコーパスでは
        # 0 バイト、大きなコーパスでも末尾が欠けていた
        temp = run_preprocess([corpus])
        assert os.path.getsize(temp.name) > 0

    def test_every_written_byte_is_visible(self, corpus):
        temp = run_preprocess([corpus])
        size_before = os.path.getsize(temp.name)
        temp.flush()
        assert os.path.getsize(temp.name) == size_before

    def test_a_single_path_string_is_accepted(self, corpus):
        assert os.path.getsize(run_preprocess(corpus).name) > 0

    def test_min_count_holds_back_the_first_occurrences(self, tmp_path):
        # 各セグメントは min_count 回目の出現から書き出される
        path = tmp_path / 'once.tsv'
        path.write_text('alpha\tbeta\n' * 3, encoding='utf-8')
        once = run_preprocess([str(path)], min_count=1)
        twice = run_preprocess([str(path)], min_count=2)
        assert os.path.getsize(once.name) > os.path.getsize(twice.name)

    def test_an_empty_result_is_reported(self, tmp_path):
        # 以前は SentencePiece 内部の読みにくいエラーとして現れていた
        path = tmp_path / 'once.tsv'
        path.write_text('alpha\tbeta\n', encoding='utf-8')
        with pytest.raises(ValueError, match='empty'):
            run_preprocess([str(path)], min_count=99)

    def test_max_segments_limits_the_input(self, corpus):
        small = run_preprocess([corpus], max_segments=5)
        full = run_preprocess([corpus])
        assert os.path.getsize(small.name) < os.path.getsize(full.name)

    def test_digits_are_split_when_asked(self, tmp_path):
        path = tmp_path / 'digits.tsv'
        path.write_text('a12\ta12\n' * 3, encoding='utf-8')
        split = run_preprocess([str(path)], split_digits=True)
        split.seek(0)
        assert b'1\n' in split.read()


class TestTrainTokenizer:
    def test_trains_a_model_from_a_small_corpus(self, corpus, tmp_path):
        # flush が無いうちは、この規模では 0 バイトの入力になり
        # SentencePiece が内部エラーで落ちていた
        prefix = str(tmp_path / 'sp')
        model_path = tokenizer.train_tokenizer(prefix, corpus, vocab_size=40)
        assert os.path.isfile(model_path)
        assert os.path.isfile(prefix + '.vocab')

    def test_the_returned_path_follows_the_prefix(self, corpus, tmp_path):
        prefix = str(tmp_path / 'sp')
        assert tokenizer.train_tokenizer(prefix, corpus, vocab_size=40) == prefix + '.model'

    @pytest.mark.parametrize('model_type', ['unigram', 'bpe', 'char'])
    def test_every_model_type_trains(self, corpus, tmp_path, model_type):
        prefix = str(tmp_path / model_type)
        tokenizer.train_tokenizer(
            prefix, corpus, vocab_size=40, model_type=model_type)
        assert os.path.isfile(prefix + '.model')

    def test_the_trained_model_round_trips_text(self, corpus, tmp_path):
        import sentencepiece as spm
        prefix = str(tmp_path / 'sp')
        processor = spm.SentencePieceProcessor()
        processor.load(tokenizer.train_tokenizer(prefix, corpus, vocab_size=40))
        assert processor.decode_ids(processor.encode_as_ids('alpha beta')) == 'alpha beta'
