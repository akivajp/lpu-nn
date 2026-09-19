'''Tests for lpu_nn.common.vocab

lpu_nn.common.vocab のテスト。

`Vocabulary` は SentencePiece モデルの薄い包みであり、seq2seq の入出力
そのものを担う。ここが静かに壊れるとモデル全体の品質が劣化するため、
符号化と復号の往復、特殊記号の付け外し、チェックポイント保存に使われる
状態の往復を中心に確認する。

`IDMap` / `LabelMap` は lpu 側の実装を再エクスポートしているだけなので、
それらの振る舞いは lpu のテストが担保する。ここでは `FieldMap` から
正しく使えることだけを見る。
'''

import shutil
from collections import OrderedDict
from pathlib import Path

import pytest

from lpu_nn.common import vocab

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / 'tests' / 'data' / 'train.tsv'

MAIN_FIELDS = OrderedDict([('x', 'seq'), ('t', 'seq')])


@pytest.fixture(scope='module')
def workdir(tmp_path_factory):
    '''Train a small SentencePiece model once for the whole module

    モジュール全体で 1 度だけ小さな SentencePiece モデルを学習する。
    '''
    path = tmp_path_factory.mktemp('vocab')
    vocab.FieldMap.train(str(path), MAIN_FIELDS, str(CORPUS), 100, {})
    return path


@pytest.fixture
def word_vocab(workdir):
    '''A Vocabulary with the default special symbols registered

    既定の特殊記号を登録済みの Vocabulary。
    '''
    return vocab.Vocabulary().load(str(workdir / 'sp.model')).set_symbols()


class TestVocabularyLoading:
    def test_train_writes_a_sentencepiece_model(self, workdir):
        assert (workdir / 'sp.model').is_file()
        assert (workdir / 'sp.vocab').is_file()

    def test_loads_accepts_the_serialized_model(self, workdir):
        buffer = (workdir / 'sp.model').read_bytes()
        loaded = vocab.Vocabulary().loads(buffer).set_symbols()
        assert len(loaded) > 0

    def test_the_special_symbols_resolve_to_ids(self, word_vocab):
        assert word_vocab.symbols['bos'] == '<s>'
        assert word_vocab.bos == word_vocab.sp.piece_to_id('<s>')
        assert word_vocab.eos == word_vocab.sp.piece_to_id('</s>')

    def test_iteration_yields_every_piece(self, word_vocab):
        pieces = list(word_vocab)
        assert len(pieces) == len(word_vocab)
        assert '<s>' in pieces


class TestVocabularyEncoding:
    def test_encode_decode_round_trip(self, word_vocab):
        text = 'one two three'
        assert word_vocab.decode(word_vocab.encode(text)) == text

    def test_encode_to_pieces_joins_back_to_the_text(self, word_vocab):
        pieces = word_vocab.encode('one two', to='pieces')
        assert all(isinstance(piece, str) for piece in pieces)
        assert word_vocab.decode(pieces) == 'one two'

    def test_encode_rejects_an_unknown_target(self, word_vocab):
        with pytest.raises(ValueError):
            word_vocab.encode('one', to='vectors')

    def test_add_symbols_surrounds_the_ids(self, word_vocab):
        ids = word_vocab.encode('one two', add_symbols=True)
        assert ids[0] == word_vocab.bos
        assert ids[-1] == word_vocab.eos

    def test_safe_add_symbols_does_not_duplicate(self, word_vocab):
        once = word_vocab.safe_add_symbols(word_vocab.encode('one'))
        assert word_vocab.safe_add_symbols(once) == once

    def test_decode_strips_the_symbols_by_default(self, word_vocab):
        ids = word_vocab.encode('one two', add_symbols=True)
        assert word_vocab.decode(ids) == 'one two'

    def test_decode_as_tokens_returns_the_pieces(self, word_vocab):
        ids = word_vocab.encode('one two')
        tokens = word_vocab.decode(ids, remove_symbols=False, as_tokens=True)
        assert isinstance(tokens, list)
        assert all(isinstance(token, str) for token in tokens)

    def test_decode_of_an_empty_sequence_is_an_empty_string(self, word_vocab):
        assert word_vocab.decode([]) == ''

    def test_clean_ids_removes_the_surrounding_symbols(self, word_vocab):
        ids = word_vocab.encode('one two')
        decorated = [word_vocab.bos, *ids, word_vocab.eos]
        assert word_vocab.clean_ids(decorated) == ids

    def test_clean_ids_truncates_at_the_first_eos(self, word_vocab):
        ids = word_vocab.encode('one two')
        assert word_vocab.clean_ids([*ids, word_vocab.eos, *ids]) == ids


class TestVocabularyConversion:
    def test_convert_between_every_representation(self, word_vocab):
        text = 'one two'
        ids = word_vocab.convert(text, 'ids')
        pieces = word_vocab.convert(text, 'pieces')
        assert word_vocab.convert(ids, 'pieces') == pieces
        assert word_vocab.convert(pieces, 'ids') == ids
        assert word_vocab.convert(ids, 'str') == text
        assert word_vocab.convert(text, 'str') == text

    def test_convert_preserves_the_sequence_type(self, word_vocab):
        ids = tuple(word_vocab.encode('one'))
        assert isinstance(word_vocab.convert(ids, 'pieces'), tuple)

    def test_convert_of_an_empty_sequence_stays_empty(self, word_vocab):
        assert word_vocab.convert([], 'ids') == []

    def test_convert_rejects_an_unsupported_input(self, word_vocab):
        with pytest.raises(ValueError):
            word_vocab.convert(3.14, 'ids')

    def test_convert_rejects_an_unknown_target(self, word_vocab):
        with pytest.raises(ValueError):
            word_vocab.convert('one', 'vectors')


class TestVocabularyHelpers:
    def test_remove_unk_drops_the_unknown_surface(self, word_vocab):
        assert word_vocab.remove_unk('a <unk> b') == 'a  b'

    def test_remove_unk_drops_the_unknown_id(self, word_vocab):
        ids = [word_vocab.unk, *word_vocab.encode('one')]
        assert word_vocab.unk not in word_vocab.remove_unk(ids)

    def test_remove_unk_rejects_an_unsupported_type(self, word_vocab):
        with pytest.raises(TypeError):
            word_vocab.remove_unk(3.14)

    def test_sample_stays_inside_the_vocabulary(self, word_vocab):
        for _ in range(50):
            sampled = word_vocab.sample()
            assert 0 <= sampled < len(word_vocab)

    def test_sample_can_be_biased_towards_given_ids(self, word_vocab):
        # additions を与えた場合もその範囲か語彙内に収まること
        for _ in range(50):
            sampled = word_vocab.sample(additions=[7])
            assert sampled == 7 or 0 <= sampled < len(word_vocab)


class TestVocabularyState:
    '''The state round trip is how a vocabulary survives a checkpoint

    状態の往復は、語彙がチェックポイントを越えて保たれる経路そのもの。
    '''

    def test_get_state_does_not_carry_the_processor(self, word_vocab):
        # SentencePiceProcessor は pickle できないため、状態には入れない
        state = word_vocab.get_state()
        assert state['sp'] is None
        assert 'sp_bytes' in state

    def test_set_state_restores_a_working_vocabulary(self, word_vocab):
        restored = vocab.Vocabulary().set_state(word_vocab.get_state())
        assert len(restored) == len(word_vocab)
        text = 'one two three'
        assert restored.decode(restored.encode(text)) == text

    def test_the_restored_ids_match_the_original(self, word_vocab):
        restored = vocab.Vocabulary().set_state(word_vocab.get_state())
        assert restored.encode('one two') == word_vocab.encode('one two')
        assert (restored.bos, restored.eos) == (word_vocab.bos, word_vocab.eos)


class TestFieldMap:
    def test_load_registers_the_sequence_vocabulary(self, workdir):
        field_map = vocab.FieldMap().load(str(workdir), MAIN_FIELDS)
        assert isinstance(field_map.dict_maps['seq'], vocab.Vocabulary)

    def test_train_is_skipped_when_the_model_exists(self, workdir):
        # 既に学習済みなら再学習しないこと (再実行が高くつかない)
        before = (workdir / 'sp.model').stat().st_mtime_ns
        vocab.FieldMap.train(str(workdir), MAIN_FIELDS, str(CORPUS), 100, {})
        assert (workdir / 'sp.model').stat().st_mtime_ns == before

    def test_the_tag_maps_come_from_lpu(self, tmp_path):
        # IDMap / LabelMap は lpu の検証済み実装を再エクスポートしている
        from lpu.common.vocab import IDMap as LpuIDMap
        from lpu.common.vocab import LabelMap as LpuLabelMap
        assert vocab.IDMap is LpuIDMap
        assert vocab.LabelMap is LpuLabelMap

    def test_train_builds_a_map_for_a_tag_column(self, workdir, tmp_path):
        # 学習済みの sp.model を置いておくと SentencePiece の学習は省かれ、
        # タグ列のマップ構築だけが走る
        shutil.copy(workdir / 'sp.model', tmp_path / 'sp.model')
        corpus = tmp_path / 'tagged.tsv'
        with open(CORPUS, encoding='utf-8') as source, \
             open(corpus, 'w', encoding='utf-8', newline='\n') as target:
            for line in source:
                words = line.split('\t')[0].split()
                tags = ['B-NUM'] + ['I-NUM'] * (len(words) - 1)
                target.write('{}\t{}\n'.format(' '.join(words), ' '.join(tags)))
        fields = OrderedDict([('x', 'seq'), ('tag', 'tags')])
        vocab.FieldMap.train(str(tmp_path), fields, str(corpus), 100, {})
        saved = tmp_path / 'map_tag.txt'
        assert saved.is_file()
        restored = vocab.IDMap().load(str(saved))
        assert 'B-NUM' in restored
        assert 'I-NUM' in restored

    def test_train_builds_a_map_for_a_label_column(self, workdir, tmp_path):
        shutil.copy(workdir / 'sp.model', tmp_path / 'sp.model')
        corpus = tmp_path / 'labelled.tsv'
        with open(CORPUS, encoding='utf-8') as source, \
             open(corpus, 'w', encoding='utf-8', newline='\n') as target:
            for i, line in enumerate(source):
                text = line.split('\t')[0]
                target.write('{}\t{}\n'.format(text, 'odd' if i % 2 else 'even'))
        fields = OrderedDict([('x', 'seq'), ('label', 'label')])
        vocab.FieldMap.train(str(tmp_path), fields, str(corpus), 100, {})
        restored = vocab.LabelMap().load(str(tmp_path / 'map_label.txt'))
        assert sorted(restored) == ['even', 'odd']
        # LabelMap は確率分布へ変換できること
        assert sum(restored.str2dist('odd')) == pytest.approx(1.0)
