'''Tests for lpu_nn.common.dataset

lpu_nn.common.dataset のテスト。

`Dataset` はファイル全体をメモリに読み込まず、各行の先頭バイト位置だけを
索引として保持し、参照のたびに seek して読み出す。テストはその索引が
正しく機能すること (ランダムアクセス、スライス、逆順) を中心に確認する。
'''

import io

import pandas as pd
import pytest

from lpu_nn.common import dataset


@pytest.fixture
def tsv_path(tmp_path):
    '''A four-row TSV with an index column, as the trainer writes it

    訓練側が書き出す形式に倣った、index 列付きの 4 行の TSV。
    '''
    path = tmp_path / 'data.tsv'
    path.write_text(
        'index\ta\tb\n'
        '0\tx0\ty0\n'
        '1\tx1\ty1\n'
        '2\tx2\ty2\n'
        '3\tx3\ty3\n',
        encoding='utf-8',
    )
    return str(path)


class TestTupleHelpers:
    def test_min_tuple_takes_the_smaller_of_each_position(self):
        assert dataset.min_tuple((1, 5), (3, 2)) == (1, 2)

    def test_max_tuple_takes_the_larger_of_each_position(self):
        assert dataset.max_tuple((1, 5), (3, 2)) == (3, 5)

    def test_inter_tuple_takes_the_midpoint_of_each_position(self):
        assert dataset.inter_tuple((0, 10), (2, 20)) == (1.0, 15.0)

    def test_the_shorter_tuple_decides_the_length(self):
        # zip の既定どおり、短い方に合わせて打ち切ること
        assert dataset.min_tuple((1, 2, 3), (0,)) == (0,)


class TestFieldHelpers:
    def test_get_indices_maps_names_to_positions(self):
        assert dataset.get_indices(['index', 'a', 'b'], ['b', 'index']) == (2, 0)

    def test_get_indices_rejects_an_unknown_name(self):
        with pytest.raises(ValueError):
            dataset.get_indices(['a'], ['missing'])

    def test_to_float_parses_a_number(self):
        assert dataset.to_float('1.5') == 1.5

    def test_to_float_falls_back_for_a_non_number(self):
        assert dataset.to_float('abc') == -1
        assert dataset.to_float('abc', default=0.0) == 0.0

    def test_get_values_reads_the_named_columns(self):
        assert dataset.get_values(['1.5', 'x', '2.5'], (0, 2)) == (1.5, 2.5)

    def test_get_values_propagates_a_missing_column(self):
        # 以前は握り潰して暗黙の None を返し、呼び出し側の比較が原因から
        # 離れた場所で TypeError になっていた
        with pytest.raises(IndexError):
            dataset.get_values(['1.5'], (0, 5))


class TestDataset:
    def test_length_excludes_the_header(self, tsv_path):
        assert len(dataset.Dataset(tsv_path)) == 4

    def test_random_access_by_index(self, tsv_path):
        data = dataset.Dataset(tsv_path)
        assert data[0] == '0\tx0\ty0'
        assert data[3] == '3\tx3\ty3'

    def test_random_access_does_not_depend_on_the_order_of_reads(self, tsv_path):
        # 索引した位置へ seek するだけなので、読み出し順に影響されないこと
        data = dataset.Dataset(tsv_path)
        assert data[2] == '2\tx2\ty2'
        assert data[0] == '0\tx0\ty0'
        assert data[2] == '2\tx2\ty2'

    def test_iteration_yields_the_header_first(self, tsv_path):
        rows = list(dataset.Dataset(tsv_path))
        assert rows[0] == 'index\ta\tb'
        assert len(rows) == 5

    def test_iteration_without_the_header(self, tsv_path):
        rows = list(dataset.Dataset(tsv_path).iter(headers=False))
        assert rows == ['0\tx0\ty0', '1\tx1\ty1', '2\tx2\ty2', '3\tx3\ty3']

    def test_slicing_returns_the_requested_range(self, tsv_path):
        assert dataset.Dataset(tsv_path)[1:3] == ['1\tx1\ty1', '2\tx2\ty2']

    def test_a_negative_start_counts_from_the_end(self, tsv_path):
        # `len(self) - start` と書かれていたため負号が打ち消され、
        # 負の start では何も返らなかった
        rows = list(dataset.Dataset(tsv_path).iter(-2, None, None, headers=False))
        assert rows == ['2\tx2\ty2', '3\tx3\ty3']

    def test_a_negative_stop_counts_from_the_end(self, tsv_path):
        rows = list(dataset.Dataset(tsv_path).iter(None, -1, None, headers=False))
        assert rows == ['0\tx0\ty0', '1\tx1\ty1', '2\tx2\ty2']

    def test_a_negative_step_walks_backwards(self, tsv_path):
        rows = list(dataset.Dataset(tsv_path).iter(None, None, -1, headers=False))
        assert rows == ['3\tx3\ty3', '2\tx2\ty2', '1\tx1\ty1', '0\tx0\ty0']

    def test_a_step_greater_than_one_skips_rows(self, tsv_path):
        rows = list(dataset.Dataset(tsv_path).iter(0, None, 2, headers=False))
        assert rows == ['0\tx0\ty0', '2\tx2\ty2']

    def test_binary_mode_yields_bytes(self, tsv_path):
        data = dataset.Dataset(tsv_path)
        assert data.getline(1, binmode=True) == b'1\tx1\ty1'

    def test_getbuffer_round_trips_through_a_text_buffer(self, tsv_path):
        buffer = dataset.Dataset(tsv_path).getbuffer()
        assert buffer.readline() == 'index\ta\tb\n'
        assert buffer.readline() == '0\tx0\ty0\n'

    def test_save_writes_a_readable_subset(self, tsv_path, tmp_path):
        out = tmp_path / 'subset.tsv'
        dataset.Dataset(tsv_path).save(str(out), 1, 3)
        written = out.read_text(encoding='utf-8').splitlines()
        assert written == ['index\ta\tb', '1\tx1\ty1', '2\tx2\ty2']

    def test_to_df_reads_the_index_column(self, tsv_path):
        frame = dataset.Dataset(tsv_path).to_df({'a': 'seq', 'b': 'seq'})
        assert isinstance(frame, pd.DataFrame)
        assert list(frame.index) == [0, 1, 2, 3]
        assert frame.loc[2, 'a'] == 'x2'

    def test_non_ascii_content_survives_the_round_trip(self, tmp_path):
        path = tmp_path / 'ja.tsv'
        path.write_text('index\ta\n0\t日本語\n', encoding='utf-8')
        assert dataset.Dataset(str(path))[0] == '0\t日本語'

    def test_priority_keys_reorder_the_index(self, tmp_path):
        # 優先度付きの読み込みでも、全行が失われずに読み出せること
        path = tmp_path / 'p.tsv'
        path.write_text(
            'index\tlen\n0\t30\n1\t10\n2\t20\n3\t40\n', encoding='utf-8')
        data = dataset.Dataset(str(path), priority_keys='len')
        assert len(data) == 4
        assert sorted(data.iter(headers=False)) == [
            '0\t30', '1\t10', '2\t20', '3\t40']


class TestBuildTrainData:
    def test_writes_a_row_per_input_line_with_the_bookkeeping_columns(self, tmp_path):
        source = tmp_path / 'train.tsv'
        source.write_text('a b\tc d\ne f\tg h\n', encoding='utf-8')
        out = tmp_path / 'built.tsv'
        dataset.build_train_data({'x': 'seq', 't': 'seq'}, str(out), str(source))
        frame = pd.read_csv(out, sep='\t', index_col='index')
        assert len(frame) == 2
        # 訓練側が使う記録用の列が揃っていること
        for column in ['len_x', 'len_t', 'len', 'cost', 'last_epoch',
                       'last_step', 'feed_count', 'criterion', 'priority']:
            assert column in frame.columns
        assert frame.loc[0, 'x'] == 'a b'
        assert frame.loc[0, 'len'] == len('a b') + len('c d')

    def test_max_length_skips_the_long_samples(self, tmp_path):
        source = tmp_path / 'train.tsv'
        source.write_text('short\tshort\n' + 'x' * 50 + '\tshort\n', encoding='utf-8')
        out = tmp_path / 'built.tsv'
        dataset.build_train_data(
            {'x': 'seq', 't': 'seq'}, str(out), str(source), max_length=10)
        assert len(pd.read_csv(out, sep='\t', index_col='index')) == 1


class TestLoadEvalData:
    def test_reads_a_path(self, tmp_path):
        path = tmp_path / 'dev.tsv'
        path.write_text('a b\tc d\n', encoding='utf-8')
        frame = dataset.load_eval_data({'x': 'seq', 't': 'seq'}, str(path))
        assert frame.loc[0, 'x'] == 'a b'
        assert frame.loc[0, 'len'] == len('a b') + len('c d')

    def test_reads_a_buffer(self):
        frame = dataset.load_eval_data(
            {'x': 'seq', 't': 'seq'}, io.StringIO('a b\tc d\n'))
        assert len(frame) == 1

    def test_the_criterion_column_accepts_float_scores(self):
        # int64 で初期化されていたため、pandas 3.0 では float の代入が
        # 拒否され評価値が静かに捨てられていた
        frame = dataset.load_eval_data(
            {'x': 'seq', 't': 'seq'}, io.StringIO('a\tb\nc\td\n'))
        frame.loc[[0, 1], 'criterion'] = [1.5, 2.5]
        assert frame['criterion'].tolist() == [1.5, 2.5]


class TestMergeTsvFiles:
    def test_keeps_one_header_and_every_body_row(self, tmp_path):
        first = tmp_path / 'a.tsv'
        first.write_text('index\tv\n0\ta\n', encoding='utf-8')
        second = tmp_path / 'b.tsv'
        second.write_text('index\tv\n1\tb\n', encoding='utf-8')
        out = tmp_path / 'merged.tsv'
        dataset.merge_tsv_files([str(first), str(second)], str(out))
        assert out.read_text(encoding='utf-8').splitlines() == [
            'index\tv', '0\ta', '1\tb']

    def test_accepts_a_single_path_string(self, tmp_path):
        source = tmp_path / 'a.tsv'
        source.write_text('index\tv\n0\ta\n', encoding='utf-8')
        out = tmp_path / 'merged.tsv'
        dataset.merge_tsv_files(str(source), str(out))
        assert out.read_text(encoding='utf-8').splitlines() == ['index\tv', '0\ta']

    def test_expands_a_glob(self, tmp_path):
        for name, row in [('p1.tsv', '0\ta'), ('p2.tsv', '1\tb')]:
            (tmp_path / name).write_text(f'index\tv\n{row}\n', encoding='utf-8')
        out = tmp_path / 'merged.tsv'
        dataset.merge_tsv_files(str(tmp_path / 'p*.tsv'), str(out))
        lines = out.read_text(encoding='utf-8').splitlines()
        assert lines[0] == 'index\tv'
        assert sorted(lines[1:]) == ['0\ta', '1\tb']


class TestDatasetEdgeCases:
    def test_an_empty_file_is_reported(self, tmp_path):
        # read_byte_line() は終端で None を返すため、以前はヘッダ行の
        # strip() が AttributeError になっていた
        path = tmp_path / 'empty.tsv'
        path.write_text('', encoding='utf-8')
        with pytest.raises(ValueError, match='header'):
            dataset.Dataset(str(path))

    def test_a_header_only_file_has_no_rows(self, tmp_path):
        path = tmp_path / 'header.tsv'
        path.write_text('index\ta\n', encoding='utf-8')
        data = dataset.Dataset(str(path))
        assert len(data) == 0
        assert list(data.iter(headers=False)) == []

    def test_blank_lines_are_not_indexed(self, tmp_path):
        path = tmp_path / 'gaps.tsv'
        path.write_text('index\ta\n0\tx\n\n1\ty\n', encoding='utf-8')
        data = dataset.Dataset(str(path))
        assert len(data) == 2
        assert list(data.iter(headers=False)) == ['0\tx', '1\ty']
