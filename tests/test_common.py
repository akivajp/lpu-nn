'''Tests for lpu_nn.common

lpu_nn.common のテスト。
'''

import pytest

from lpu_nn.common import args
from lpu_nn.common.files import PastedFile


class TestStrToBool:
    @pytest.mark.parametrize('text', ['y', 'Yes', 'T', 'true', 'ON', '1'])
    def test_truthy_spellings(self, text):
        assert args.strtobool(text) is True

    @pytest.mark.parametrize('text', ['n', 'No', 'F', 'false', 'OFF', '0'])
    def test_falsy_spellings(self, text):
        assert args.strtobool(text) is False

    def test_surrounding_whitespace_is_ignored(self):
        assert args.strtobool('  yes  ') is True

    def test_an_unknown_spelling_is_rejected(self):
        with pytest.raises(ValueError):
            args.strtobool('maybe')


class TestPastedFile:
    @pytest.fixture
    def parallel(self, tmp_path):
        '''Two files of three lines each, plus a shorter third one

        3 行ずつの 2 ファイルと、それより短い 3 つ目のファイル。
        '''
        left = tmp_path / 'left.txt'
        left.write_text('a1\na2\na3\n', encoding='utf-8')
        right = tmp_path / 'right.txt'
        right.write_text('b1\nb2\nb3\n', encoding='utf-8')
        short = tmp_path / 'short.txt'
        short.write_text('c1\n', encoding='utf-8')
        return str(left), str(right), str(short)

    def test_joins_lines_with_a_tab_in_binary_mode(self, parallel):
        left, right, _short = parallel
        with PastedFile([left, right]) as fobj:
            assert fobj.readline() == b'a1\tb1\n'
            assert fobj.readline() == b'a2\tb2\n'

    def test_yields_str_in_text_mode(self, parallel):
        left, right, _short = parallel
        with PastedFile([left, right], 'rt') as fobj:
            assert fobj.readline() == 'a1\tb1\n'

    def test_honors_a_custom_separator(self, parallel):
        # 元実装は sep を保持しながらタブ固定で結合していた
        left, right, _short = parallel
        with PastedFile([left, right], 'rt', sep='|') as fobj:
            assert fobj.readline() == 'a1|b1\n'

    def test_accepts_a_single_path(self, parallel):
        left, _right, _short = parallel
        with PastedFile(left, 'rt') as fobj:
            assert fobj.readline() == 'a1\n'

    def test_stops_at_the_shortest_input_by_default(self, parallel):
        left, right, short = parallel
        with PastedFile([left, right, short], 'rt') as fobj:
            assert fobj.readline() == 'a1\tb1\tc1\n'
            # 3 つ目が尽きた時点で終端。型はモードに従うこと
            assert fobj.readline() == ''

    def test_continues_to_the_longest_input_when_asked(self, parallel):
        left, right, short = parallel
        with PastedFile([left, right, short], 'rt', longest=True) as fobj:
            fobj.readline()
            assert fobj.readline() == 'a2\tb2\t\n'

    def test_binary_mode_ends_with_empty_bytes(self, parallel):
        _left, _right, short = parallel
        with PastedFile([short]) as fobj:
            assert fobj.readline() == b'c1\n'
            assert fobj.readline() == b''

    def test_read_returns_one_line(self, parallel):
        left, right, _short = parallel
        with PastedFile([left, right], 'rt') as fobj:
            assert fobj.read(4096) == 'a1\tb1\n'

    def test_close_closes_every_underlying_file(self, parallel):
        left, right, _short = parallel
        fobj = PastedFile([left, right])
        handles = list(fobj.fobjs)
        fobj.close()
        assert all(handle.closed for handle in handles)
