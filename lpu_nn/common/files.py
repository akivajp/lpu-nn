#!/usr/bin/env python3

'''File helpers specific to this package

本パッケージ固有のファイル補助。
汎用的なファイル操作は lpu.common.files を参照すること。
'''

import io


class PastedFile(io.IOBase):
    '''A read-only file object joining parallel files line by line

    並列に並んだ複数ファイルを 1 行ずつ結合して読み出す、読み取り専用の
    ファイルオブジェクト。Unix の `paste` コマンドに相当する。

    This is how a parallel corpus kept as one file per language is fed to a
    trainer that expects a single TSV stream, without materializing the
    joined file on disk.

    言語ごとに別ファイルへ分かれた対訳コーパスを、結合済みファイルを
    ディスク上に作ることなく、単一の TSV ストリームを期待する訓練側へ
    渡すために用いる。
    '''

    def __init__(
        self,
        paths: 'str | list[str]',
        mode: str = 'rb',
        sep: str = '\t',
        longest: bool = False,
    ) -> None:
        '''Open every given file for reading

        指定された全ファイルを読み取り用に開く。

        Args:
            paths: The paths of the files to join; a single path is accepted.
                結合するファイルのパス。単一のパスも受け付ける。
            mode: 'rb' yields bytes, a mode containing 't' yields str.
                'rb' は bytes を、't' を含むモードは str を返す。
            sep: The separator inserted between the joined fields.
                結合したフィールドの間に挿入する区切り文字。
            longest: Stop at the longest input instead of the shortest.
                最短ではなく最長の入力に合わせて読み終えるかどうか。
        '''
        if isinstance(paths, str):
            paths = [paths]
        self.paths = paths
        self.mode = mode
        self.sep = sep
        self.longest = longest
        self.fobjs = [open(path, 'rb') for path in paths]

    @property
    def _text_mode(self) -> bool:
        '''Whether this object yields str rather than bytes

        bytes ではなく str を返すモードかどうか。
        '''
        return 't' in self.mode

    def read(self, size: 'int | None' = None) -> 'bytes | str':
        '''Read one joined line, ignoring the requested size

        結合した 1 行を読み出す。要求サイズは無視する。

        A joined line cannot be split across reads without buffering the
        remainder, and the callers of this class consume it line by line, so
        a read returns exactly one line.

        結合後の行は、残りを保持しない限り読み出しを跨いで分割できず、
        本クラスの利用側は行単位で消費するため、1 回の読み出しは
        ちょうど 1 行を返す。
        '''
        return self.readline()

    def readline(self, size: 'int | None' = None) -> 'bytes | str':  # type: ignore[override]
        '''Read one line from every file and join them

        各ファイルから 1 行ずつ読み出して結合する。

        Returns:
            The joined line, or an empty bytes / str at the end of input.
                結合した行。入力の終端では空の bytes / str。
        '''
        lines = [fobj.readline() for fobj in self.fobjs]
        # longest=False なら、どれか 1 つでも尽きた時点で終端とする
        check = any if self.longest else all
        if not check(lines):
            return '' if self._text_mode else b''
        separator = self.sep.encode('utf-8')
        joined = bytes.join(separator, [line.strip() for line in lines]) + b'\n'
        if self._text_mode:
            return joined.decode('utf-8', 'backslashreplace')
        return joined

    def readable(self) -> bool:
        return True

    def close(self) -> None:
        '''Close every underlying file

        開いている全ファイルを閉じる。
        '''
        for fobj in self.fobjs:
            fobj.close()
        super().close()
