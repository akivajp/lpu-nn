#!/usr/bin/env python3

'''Command line argument helpers

コマンドライン引数の補助関数。
'''


# The truthy / falsy spellings accepted by `strtobool`, matching the set that
# `distutils.util.strtobool` accepted before it was removed in Python 3.12.
# (`strtobool` が受け付ける真偽の綴り。Python 3.12 で削除された
#  `distutils.util.strtobool` と同じ集合に合わせている)
_TRUE_WORDS = frozenset({'y', 'yes', 't', 'true', 'on', '1'})
_FALSE_WORDS = frozenset({'n', 'no', 'f', 'false', 'off', '0'})


def strtobool(value: str) -> bool:
    '''Convert a truth string to a bool

    真偽を表す文字列を bool に変換する。

    This replaces `distutils.util.strtobool`, which was removed together with
    `distutils` in Python 3.12. It accepts the same spellings, but returns a
    real `bool` instead of the `int` the original returned.

    Python 3.12 で `distutils` ごと削除された `distutils.util.strtobool` の
    置き換え。受け付ける綴りは同じだが、元が返していた `int` ではなく
    本来の `bool` を返す。

    Args:
        value: The string to convert, e.g. 'yes' or 'off'.
            変換する文字列。例: 'yes', 'off'。

    Returns:
        The boolean the string denotes.
            文字列が表す真偽値。

    Raises:
        ValueError: If the string is not a recognized truth value.
            真偽値として解釈できない文字列の場合。

    Examples:
        >>> strtobool('Yes'), strtobool('0')
        (True, False)
    '''
    normalized = value.strip().lower()
    if normalized in _TRUE_WORDS:
        return True
    if normalized in _FALSE_WORDS:
        return False
    raise ValueError(f"invalid truth value: {value!r}")
