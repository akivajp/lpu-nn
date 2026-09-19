#!/usr/bin/env python3

'''Neural language processing models on PyTorch, built on LPU

LPU の上に構築された、PyTorch によるニューラル言語処理モデル。
'''

__all__ = [
    'commands',
    'common',
    'modeling',
    'optimizers',
]

import os.path

from lpu.common import logging

version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
# encoding を明示し、ハンドルを確実にクローズする
with open(version_file, encoding='utf-8') as _version_fp:
    __version__ = _version_fp.read().strip()

logger = logging.getColorLogger(__name__)

if logging.get_quiet_status():
    logger.setLevel(logging.ERROR)
elif logging.get_debug_status():
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
