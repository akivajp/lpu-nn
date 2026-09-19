#!/usr/bin/env python

# Submodules, plus the names re-exported below for convenience.
# `match_ranker` and `re2` were listed here but are not ported yet, so
# `from lpu_nn.modeling import *` used to fail.
# (サブモジュールと、下部で再エクスポートしている名前。`match_ranker` と
#  `re2` は未移植のまま列挙されており、ワイルドカード import が失敗していた)
__all__ = [
    'Linear',
    'Module',
    'ModuleArray',
    'ModuleList',
    'SequenceConvolution1d',
    'activation',
    'attention',
    'convolution',
    'embeddings',
    'encoder_decoder',
    'linear',
    'lstm',
    'module',
    'transformer',
    'universal_transformer',
]

from lpu.common import logging
logger = logging.getColorLogger(__name__)

from .module import Module
from .module import ModuleArray
from .module import ModuleList
from .linear import Linear
from .convolution import SequenceConvolution1d
