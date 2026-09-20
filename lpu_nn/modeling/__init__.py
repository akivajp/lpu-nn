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
    'apply_to',
    'attention',
    'bert',
    'bert_classifier',
    'bert_ranker',
    'compare_aggregate',
    'convolution',
    'embeddings',
    'encoder_decoder',
    'format_module',
    'linear',
    'lstm',
    'module',
    'pooling',
    're2',
    'sequence_matcher',
    'sequence_tagger',
    'transformer',
    'universal_transformer',
]

from lpu.common import logging
logger = logging.getColorLogger(__name__)

from .module import Module
from .module import apply_to
from .module import format_module
from .module import ModuleArray
from .module import ModuleList
from .linear import Linear
from .convolution import SequenceConvolution1d
