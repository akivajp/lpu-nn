#!/usr/bin/env python

# The optimizers the trainer selects by name, re-exported from this package.
# (訓練側が名前で選択する最適化器。本パッケージから再エクスポートする)
__all__ = [
    'SGD',
    'AdaBound',
    'AdaBoundW',
    'Adamax',
    'Lamb',
    'adabound',
    'lamb',
]

from lpu.common import logging
logger = logging.getColorLogger(__name__)
logger.debug(f"initialized {__name__} logger")

from . adabound import AdaBound
from . adabound import AdaBoundW
from . lamb import Lamb
from torch.optim import Adamax
#from torch.optim import AdamW
from torch.optim import SGD
