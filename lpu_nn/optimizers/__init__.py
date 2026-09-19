#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    'adabound',
]

from lpu.common import logging
logger = logging.getColorLogger(__name__)
logger.debug("initialized {} logger".format(__name__))

from . adabound import AdaBound
from . adabound import AdaBoundW
from . lamb import Lamb
from torch.optim import Adamax
#from torch.optim import AdamW
from torch.optim import SGD
