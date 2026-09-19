#!/usr/bin/env python3

"""
    Weight initialization functions
"""

# 3rd
from torch import nn

# local
from lpu.common import logging
from lpu_nn.modeling.convolution import SequenceConvolution1d
from lpu_nn.modeling.linear import Linear
from lpu_nn.modeling.embeddings import Embedding

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

def init_weights(m, name=None):
    """
    :param nn.Module m: any module
    :return: None
    """
    #dprint(m)
    # optimal initialization depends on activation functions
    ignore_list = [
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
    ]
    if hasattr(m, 'init_weights'):
        if not isinstance(m, (Embedding,Linear,SequenceConvolution1d)):
            logger.debug(f"initializing {m.__class__.__name__} weights")
        try:
            m.init_weights(name=name)
        except:
            m.init_weights()
    elif isinstance(m, nn.Embedding):
        logger.debug("initializing Embedding weight orthogonally")
        nn.init.orthogonal_(m.weight)
    elif isinstance(m, nn.LSTM):
        #dprint("initializing LSTM")
        init_lstm_weights(m)
    elif isinstance(m, tuple(ignore_list)):
        # nothing to do (primitive module should be initialized with it's parent module)
        pass
    else:
        if hasattr(m, 'weight'):
            if m.weight.dim() > 1:
                logger.debug(f"initializing unknown ({m.__class__.__name__}) weight orthogonally")
                nn.init.orthogonal_(m.weight)
                #nn.init.xavier_uniform_(m.weight)
        pass

def init_lstm_weights(m):
    for key, param in m.named_parameters():
        if key.find('weight_ih') == 0:
            hsize = param.shape[0] // 4
            logger.debug(f"initializing LSTM {key} orthogonally with gain 2 (for sigmoid)")
            for i in range(4):
                nn.init.orthogonal_(param[hsize*i:hsize*(i+1)], gain=2.0)
        elif key.find('weight_hh') == 0:
            hsize = param.shape[0] // 4
            logger.debug(f"initializing LSTM {key} orthogonally with gain 5/3 (for tanh)")
            for i in range(4):
                nn.init.orthogonal_(param[hsize*i:hsize*(i+1)], gain=5.0/3)
        else:
            param.data.fill_(0)

def apply_init_weights(top):
    logger.info("initializing module parameters:")
    for name, module in top.named_modules():
        #logger.info("  module: {}, {}".format(name, module.__class__.__name__))
        depth = name.count('.') + 1
        indent = '  ' * depth
        base = name.split('.')[-1]
        if not name:
            logger.info(f"  * {module.__class__.__name__}({module.extra_repr()})")
        else:
            logger.info(f"  {indent} * {base}: {module.__class__.__name__}({module.extra_repr()})")
        init_weights(module, name)
