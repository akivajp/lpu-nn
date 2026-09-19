#!/usr/bin/env python
# -*- coding: utf-8 -*-

# system
import copy
import torch

# 3rd
from torch import nn
from torch.nn.modules.module import _addindent

# local
from lpu.common import logging
logger = logging.getColorLogger(__name__)

class Module(nn.Module):
    def __init__(self):
        super(Module,self).__init__()
        self.device = torch.device('cpu')
        self.dtype = torch.float32

    def dropout(self, x, dropout_ratio=None):
        if dropout_ratio is None:
            if hasattr(self, 'dropout_ratio'):
                dropout_ratio = self.dropout_ratio
            if dropout_ratio is None:
                dropout_ratio = 0.1
        return torch.dropout(x, dropout_ratio, self.training)

    def to(self, *args, **kwargs):
        # `to()` の多重定義された引数を解釈する公開 API が無いため、torch の
        # 私的関数を用いる。返す要素数は torch のバージョンで増えており
        # (1.5 で memory_format が加わり 3 -> 4)、固定個数での展開は壊れるため
        # 先頭 3 つだけを取り出す
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)[:3]
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        if non_blocking is not None:
            self.non_blocking = non_blocking
        #super(Module,self).to(*args, **kwargs)
        nn.Module.to(self, *args, **kwargs)
        for i, module in enumerate(self.modules()):
            if i > 0:
                module.to(*args, **kwargs)
        return self

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        last_mod_str = {}
        for key, module in self._modules.items():
            mod_name = module._get_name()
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            if mod_name not in last_mod_str:
                last_mod_str[mod_name] = mod_str
            elif last_mod_str[mod_name] == mod_str:
                mod_str = module._get_name() + '(...)'
            else:
                last_mod_str[mod_name] = mod_str
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

class ModuleList(nn.ModuleList):
    def __repr__(self):
        return Module.__repr__(self)
    def to(self, *args, **kwargs):
        return Module.to(self, *args, **kwargs)

class ModuleArray(nn.ModuleList):
    def __init__(self, module, count):
        super(ModuleArray,self).__init__()
        self.count = count
        for _ in range(count):
            self.append(copy.deepcopy(module))

    def __repr__(self):
        name = self.__class__.__name__
        first = _addindent(repr(self[0]), 2)
        return "{}(\n  {} x [{}]\n)".format(name, self.count, first)

    def to(self, *args, **kwargs):
        return Module.to(self, *args, **kwargs)
