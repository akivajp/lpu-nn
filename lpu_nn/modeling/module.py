#!/usr/bin/env python

# system
import copy
from typing import Any, TypeVar

import torch

# 3rd
from torch import nn
from torch.nn.modules.module import _addindent

# local
from lpu.common import logging
logger = logging.getColorLogger(__name__)

# `to()` and `__repr__` are shared by classes that do not all derive from
# `Module`: `ModuleList` and `ModuleArray` extend `nn.ModuleList`, and
# `Linear` / `SequenceConvolution1d` extend their torch counterparts. They
# used to borrow the unbound methods (`Module.to(self, ...)`), which works
# only because Python does not check the receiver's type. Plain functions
# say the same thing honestly.
# (`to()` と `__repr__` は、必ずしも `Module` を継承しないクラス間で共有
#  される。`ModuleList` / `ModuleArray` は `nn.ModuleList` を、`Linear` /
#  `SequenceConvolution1d` は torch の対応クラスを継承している。以前は
#  未束縛メソッドを借用していたが (`Module.to(self, ...)`)、これは Python
#  が受け手の型を検査しないから動いていただけである。通常の関数として
#  切り出せば、同じことを正直に表現できる)
_ModuleT = TypeVar('_ModuleT', bound=nn.Module)


def apply_to(module: _ModuleT, *args: Any, **kwargs: Any) -> _ModuleT:
    """Move a module, recording the device and dtype on it

    モジュールを移動し、その配置先とデータ型をモジュール自身に記録する。
    """
    # `to()` の多重定義された引数を解釈する公開 API が無いため、torch の
    # 私的関数を用いる。返す要素数は torch のバージョンで増えており
    # (1.5 で memory_format が加わり 3 -> 4)、固定個数での展開は壊れるため
    # 先頭 3 つだけを取り出す
    device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)[:3]
    # nn.Module.__setattr__ は Parameter / Module のみを想定した型を持つ
    # ため、通常の値の代入は型検査に引っかかる。setattr で回避すると今度は
    # ruff の B010 が直接代入へ書き戻すので、無視指示で明示する
    if device is not None:
        module.device = device  # type: ignore[assignment]
    if dtype is not None:
        module.dtype = dtype  # type: ignore[assignment]
    if non_blocking is not None:
        module.non_blocking = non_blocking  # type: ignore[assignment]
    nn.Module.to(module, *args, **kwargs)
    # nn.Module.to() は再帰的に適用するが、上の属性までは伝えないため、
    # 入れ子のモジュールにも改めて適用する
    for i, child in enumerate(module.modules()):
        if i > 0:
            child.to(*args, **kwargs)
    return module


def format_module(module: nn.Module) -> str:
    """Render a module, collapsing repeated children into one line

    モジュールを文字列化する。同じ内容の子モジュールが続く場合は
    1 行にまとめる。
    """
    # We treat the extra repr like the sub-module, one item per line
    extra_lines = []
    extra_repr: str = module.extra_repr()
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split('\n')
    child_lines = []
    last_mod_str: dict[str, str] = {}
    for key, child in module._modules.items():
        if child is None:
            # torch は削除済みの子を None として残すことがある
            continue
        mod_name = child._get_name()
        mod_str = repr(child)
        mod_str = _addindent(mod_str, 2)
        if mod_name not in last_mod_str:
            last_mod_str[mod_name] = mod_str
        elif last_mod_str[mod_name] == mod_str:
            mod_str = child._get_name() + '(...)'
        else:
            last_mod_str[mod_name] = mod_str
        child_lines.append('(' + key + '): ' + mod_str)
    lines = extra_lines + child_lines

    # _get_name() は torch の私的メソッドで型情報を持たない
    main_str = str(module._get_name()) + '('
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

    main_str += ')'
    return main_str


class Module(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device('cpu')
        self.dtype = torch.float32

    def dropout(self, x: torch.Tensor, dropout_ratio: "float | None" = None) -> torch.Tensor:
        if dropout_ratio is None:
            # 既定値は、モジュールが自分の dropout_ratio を持っていれば
            # それに従う。`or` で畳むと 0.0 (偽値) が既定値に化けるため、
            # None かどうかで分岐する
            ratio = getattr(self, 'dropout_ratio', None)
            dropout_ratio = 0.1 if ratio is None else float(ratio)
        return torch.dropout(x, dropout_ratio, self.training)

    def to(self, *args: Any, **kwargs: Any) -> "Module":
        return apply_to(self, *args, **kwargs)

    def __repr__(self) -> str:
        return format_module(self)

class ModuleList(nn.ModuleList):
    def __repr__(self) -> str:
        return format_module(self)

    def to(self, *args: Any, **kwargs: Any) -> "ModuleList":
        return apply_to(self, *args, **kwargs)

class ModuleArray(nn.ModuleList):
    def __init__(self, module: nn.Module, count: int) -> None:
        super().__init__()
        self.count = count
        for _ in range(count):
            self.append(copy.deepcopy(module))

    def __repr__(self) -> str:
        name = self.__class__.__name__
        first = _addindent(repr(self[0]), 2)
        return f"{name}(\n  {self.count} x [{first}]\n)"

    def to(self, *args: Any, **kwargs: Any) -> "ModuleArray":
        return apply_to(self, *args, **kwargs)
