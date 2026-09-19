#!/usr/bin/env python3

# system
from collections.abc import Sequence
from typing import Any

# 3rd
import torch
from torch import nn

# local
from lpu.common import logging

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

def copy_model(src: nn.Module, dst: nn.Module, depth: int = 0) -> None:
    """
    :param nn.Module src: source module
    :param nn.Module dst: target module
    :param int depth: depth for indentation (for debug use)
    """
    '  ' * depth
    assert isinstance(src, nn.Module)
    assert isinstance(dst, nn.Module)
    param_dict = {}
    for name, param in src.named_parameters():
        param_dict[name] = param
    for name, param in dst.named_parameters():
        if name in param_dict:
            logger.debug(f"copying parameter: {name}")
            param.data = param_dict[name].data

def flip(t: torch.Tensor, dims: "int | Sequence[int]") -> torch.Tensor:
    """Reverse a tensor along the given dimension(s)

    指定した次元に沿ってテンソルを反転する。

    This used to build a list of index tensors and apply it as `t[slices]`,
    working around `torch.flip` not supporting bool tensors on the CPU in
    torch 1.3. That restriction is long gone, and indexing with a non-tuple
    sequence is deprecated: with more than one dimension it is already
    interpreted as advanced indexing and raises an IndexError.

    以前は添字テンソルのリストを組み立てて `t[slices]` として適用していた。
    torch 1.3 の `torch.flip` が CPU 上の bool テンソルを扱えなかったことへの
    回避策だが、その制限は既に無く、非タプル列による添字指定は非推奨である。
    2 次元以上を指定すると既に高度な添字指定として解釈され IndexError になる。
    """
    if isinstance(dims, int):
        dims = [dims]
    return torch.flip(t, list(dims))

def purge_tensor(tensor: torch.Tensor, mask: torch.Tensor, else_value: float = 0.0) -> torch.Tensor:
    """
    :param torch.Tensor tensor: tensor
    :param torch.Tensor mask: mask tensor
    :param float else_value:
    :rtype: torch.Tensor
    """
    device = tensor.device
    dtype  = tensor.dtype
    #else_tensor = torch.tensor(float(else_value), device=device, dtype=dtype)
    #if isinstance(else_value, int):
    #    else_tensor = torch.tensor(else_value, device=device)
    #else:
    #    else_tensor = torch.tensor(float(else_value), device=device, dtype=dtype)
    else_tensor = torch.tensor(else_value, device=device, dtype=dtype)
    if tensor.dim() == mask.dim():
        return torch.where(mask, tensor, else_tensor)
    elif tensor.shape[:2] == mask.shape:
        return torch.where(mask[:,:,None], tensor, else_tensor)
    else:
        raise ValueError(f"tensor.shape: {tensor.shape}, mask.shape: {mask.shape}")

def make_attention_mask(query_id_seq: torch.Tensor, key_id_seq: torch.Tensor, padding: int = 0) -> torch.Tensor:
    """
    :param torch.Tensor query_id_seq: tensor (B, LenQ)
    :param torch.Tensor key_id_seq: tensor (B, LenK)
    :param int padding:
    :rtype: torch.Tensor
    :return: mask tensor (B, LenQ, LenK)
    """
    mask_query = (query_id_seq != padding)[:, :, None] # (B, LenQ, 1)
    mask_key   = (key_id_seq   != padding)[:, None, :] # (B, 1, LenK)
    return mask_query * mask_key # (B, LenQ, LenK)

def make_history_mask(id_seq: torch.Tensor) -> torch.Tensor:
    """
    :param torch.Tensor id_seq: tensor (B, Len)
    :rtype: torch.Tensor
    :return: mask tensor (B, Len, Len)
    """
    device = id_seq.device
    batch_size, seq_len = id_seq.shape
    #return torch.ones(batch_size, seq_len, seq_len).to(torch.uint8).tril().to(device)
    ones = torch.ones(batch_size, seq_len, seq_len, dtype=torch.uint8) # (B, Len, Len)
    mask = ones.tril() > 0
    return mask.to(device)

def parameterize(val: "float | list[float] | tuple[float, ...] | torch.Tensor") -> nn.Parameter:
    """
    :param float or torch.Tensor val:
    :rtype: nn.Parameter
    """
    if isinstance(val, torch.Tensor):
        return nn.Parameter(val)
    elif isinstance(val, (tuple, list)):
        return nn.Parameter(torch.tensor(val).to(torch.float))
    else:
        return nn.Parameter(torch.tensor(float(val)))

def format_state(state: Any, indent: int = 0) -> str:
    #if indent > 50:
    #    return '?'
    str_indent = "  " * indent
    if state is None:
        return str_indent + "None"
    elif isinstance(state, torch.Tensor):
        str_shape = str.join(', ', map(str, state.shape))
        return str_indent + f"Tensor[{str_shape}]"
    elif isinstance(state, (list,tuple)):
        if isinstance(state, list):
            s = str_indent + '[\n'
        else:
            s = str_indent + '(\n'
        for elem in state:
            str_elem = format_state(elem, indent+1)
            s = s + f"{str_elem},\n"
        if isinstance(state, list):
            s = s + str_indent + ']'
        else:
            s = s + str_indent + ')'
        return s
    elif isinstance(state, dict):
        s = str_indent + '{\n'
        for key, val in state.items():
            s = s + ("  " * (indent+1))+ f"{key}:\n"
            str_val = format_state(val, indent+2)
            s = s + f"{str_val}\n"
        s = s + str_indent + '}'
        return s
    else:
        raise TypeError(f"unsupported type: {type(state).__name__}")

def format_state_list(state_list: "Sequence[Any] | None") -> str:
    if state_list is None:
        return "None"
    elif len(state_list) == 0:
        return "[]"
    s = "[\n"
    s += format_state(state_list[0], 1) + "\n"
    s += f"] * {len(state_list)}"
    return s

def get_sample_state(state: Any, i: int) -> Any:
    if isinstance(state, torch.Tensor):
        if state.shape[0] == 1:
            return state[0]
        else:
            return state[i]
    elif isinstance(state, (list, tuple)):
        return type(state)(get_sample_state(elem, i) for elem in state)
    elif isinstance(state, dict):
        return {key: get_sample_state(val, i) for key, val in state.items()}
    elif state is None:
        return None
    else:
        raise TypeError(f"unsupported type: {type(state).__name__}")

def split_state(state: Any, batch_size: int) -> list[Any]:
    return [get_sample_state(state, i) for i in range(batch_size)]

def stack_state_list(state_list: "Sequence[Any] | None") -> Any:
    if state_list is None:
        return None
    state_list = list(state_list)
    if len(state_list) == 0:
        return []
    first = state_list[0]
    if first is None:
        return None
    elif isinstance(first, torch.Tensor):
        return torch.stack(state_list, dim=0)
    elif isinstance(first, (list,tuple)):
        indices = range(len(first))
        return type(first)(stack_state_list([state[i] for state in state_list]) for i in indices)
    elif isinstance(first, dict):
        keys = first.keys()
        return {key: stack_state_list([state[key] for state in state_list]) for key in keys}
    else:
        raise TypeError(f"unsupported type: {type(first).__name__}")

