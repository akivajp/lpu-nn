#!/usr/bin/env python3

# system
import unicodedata
from collections.abc import Mapping
from typing import Any, cast

# 3rd
import pandas as pd
import torch
from torch import nn

# local
from lpu.common import logging
from lpu_nn.common.vocab import dict_str2vector
from lpu_nn import modeling
from lpu_nn.modeling import lstm

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

def extract_vector(token: str, str2vector: Mapping[str, torch.Tensor]) -> "torch.Tensor | None":
    replaced = token.replace('▁', '')
    test_tokens = [
        token,
        unicodedata.normalize('NFKC', token),
        token.lower(),
        unicodedata.normalize('NFKC', token.lower()),
        replaced,
        unicodedata.normalize('NFKC', replaced),
        replaced.lower(),
        unicodedata.normalize('NFKC', replaced.lower()),
    ]
    for tok in test_tokens:
        if tok and tok in str2vector:
            return str2vector[tok]
    return None

#class Embedding(models.Module):
class Embedding(nn.Embedding):
    def __init__(self, num_ids: int, embed_size: int, padding: "int | None" = None,
                 idmap: Any = None, **kwargs: Any) -> None:
        self.hparams = kwargs
        super().__init__(num_ids, embed_size)
        weight_mask = torch.zeros(self.weight.shape)
        self.weight_mask = nn.Parameter(weight_mask, requires_grad=False)
        self.idmap = idmap
        # hyper parameters
        self.num_ids = num_ids
        self.embed_size = embed_size
        self.scale_embed = embed_size ** 0.5
        self.padding = padding
        #self.weight.requires_grad = False

    def init_weights(self, name: "str | None" = None) -> None:
        name = self.__class__.__name__
        initializer = self.hparams.get('initializer')
        if initializer in ['orthogonal']:
            logger.debug(f"initializing {name} weight orthogonally")
            nn.init.orthogonal_(self.weight, gain=1.0)
        elif initializer in ['he-normal']:
            logger.debug(f"initializing {name} weight with Kaiming He's Normal")
            #std = 1.0 / (self.num_embeddings ** 0.5)
            std = 1.0
            nn.init.normal_(self.weight, std=std)
        else: # if initilizer in ['pytorch', None]:
            logger.debug(f"initializing {name} weight with PyTorch's default method")
            pass # pytorch default initializer
        #dprint(self.idmap)
        #dprint(len(dict_str2vector))
        #dprint(self.weight.shape)
        #dprint(dict(self.named_parameters()))
        if self.idmap is not None:
            #dprint(dict(self.named_parameters()))
            if len(dict_str2vector) > 0:
                #weight_mask = torch.zeros(self.weight.shape, dtype=torch.bool)
                #dprint(weight_mask)
                #self.weight_mask = nn.Parameter(weight_mask, requires_grad=False)
                count = 0
                #logger.debug("importing Embedding weight with pre-trained vectors")
                for i, s in enumerate(self.idmap):
                    #dprint(s)
                    #dprint(s in dict_str2vector)
                    vec = extract_vector(s, dict_str2vector)
                    #if s in dict_str2vector:
                    if vec is not None:
                        count += 1
                        #vec = dict_str2vector[s]
                        dim = min(vec.shape[0], self.embed_size)
                        self.weight.data[i,0:dim] = vec[0:dim]
                        #self.weight.data[i,0:dim] = vec[0:dim] / self.scale_embed
                        #self.weight.data[i,0:dim] = vec[0:dim] / dim
                        #self.weight_mask.data[i,0:dim] = True
                        self.weight_mask.data[i,0:dim] = 1.0
                    #else:
                    #    dprint(s)
                # normalizing
                logger.debug(f"imported {count} pre-trained vectors")
                #dprint(dict(self.named_parameters()))
        # he-normal の分岐で使った std と名前が衝突していたため別名にする
        logger.debug("re-scaling the initial weights")
        dprint(self.weight.data.abs().max())
        dprint(self.weight.data.std())
        #max_abs = self.weight.data.abs().max()
        weight_std = self.weight.data.std()
        #self.weight.data /= max_abs
        self.weight.data /= weight_std
        dprint(self.weight.data.abs().max())
        dprint(self.weight.data.std())
        self.weight.data /= self.scale_embed
        dprint(self.weight.data.abs().max())
        dprint(self.weight.data.std())

    def forward(self, ids: torch.Tensor, fix_vectors: bool = False) -> torch.Tensor:
        if self.padding is not None:
            zero = torch.tensor(0).to(ids.device)
            mask = ids != self.padding # (D1,...,Dn)
            ids = torch.where(mask, ids, zero) # dummy
        emb = super().forward(ids) # (D1,...,Dn, E)
        if fix_vectors and self.training:
            if hasattr(self, 'weight_mask'):
                fix_mask = nn.functional.embedding(ids, self.weight_mask) > 0.0
                #dprint(emb)
                #dprint(w)
                #dprint(w > 0.0)
                #dprint(emb.shape)
                #dprint(w.shape)
                emb = torch.where(fix_mask, emb.detach(), emb)
                #dprint(emb)
        if self.padding is not None:
            zero = torch.tensor(0.0).to(emb.device, emb.dtype)
            #mask = mask.reshape(mask.shape + (1,))
            emb = torch.where(mask.unsqueeze(-1), emb, zero)
        #return emb
        return cast(torch.Tensor, emb * self.scale_embed)

    def extra_repr(self) -> str:
        if self.padding is None:
            return f'{self.num_ids}, {self.embed_size}'
        else:
            return f'{self.num_ids}, {self.embed_size}, padding={self.padding}'

    def to(self, *args: Any, **kwargs: Any) -> "Embedding":
        return modeling.apply_to(self, *args, **kwargs)

class CharacterEmbedding(Embedding):
    def __init__(self, embed_size: int) -> None:
        self.offset = 2 # <bos>, <eos>
        self.num_ids = self.offset + 256
        self.padding = -1
        self.bos = 0
        self.eos = 1
        self.embed_size = embed_size
        super().__init__(self.num_ids, self.embed_size, self.padding)

    def bytes2tensor(self, codes: bytes, add_symbols: bool = True) -> torch.Tensor:
        assert isinstance(codes, bytes)
        device = self.weight.device
        seq = [self.bos] + [code + self.offset for code in codes] + [self.eos]
        return torch.tensor(seq).to(device)

    def str2tensor(self, string: str, add_symbols: bool = True) -> torch.Tensor:
        assert isinstance(string, str)
        return self.bytes2tensor(bytes(string, 'utf-8'), add_symbols)

    def tensor2bytes(self, tensor: torch.Tensor, remove_symbols: bool = True) -> bytes:
        assert isinstance(tensor, torch.Tensor)
        # ndim はプロパティであり呼び出せない (元の書き方は TypeError)
        assert tensor.ndim == 1
        seq = tensor.tolist()
        if remove_symbols:
            if seq[0] == self.bos:
                seq.pop(0)
            while seq[-1] == self.padding:
                seq.pop(-1)
            if seq[-1] == self.eos:
                seq.pop(-1)
        codes = [code - self.offset for code in seq]
        return bytes(codes)

    def tensor2str(self, tensor: torch.Tensor, remove_symbols: bool = True) -> str:
        return str(self.tensor2bytes(tensor, remove_symbols), 'utf-8', errors='backslashreplace')

    def prepare_batch(self, seq: Any) -> torch.Tensor:
        """Pad a batch of strings into one tensor of character ids

        文字列の集まりを、文字 ID の 1 つのテンソルへパディングして揃える。

        This body was on `ContextualStringEmbedding`, where `self.weight`
        and `self.str2tensor` do not exist; it belongs here.

        この処理は `ContextualStringEmbedding` 側に置かれていたが、そこには
        `self.weight` も `self.str2tensor` も無く、本来こちらのものである。
        """
        device = self.weight.device
        if isinstance(seq, torch.Tensor):
            return seq
        batch: list[torch.Tensor]
        if isinstance(seq, str):
            batch = [self.str2tensor(seq, add_symbols=True)]
        elif isinstance(seq, pd.Series):
            if len(seq) == 0:
                batch = []
            elif isinstance(seq.iloc[0], str): # assuming codes
                batch = [self.str2tensor(string, add_symbols=True) for string in seq]
            else:
                raise TypeError(f"unsupported type: {type(seq.iloc[0])}")
        elif isinstance(seq, (list, tuple)):
            batch = [self.str2tensor(string, add_symbols=True) for string in seq]
        else:
            # 元実装はここで batch が未定義のまま次へ進んでいた
            raise TypeError(f"unsupported type: {type(seq).__name__}")
        # CharacterEmbedding は必ず整数のパディング ID を持つ
        assert self.padding is not None
        padded = nn.utils.rnn.pad_sequence(batch, True, float(self.padding))
        return cast(torch.Tensor, padded.to(device))

class ContextualStringEmbedding(modeling.Module):
    def __init__(self, **params: Any) -> None:
        # nn.Module.__init__ を先に通さないと、下でのモジュール代入が
        # AttributeError になる (元実装はこの呼び出しを欠いており、
        # 本クラスは一度も構築できなかった)
        super().__init__()
        # parameters
        hparams = self.get_config(**params)
        self.padding = hparams['padding']
        self.char_embed_size = hparams['char_embed_size']
        self.hidden_size = hparams['hidden_size']
        # modules
        #self.embed_char = Embedding(256, self.char_embed_size, self.padding)
        self.embed_char = CharacterEmbedding(self.char_embed_size)
        self.mod_rnn_forward  = lstm.MultiLayerLSTM(self.char_embed_size, self.hidden_size, **params)
        self.mod_rnn_backward = lstm.MultiLayerLSTM(self.char_embed_size, self.hidden_size, **params)

    @classmethod
    def get_config(cls, **params: Any) -> dict[str, Any]:
        params.setdefault('padding', 1)
        params.setdefault('char_embed_size', 256)
        params.setdefault('hidden_size', 256)
        params['out_size'] = params['hidden_size'] * 2
        #params.setdefault('num_layers', 1)
        #params.setdefault('dropout_ratio', 0.1)
        return params

    def forward(self, block_ids: torch.Tensor) -> torch.Tensor:
        # ids.shape : (B, NumTokens, NumChars)
        batch_size, num_tokens, num_chars = block_ids.shape
        mask_block = block_ids != self.padding # (B, NumTokens, NumChars)
        emb_block = self.embed_char(block_ids) # (B, NumTokens, NumChars, E)
        mask_seq = mask_block.reshape(batch_size, num_tokens * num_chars) # (B, NT*NC)
        emb_seq = emb_block.reshape(batch_size, num_tokens * num_chars, -1) # (B, NT*NC, E)
        h_seq_forward  = self.mod_rnn_forward(emb_seq, mask_seq) # (B, NT*NC, H)
        h_seq_backward = self.mod_rnn_backward(emb_seq.flip(1), mask_seq.flip(1)).flip(1) # (B, NT*NC, H)
        h_block_forward  = h_seq_forward.reshape(batch_size, num_tokens, num_chars, -1) # (B, NT, NC, H)
        h_block_backward = h_seq_backward.reshape(batch_size, num_tokens, num_chars, -1) # (B, NT, NC, H)
        # 各テンソルは (B, NT, H) の 3 次元なので、連結軸は 2。
        # dim=3 は範囲外で IndexError になっていた
        return torch.cat([h_block_backward[:,:,0,:],h_block_forward[:,:,-1,:]], dim=2) # (B, NumTokens, 2*H)

    def reset_state(self) -> None:
        self.mod_rnn_forward.reset_state()
        self.mod_rnn_backward.reset_state()

    def prepare_batch(self, seq: Any) -> torch.Tensor:
        """Delegate to the character embedding this module wraps

        本モジュールが内部に持つ文字埋め込みへ委譲する。
        """
        return self.embed_char.prepare_batch(seq)

