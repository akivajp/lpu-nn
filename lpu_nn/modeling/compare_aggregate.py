#!/usr/bin/env python3

# system
import math
from typing import Any, cast

# 3rd
import torch
from torch import nn

# local
from lpu.common import logging
from lpu_nn.common.utils import make_attention_mask
from lpu_nn.common.utils import purge_tensor
from lpu_nn import modeling
from lpu_nn.modeling import embeddings
from lpu_nn.modeling.activation import get_activator
from lpu_nn.modeling.activation import get_gain
from lpu_nn.modeling.pooling import get_sequence_pooler

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

class NGramPooler(modeling.Module):
    def __init__(self, ngram_order: int, input_size: int, **hparams: Any) -> None:
        hparams = self.get_config(**hparams)
        self.pooling = hparams['sequence_pooling']
        self.ngram_order  = ngram_order
        self.input_size   = input_size
        self.hidden_size  = hparams['hidden_size']
        self.activation   = hparams['activation']
        super().__init__()
        self.mod_conv = nn.Conv2d(
            in_channels  = 1,
            out_channels = self.hidden_size,
            kernel_size  = (self.ngram_order, self.input_size),
            bias         = True,
        )
        #self.mod_conv = nn.Sequential(
        #    conv2d,
        #    get_activator(self.activation, self.hidden_size),
        #)
        self.mod_activate = get_activator(self.activation, self.hidden_size)
        self.mod_pool = get_sequence_pooler(self.pooling, self.hidden_size)

    def init_weights(self) -> None:
        gain = get_gain(self.activation)
        nn.init.orthogonal_(self.mod_conv.weight, gain=gain)
        if self.mod_conv.bias is not None:
            nn.init.zeros_(self.mod_conv.bias)

    @classmethod
    def get_config(cls, **params: Any) -> dict[str, Any]:
        if 'pooling' in params:
            params.setdefault('sequence_pooling', params['pooling'])
        params.setdefault('sequence_pooling', 'average')
        params.setdefault('hidden_size', 150)
        params.setdefault('activation', 'relu')
        return params

    def forward(self, seq: torch.Tensor, mask_seq: torch.Tensor) -> torch.Tensor:
        # seq.shape : (B, Len, I)
        batch_size, length, _ = seq.shape
        seq = seq.unsqueeze(1) # (B, 1, Len, I)
        if length < self.ngram_order:
            # 系列長が n-gram 長より短いと畳み込みが成立しない
            # (Conv2d が RuntimeError を送出する) ため、右側を 0 で埋めて
            # 少なくとも 1 出力が得られるようにする
            seq = nn.functional.pad(seq, (0, 0, 0, self.ngram_order - length))
        #seq = self.mod_activate(self.mod_conv(seq)) # (B, H, Len-N+1, 1) 
        seq = self.mod_conv(seq) # (B, H, Len-N+1, 1)
        seq = seq.squeeze(3).transpose(1, 2) # (B, Len-N+1, H)
        seq = self.mod_activate(seq)
        num_pad = length - seq.shape[1]
        if num_pad > 0:
            # 出力長を入力長に揃え、mask_seq と位置を対応付ける
            zeros = torch.zeros([batch_size, num_pad, self.hidden_size], dtype=seq.dtype, device=seq.device)
            seq = torch.cat([seq, zeros], dim=1) # (B, Len, H)
        return cast(torch.Tensor, self.mod_pool(seq, mask_seq)) # (B, H)
class MultiFilterPooler(modeling.Module):
    def __init__(self, input_size: int, **hparams: Any) -> None:
        hparams = self.get_config(**hparams)
        self.ngram_orders = hparams['ngram_orders']
        self.hidden_size  = hparams['hidden_size']
        self.num_filters = len(self.ngram_orders)
        super().__init__()
        filter_modules = []
        for n in self.ngram_orders:
            filter_modules.append(NGramPooler(n, input_size, **hparams))
        self.mods_filter = nn.ModuleList(filter_modules)
        self.mod_select = nn.Linear(self.num_filters*self.hidden_size, self.hidden_size, bias=True)

    def init_weights(self) -> None:
        nn.init.orthogonal_(self.mod_select.weight, gain=get_gain('tanh'))
        nn.init.zeros_(self.mod_select.bias)

    @classmethod
    def get_config(cls, **params: Any) -> dict[str, Any]:
        params.setdefault('pooling', 'average')
        params.setdefault('hidden_size', 150)
        params.setdefault('ngram_orders', [1,2,3,4,5])
        if isinstance(params['ngram_orders'], int):
            params['ngram_orders'] = [params['ngram_orders']]
        params.setdefault('activation', 'relu')
        params = NGramPooler.get_config(**params)
        return params

    def forward(self, seq: torch.Tensor, mask_seq: torch.Tensor) -> torch.Tensor:
        # seq.shape : (B, Len, I)
        h_list = [self.mods_filter[i](seq, mask_seq) for i in range(self.num_filters)] # List[(B, H), N]
        h = torch.cat(h_list, dim=1) # (B, H*N)
        return torch.tanh(self.mod_select(h)) # (B, H)

class CompareAggregatePooler(modeling.Module):
    def __init__(self, idmaps: "dict[str, Any]", **hparams: Any) -> None:
        hparams = self.config = self.get_config(idmaps=idmaps, **hparams)
        self.dropout_ratio = hparams['dropout_ratio']
        self.embed_size    = hparams['embed_size']
        self.hidden_size   = hparams['hidden_size']
        self.fix_imported_vectors = hparams['fix_imported_vectors']
        vocab = idmaps['seq']
        self.idmaps = idmaps
        self.vocab = vocab 
        self.vocab_size = vocab_size = len(vocab)
        self.padding = vocab.pad
        super().__init__()
        self.mod_embed = embeddings.Embedding(vocab_size, self.embed_size, padding=self.padding, idmap=vocab)
        self.mod_input  = nn.Linear(self.embed_size, self.hidden_size, bias=True)
        self.mod_update = nn.Linear(self.embed_size, self.hidden_size, bias=True)
        self.mod_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.mod_nn  = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.mod_pool = MultiFilterPooler(self.hidden_size, **hparams)

    def init_weights(self) -> None:
        nn.init.orthogonal_(self.mod_input.weight, gain=get_gain('sigmoid'))
        nn.init.zeros_(self.mod_input.bias)
        nn.init.orthogonal_(self.mod_update.weight, gain=get_gain('tanh'))
        nn.init.zeros_(self.mod_update.bias)
        nn.init.orthogonal_(self.mod_weight.weight)
        nn.init.zeros_(self.mod_weight.bias)
        nn.init.orthogonal_(self.mod_nn.weight, gain=get_gain('relu'))
        nn.init.zeros_(self.mod_nn.bias)

    @classmethod
    def get_config(cls, idmaps: "dict[str, Any] | None" = None, **params: Any) -> dict[str, Any]:
        params.setdefault('embed_size', 128)
        params.setdefault('hidden_size', 150)
        params.setdefault('padding', -1)
        params.setdefault('dropout_ratio', 0.1)
        params.setdefault('fix_imported_vectors', True)
        params['fix_imported_vectors'] = bool(params['fix_imported_vectors'])
        params = MultiFilterPooler.get_config(**params)
        return params

    def preproc(self, x: torch.Tensor) -> torch.Tensor:
        input = torch.sigmoid(self.mod_input(x))
        input = self.dropout(input)
        update = torch.tanh(self.mod_update(x))
        update = self.dropout(update)
        return input * update

    def forward(self, q: torch.Tensor, a: torch.Tensor, encoded: bool = False) -> torch.Tensor:
        mask_a = a != self.padding
        mask_qa = make_attention_mask(q, a, self.padding) # (B, LenQ, LenA)
        if not encoded:
            q_emb = self.mod_embed(q, self.fix_imported_vectors) # (B,LenQ,E)
            a_emb = self.mod_embed(a, self.fix_imported_vectors) # (B,LenA,E)
            q_emb = self.dropout(q_emb)
            a_emb = self.dropout(a_emb)
            q_preproc = self.preproc(q_emb) # (B,LenQ,H)
            a_preproc = self.preproc(a_emb) # (B,LenA,H)
        else:
            q_preproc = q
            a_preproc = a
        q_weighted = self.mod_weight(q_preproc)
        prod_qa = torch.matmul(q_weighted, a_preproc.transpose(1, 2)) # (B, LenQ, H) * (B, H, LenA) -> (B, LenQ, LenA)
        prod_qa = purge_tensor(prod_qa, mask_qa, -math.inf)
        weight_qa = torch.softmax(prod_qa, dim=1) # (B, LenQ, LenA)
        weight_qa = purge_tensor(weight_qa, mask_qa)
        h_attn = torch.matmul(weight_qa.transpose(1, 2), q_preproc) # (B, LenA, LenQ) * (B, LenQ, H) -> (B, LenA, H)
        h_attn = purge_tensor(h_attn, mask_a)
        sub = (a_preproc - h_attn) ** 2
        mult = a_preproc * h_attn
        sub_mult = torch.cat([sub, mult], dim=2) # (B, LenA, 2*H)
        h_nn = torch.relu(self.mod_nn(sub_mult))
        h_nn = self.dropout(h_nn)
        return cast(torch.Tensor, self.mod_pool(h_nn, mask_a)) # (B, H)
