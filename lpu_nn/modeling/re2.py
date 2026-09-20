#!/usr/bin/env python3

# system
from typing import Any, cast

# 3rd
import torch
from torch import nn

# local
from lpu.common import logging
from lpu_nn.common.utils import make_attention_mask
from lpu_nn import modeling
from lpu_nn.modeling import embeddings
from lpu_nn.modeling.convolution import SequenceConvolution1d
from lpu_nn.modeling.pooling import get_sequence_pooler

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

class MultiFilterEncoderLayer(modeling.Module):
    def __init__(self, input_size: int, **kwargs: Any) -> None:
        hparams = self.get_config(**kwargs)
        self.orders = hparams['ngram_orders']
        self.input_size = input_size
        self.hidden_size = hparams['hidden_size']
        self.activation   = hparams['activation']
        self.dropout_ratio = hparams['dropout_ratio']
        self.num_filters = len(self.orders)
        assert self.hidden_size % self.num_filters == 0
        self.hparams = hparams
        super().__init__()
        cnn_modules = []
        for n in self.orders:
            mod_conv = SequenceConvolution1d(
                input_size  = self.input_size,
                output_size = self.hidden_size // self.num_filters,
                ngram_order = n,
                **hparams,
            )
            #cnn_modules.append(mod_conv)
            cnn_modules.append(nn.utils.weight_norm( mod_conv) )
        self.mods_conv = modeling.ModuleList(cnn_modules)
        #self.mod_activate = get_activator(self.activation, self.hidden_size)

    @classmethod
    def get_config(cls, **hparams: Any) -> dict[str, Any]:
        preset = hparams.get('preset')
        if preset in ['ref', 'reference', 're2']:
            hparams.setdefault('hidden_size', 150)
            hparams.setdefault('activation', 'gelu')
        elif preset in ['re2-wikiqa']:
            hparams.setdefault('hidden_size', 200)
            hparams.setdefault('activation', 'gelu')
        else:
            hparams.setdefault('hidden_size', 150)
            hparams.setdefault('activation', 'swish')
        hparams.setdefault('dropout_ratio', 0.2)
        hparams.setdefault('ngram_orders', [3])
        return hparams

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out_list = []
        for mod_cnn in self.mods_conv:
            out_seq = mod_cnn(seq) # (B, L, H/N)
            out_list.append(out_seq)
        out_seq = torch.cat(out_list, dim=2) # List[(B,L,H/N),N] -> (B,L,H)
        #return self.mod_activate(out_seq)
        return out_seq
class MultiFilterEncoder(modeling.Module):
    def __init__(self, input_size: int, **params: Any) -> None:
        params = self.get_config(**params)
        self.input_size = input_size
        self.hidden_size = params['hidden_size']
        self.dropout_ratio = params['dropout_ratio']
        self.num_layers = params['num_layers']
        super().__init__()
        mods_encode = []
        for i in range(self.num_layers):
            if i == 0:
                mod_encode = MultiFilterEncoderLayer(self.input_size, **params)
            else:
                mod_encode = MultiFilterEncoderLayer(self.hidden_size, **params)
            mods_encode.append(mod_encode)
        self.mods_encode = modeling.ModuleList(mods_encode)

    @classmethod
    def get_config(cls, **hparams: Any) -> dict[str, Any]:
        preset = hparams.get('preset')
        if preset in ['ref', 'reference', 're2']:
            hparams.setdefault('num_layers', 2)
        elif preset in ['re2-wikiqa']:
            hparams.setdefault('num_layers', 3)
        else:
            hparams.setdefault('num_layers', 2)
        hparams.setdefault('dropout_ratio', 0.2)
        return hparams

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # seq : (B, I, H)
        for i, mod_encode in enumerate(self.mods_encode):
            if i > 0:
                seq = self.dropout(seq)
            seq = seq.masked_fill(~mask[:,:,None], 0.0)
            seq = mod_encode(seq) # (B, L, H)
        return cast(torch.Tensor, seq)

class Alignment(modeling.Module):
    def __init__(self, input_size: int, **kwargs: Any) -> None:
        self.hparams = hparams = self.get_config(**kwargs)
        self.input_size = input_size
        #self.activation = hparams.get('activation', 'none')
        self.map = hparams['map']
        self.hidden_size = hparams['hidden_size']
        self.dropout_ratio = hparams['dropout_ratio']
        super().__init__()
        self.scale_dot: float
        if self.map == 'linear':
            self.mod_map = nn.utils.weight_norm(
                modeling.Linear(input_size, self.hidden_size, bias=True, **hparams)
            )
            self.scale_dot = float(self.hidden_size) ** -0.5
        else:
            self.scale_dot = float(self.input_size) ** -0.5

    @classmethod
    def get_config(cls, **params: Any) -> dict[str, Any]:
        params.setdefault('map', 'linear')
        params.setdefault('hidden_size', 256)
        params.setdefault('dropout_ratio', 0.1)
        if params['map'] == 'linear':
            params.setdefault('activation', 'gelu')
        return params

    def calc_scores(self, seq1: torch.Tensor, seq2: torch.Tensor, mask_align: torch.Tensor) -> torch.Tensor:
        # (B, Len1, H) * (B, H, Len2) -> (B, Len1, Len2)
        scores = torch.matmul(seq1, seq2.transpose(1, 2)) * self.scale_dot
        #scores.masked_fill_(~mask_align, -math.inf)
        scores.masked_fill_(~mask_align, -10**7)
        return scores

    def forward(self, seq1: torch.Tensor, seq2: torch.Tensor, mask1: torch.Tensor,
                mask2: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
        mask_align = make_attention_mask(mask1, mask2) # (B, Len1, Len2)
        if hasattr(self, 'mod_map'):
            seq1 = self.dropout(seq1)
            seq2 = self.dropout(seq2)
            mapped_seq1 = self.mod_map(seq1)
            mapped_seq2 = self.mod_map(seq2)
        else: # identity
            mapped_seq1 = seq1
            mapped_seq2 = seq2
        scores = self.calc_scores(mapped_seq1, mapped_seq2, mask_align)
        weight1 = scores.softmax(2) # (B, Len1, Len2)
        weight1 = weight1.masked_fill(~mask_align, 0.0)
        weight2 = scores.softmax(1) # (B, Len1, Len2)
        weight2 = weight2.masked_fill(~mask_align, 0.0)
        weight2 = weight2.transpose(1, 2) # (B, Len2, Len1)
        align_seq1 = torch.matmul(weight1, seq2) # (B,Len1,Len2) * (B,Len2,H) -> (B,Len1,H)
        align_seq2 = torch.matmul(weight2, seq1) # (B,Len2,Len1) * (B,Len1,H) -> (B,Len2,H)
        #align_seq1 = self.calc_attention(mapped_seq1, mapped_seq2, seq2, mask1, mask2)
        #align_seq2 = self.calc_attention(mapped_seq2, mapped_seq1, seq1, mask2, mask1)
        return align_seq1, align_seq2

class Fusion(modeling.Module):
    def __init__(self, input_size: int, **kwargs: Any) -> None:
        #hparams = self.config = self.get_config(**hparams)
        self.hparams = hparams = self.get_config(**kwargs)
        self.input_size = input_size
        self.activation = hparams['activation']
        self.hidden_size = hparams['hidden_size']
        self.dropout_ratio = hparams['dropout_ratio']
        super().__init__()
        self.mod_direct = nn.utils.weight_norm(
            modeling.Linear(input_size*2, self.hidden_size, bias=True, **hparams)
        )
        self.mod_sub = nn.utils.weight_norm(
            modeling.Linear(input_size*2, self.hidden_size, bias=True, **hparams)
        )
        self.mod_mult = nn.utils.weight_norm(
            modeling.Linear(input_size*2, self.hidden_size, bias=True, **hparams)
        )
        self.mod_concat = nn.utils.weight_norm(
            modeling.Linear(self.hidden_size*3, self.hidden_size, bias=True, **hparams)
        )

    @classmethod
    def get_config(cls, **params: Any) -> dict[str, Any]:
        params.setdefault('activation', 'gelu')
        params.setdefault('dropout_ratio', 0.2)
        params.setdefault('hidden_size', 256)
        return params

    def forward(self, seq1: torch.Tensor, seq2: torch.Tensor) -> torch.Tensor:
        seq_direct = self.mod_direct(torch.cat([seq1,seq2], dim=2))
        seq_sub    = self.mod_sub(torch.cat([seq1,seq1-seq2], dim=2))
        seq_mult   = self.mod_mult(torch.cat([seq1,seq1*seq2], dim=2))
        seq_concat = torch.cat([seq_direct, seq_sub, seq_mult], dim=2)
        seq_concat = self.dropout(seq_concat)
        return cast(torch.Tensor, self.mod_concat(seq_concat))

class CombinationPooler(modeling.Module):
    def __init__(self, input_size: int, **kwargs: Any) -> None:
        self.hparams = hparams = self.get_config(**kwargs)
        self.input_size = input_size
        self.activation = hparams['activation']
        self.hidden_size = hparams['hidden_size']
        self.dropout_ratio = hparams['dropout_ratio']
        self.symmetric= hparams['symmetric']
        self.method = hparams['method']
        super().__init__()
        if self.method == 'simple':
            #self.mod_linear = nn.Sequential(
            #    nn.utils.weight_norm( nn.Linear(input_size * 2, self.hidden_size, bias=True) ),
            #    get_activator(self.activation, self.hidden_size),
            #)
            self.mod_linear = modeling.Linear(
                input_size*2, self.hidden_size, bias=True, **hparams
            )
        elif self.method == 'full':
            #self.mod_linear = nn.Sequential(
            #    nn.utils.weight_norm( nn.Linear(input_size * 4, self.hidden_size, bias=True) ),
            #    #nn.Linear(input_size * 4, self.hidden_size, bias=True),
            #    get_activator(self.activation, self.hidden_size),
            #)
            self.mod_linear = modeling.Linear(
                input_size*4, self.hidden_size, bias=True, **hparams
            )
        else:
            raise ValueError(f"Unknown pooling method: {self.method}")

    #def init_weights(self):
    #    gain = get_gain(self.activation)
    #    #nn.init.orthogonal_(self.mod_linear[0].weight, gain=gain)
    #    linear = self.mod_linear[0]
    #    nn.init.normal_(linear.weight, std=(2.0/linear.in_features)**0.5) # RE2
    #    nn.init.zeros_(linear.bias)

    @classmethod
    def get_config(cls, **params: Any) -> dict[str, Any]:
        params.setdefault('activation', 'gelu')
        params.setdefault('method', 'simple')
        params.setdefault('dropout_ratio', 0.2)
        params.setdefault('hidden_size', 256)
        params.setdefault('symmetric', False)
        return params

    def forward(self, pool1: torch.Tensor, pool2: torch.Tensor) -> torch.Tensor:
        if self.method == 'simple':
            pool_concat = torch.cat([pool1, pool2], dim=1)
        elif self.method == 'full':
            pool_sub = pool1 - pool2
            if self.symmetric:
                pool_sub = pool_sub.abs()
            pool_concat = torch.cat([pool1, pool2, pool_sub, pool1*pool2], dim=1)
        pool_concat = self.dropout(pool_concat)
        return cast(torch.Tensor, self.mod_linear(pool_concat))

class RE2Block(modeling.Module):
    def __init__(self, input_size: int, **kwargs: Any) -> None:
        hparams = self.get_config(**kwargs)
        self.hidden_size   = hparams['hidden_size']
        super().__init__()
        self.mod_encode = MultiFilterEncoder(input_size, **hparams)
        self.mod_align = Alignment(input_size + self.hidden_size, **hparams)
        self.mod_fusion = Fusion(input_size + self.hidden_size, **hparams)

    @classmethod
    def get_config(cls, **hparams: Any) -> dict[str, Any]:
        preset = hparams.get('preset')
        if preset in ['ref', 'reference', 're2']:
            hparams.setdefault('num_encoder_layers', 2)
        elif preset in ['re2-wikiqa']:
            hparams.setdefault('num_encoder_layers', 3)
        else:
            hparams.setdefault('num_encoder_layers', 2)
        hparams.setdefault('num_layers', hparams['num_encoder_layers'])
        hparams.setdefault('hidden_size', 128)
        hparams.setdefault('dropout_ratio', 0.1)
        hparams.setdefault('num_encoder_layers', hparams['num_layers'])
        hparams['num_layers'] = hparams['num_encoder_layers']
        hparams = MultiFilterEncoderLayer.get_config(**hparams)
        return hparams

    def forward(self, seq1: torch.Tensor, seq2: torch.Tensor, mask1: torch.Tensor,
                mask2: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
        enc_seq1 = self.mod_encode(seq1, mask1) # (B,Len1,I)
        enc_seq2 = self.mod_encode(seq2, mask2) # (B,Len2,I)
        seq1 = torch.cat([enc_seq1, seq1], dim=2) # (B,Len1,I+H)
        seq2 = torch.cat([enc_seq2, seq2], dim=2) # (B,Len2,I+H)
        align_seq1, align_seq2 = self.mod_align(seq1, seq2, mask1, mask2) # (B,Len1,I+H), (B,Len2,I+H)
        fusion_seq1 = self.mod_fusion(seq1, align_seq1) # (B,Len1,H)
        fusion_seq2 = self.mod_fusion(seq2, align_seq2) # (B,Len2,H)
        return fusion_seq1, fusion_seq2

class RE2Pooler(modeling.Module):
    def __init__(self, idmaps: "dict[str, Any]", **hparams: Any) -> None:
        #hparams = self.config = self.get_config(**hparams)
        self.hparams = hparams = self.get_config(**hparams)
        self.dropout_ratio = hparams['dropout_ratio']
        self.embed_size    = hparams['embed_size']
        self.hidden_size   = hparams['hidden_size']
        #self.ngram_orders        = hparams['ngram_orders']
        self.sequence_pooling = hparams['sequence_pooling']
        self.num_blocks = hparams['num_blocks']
        self.fix_imported_vectors = hparams['fix_imported_vectors']
        vocab = idmaps['seq']
        self.idmaps = idmaps
        self.vocab = vocab 
        self.vocab_size = vocab_size = len(vocab)
        self.padding = vocab.pad
        initializer = hparams.get('initializer')
        super().__init__()
        #self.mod_embed = embeddings.Embedding(vocab_size, self.embed_size, padding=self.padding, idmap=vocab)
        self.mod_embed = embeddings.Embedding(
            vocab_size, self.embed_size, padding=self.padding, idmap=vocab, initializer=initializer
        )
        blocks = []
        for i in range(self.num_blocks):
            if i == 0:
                block = RE2Block(self.embed_size, **hparams)
            else:
                block = RE2Block(self.embed_size + self.hidden_size, **hparams)
            blocks.append(block)
        self.mods_blocks = modeling.ModuleList(blocks)
        self.mod_pool_sequence = get_sequence_pooler(self.sequence_pooling, self.hidden_size, **hparams)
        self.mod_compare  = CombinationPooler(self.hidden_size, **hparams)

    @classmethod
    def get_preset_choices(cls, **hparams: Any) -> dict[str, Any]:
        preset_choices = hparams.setdefault('preset_choices', set())
        preset_choices.add('re2')
        preset_choices.add('re2-wikiqa')
        return hparams

    @classmethod
    def get_config(cls, **hparams: Any) -> dict[str, Any]:
        hparams = cls.get_preset_choices(**hparams)
        # 別名の解決はプリセットの既定値より先に行う。
        # 後ろに置くと下の setdefault が先に埋めてしまい、
        # pooling= で渡した指定が黙って無視されていた。
        if 'pooling' in hparams:
            hparams.setdefault('sequence_pooling', hparams['pooling'])
        preset = hparams.get('preset')
        if preset in ['ref', 'reference', 're2']:
            hparams.setdefault('embed_size', 300)
            hparams.setdefault('hidden_size', 150)
            hparams.setdefault('sequence_pooling', 'max')
            hparams.setdefault('optimizer', 'adam')
            hparams.setdefault('initializer', 'he-normal')
        elif preset in ['re2-wikiqa']:
            hparams.setdefault('embed_size', 300)
            hparams.setdefault('hidden_size', 200)
            hparams.setdefault('sequence_pooling', 'max')
            hparams.setdefault('optimizer', 'adam')
            hparams.setdefault('initializer', 'he-normal')
        else:
            hparams.setdefault('embed_size', 128)
            hparams.setdefault('hidden_size', hparams['embed_size'])
            hparams.setdefault('sequence_pooling', 'attention')
            hparams.setdefault('initializer', 'orthogonal')
        hparams.setdefault('padding', -1)
        hparams.setdefault('dropout_ratio', 0.2)
        #params.setdefault('orders', [3])
        hparams.setdefault('num_blocks', 2)
        hparams.setdefault('fix_imported_vectors', True)
        hparams['fix_imported_vectors'] = bool(hparams['fix_imported_vectors'])
        hparams = RE2Block.get_config(**hparams)
        #dprint(hparams)
        return hparams

    def forward(self, id_seq1: torch.Tensor, id_seq2: torch.Tensor) -> torch.Tensor:
        mask1 = id_seq1 != self.padding
        mask2 = id_seq2 != self.padding
        #dprint(self.mod_embed.weight[0:4,0:5])
        emb_seq1 = self.mod_embed(id_seq1, self.fix_imported_vectors) # (B,Len1,E)
        emb_seq2 = self.mod_embed(id_seq2, self.fix_imported_vectors) # (B,Len2,E)
        emb_seq1 = self.dropout(emb_seq1)
        emb_seq2 = self.dropout(emb_seq2)
        seq1 = emb_seq1 # (B,L,E)
        seq2 = emb_seq2 # (B,L,E)
        # i == 0 の回で必ず上書きされるため、この初期値が読まれることは無い
        # (字句上の定義前参照と num_blocks=0 での NameError を避けるための初期化)
        res_seq1 = emb_seq1
        res_seq2 = emb_seq2
        for i, mod_block in enumerate(self.mods_blocks):
            if i == 0:
                seq1 = emb_seq1 # (B,L,E)
                seq2 = emb_seq2 # (B,L,E)
            else: # i >= 1:
                #seq1 = torch.cat([emb_seq1, seq1], dim=2) # (B,L,E+H)
                #seq2 = torch.cat([emb_seq2, seq2], dim=2) # (B,L,E+H)
                seq1 = torch.cat([emb_seq1, res_seq1], dim=2) # (B,L,E+H)
                seq2 = torch.cat([emb_seq2, res_seq2], dim=2) # (B,L,E+H)
            seq1, seq2 = mod_block(seq1, seq2, mask1, mask2)
            #seq1 = self.dropout(seq1)
            #seq2 = self.dropout(seq2)
            if i == 0:
                res_seq1 = seq1
                res_seq2 = seq2
            else: # i >= 1:
                res_seq1 = (res_seq1 + seq1) * (2 ** -0.5)
                res_seq2 = (res_seq2 + seq2) * (2 ** -0.5)
        pool1 = self.mod_pool_sequence(seq1, mask1) # (B,Len1,H) -> (B,H)
        pool2 = self.mod_pool_sequence(seq2, mask2) # (B,Len2,H) -> (B,H)
        return cast(torch.Tensor, self.mod_compare(pool1, pool2)) # (B,H)
