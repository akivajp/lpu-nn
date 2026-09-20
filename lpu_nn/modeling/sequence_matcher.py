#!/usr/bin/env python3

# system
from collections.abc import Sequence
from typing import Any, cast

# 3rd 
import pandas as pd
import torch
from torch import nn

# local
from lpu.common import logging
from lpu_nn import modeling
from lpu_nn.modeling.re2 import RE2Pooler
from lpu_nn.modeling.compare_aggregate import CompareAggregatePooler

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

class SequenceMatcher(modeling.Module):
    def __init__(self, idmaps: "dict[str, Any]", **hparams: Any) -> None:
        hparams = self.config = self.get_config(idmaps=idmaps, **hparams)
        self.device = torch.device('cpu')
        self.dropout_ratio = hparams['dropout_ratio']
        self.embed_size    = hparams['embed_size']
        self.hidden_size   = hparams['hidden_size']
        self.num_classes   = hparams['num_classes']
        self.match_pooler_type = hparams['match_pooler_type']
        vocab = idmaps['seq']
        self.idmaps = idmaps
        self.vocab = vocab 
        self.vocab_size = len(vocab)
        self.padding = vocab.pad
        initializer   = hparams.get('initializer')
        super().__init__()
        self.mod_match: modeling.Module
        if self.match_pooler_type == 're2':
            self.mod_match = RE2Pooler(idmaps, **hparams)
        elif self.match_pooler_type == 'compare-aggregate':
            self.mod_match = CompareAggregatePooler(idmaps, **hparams)
        else:
            raise ValueError(f"unsupported pooler type: {self.match_pooler_type}")
        if self.num_classes:
            #self.mod_pred = nn.Linear(self.hidden_size, self.num_classes, bias=True)
            self.mod_pred = nn.utils.weight_norm( nn.Linear(self.hidden_size, self.num_classes, bias=True) )
        else:
            #self.mod_pred = nn.Linear(self.hidden_size, 1, bias=True)
            self.mod_pred = modeling.Linear(self.hidden_size, 1, bias=True, initializer=initializer)

    #def init_weights(self):
    #    #nn.init.orthogonal_(self.mod_pred.weight)
    #    #nn.init.zeros_(self.mod_pred.bias)
    #    linear = self.mod_pred
    #    nn.init.normal_(linear.weight, std=(1.0/linear.in_features)**0.5) # RE2
    #    nn.init.zeros_(linear.bias)

    @classmethod
    def get_preset_choices(cls, **hparams: Any) -> dict[str, Any]:
        preset_choices = hparams.setdefault('preset_choices', set())
        preset_choices.add('re2')
        preset_choices.add('compare-aggregate')
        return hparams

    @classmethod
    def get_config(cls, idmaps: "dict[str, Any] | None" = None, **hparams: Any) -> dict[str, Any]:
        hparams = cls.get_preset_choices(**hparams)
        hparams.setdefault('match_pooler_type', 're2')
        if hparams['match_pooler_type'].lower() in ['re2']:
            hparams['match_pooler_type'] = 're2'
            hparams = RE2Pooler.get_config(**hparams)
        elif hparams['match_pooler_type'].lower() in ['ca', 'compare-aggregate']:
            hparams['match_pooler_type'] = 'compare-aggregate'
            hparams = CompareAggregatePooler.get_config(**hparams)
        else:
            # ここで弾かないと既定値が一切入らないまま __init__ に進み、
            # 種別ではなく dropout_ratio の KeyError として表面化する
            raise ValueError("unsupported pooler type: {}".format(
                hparams['match_pooler_type']))
        #params.setdefault('embed_size', 128)
        #params.setdefault('hidden_size', params['embed_size'])
        #params.setdefault('padding', -1)
        #params.setdefault('dropout_ratio', 0.2)
        #params.setdefault('num_classes', None)
        if idmaps is not None:
            if 't' in idmaps:
                hparams['num_classes'] = len(idmaps['t'])
        # ラベル表が無い場合は回帰 (スコア直接予測) として扱う。
        # setdefault が無いと __init__ の hparams['num_classes'] が KeyError になる。
        hparams.setdefault('num_classes', None)
        return hparams

    def prepare_batch(self, name: str,
                      iterable: "Sequence[Any] | pd.Series | torch.Tensor") -> torch.Tensor:
        dtype = self.dtype
        device = self.device
        if isinstance(iterable, torch.Tensor):
            return iterable
        if isinstance(iterable, pd.Series):
            iterable = list(iterable)
        if name in ['s1', 's2']:
            idmap = self.idmaps[name]
            if isinstance(iterable[0], str):
                iterable = [idmap.encode(sent) for sent in iterable]
            iterable = [idmap.safe_add_symbols(idvec) for idvec in iterable]
            iterable = [torch.tensor(idvec) for idvec in iterable]
            batch = nn.utils.rnn.pad_sequence(iterable, True, idmap.pad).to(device)
        elif name in ['t']:
            if self.num_classes is None:
                batch = torch.tensor([float(score) for score in iterable], dtype=dtype, device=device)
            else:
                idmap = self.idmaps[name]
                batch = torch.tensor([idmap.str2dist(str(label)) for label in iterable]).to(device)
        else:
            raise TypeError(f"unsupported type: {type(iterable).__name__}")
        return batch

    def forward(self, id_seq1: torch.Tensor, id_seq2: torch.Tensor) -> torch.Tensor:
        pooled = self.mod_match(id_seq1, id_seq2) # (B,H)
        pooled = self.dropout(pooled)
        pred = cast(torch.Tensor, self.mod_pred(pooled)) # (B,C)
        if self.num_classes:
            return pred
        else:
            return pred[:,0] # (B,)

    def logit_to_score(self, logits: torch.Tensor, correct_label: str = '1') -> torch.Tensor:
        batch_size = logits.shape[0]
        idmap = self.idmaps['t']
        probs = logits.softmax(dim=1) # (B, C)
        #dprint(correct_label in idmap)
        #dprint(idmap[correct_label])
        if correct_label in idmap:
            return probs[:,idmap[correct_label]] # (B)
        else:
            scores = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
            for i, label in enumerate(idmap):
                try:
                    f = float(label)
                except ValueError:
                    # 数値として解釈できないラベル (特殊記号など) は寄与させない
                    continue
                scores = scores + f * probs[:, i]
            return scores

    #def score(self, q, a, encoded=False):
    def score(self, id_seq1: torch.Tensor, id_seq2: torch.Tensor,
              correct_label: str = '1') -> torch.Tensor:
        if self.num_classes is None:
            return cast(torch.Tensor, self(id_seq1, id_seq2))
        else:
            logits = self(id_seq1, id_seq2) # (B, C)
            return self.logit_to_score(logits, correct_label)
