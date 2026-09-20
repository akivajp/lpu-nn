#!/usr/bin/env python3

from typing import Any, cast

import torch
from torch import nn

from lpu_nn import modeling
from lpu_nn.modeling import bert

class BertRanker(modeling.Module):
    # 引数は語彙そのものではなくフィールド名 -> 語彙の対応表
    def __init__(self, idmaps: "dict[str, Any]", **params: Any) -> None:
        super().__init__()
        # parameters
        hparams = self.get_config(**params)
        self.idmaps = idmaps
        self.vocab = idmaps['seq']
        self.padding = self.vocab.pad
        self.embed_size  = hparams['embed_size']
        # modules
        self.mod_bert  = bert.Bert(idmaps=idmaps, **hparams)
        self.mod_score = nn.Linear(self.embed_size, 1)

    @classmethod
    def get_config(cls, **params: Any) -> dict[str, Any]:
        params.setdefault('embed_size', 512)
        params = bert.Bert.get_config(**params)
        return params

    def prepare_input(self, seq1: Any, seq2: Any) -> "Any":
        return self.mod_bert.prepare_input(seq1, seq2)

    def prepare_batch(self, seq: Any) -> torch.Tensor:
        return self.mod_bert.prepare_batch(seq)

    def forward(self, seq: Any, segment_info: "torch.Tensor | None" = None,
                reset_state: bool = True, **features: Any) -> torch.Tensor:
        if reset_state:
            self.reset_state()
        #transformed, extra_output = self.L_bert(seq, segment_id_seq=segment_id_seq, require_extra_info=True)
        self.mod_bert(seq, segment_info=segment_info, **features)
        #score = self.L_score(transformed[:,0])
        #score = score[:,0]
        pooled = self.mod_bert.get_pooled() # (B, H)
        score = cast(torch.Tensor, self.mod_score(pooled))[:,0] # (B,)
        #return score, extra_output
        return score

    def reset_state(self) -> "BertRanker":
        self.mod_bert.reset_state()
        return self

