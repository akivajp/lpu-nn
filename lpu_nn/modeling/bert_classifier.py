#!/usr/bin/env python3

from typing import Any, cast

import torch
from torch import nn

from lpu_nn import modeling
from lpu_nn.modeling import bert

class BertClassifier(modeling.Module):
    # 引数は語彙そのものではなくフィールド名 -> 語彙の対応表。
    # 旧称 vocab のままだったため vocab.pad が AttributeError になっていた
    def __init__(self, idmaps: "dict[str, Any]", **params: Any) -> None:
        super().__init__()
        # hyper parameters
        hparams = self.get_config(**params)
        self.idmaps = idmaps
        self.vocab = idmaps['seq']
        self.padding = self.vocab.pad
        self.embed_size  = hparams['embed_size']
        self.num_classes = hparams['num_classes']
        # modules
        self.mod_bert = bert.Bert(idmaps, **hparams)
        self.mod_classify = nn.Linear(self.embed_size, self.num_classes)

    @classmethod
    def get_config(cls, **params: Any) -> dict[str, Any]:
        params.setdefault('embed_size', 512)
        params.setdefault('num_classes', 2)
        params = bert.Bert.get_config(**params)
        return params

    def prepare_input(self, seq1: Any) -> "Any":
        return self.mod_bert.prepare_input(seq1)

    def prepare_batch(self, seq: Any) -> torch.Tensor:
        return self.mod_bert.prepare_batch(seq)

    def forward(self, seq: Any, segment_info: "torch.Tensor | None" = None,
                reset_state: bool = True) -> torch.Tensor:
        if reset_state:
            self.reset_state()
        self.mod_bert(seq, segment_info=segment_info)
        pooled = self.mod_bert.get_pooled()
        return cast(torch.Tensor, self.mod_classify(pooled))

    def reset_state(self) -> "BertClassifier":
        self.mod_bert.reset_state()
        return self
