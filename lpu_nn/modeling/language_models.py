#!/usr/bin/env python3

# system

# system
from typing import Any, cast

# 3rd
import pandas as pd
import torch
from torch import nn

# local
from lpu.common import logging
from lpu_nn import modeling
from lpu_nn.modeling import embeddings
from lpu_nn.modeling import transformer
from lpu_nn.modeling.encoder_decoder import LSTMDecoder

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

class LanguageModel(modeling.Module):
    def __init__(self, idmaps: "dict[str, Any]", **params: Any) -> None:
        """
        :param FieldMap idmaps:
        :param dict params:
        """
        params = self.get_config(**params)
        vocab = idmaps['x']
        self.idmaps = idmaps
        self.vocab = vocab
        self.vocab_size = len(vocab)
        #self.padding = vocab.pad
        self.padding       = params['padding']
        self.embed_size    = params['embed_size']
        self.direction     = params['direction']
        self.share_embedding = params['share_embedding']
        self.decoder_type = params['decoder_type']
        #params.setdefault('padding', self.padding)
        #self.device = torch.device('cpu')
        super().__init__()
        if self.share_embedding:
            #self.mod_embed_tok = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=vocab.pad)
            self.mod_embed_tok = embeddings.Embedding(self.vocab_size, self.embed_size, padding=self.padding)
            params['shared_embed_tok'] = self.mod_embed_tok
        if params.get('using_transformer'):
            if params['embed_positions']:
                self.mod_embed_pos = transformer.EmbedPosition(**params)
                params['shared_embed_pos'] = self.mod_embed_pos
            else:
                self.mod_encode_pos = transformer.PositionalEncoder()
                params['shared_encode_pos'] = self.mod_encode_pos
            if params['relative_attention']:
                self.mod_embed_rel_pos = transformer.EmbedRelativePosition(**params)
                params['shared_embed_rel_pos'] = self.mod_embed_rel_pos
        self.mod_decode: (LSTMDecoder | transformer.Decoder)
        if self.decoder_type == 'lstm':
            self.mod_decode = LSTMDecoder(idmaps, **params)
        elif self.decoder_type == 'transformer':
            self.mod_decode = transformer.Decoder(idmaps, **params)
        else:
            raise ValueError(f"unknown decoder type: {self.decoder_type}")

    @classmethod
    def get_config(cls, **params: Any) -> dict[str, Any]:
        #dprint(params)
        params.setdefault('direction', 'bidirectional')
        params.setdefault('padding', -1)
        params.setdefault('embed_size', 512)
        params.setdefault('share_embedding', True)
        params.setdefault('dropout_ratio', 0.1)
        params.setdefault('main_component', 'lstm')
        params.setdefault('decoder_type', params['main_component'])
        if params['decoder_type'] == 'lstm':
            params.setdefault('attention_type', 'none')
            params = LSTMDecoder.get_config(**params)
        elif params['decoder_type'] == 'transformer':
            params = transformer.Decoder.get_config(**params)
        return params

    def decode_one(self, memory: torch.Tensor, y: torch.Tensor, **features: Any) -> torch.Tensor:
        return self.mod_decode.decode_one(memory, y, **features)

    def prepare_batch(self, seq: Any) -> torch.Tensor:
        device = self.device
        vocab = self.vocab
        batch: list[torch.Tensor]
        if isinstance(seq, torch.Tensor):
            return seq
        elif isinstance(seq, str):
            batch = [torch.tensor(vocab.encode(seq, add_symbols=True))]
        elif isinstance(seq, pd.Series):
            if len(seq) == 0:
                batch = []
            elif isinstance(seq.iloc[0], (list,tuple)):
                batch = [torch.tensor(vocab.safe_add_symbols(idvec)) for idvec in seq]
            else:
                batch = [torch.tensor(vocab.encode(sent, add_symbols=True)) for sent in seq]
        else:
            raise TypeError(f"unsupported type: {type(seq).__name__}")
        return nn.utils.rnn.pad_sequence(batch, True, self.padding).to(device)

    def prepare_features(self, **features: Any) -> dict[str, Any]:
        max_steps = getattr(self, 'max_steps', None)
        if max_steps is not None:
            features['max_steps'] = max_steps
        #return self.mod_encode.prepare_features(**features)
        return self.mod_decode.prepare_features(**features)

    def restore_batch(self, batch: Any, to: Any = str, squeeze: bool = True) -> Any:
        vocab = self.vocab
        list_ids = [vocab.clean_ids(ids) for ids in batch.tolist()]
        if to in [str, 'str', 'string']:
            list_str = [vocab.decode(ids) for ids in list_ids]
            if squeeze:
                if len(list_str) == 1:
                    return list_str[0]
            return list_str
        elif to in [list, 'list']:
            return list_ids
        elif to in [tuple, 'tuple']:
            return tuple(list_ids)
        elif to in [pd.Series, 'series']:
            return pd.Series(list_ids)
        else:
            raise TypeError(f"unknown conversion type: {to}")

    #def forward(self, x_seq, t_seq=None):
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        x_seq = self.prepare_batch(x_seq)
        features = self.prepare_features(x=x_seq)
        #t_seq = self.prepare_batch(t_seq)
        #memory, encoder_features = self.encode(x_seq, **features)
        #memory  = self.mod_encode(x_seq, **features)
        #logits = self.mod_decode(memory, t_seq, **features) # (B, V, L)
        logits = self.mod_decode(x_seq, None, **features) # (B, V, L)
        return cast(torch.Tensor, logits)

    def get_state(self) -> dict[str, Any]:
        if hasattr(self.mod_decode, 'get_state'):
            self.last_state['decoder_state'] = self.mod_decode.get_state()
        return self.last_state

    def reset_state(self) -> "LanguageModel":
        self.last_state: dict[str, Any] = {}
        # 値を捨てる式ではなく、条件として書く
        if hasattr(self.mod_decode, 'reset_state'):
            self.mod_decode.reset_state()
        return self

    def set_state(self, state: Any) -> Any:
        if state is None:
            return self.reset_state()
        self.mod_decode.set_state(state['decoder_state'])
        return self
