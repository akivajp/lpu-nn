#!/usr/bin/env python3

# system
from typing import Any, cast

# 3rd
import pandas as pd
import torch
from torch import nn

# local
from lpu.common import logging
from lpu_nn.common import utils
from lpu_nn import modeling
from lpu_nn.modeling import embeddings
from lpu_nn.modeling import transformer
from lpu_nn.modeling import universal_transformer
from lpu_nn.modeling.activation import get_gain

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

class Bert(modeling.Module):
    #def __init__(self, vocab, **params):
    def __init__(self, idmaps: "dict[str, Any]", **params: Any) -> None:
        super().__init__()
        # parameters
        hparams = self.get_config(**params)
        #vocab = idmaps['x']
        vocab = idmaps['seq']
        self.idmaps = idmaps
        self.vocab = vocab
        self.padding = vocab.pad
        self.embed_size   = hparams['embed_size']
        self.num_token_types = hparams['num_token_types']
        if self.num_token_types is None:
            self.num_token_types = 0
        self.embed_positions = hparams['embed_positions']
        self.universal    = hparams['universal']
        self.relative_attention = hparams['relative_attention']
        self.vocab_size = len(vocab)
        # 埋め込みのスケーリングは embeddings.Embedding が内部で行う
        self.scale_emb = self.embed_size ** 0.5
        # modules
        self.mod_embed_tok = embeddings.Embedding(
            self.vocab_size, self.embed_size, padding=self.padding)
        if self.embed_positions:
            self.mod_embed_pos = transformer.EmbedPosition(**hparams)
            hparams['shared_embed_pos'] = self.mod_embed_pos
        else:
            self.mod_encode_pos = transformer.PositionalEncoder()
            hparams['shared_encode_pos'] = self.mod_encode_pos
        if self.relative_attention:
            self.mod_embed_rel_pos = transformer.EmbedRelativePosition(**hparams)
            hparams['shared_embed_rel_pos'] = self.mod_embed_rel_pos
        self.mod_embed_type: embeddings.Embedding | None
        if self.num_token_types >= 2:
            #self.mod_embed_type = nn.Embedding(self.num_token_types, self.embed_size, padding_idx=self.padding)
            self.mod_embed_type = embeddings.Embedding(
                1+self.num_token_types, self.embed_size, padding=self.padding)
        else:
            self.mod_embed_type = None
        #params['combine'] = False
        self.mod_transform: (universal_transformer.UniversalTransformer
                             | transformer.MultiStepTransformer)
        if self.universal:
            #self.mod_transform = transformer.UniversalTransformer(combine=False, **hparams)
            self.mod_transform = universal_transformer.UniversalTransformer(conditioned=False, **hparams)
        else:
            self.mod_transform = transformer.MultiStepTransformer(conditioned=False, **hparams)
        self.mod_pool = nn.Linear(self.embed_size, self.embed_size)

    @classmethod
    def get_config(cls, **params: Any) -> dict[str, Any]:
        #dprint(params)
        params.setdefault('embed_size', 512)
        #if params.get('hidden_size') is None:
        #    params['hidden_size'] = params['embed_size'] * 4
        params.setdefault('activation', 'gelu')
        params.setdefault('num_layers', 8)
        #params.setdefault('share_embedding', True)
        params.setdefault('dropout_ratio', 0.1)
        params.setdefault('embed_positions', True)
        params.setdefault('relative_attention', False)
        params.setdefault('universal', False)
        params.setdefault('max_length', 512)
        params.setdefault('num_token_types', 0)
        #params.setdefault('num_token_types', 3)
        if params['universal']:
            params = universal_transformer.UniversalTransformer.get_config(**params)
        else:
            params.setdefault('sublayer_preprocess', '')
            params.setdefault('sublayer_postprocess', 'dan')
            params = transformer.MultiStepTransformer.get_config(**params)
        if params['relative_attention']:
            params = transformer.EmbedRelativePosition.get_config(**params)
        #transformer.Encoder.get_parameters(**params)
        #dprint(params)
        return params

    def init_weights(self) -> None:
        gain = get_gain('tanh')
        nn.init.orthogonal_(self.mod_pool.weight, gain=gain)
        self.mod_pool.bias.data.zero_()

    def prepare_input(self, seq1: Any, seq2: Any = None) -> "pd.Series":
        input = pd.Series()
        x: tuple[int, ...] = (self.vocab.cls,)
        segment_info: tuple[int, ...] = (1,)
        seq1 = self.vocab.convert(seq1, 'ids')
        x += tuple(seq1)
        segment_info += (1,) * len(seq1)
        if seq2 is not None:
            seq2 = self.vocab.convert(seq2, 'ids')
            x += tuple(seq2)
            segment_info += (2,) * len(seq2)
        input['x'] = x
        input['segment_info'] = segment_info
        input['s1'] = seq1
        if seq2 is not None:
            input['s2'] = seq2
        return input

    def prepare_batch(self, seq: Any) -> torch.Tensor:
        vocab = self.vocab
        if isinstance(seq, torch.Tensor):
            return seq
        elif isinstance(seq, str):
            #batch = [xp.array(vocab.safe_add_symbols(vocab.encode(seq)), dtype=xp.int32)]
            batch = [torch.tensor(vocab.safe_add_symbols(vocab.encode(seq)))]
        elif isinstance(seq, (list,tuple)):
            if isinstance(seq[0], int):
                # ids
                batch = [torch.tensor(seq)]
            elif isinstance(seq[0], str):
                batch = [torch.tensor(vocab.encode(s)) for s in seq]
            elif isinstance(seq[0], (list, tuple)):
                # should be list/tuple of ids
                batch = [torch.tensor(idvec) for idvec in seq]
            else:
                raise TypeError(f"unknown element type: {type(seq[0])}")
        elif isinstance(seq, pd.Series):
            if len(seq) == 0:
                batch = []
            elif isinstance(seq.iloc[0], (list,tuple)):
                batch = [torch.tensor(idvec) for idvec in seq]
            else:
                #dprint(list(seq))
                #dprint(seq[0])
                #dprint(type(seq[0]))
                batch = [torch.tensor(vocab.encode(sent, add_symbols=False)) for sent in seq]
        return nn.utils.rnn.pad_sequence(batch, True, self.padding).to(self.device)

    def prepare_features(self, seq: "torch.Tensor | None", **features: Any) -> dict[str, Any]:
        if seq is not None:
            if seq.dim() == 2:
                features['id_seq'] = seq
                #features['mask_self'] = transformer.make_attention_mask(self.xp, self.padding, seq, seq)
                features['mask_self'] = utils.make_attention_mask(seq, seq, self.padding)
        max_steps = getattr(self, 'max_steps', None)
        if max_steps is not None:
            features['max_steps'] = max_steps
        return features

    def forward(self, id_seq: Any, segment_info: "torch.Tensor | None" = None,
                **features: Any) -> torch.Tensor:
        id_seq = self.prepare_batch(id_seq)
        features = self.prepare_features(seq=id_seq, **features)
        embed_seq = self.mod_embed_tok(id_seq)
        if segment_info is not None:
            if self.mod_embed_type is not None:
                #dprint(segment_info)
                type_embed_seq = self.mod_embed_type(segment_info)
                embed_seq = embed_seq + type_embed_seq
                #dprint(type_id_seq[0].data)
        h = cast(torch.Tensor, self.mod_transform(embed_seq, **features))
        pooled = torch.tanh(self.mod_pool(h[:, 0]))
        self.last_state['pooled'] = pooled
        return h

    def get_pooled(self) -> "torch.Tensor | None":
        return self.last_state.get('pooled')

    def get_state(self) -> dict[str, Any]:
        self.last_state['transformer_state'] = self.mod_transform.get_state()
        return self.last_state

    def reset_state(self) -> None:
        self.last_state: dict[str, Any] = {}
        self.mod_transform.reset_state()

class BertLanguageModel(modeling.Module):
    #def __init__(self, vocab, **params):
    def __init__(self, idmaps: "dict[str, Any]", **params: Any) -> None:
        super().__init__()
        # hyper parameters
        params = self.get_config(**params)
        vocab = idmaps['seq']
        self.idmaps = idmaps
        self.vocab = vocab
        self.padding = self.vocab.pad
        self.vocab_size = len(vocab)
        self.embed_size   = params['embed_size']
        self.share_embedding = params['share_embedding']
        self.dropout_ratio = params['dropout_ratio']
        # modules
        #self.mod_bert = Bert(vocab, **params)
        self.mod_bert = Bert(idmaps, **params)
        self.mod_predict_next_sentence = nn.Linear(self.embed_size, 2)
        self.mod_generate = nn.Linear(self.embed_size, self.vocab_size, bias=False)
        if self.share_embedding:
            self.mod_generate.weight = self.mod_bert.mod_embed_tok.weight

    @classmethod
    def get_config(cls, **params: Any) -> dict[str, Any]:
        #dprint(params)
        params.setdefault('embed_size', 512)
        params.setdefault('share_embedding', True)
        params.setdefault('dropout_ratio', 0.1)
        if params.get('relative_attention'):
            params.setdefault('embed_positions', False)
        else:
            params.setdefault('embed_positions', True)
        #params.setdefault('num_classes', 2)
        #params = BertBase.get_parameters(**params)
        params = Bert.get_config(**params)
        return params

    def init_weights(self) -> None:
        nn.init.orthogonal_(self.mod_predict_next_sentence.weight)
        self.mod_predict_next_sentence.bias.data.zero_()
        if self.mod_generate.weight is not self.mod_bert.mod_embed_tok.weight:
            nn.init.orthogonal_(self.mod_generate.weight)

    def prepare_batch(self, seq: Any) -> torch.Tensor:
        return self.mod_bert.prepare_batch(seq)

    def forward(self, id_seq: Any, type_id_seq: "torch.Tensor | None" = None,
                **features: Any) -> torch.Tensor:
        h = self.mod_bert(id_seq, type_id_seq, **features)
        return cast(torch.Tensor, h)

    def predict_next_sentence(self, h: torch.Tensor) -> torch.Tensor:
        pooled = self.mod_bert.get_pooled()
        return cast(torch.Tensor, self.mod_predict_next_sentence(pooled)) # (B, 2)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        logits = cast(torch.Tensor, self.mod_generate(h[:, 1:None])) # (B, L, V)
        return logits.transpose(1,2) # (B, V, L)

    def get_state(self) -> dict[str, Any]:
        self.last_state['bert_state'] = self.mod_bert.get_state()
        return self.last_state

    def reset_state(self) -> "BertLanguageModel":
        self.last_state: dict[str, Any] = {}
        self.mod_bert.reset_state()
        return self

