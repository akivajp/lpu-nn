#!/usr/bin/env python3

# system
import time

# 3rd
import pandas as pd
import torch
from torch import nn

# local
from lpu.common import logging
from lpu_nn.common.utils import flip
from lpu_nn.common.utils import split_state
from lpu_nn.common.utils import stack_state_list
from lpu_nn import modeling
from lpu_nn.modeling import embeddings
from lpu_nn.modeling import transformer
from lpu_nn.modeling.activation import get_gain
from lpu_nn.modeling.attention import get_attention
from lpu_nn.modeling.lstm import MultiLayerLSTM

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

class LSTMEncoder(modeling.Module):
    #def __init__(self, vocab, **params):
    def __init__(self, idmaps, **params):
        params = LSTMEncoder.get_config(**params)
        #dprint(params)
        #self.padding = vocab.pad
        vocab = idmaps['x']
        self.idmaps = idmaps
        self.vocab = vocab
        #self.padding = vocab.pad
        self.vocab_size = len(vocab)
        #self.vocab_size    = params['vocab_size']
        self.padding       = params['padding']
        self.embed_size    = params['embed_size']
        self.hidden_size   = params['hidden_size']
        self.dropout_ratio = params['dropout_ratio']
        self.bidirectional = params.get('bidirectional_encoder')
        super().__init__()
        shared_embed_tok = params.get('shared_embed_tok')
        self.mod_rnn_forward = MultiLayerLSTM(**params)
        if self.bidirectional:
            self.mod_rnn_backward = MultiLayerLSTM(**params)
        if shared_embed_tok is None:
            #self.mod_embed_tok = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=self.padding)
            self.mod_embed_tok = embeddings.Embedding(self.vocab_size, self.embed_size, padding=self.padding)
        else:
            self.mod_embed_tok = shared_embed_tok
        #self.mod_dropout = nn.Dropout(self.dropout_ratio)

    @classmethod
    def get_config(cls, **params):
        #dprint(params)
        params.setdefault('padding', -1)
        params.setdefault('embed_size', 512)
        params.setdefault('hidden_size', params['embed_size'] * 2)
        params['in_size']  = params['embed_size']
        params['out_size'] = params['hidden_size']
        params.setdefault('dropout_ratio', 0.1)
        if params.get('bidirectional_encoder') is None:
            params.setdefault('bidirectional', True)
            params['bidirectional_encoder'] = params['bidirectional']
        else:
            params.setdefault('bidirectional', params.get('bidirectional_encoder'))
        if params['bidirectional_encoder']:
            params['memory_size'] = params['hidden_size'] * 2
        else:
            params['memory_size'] = params['hidden_size']
        if params.get('num_encoder_layers') is None:
            params.setdefault('num_layers', 1)
            params['num_encoder_layers'] = params['num_layers']
        else:
            params.setdefault('num_layers', params.get('num_encoder_layers'))
        params = MultiLayerLSTM.get_config(**params)
        #dprint(params)
        return params

    #def prepare_features(self, x=None, **features):
    def prepare_features(self, seq=None, **features):
        #if x is not None:
        if seq is not None:
            #if x.dim() == 2:
            if seq.dim() == 2:
                #features['x_id_seq'] = x
                #features['mask_x'] = (x.data != self.padding)
                features['mem_id_seq'] = seq
                features['mask_mem'] = (seq.data != self.padding)
        #dprint(features.keys())
        return features

    #def forward(self, x_seq, **features):
    def forward(self, seq, **features):
        #batch_size, len_x = x_seq.shape
        batch_size, len_seq = seq.shape
        #mask_x = features.get('mask_x')
        #mask_mem = features.get('mask_mem')
        mask_mem = features['mask_mem'] # mandatory
        embed_size = self.embed_size
        #x_seq_emb = self.mod_embed_tok(x_seq) * (embed_size ** 0.5)
        #x_seq_emb = torch.dropout(x_seq_emb, self.dropout_ratio, self.training)
        #seq_emb = self.mod_embed_tok(seq) * (embed_size ** 0.5)
        seq_emb = self.mod_embed_tok(seq)
        seq_emb = torch.dropout(seq_emb, self.dropout_ratio, self.training)
        #h_forward = self.mod_recurrent_forward(x_seq_emb) # (B, L, H)
        #h_forward = self.mod_rnn_forward(x_seq_emb, mask_x) # (B, L, H)
        h_forward = self.mod_rnn_forward(seq_emb, mask_mem) # (B, L, H)
        if self.bidirectional:
            #h_backward = self.mod_rnn_backward(seq_emb.flip(1), mask_mem.flip(1)).flip(1) # (B, L, H)
            h_backward = flip(self.mod_rnn_backward(flip(seq_emb,1), flip(mask_mem,1)),1) # (B, L, H)
            return torch.cat([h_forward, h_backward], dim=2) # (B, L, 2H)
        else:
            #return h_forward, features
            return h_forward

    def get_state(self):
        self.last_state['forward_rnn_state'] = self.mod_rnn_forward.get_state()
        if self.bidirectional:
            self.last_state['backward_rnn_state'] = self.mod_rnn_backward.get_state()
        return self.last_state

    def reset_state(self):
        self.last_state = {}
        self.mod_rnn_forward.reset_state()
        if self.bidirectional:
            self.mod_rnn_backward.reset_state()

class LSTMDecoder(modeling.Module):
    def __init__(self, idmaps, **params):
        params = LSTMDecoder.get_config(**params)
        vocab = idmaps['x']
        self.idmaps = idmaps
        self.padding = vocab.pad
        self.vocab_size = params['vocab_size']
        self.padding    = params['padding']
        self.embed_size = params['embed_size']
        self.hidden_size = params['hidden_size']
        self.input_feeding = params['input_feeding']
        self.dropout_ratio = params['dropout_ratio']
        self.memory_size = params['memory_size']
        self.attention_type = params['attention_type'] # (none, dot, concat, general, mlp)
        self.share_embedding = params['share_embedding']
        super().__init__()
        mod_embed_tok = params.get('shared_embed_tok')
        if mod_embed_tok is None:
            #self.mod_embed_tok = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=self.padding)
            self.mod_embed_tok = embeddings.Embedding(self.vocab_size, self.embed_size, padding=self.padding)
        else:
            self.mod_embed_tok = mod_embed_tok # shared
        self.mod_rnn_forward = MultiLayerLSTM(**params)
        self.mod_attention = get_attention(self.attention_type, **params)
        if self.attention_type in ['concat', 'general', 'mlp']:
            self.mod_combine = nn.Linear(self.hidden_size + self.memory_size, self.embed_size)
        elif self.attention_type in ['dot']:
            self.mod_combine = nn.Linear(self.hidden_size * 2, self.embed_size)
        else:
            self.mod_reduce = nn.Linear(self.hidden_size, self.embed_size)
        self.mod_output = nn.Linear(self.embed_size, self.vocab_size, bias=False)
        if self.share_embedding:
            pass # using self.embed_tok.W
            self.mod_output.weight = self.mod_embed_tok.weight
        self.mod_dropout = nn.Dropout(self.dropout_ratio)

    @staticmethod
    def get_config(**params):
        #dprint(params)
        params.setdefault('padding', -1)
        params.setdefault('embed_size', 512)
        params.setdefault('hidden_size', params['embed_size'] * 2)
        params.setdefault('activation', 'relu')
        #params.setdefault('attention_type', 'general')
        params.setdefault('attention_type', 'mlp')
        params.setdefault('local_attention', False)
        params.setdefault('input_feeding', True)
        if params['attention_type'] == 'none':
            params['memory_size'] = None
        if params.get('bidirectional_encoder'):
            params['encoder_hidden_size'] = params['hidden_size'] * 2
        if params.get('input_feeding'):
            params['in_size']  = params['embed_size'] * 2
            #params['in_size']  = params['embed_size'] + params['memory_size']
        else:
            params['in_size']  = params['embed_size']
        params['out_size'] = params['hidden_size']
        params.setdefault('dropout_ratio', 0.1)
        if params.get('num_decoder_layers') is None:
            params.setdefault('num_layers', 1)
            params['num_decoder_layers'] = params['num_layers']
        else:
            params.setdefault('num_layers', params.get('num_decoder_layers'))
        params = MultiLayerLSTM.get_config(**params)
        #dprint(params)
        return params

    def init_weights(self):
        nn.init.orthogonal_(self.mod_embed_tok.weight)
        if hasattr(self, 'mod_combine'):
            nn.init.orthogonal_(self.mod_combine.weight, gain=get_gain('tanh'))
            self.mod_combine.bias.data.zero_()
        nn.init.orthogonal_(self.mod_output.weight)

    #def prepare_features(self, y=None, prev_y=None, **features):
    def prepare_features(self, seq=None, prev_seq=None, **features):
        #if y is not None:
        if seq is not None:
            #if y.dim() == 1:
            if seq.dim() == 1:
                #features['y_id_seq'] = y[:,None] # (B, 1)
                features['id_seq'] = seq[:,None] # (B, 1)
            #elif y.ndim == 2:
            elif seq.ndim == 2:
                #features['y_id_seq'] = y # (B, L)
                features['id_seq'] = seq # (B, L)
        return features

    def forward(self, y, memory, **features):
        return self.decode(y, memory, **features)

    #def decode_one(self, y, memory, **features):
    def decode_one(self, seq, memory, **features):
        device = self.device
        dtype  = self.dtype
        #features = self.prepare_features(y=y, **features)
        features = self.prepare_features(seq_enc=seq, **features)
        #batch_size = y.shape[0]
        batch_size = seq.shape[0]
        #y_emb = self.mod_embed_tok(y) * (self.embed_size ** 0.5)
        #y_emb = torch.dropout(y_emb, self.dropout_ratio, self.training)
        #seq_emb = self.mod_embed_tok(seq) * (self.embed_size ** 0.5)
        seq_emb = self.mod_embed_tok(seq)
        seq_emb = torch.dropout(seq_emb, self.dropout_ratio, self.training)
        #mask = y != self.padding
        mask_seq = seq != self.padding
        if self.input_feeding:
            input_feed = self.last_state.get('input_feed')
            #input_feed = self.input_feed
            if input_feed is None:
                #input_feed = torch.zeros([batch_size, self.embed_size]).to(device)
                input_feed = torch.zeros([batch_size, self.embed_size]).to(device, dtype)
            #h_dec = self.mod_rnn_forward(torch.cat([y_emb, input_feed], dim=1)).squeeze(1) # (B, H_dec+H_mem)
            h_dec = self.mod_rnn_forward(torch.cat([seq_emb, input_feed], dim=1)).squeeze(1) # (B, H_dec+H_mem)
            #h_dec = self.mod_recurrent_forward(torch.cat([y_emb, input_feed], dim=1), mask).squeeze(1) # (B, H_dec+H_mem)
        else:
            #h_dec = self.mod_rnn_forward(y_emb).squeeze(1) # (B, H_dec)
            h_dec = self.mod_rnn_forward(seq_emb).squeeze(1) # (B, H_dec)
            #h_dec = self.mod_recurrent_forward(y_emb, mask).squeeze(1) # (B, H_dec)
        if self.attention_type != "none":
            #context = self.mod_attention(memory, h_dec, **features) # (B, H_mem)
            context = self.mod_attention(h_dec, memory, **features) # (B, H_mem)
            h_combine = torch.tanh(self.mod_combine(torch.cat([context, h_dec], dim=1))) # (B, E)
            self.last_state['input_feed'] = h_combine
            h_combine = self.mod_dropout(h_combine)
            h_out = self.mod_output(h_combine) # (B, V)
        else:
            h_reduce = torch.tanh(self.mod_reduce(h_dec)) # (B, H) -> (B, E)
            h_out = self.mod_output(h_reduce) # (B, V)
        #self.input_feed = h_combine
        return h_out

    #def decode(self, memory, y_seq, **features):
    #def decode(self, y_seq, memory, **features):
    def decode(self, seq, memory, **features):
        #dprint(features.keys())
        #y_list = [y.squeeze(1) for y in y_seq.split(1, dim=1)] # List[B, E]
        token_batch_list = [tokens.squeeze(1) for tokens in seq.split(1, dim=1)] # List[B, E]
        h_out_list = []
        #for y in y_list:
        for token_batch in token_batch_list:
            #h_out = self.decode_one(memory, y, **features)
            #h_out = self.decode_one(y, memory, **features)
            h_out = self.decode_one(token_batch, memory, **features)
            h_out_list.append(h_out)
        h_out_seq = torch.stack(h_out_list, dim=2) # (B, V, L)
        return h_out_seq

    def get_state(self):
        self.last_state['rnn_state'] = self.mod_rnn_forward.get_state()
        return self.last_state

    def reset_state(self):
        self.last_state = {}
        #self.cache = Cache(self.xp)
        #self.input_feed = None
        self.mod_rnn_forward.reset_state()

    def set_state(self, state):
        self.last_state = state
        self.mod_rnn_forward.set_state(state['rnn_state'])

class EncoderDecoder(modeling.Module):
    #def __init__(self, vocab, **params):
    def __init__(self, idmaps, **params):
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
        self.share_embedding = params['share_embedding']
        self.encoder_type = params['encoder_type']
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
        if self.encoder_type == 'lstm':
            #self.mod_encode = LSTMEncoder(vocab, **params)
            self.mod_encode = LSTMEncoder(idmaps, **params)
        elif self.encoder_type == 'transformer':
            #self.mod_encode = transformer.Encoder(vocab, **params)
            self.mod_encode = transformer.Encoder(idmaps, **params)
        else:
            raise ValueError(f"unsupported encoder type: {self.encoder_type}")
        if self.decoder_type == 'lstm':
            #self.mod_decode = LSTMDecoder(vocab, **params)
            self.mod_decode = LSTMDecoder(idmaps, **params)
        elif self.decoder_type == 'transformer':
            #self.mod_decode = transformer.Decoder(vocab, **params)
            self.mod_decode = transformer.Decoder(idmaps, **params)
        else:
            raise ValueError(f"unsupported decoder type: {self.decoder_type}")

    @classmethod
    def fix_component_name(cls, name):
        if name.lower() in ['lstm', 'rnn']:
            return 'lstm'
        elif name.lower() in ['transformer', 'trans']:
            return 'transformer'
        elif name.lower() in ['bert']:
            return 'bert'
        else:
            raise ValueError(f"Unsupported component name: {name}")

    @classmethod
    def get_config(cls, **params):
        #dprint(params)
        params.setdefault('padding', -1)
        params.setdefault('embed_size', 512)
        params.setdefault('share_embedding', True)
        params.setdefault('dropout_ratio', 0.1)
        #params.setdefault('main_component', 'lstm')
        params.setdefault('main_component', 'transformer')
        params.setdefault('encoder_type', params['main_component'])
        params.setdefault('decoder_type', params['main_component'])
        if any(params[key] == 'transformer' for key in ['encoder_type', 'decoder_type']):
            # transformer specific
            params['using_transformer'] = True
            params.setdefault('embed_positions', False)
            #params.setdefault('relative_attention', False)
            params.setdefault('relative_attention', True)
            params.setdefault('universal', True)
            if params['relative_attention']:
                params = transformer.EmbedRelativePosition.get_config(**params)
        params['encoder_type'] = cls.fix_component_name(params['encoder_type'])
        params['decoder_type'] = cls.fix_component_name(params['decoder_type'])
        if params['encoder_type'] == 'lstm':
            params = LSTMEncoder.get_config(**params)
        elif params['encoder_type'] == 'transformer':
            params = transformer.Encoder.get_config(**params)
        if params['decoder_type'] == 'lstm':
            params = LSTMDecoder.get_config(**params)
        elif params['decoder_type'] == 'transformer':
            params = transformer.Decoder.get_config(**params)
        #dprint(params)
        return params

    #def decode_one(self, memory, y, **features):
    def decode_one(self, y, memory, **features):
        #return self.mod_decode.decode_one(memory, y, **features)
        return self.mod_decode.decode_one(y, memory, **features)

    def prepare_batch(self, seq):
        device = self.device
        vocab = self.vocab
        if isinstance(seq, torch.Tensor):
            return seq
        elif isinstance(seq, str):
            batch = torch.tensor([vocab.encode(seq, add_symbols=True)])
        elif isinstance(seq, (pd.Series,list)):
            seq = list(seq)
            if len(seq) == 0:
                batch = []
            elif isinstance(seq[0], (list,tuple)):
                batch = [torch.tensor(vocab.safe_add_symbols(idvec)) for idvec in seq]
            else:
                batch = [torch.tensor(vocab.encode(sent, add_symbols=True)) for sent in seq]
        return nn.utils.rnn.pad_sequence(batch, True, self.padding).to(device)

    #def prepare_features(self, **features):
    def prepare_features(self, seq_enc=None, seq_dec=None, **features):
        max_steps = getattr(self, 'max_steps', None)
        if max_steps is not None:
            features['max_steps'] = max_steps
        features = self.mod_encode.prepare_features(seq=seq_enc, **features)
        features = self.mod_decode.prepare_features(seq=seq_dec, **features)
        return features

    def restore_batch(self, batch, to=str, squeeze=True):
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

    def generate(self, x, max_length=100, timeout=None):
        device = self.device
        start = time.time()
        with torch.no_grad():
            x_id_seq = self.prepare_batch(x)
            batch_size, len_x = x_id_seq.shape
            self.reset_state()
            #features = self.prepare_features(x=x_id_seq)
            features = self.prepare_features(seq_enc=x_id_seq)
            memory = self.mod_encode(x_id_seq, **features)
            y = torch.tensor([self.vocab.bos] * batch_size).to(device) # (B,)
            y_id_seq = y[:,None] # (B, 1)
            stopped = torch.tensor([False] * batch_size).to(device) # (B,)
            try:
                for i in range(max_length):
                    #if i >= 2:
                    #    break
                    #dprint("-----")
                    prev_y = y_id_seq[:,0:-1] # (B, Len-1)
                    #dprint(prev_y)
                    #h_out = self.decode_one(memory, y, prev_y=prev_y, **features) # (B, V)
                    #h_out = self.decode_one(y, memory, prev_y=prev_y, **features) # (B, V)
                    h_out = self.decode_one(y, memory, prev_seq=prev_y, **features) # (B, V)
                    #dprint(h_out.shape)
                    #dprint(h_out[:5])
                    #self.mod_decode.reset_state()
                    #h_out = self.decode_one(memory, y_id_seq, **features) # (B, V)
                    #dprint(h_out.shape)
                    #dprint(h_out[:5])
                    y = h_out.argmax(1) # (B,)
                    y_id_seq = torch.cat([y_id_seq,y[:,None]], dim=1) # (B, Len+1)
                    stopped = stopped | (y == self.vocab.eos)
                    if stopped.all():
                        break
                    if timeout is not None and time.time() - start > timeout:
                        logger.debug("timed out")
                        break
            except Exception as e:
                logger.exception(e)
            return y_id_seq

    def beam_search(self, x,
                    beam_width=5, incomplete_cost=100, max_length=100, repetition_cost=0, normalize=False, timeout=None):
        start = time.time()
        device = self.device
        with torch.no_grad():
            x_id_seq = self.prepare_batch(x)
            features = self.prepare_features(seq_enc=x_id_seq)
            batch_size, len_x = x_id_seq.shape
            if batch_size > 1:
                dprint(x_id_seq.shape)
                raise ValueError("beam search currently supports only for sequences with input batch size 1")
            self.reset_state()
            memory = self.mod_encode(x_id_seq, **features) # (1, Len_x, H)
            batch_y = torch.tensor([self.vocab.bos])[:,None].to(device) # (1, 1)
            nbest_comp_list = []
            nbest_incomp_list = [(0, batch_y[0], None)]
            for i in range(1, int(max_length)):
                candidates = []
                dprint(i)
                prev_scores, list_y_id_seq, state_list = zip(*nbest_incomp_list)
                #dprint(format_state_list(state_list))
                batch_size = len(list_y_id_seq)
                batch_state = stack_state_list(state_list)
                #dprint(format_state(batch_state))
                self.set_state(batch_state)
                #batch_x_id_seq = F.repeat(x_id_seq, batch_size, axis=0) # (B, Len_x)
                batch_x_id_seq = x_id_seq.repeat(batch_size, 1) # (B, Len_x)
                features = self.prepare_features(seq_enc=batch_x_id_seq)
                batch_y_id_seq = nn.utils.rnn.pad_sequence(list_y_id_seq, True, self.padding) # (B, Len_y)
                #dprint(batch_y_id_seq)
                #dprint(list(self.vocab.convert(y_id_seq.tolist(), 'tokens') for y_id_seq in list_y_id_seq))
                batch_memory = memory.repeat(batch_size, 1, 1) # (B, LenMem, H)
                batch_prev_y = batch_y_id_seq[:,0:-1] # (B, L-1)
                #dprint(batch_prev_y)
                batch_y = batch_y_id_seq[:,-1] # (B, 1)
                #logits = self.decode_one(batch_y, batch_memory, prev_y = batch_prev_y, **features)
                logits = self.decode_one(batch_y, batch_memory, prev_seq = batch_prev_y, **features)
                #dprint(format_state(self.get_state()))
                state_list = split_state(self.get_state(), batch_size)
                #dprint(format_state(split_state(self.get_state(), 2)))
                #dprint(state_list)
                #dprint(format_state_list(state_list))
                batch_log_probs = -logits.log_softmax(dim=1) # (B, V)
                array_log_probs = batch_log_probs.data
                for i, prev_score in enumerate(prev_scores):
                    #dprint(array_log_probs.shape)
                    trg_id_seq = list_y_id_seq[i]
                    log_probs = array_log_probs[i]
                    sample_state = state_list[i]
                    best_values, best_indices = log_probs.topk(beam_width, dim=0, largest=False)
                    indices = best_indices.tolist()
                    if self.vocab.eos in indices:
                        sent_score = prev_score + float(log_probs[self.vocab.eos])
                    else:
                        sent_score = prev_score + float(log_probs[self.vocab.eos]) * (1 + incomplete_cost)
                    if len(nbest_comp_list) < beam_width:
                        nbest_comp_list.append( (sent_score, trg_id_seq.data.tolist()) )
                        nbest_comp_list = sorted(nbest_comp_list, key=lambda t: t[0])[:beam_width]
                    else:
                        if sent_score < nbest_comp_list[-1][0]:
                            nbest_comp_list.append( (sent_score, trg_id_seq.data.tolist()) )
                            nbest_comp_list = sorted(nbest_comp_list, key=lambda t: t[0])[:beam_width]
                    if self.vocab.eos in indices:
                        indices.remove(self.vocab.eos)
                    scores = log_probs[indices]
                    for index, score in zip(indices, scores.tolist()):
                        vocab_ids = torch.tensor([index]).to(device)
                        trg_id_seq_concat = torch.cat([trg_id_seq, vocab_ids], dim=0)
                        if repetition_cost > 0:
                            prev_idvec = trg_id_seq.data.reshape(-1)
                            weight = torch.arange(1, prev_idvec.shape[0]+1).float().to(device) / float(prev_idvec.shape[0])
                            cost = ((prev_idvec == index).float() * weight * repetition_cost).sum()
                            score = prev_score + score * (1 + cost)
                        else:
                            score = prev_score + score
                        if len(nbest_comp_list) < beam_width or score < nbest_comp_list[-1][0]:
                            candidates.append( (score, trg_id_seq_concat, sample_state) )
                        else:
                            break
                nbest_incomp_list = sorted(candidates, key=lambda t: t[0])[:beam_width]
                for score, id_seq in nbest_comp_list[:5]:
                    sent_comp = self.vocab.decode(id_seq) + "</s>"
                    #sent_comp = self.vocab.convert(id_seq, 'tokens') + ["</s>"]
                    dprint( (score, sent_comp) , )
                for score, id_seq, sample_state in nbest_incomp_list[:5]:
                    sent_incomp = self.vocab.decode(id_seq.data.tolist())
                    #sent_incomp = self.vocab.convert(id_seq.data.tolist(), 'tokens')
                    dprint( (score, sent_incomp), )
                if len(nbest_incomp_list) == 0:
                    break
                if timeout is not None and time.time() - start > timeout:
                    logger.debug("timed out")
                    break
        results = []
        if normalize:
            #dprint(nbest_comp_list)
            nbest_comp_list = sorted([(score/len(seq), seq) for score, seq in nbest_comp_list])
            #dprint(nbest_comp_list)
        for score, id_seq in nbest_comp_list:
            #generated = restore(id_seq)
            #results.append( (generated, score) )
            results.append( (id_seq, score) )
        return results

    def forward(self, x_seq, t_seq=None):
        if t_seq is not None:
            x_seq = self.prepare_batch(x_seq)
            features = self.prepare_features(seq_enc=x_seq)
            t_seq = self.prepare_batch(t_seq)
            memory  = self.mod_encode(x_seq, **features)
            logits = self.mod_decode(t_seq, memory, **features) # (B, V, L)
            return logits
        else:
            return self.generate(x_seq)

    def get_state(self):
        if hasattr(self.mod_encode, 'get_state'):
            self.last_state['encoder_state'] = self.mod_encode.get_state()
        if hasattr(self.mod_decode, 'get_state'):
            self.last_state['decoder_state'] = self.mod_decode.get_state()
        return self.last_state

    def reset_state(self):
        self.last_state = {}
        hasattr(self.mod_encode, 'reset_state') and self.mod_encode.reset_state()
        hasattr(self.mod_decode, 'reset_state') and self.mod_decode.reset_state()
        return self

    def set_state(self, state):
        if state is None:
            return self.reset_state()
        self.mod_decode.set_state(state['decoder_state'])
        return self
