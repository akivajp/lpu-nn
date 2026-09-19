#!/usr/bin/env python3

# system
import math

# 3rd
import torch
from torch import nn

# local
from lpu.common import logging
from lpu_nn.common.utils import make_attention_mask
from lpu_nn.common.utils import make_history_mask
from lpu_nn.common.utils import purge_tensor
from lpu_nn import modeling
from lpu_nn.modeling import embeddings
from lpu_nn.modeling.activation import get_activator
from lpu_nn.modeling.activation import get_gain
from lpu_nn.modeling.universal_transformer import UniversalTransformer

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

class EmbedPosition(modeling.Module):
    def __init__(self, **params):
        super().__init__()
        params = EmbedPosition.get_config(**params)
        self.max_length = params.get('max_length')
        self.embed_size = params.get('embed_size')
        self.hidden_size = params.get('hidden_size')
        #self.embed_pos = nn.Embedding(self.max_length, self.embed_size)
        self.embed_pos = embeddings.Embedding(self.max_length, self.hidden_size)
        #self.scale_emb = self.hidden_size ** 0.5

    @classmethod
    def get_config(cls, **params):
        #dprint(params)
        params.setdefault('max_length', 256)
        params.setdefault('embed_size', 512)
        params.setdefault('embed_size', params['hidden_size'])
        #dprint(params)
        return params

    #def forward(self, batch_size, length, start=0):
    def forward(self, length, start=0):
        positions = torch.arange(start, start+length).to(self.device) # (L,)
        position_embed_seq = self.embed_pos(positions) # (L, H)
        #position_embed_seq = position_embed_seq * self.scale_emb
        return position_embed_seq

class EmbedRelativePosition(modeling.Module):
    def __init__(self, **params):
        super().__init__()
        params = EmbedRelativePosition.get_config(**params)
        self.embed_size = params['embed_size']
        self.hidden_size = params['hidden_size']
        self.num_heads  = params['num_heads']
        self.clip_distance = params['clip_distance']
        K = self.clip_distance
        logger.debug(f"using K={K} (clipping distance) for relative positional embedding for self-attention of transformer")
        #self.key_size = self.embed_size // self.num_heads
        self.key_size = self.hidden_size // self.num_heads
        logger.debug("using relative positional enbedding for self-attention")
        #self.mod_embed_relative_position = nn.Embedding(2 * K + 1, self.key_size)
        self.mod_embed_relative_position = embeddings.Embedding(2 * K + 1, self.key_size)
        #self.mod_embed_relative_key_position = self.mod_embed_relative_position # shared
        #logger.debug("using relative positional enbedding of key sequences for self-attention of transformer")
        #self.mod_embed_relative_value_position = self.mod_embed_relative_position # shared
        #logger.debug("using relative positional enbedding of value sequences for self-attention of transformer")
        #self.scale_emb = self.embed_size ** 0.5
        #self.scale_emb = self.hidden_size ** 0.5

    @staticmethod
    def get_config(**params):
        params.setdefault('clip_distance', 8)
        params.setdefault('embed_size', 512)
        params.setdefault('hidden_size', params['embed_size'])
        hidden_size = params['hidden_size']
        if params.get('num_heads'):
            num_heads = params['num_heads']
            #params.setdefault('key_size', embed_size // num_heads)
            params.setdefault('key_size', hidden_size // num_heads)
        else:
            params.setdefault('key_size', 64)
            key_size = params['key_size']
            #params.setdefault('num_heads', embed_size // key_size)
            params.setdefault('num_heads', hidden_size // key_size)
        return params

    def forward(self, positions):
        emb_positions = self.mod_embed_relative_position(positions)
        #return emb_positions * self.scale_emb
        return emb_positions

class FeedForward(modeling.Module):
    def __init__(self, **params):
        super().__init__()
        # parameters
        params = self.get_config(**params)
        self.dropout_ratio = params['dropout_ratio']
        #self.embed_size    = params['embed_size']
        self.hidden_size   = params['hidden_size']
        self.inner_size    = params['inner_size']
        self.activation    = params['activation']
        # module
        self.mod_seq = nn.Sequential(
            #nn.Linear(self.embed_size, self.hidden_size),
            #nn.Linear(self.embed_size, self.inner_size),
            nn.Linear(self.hidden_size, self.inner_size),
            #get_activator(self.activation, self.hidden_size),
            get_activator(self.activation, self.inner_size),
            nn.Dropout(self.dropout_ratio),
            #nn.Linear(self.hidden_size, self.embed_size),
            #nn.Linear(self.inner_size, self.embed_size),
            nn.Linear(self.inner_size, self.hidden_size),
        )

    @classmethod
    def get_config(cls, **params):
        params.setdefault('embed_size', 512)
        params.setdefault('hidden_size', params['embed_size'])
        #params.setdefault('hidden_size', params['embed_size'] * 4)
        #params.setdefault('inner_size', params['embed_size'] * 4)
        params.setdefault('inner_size', params['hidden_size'] * 4)
        params.setdefault('dropout_ratio', 0.1)
        params.setdefault('activation', 'swish')
        return params

    def init_weights(self):
        gain = get_gain(self.activation)
        nn.init.orthogonal_(self.mod_seq[0].weight, gain=gain)
        self.mod_seq[0].bias.data.zero_()
        nn.init.orthogonal_(self.mod_seq[3].weight)
        self.mod_seq[3].bias.data.zero_()

    def forward(self, seq):
        return self.mod_seq(seq)

class MultiHeadAttention(modeling.Module):
    def __init__(self, **params):
        super().__init__()
        # parameters
        params = MultiHeadAttention.get_config(**params)
        self.dropout_ratio = params['dropout_ratio']
        #self.embed_size    = params['embed_size']
        self.hidden_size    = params['hidden_size']
        self.num_heads     = params['num_heads']
        #embed_size = self.embed_size
        hidden_size = self.hidden_size
        num_heads = self.num_heads
        #self.key_size = key_size = int(embed_size / num_heads)
        self.key_size = key_size = int(hidden_size / num_heads)
        #assert embed_size == num_heads * key_size, "got E:{}, H:{}, K:{}".format(embed_size, num_heads, key_size)
        assert hidden_size == num_heads * key_size, f"got Hidden:{hidden_size}, NumHeads:{num_heads}, KeySize:{key_size}"
        self.scale_dot = key_size ** -0.5
        # modules
        self.mod_embed_rel_pos = params.get('shared_embed_rel_pos')
        #self.mod_query  = nn.Linear(embed_size, embed_size, bias=False)
        #self.mod_key    = nn.Linear(embed_size, embed_size, bias=False)
        #self.mod_value  = nn.Linear(embed_size, embed_size, bias=False)
        #self.mod_output = nn.Linear(embed_size, embed_size, bias=False)
        self.mod_query  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mod_key    = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mod_value  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mod_output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.reset_state()

    @classmethod
    def get_config(cls, **params):
        params.setdefault('dropout_ratio', 0.1)
        params.setdefault('embed_size', 512)
        params.setdefault('hidden_size', params['embed_size'])
        #embed_size = params['embed_size']
        hidden_size = params['hidden_size']
        if params.get('num_heads'):
            num_heads = params['num_heads']
            #params.setdefault('key_size', embed_size // num_heads)
            params.setdefault('key_size', hidden_size // num_heads)
        else:
            params.setdefault('key_size', 64)
            key_size = params['key_size']
            params.setdefault('num_heads', hidden_size // key_size)
        return params

    def init_weights(self):
        nn.init.orthogonal_(self.mod_query.weight)
        nn.init.orthogonal_(self.mod_key.weight)
        nn.init.orthogonal_(self.mod_value.weight)
        nn.init.orthogonal_(self.mod_output.weight)

    def embed_relative_position(self, len_query, len_key_val, offset=0):
        device = self.device
        K = self.mod_embed_rel_pos.clip_distance
        arange_key   = torch.arange(len_key_val).to(device) # (LenK)
        arange_query = torch.arange(len_query).to(device)   # (LenQ)
        relative_pos  = arange_key[None,:] - arange_query[:,None] - offset # (LenQ, LenK)
        #dprint(relative_pos)
        clipped_relative_pos = torch.clamp(relative_pos, -K, K) + K
        emb_relative_pos = self.mod_embed_rel_pos(clipped_relative_pos) # (LenQ, LenK, E/H)
        return emb_relative_pos

    def add_relative_key_position_embedding(self, prod, query_slice, is_self_attention, offset):
        #dprint(is_self_attention)
        emb_relative_pos = self.embed_relative_position(prod.shape[1], prod.shape[2], offset) # (LenQ, LenK, E/H)
        # (LenQ, LenK, E/H) -> (LenQ, E/H, LenK)
        emb_relative_pos_trans = emb_relative_pos.transpose(1, 2) # (LenQ, E/H, LenK)
        # (B*H, LenQ, E/H) -> (LenQ, B*H, E/H)
        query_trans = query_slice.transpose(0, 1) # (LenQ, B*H, E/H)
        # (LenQ, B*H, E/H) * (LenQ, E/H, LenK) -> (LenQ, B*H, LenK)
        #dprint(query_trans.shape)
        #dprint(emb_relative_pos_trans.shape)
        prod_relative_key_trans = torch.matmul(query_trans, emb_relative_pos_trans) # (LenQ, B*H, LenK)
        prod_relative_key_trans = prod_relative_key_trans * self.scale_dot # scale product
        # (LenQ, B*H, LenK) -> (B*H, LenQ, LenK)
        prod_relative_key = prod_relative_key_trans.transpose(0, 1) # (B*H, LenQ, LenK)
        prod = prod + prod_relative_key
        return prod

    def add_relative_value_position_embedding(self, attention, weight, is_self_attention, offset):
        #dprint(is_self_attention)
        emb_relative_pos = self.embed_relative_position(weight.shape[1], weight.shape[2], offset) # (LenQ, LenV, E/H)
        # (B*H, LenQ, LenK) -> (LenQ, B*H, LenK)
        weight_trans = weight.transpose(0, 1) # (LenQ, B*H, LenK)
        # (LenQ, B*H, LenV) * (LenQ, LenV, E/H) -> (LenQ, B*H, E/H)
        relative_attention_trans = torch.matmul(weight_trans, emb_relative_pos) # (LenQ, B*H, E/H)
        # (LenQ, B*H, E/H) -> (B*H, LenQ, E/H)
        relative_attention = relative_attention_trans.transpose(0, 1) # (B*H, LenQ, E/H)
        attention = attention + relative_attention
        return attention

    def forward(self, query_seq, key_seq, value_seq, mask, offset=0):
        batch_size, batch_src_len, hidden_size = value_seq.shape
        _, batch_trg_len, _ = query_seq.shape
        self_attention = (query_seq is key_seq)
        assert value_seq.shape == key_seq.shape
        key_size = self.key_size
        num_heads = self.num_heads
        query_seq = self.mod_query(query_seq)
        key_seq   = self.mod_key  (key_seq)
        value_seq = self.mod_value(value_seq)
        # reshaping (B,L,E) -> (B*H,L,E/H) to apply standard attention for each head
        query_slice = torch.cat(query_seq.split(key_size, dim=2), dim=0)
        key_slice   = torch.cat(key_seq.split(key_size, dim=2), dim=0)
        value_slice = torch.cat(value_seq.split(key_size, dim=2), dim=0)
        # (B*H, LenQ, H) * (B*H, H, LenK) -> (B*H, LenQ, LenK=LenV)
        prod = torch.matmul(query_slice, key_slice.transpose(1,2)) * self.scale_dot
        if self.mod_embed_rel_pos is not None:
            prod = self.add_relative_key_position_embedding(prod, query_slice, self_attention, offset)
        assert prod.shape == (batch_size*num_heads, batch_trg_len, batch_src_len)
        #dprint(mask)
        # mask.shape : (B, LenQ, LenK)
        if mask is not None:
            assert mask.shape == (batch_size, batch_trg_len, batch_src_len)
            mask_stack = torch.cat([mask]*num_heads, dim=0) # (B*H, LenQ, LenK)
            prod = purge_tensor(prod, mask=mask_stack, else_value=-math.inf)
        weight = torch.softmax(prod, dim=2)
        if mask is not None:
            weight = purge_tensor(weight, mask=mask_stack)
        weight = torch.dropout(weight, self.dropout_ratio, self.training)
        #self.last_weight = weight
        self.last_state['attention_weight'] = weight.reshape(batch_size, num_heads, batch_trg_len, batch_src_len)
        # (B*H,L2,L1) * (B*H,L1,E/H) -> (B*H,L2,E/H)
        attention = torch.matmul(weight, value_slice) # (B*H, L_q, E/H)
        if self.mod_embed_rel_pos is not None:
            attention = self.add_relative_value_position_embedding(attention, weight, self_attention, offset)
        # reshaping (B*H,L_q,E/H) -> (B,L_q,E)
        #dprint(attention.shape)
        #attention = torch.cat(attention.split(num_heads, dim=0), dim=2)
        attention = torch.cat(attention.split(batch_size, dim=0), dim=2)
        #dprint(attention.shape)
        #out = feed_seq(self.mod_output, attention)
        out = self.mod_output(attention)
        #out = purge_variables(out, mask=mask[:,:,0])
        return out

    def get_state(self):
        return self.last_state

    def reset_state(self):
        #self.last_weight = None
        self.last_state = {}

class PositionalEncoder(modeling.Module):
    def __init__(self):
        self.cache = {}
        super().__init__()

    #def __call__(self, shape, step=0, remember=True):
    def forward(self, shape, step=0, remember=True):
        #batch_size, seq_len, embed_size = shape
        batch_size, seq_len, hidden_size = shape
        #start = 0
        start = 1
        last_encoding = None
        last_seq_len = 0
        #if (embed_size, step) in self.cache:
        if (hidden_size, step) in self.cache:
            #last_encoding = self.cache[embed_size, step]
            last_encoding = self.cache[hidden_size, step]
            last_seq_len, _ = last_encoding.shape
        if last_encoding is not None and seq_len <= last_seq_len:
            encoding = last_encoding
        else:
            pos   = torch.arange(start, seq_len+start)[:,None].float()
            #steps = torch.ones([seq_len,embed_size]).float() * step
            steps = torch.ones([seq_len,hidden_size]).float() * step
            #emb_i = torch.arange(0, embed_size)[None,:].float()
            emb_i = torch.arange(0, hidden_size)[None,:].float()
            #encoding = torch.empty([seq_len,embed_size])
            encoding = torch.empty([seq_len,hidden_size])
            L = torch.tensor(10000.0)
            #encoding[:,0::2]  = torch.sin(pos[:,0::2]   * torch.exp( -emb_i[:,0::2]    * torch.log(L) / embed_size))
            #encoding[:,0::2] += torch.sin(steps[:,0::2] * torch.exp( -emb_i[:,0::2]    * torch.log(L) / embed_size))
            #encoding[:,1::2]  = torch.cos(pos[:,0::2]   * torch.exp(-(emb_i[:,1::2]-1) * torch.log(L) / embed_size))
            #encoding[:,1::2] += torch.cos(steps[:,0::2] * torch.exp(-(emb_i[:,1::2]-1) * torch.log(L) / embed_size))
            encoding[:,0::2]  = torch.sin(pos[:,0::2]   * torch.exp( -emb_i[:,0::2]    * torch.log(L) / hidden_size))
            encoding[:,0::2] += torch.sin(steps[:,0::2] * torch.exp( -emb_i[:,0::2]    * torch.log(L) / hidden_size))
            encoding[:,1::2]  = torch.cos(pos[:,0::2]   * torch.exp(-(emb_i[:,1::2]-1) * torch.log(L) / hidden_size))
            encoding[:,1::2] += torch.cos(steps[:,0::2] * torch.exp(-(emb_i[:,1::2]-1) * torch.log(L) / hidden_size))
            encoding = encoding * (hidden_size ** 0.5)
            #dprint(encoding[:5,:5])
            if remember:
                #self.cache[embed_size, step] = encoding
                self.cache[hidden_size, step] = encoding
        #return encoding[None,0:seq_len,0:embed_size].to(self.device, self.dtype)
        return encoding[None,0:seq_len,0:hidden_size].to(self.device, self.dtype)

class ModuleConnection(modeling.Module):
    def __init__(self, layer_size, **params):
        params = self.get_parameters(**params)
        self.dropout_ratio = params['dropout_ratio']
        super().__init__()
        init_gamma = params.get('init_gamma')
        self.pre  = params['sublayer_preprocess']
        self.post = params['sublayer_postprocess']
        self.mod_norm = nn.LayerNorm(layer_size)

    @classmethod
    def get_parameters(cls, **params):
        params.setdefault('dropout_ratio', 0.1)
        if params.get('num_layers'):
            num_layers = params['num_layers']
            init_gamma = 1.0 * (num_layers ** -0.5)
            params.setdefault('init_gamma', init_gamma)
        ### operation order, 'd' -> dropout, 'a' -> residual connection, 'n' -> normalization
        ### original paper's setting
        ### "Attention is All You Need" [Vaswani et al., NIPS17]
        params.setdefault('sublayer_preprocess',  '')
        params.setdefault('sublayer_postprocess', 'dan')
        ### better setting following reference implementation
        ### https://github.com/tensorflow/tensor2tensor/blob/v1.6.5/tensor2tensor/layers/common_hparams.py#L110-L112
        ### https://github.com/tensorflow/tensor2tensor/blob/v1.6.5/tensor2tensor/models/transformer.py#L1133-L1134
        #params.setdefault('sublayer_preprocess',  'n')
        #params.setdefault('sublayer_postprocess', 'da')
        return params

    def forward(self, seq, sublayer):
        input_seq = seq
        #dprint("--")
        #dprint(input_seq[0,:,0].data.tolist(),)
        for char in self.pre:
            if char == 'd':
                seq = torch.dropout(seq, self.dropout_ratio, self.training)
            if char == 'n':
                #seq = feed_seq(self.normalize, seq)
                seq = self.mod_norm(seq)
        #dprint(seq[0,:,0].data.tolist(),)
        sub = sublayer(seq)
        #dprint(sub[0,:,0].data.tolist(),)
        for char in self.post:
            if char == 'a':
                #sub = seq + sub
                sub = input_seq + sub
            if char == 'd':
                sub = torch.dropout(sub, self.dropout_ratio, self.training)
            if char == 'n':
                #sub = feed_seq(self.normalize, sub)
                sub = self.mod_norm(sub)
        #dprint(sub[0,:,0].data.tolist(),)
        return sub

class Transformer(modeling.Module):
    def __init__(self, conditioned=False, **params):
        super().__init__()
        # parameters
        params = self.get_config(**params)
        #self.embed_size    = params['embed_size']
        self.hidden_size    = params['hidden_size']
        self.dropout_ratio = params['dropout_ratio']
        self.conditioned = conditioned
        # modules
        self.mod_self_attention = MultiHeadAttention(**params)
        #self.mod_add_self_attention = ModuleConnection(self.embed_size, **params)
        self.mod_add_self_attention = ModuleConnection(self.hidden_size, **params)
        if self.conditioned:
            self.mod_memory_attention     = MultiHeadAttention(**params)
            #self.mod_add_memory_attention = ModuleConnection(self.embed_size, **params)
            self.mod_add_memory_attention = ModuleConnection(self.hidden_size, **params)
        self.mod_feed_forward   = FeedForward(**params)
        #self.mod_add_feed_forward   = ModuleConnection(self.embed_size, **params)
        self.mod_add_feed_forward   = ModuleConnection(self.hidden_size, **params)
        self.reset_state()

    @classmethod
    def get_config(cls, **params):
        #params.setdefault('embed_size', 512)
        params.setdefault('hidden_size', 512)
        params.setdefault('dropout_ratio', 0.1)
        #params['memory_size'] = params['hidden_size']
        params.setdefault('memory_size', params['hidden_size'])
        params = MultiHeadAttention.get_config(**params)
        params = FeedForward.get_config(**params)
        params = ModuleConnection.get_parameters(**params)
        return params

    def prepare_features(self, seq=None, **features):
        #dprint(y)
        prev_seq = features.get('prev_seq')
        if seq is not None:
            if seq.dim() == 2:
                features['id_seq'] = seq # (B, L)
        return features

    def forward(self, seq_input, memory=None, mask_self=None, mask_combine=None, step=None, **features):
        seq = seq_input
        #features = self.prepare_features(seq=seq_input, step=step, **features)
        features = self.prepare_features(seq=seq_input, **features)
        last_input = self.last_state.get('input')
        if last_input is None:
            #dprint(last_input)
            all_input = seq_input
            offset = 0
        else:
            #dprint(last_input.shape)
            #dprint(seq_input.shape)
            #self.last_input, seq_input = torch.broadcast_tensors(self.last_input, seq_input)
            #all_input = torch.cat([self.last_input, seq_input], dim=1) # (B, Len_prev+Len_new, H)
            all_input = torch.cat([last_input, seq_input], dim=1) # (B, Len_prev+Len_new, H)
            #offset = self.last_input.shape[1]
            offset = last_input.shape[1] # (Len_prev)
        #if self.combine:
        ##if last_input is not None:
        #    if step == 1:
        #        dprint("--")
        #    dprint(step)
        #    dprint(all_input.shape)
        #    dprint(all_input[0,:,0:5],)
        #    dprint(seq.shape)
        #    dprint(seq[0,:,0:5],)
        #    #dprint(mask_self.shape)
        #    #dprint(mask_self[:5])
        seq = self.mod_add_self_attention(seq, lambda seq: self.mod_self_attention(seq, all_input, all_input, mask_self, offset))
        #self.status['self_attention_weight'] = self.mod_self_attention.last_weight
        #self.last_state['self_attention_weight'] = self.mod_self_attention.last_weight
        #dprint(self.last_self_attention_weight[0,:,:].data,)
        #dprint(F.argmax(self.last_self_attention_weight[0,:,:],axis=1).data,)
        if self.conditioned and memory is not None:
            #dprint(memory.shape)
            #dprint(mask_combine)
            seq = self.mod_add_memory_attention(seq, lambda seq: self.mod_memory_attention(seq, memory, memory, mask_combine, offset))
            #self.last_memory_attention_weight = self.mod_memory_attention.last_weight
            #self.status['memory_attention_weight'] = self.mod_memory_attention.last_weight
            #dprint(self.last_memory_attention_weight[0,:,:].data,)
            #dprint(F.argmax(self.last_memory_attention_weight[0,:,:],axis=1).data,)
        seq = self.mod_add_feed_forward(seq, self.mod_feed_forward)
        #self.last_input = all_input
        self.last_state['input'] = all_input
        last_output = self.last_state.get('output')
        #if self.last_output is None:
        if last_output is None:
            #self.last_output = seq
            self.last_state['output'] = seq
        else:
            #self.last_output = F.concat([self.last_output, seq], axis=1) # (B, Len_prev+Len_new, H)
            #self.last_output = torch.cat([self.last_output, seq], dim=1) # (B, Len_prev+Len_new, H)
            self.last_state['output'] = torch.cat([last_output, seq], dim=1) # (B, Len_prev+Len_new, H)
        #if not chainer.config.train:
        #    self.cache_features(step=step, **features)
        #if self.combine:
        #    dprint(self.last_state['output'][0,:,0:5],)
        return seq

    def extra_repr(self):
        return f"conditioned = {self.conditioned}"

    def get_state(self):
        #status = {}
        #status['last_input']  = self.last_input
        #status['last_output'] = self.last_output
        #return status
        self.last_state['self_attention_state'] = self.mod_self_attention.get_state()
        if self.conditioned:
            self.last_state['memory_attention_state'] = self.mod_memory_attention.get_state()
        return self.last_state

    def reset_state(self):
        self.last_state = {}
        self.mod_self_attention.reset_state()
        if self.conditioned:
            self.mod_memory_attention.reset_state()
        return self

    def set_state(self, state):
        if state is None:
            return self.reset_state()
        self.last_state = state
        return self

class MultiStepTransformer(modeling.Module):
    #def __init__(self, **hparams):
    #def __init__(self, combine=False, **hparams):
    def __init__(self, conditioned=False, **hparams):
        super().__init__()
        # parameters
        hparams = self.get_config(**hparams)
        self.num_layers      = hparams['num_layers']
        self.embed_positions = hparams['embed_positions']
        self.dropout_ratio   = hparams['dropout_ratio']
        #self.embed_size      = hparams['embed_size']
        self.hidden_size      = hparams['hidden_size']
        self.pre  = hparams.get('sublayer_preprocess')
        self.post = hparams.get('sublayer_postprocess')
        # modules
        #self.mods_transform = nn.ModuleList()
        #for i in range(self.num_layers):
        #    #self.mods_transform.append(Transformer(**hparams))
        #    self.mods_transform.append(Transformer(combine, **hparams))
        self.mods_transform = modeling.ModuleArray(Transformer(conditioned, **hparams), self.num_layers)
        if self.pre and 'n' not in self.pre:
            #self.mod_norm_input  = nn.LayerNorm(self.embed_size)
            self.mod_norm_input  = nn.LayerNorm(self.hidden_size)
        if self.post and 'n' not in self.post:
            #self.mod_norm_output = nn.LayerNorm(self.embed_size)
            self.mod_norm_output = nn.LayerNorm(self.hidden_size)
        if self.embed_positions:
            mod_embed_pos  = hparams.get('shared_embed_pos')
            if mod_embed_pos is None:
                self.mod_embed_pos = EmbedPosition(**hparams)
            else:
                self.mod_embed_pos = mod_embed_pos
        else:
            mod_encode_pos = hparams.get('shared_encode_pos')
            # positional encoding (sinusoidal)
            if mod_encode_pos is None:
                self.mod_encode_pos = PositionalEncoder()
            else:
                self.mod_encode_pos = mod_encode_pos
        #self.mod_dropout = nn.Dropout(self.dropout_ratio)

    @classmethod
    def get_config(cls, **params):
        params.setdefault('num_layers', 1)
        params.setdefault('embed_positions', False)
        params.setdefault('embed_size', 512)
        params.setdefault('hidden_size', params['embed_size'])
        params.setdefault('dropout_ratio', 0.1)
        params = Transformer.get_config(**params)
        return params

    def add_positional_encoding(self, seq, **features):
        device = seq.device
        #batch_size, len_seq, embed_size = seq.shape
        batch_size, len_seq, hidden_size = seq.shape
        if self.embed_positions:
            if 'prev_seq' in features:
                start = features['prev_seq'].shape[1]
            else:
                start = 0
            #pos_emb = self.mod_embed_pos(batch_size, len_seq, start)
            pos_emb = self.mod_embed_pos(len_seq, start)
            seq = seq + pos_emb
            #dprint(start)
            #dprint(pos_emb[0,-1,0:5],)
        else:
            # Positional encoding (sinusoidal) / embedding
            if 'all_seq' in features:
                #dprint(features['all_seq']
                #shape = features['all_seq'].shape + (embed_size,)
                shape = features['all_seq'].shape + (hidden_size,)
                all_pos_enc = self.mod_encode_pos(shape)
                #all_pos_enc = self.mod_encode_pos(shape).to(device)
                pos_enc = all_pos_enc[:,-len_seq:None,:]
                #dprint(pos_enc[0,:,0:4])
            else:
                pos_enc = self.mod_encode_pos(seq.shape)
                #pos_enc = self.mod_encode_pos(seq.shape).to(device)
            seq = seq + pos_enc
        if hasattr(self, 'normalize_input'):
            seq = self.mod_norm_input(seq)
        #seq = F.dropout(seq, ratio=self.dropout_ratio)
        #seq = self.mod_dropout(seq)
        seq = torch.dropout(seq, self.dropout_ratio, self.training)
        #dprint(seq[0,:,0])
        return seq

    def forward(self, seq_input, memory=None, mask_self=None, mask_combine=None, **features):
        #batch_size, len_seq, embed_size = seq_input.shape
        seq = seq_input
        if features.get('max_steps'):
            max_steps = min(features['max_steps'], self.num_layers)
        else:
            max_steps = self.num_layers
        seq = self.add_positional_encoding(seq, **features)
        #output_list = []
        for i in range(max_steps):
            #transform = getattr(self, 'transform'+str(i+1))
            mod_trans = self.mods_transform[i]
            #seq = transform(seq, memory=memory, mask_self=mask_self, mask_combine=mask_combine, step=i+1, **features)
            seq = mod_trans(seq, memory=memory, mask_self=mask_self, mask_combine=mask_combine, step=i+1, **features)
            #output_list.append(seq)
        if hasattr(self, 'normalize_output'):
            #logger.debug("--normalize output--")
            #dprint(seq[0,:5,0],)
            seq = self.mod_norm_output(seq)
            #dprint(seq[0,:5,0],)
        #self.status['output_list'] = output_list
        return seq
        #return seq, features

    def get_state(self):
        list_state = []
        for i in range(self.num_layers):
            list_state.append(self.mods_transform[i].get_state())
        return list_state

    def reset_state(self):
        #self.status = {}
        #self.last_state = {}
        for i in range(self.num_layers):
            #getattr(self, 'transform'+str(i+1)).reset_state()
            self.mods_transform[i].reset_state()
        return self

    def set_state(self, state):
        if state is None:
            return self.reset_state()
        for i, layer_state in enumerate(state):
            #self.mod_trans[i].set_state(state)
            #self.mods_trans[i].set_state(state)
            self.mods_transform[i].set_state(state[i])

class Encoder(modeling.Module):
    #def __init__(self, vocab, **params):
    def __init__(self, idmaps, **params):
        params = Encoder.get_config(**params)
        vocab = idmaps['x']
        self.idmaps = idmaps
        self.padding = vocab.pad
        self.vocab_size      = params['vocab_size']
        self.embed_size      = params['embed_size']
        self.hidden_size     = params['hidden_size']
        self.dropout_ratio   = params['dropout_ratio']
        self.universal       = params['universal']
        super().__init__()
        mod_embed_tok  = params.get('shared_embed_tok')
        #params['combine'] = False
        params['conditioned'] = False
        if self.universal:
            self.mod_transform = UniversalTransformer(**params)
        else:
            self.mod_transform = MultiStepTransformer(**params)
        if mod_embed_tok is None:
            #self.mod_embed_tok = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=self.padding)
            self.mod_embed_tok = embeddings.Embedding(self.vocab_size, self.embed_size, padding_idx=self.padding)
        else:
            self.mod_embed_tok = mod_embed_tok
        if self.embed_size != self.hidden_size:
            self.mod_embed2hidden = nn.Linear(self.embed_size, self.hidden_size, bias=False)
        self.reset_state()

    @classmethod
    def get_config(cls, **params):
        params.setdefault('embed_size', 512)
        params.setdefault('hidden_size', params['embed_size'])
        params.setdefault('dropout_ratio', 0.1)
        params.setdefault('universal', False)
        if params['universal']:
            params = UniversalTransformer.get_config(**params)
        else:
            if params.get('num_encoder_layers') is None:
                params.setdefault('num_layers', 1)
            else:
                params['num_layers'] = params['num_encoder_layers']
            params = MultiStepTransformer.get_config(**params)
        #dprint(params)
        return params

    #def prepare_features(self, x=None, **features):
    def prepare_features(self, seq=None, **features):
        #if x is not None:
        if seq is not None:
            #if x.dim() == 2:
            if seq.dim() == 2:
                #features['x_id_seq'] = x
                features['mem_id_seq'] = seq
                #features['mask_seq'] = (x.data != self.padding)
                #features['mask_mem'] = (x.data != self.padding)
                features['mask_mem'] = (seq.data != self.padding)
                #features['mask_self'] = make_attention_mask(x, x, self.padding)
                features['mask_self'] = make_attention_mask(seq, seq, self.padding)
        return features

    #def forward(self, x_seq, **features):
    def forward(self, seq, **features):
        #features = self.prepare_features(x=x_seq, **features)
        features = self.prepare_features(seq=seq, **features)
        #batch_size, len_x = x_seq.shape
        batch_size, len_seq = seq.shape
        embed_size = self.embed_size
        hidden_size = self.hidden_size
        #x_seq_emb = self.mod_embed_tok(x_seq) * (embed_size ** 0.5)
        #seq_emb = self.mod_embed_tok(seq) * (embed_size ** 0.5)
        seq_emb = self.mod_embed_tok(seq)
        #h = self.mod_transform(seq_input=x_seq_emb, **features)
        h = self.mod_transform(seq_input=seq_emb, **features)
        return h

    def get_state(self):
        self.last_state['transformer_state'] = self.mod_transform.get_state()
        return self.last_state

    def reset_state(self):
        self.last_state = {}
        self.mod_transform.reset_state()

class Decoder(modeling.Module):
    def __init__(self, idmaps, **hparams):
        hparams = Decoder.get_config(**hparams)
        vocab = idmaps['x']
        self.idmaps = idmaps
        self.padding = vocab.pad
        self.vocab_size      = hparams['vocab_size']
        self.embed_size      = hparams['embed_size']
        self.hidden_size     = hparams['hidden_size']
        self.memory_size     = hparams['memory_size']
        self.dropout_ratio   = hparams['dropout_ratio']
        self.share_embedding = hparams['share_embedding']
        self.universal       = hparams['universal']
        super().__init__()
        mod_embed_tok  = hparams.get('shared_embed_tok')
        hparams['conditioned'] = True
        if self.universal:
            self.mod_transform = UniversalTransformer(**hparams)
        else:
            self.mod_transform = MultiStepTransformer(**hparams)
        if mod_embed_tok is None:
            #self.mod_embed_tok = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=self.padding)
            self.mod_embed_tok = embeddings.Embedding(self.vocab_size, self.embed_size, padding_idx=self.padding)
        else:
            self.mod_embed_tok = mod_embed_tok
        if self.embed_size != self.hidden_size:
            self.mod_embed2hidden = nn.Linear(self.embed_size, self.hidden_size, bias=False)
        if self.hidden_size != self.memory_size:
            self.mod_memory2hidden = nn.Linear(self.memory_size, self.hidden_size, bias=False)
        if self.share_embedding:
            if self.embed_size != self.hidden_size:
                self.mod_hidden2embed = nn.Linear(self.hidden_size, self.embed_size, bias=False)
            self.mod_generate = nn.Linear(self.embed_size, self.vocab_size, bias=False)
            self.mod_generate.weight = self.mod_embed_tok.weight
        else:
            self.mod_generate = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    @classmethod
    def get_config(cls, **params):
        params.setdefault('embed_size', 512)
        params.setdefault('hidden_size', params['embed_size'])
        if params['embed_size'] > params['hidden_size']:
            params['embed_size'] = params['hidden_size']
        params.setdefault('memory_size', params['hidden_size'])
        params.setdefault('dropout_ratio', 0.1)
        params.setdefault('share_embedding', True)
        params.setdefault('universal', False)
        if params['universal']:
            params = UniversalTransformer.get_config(**params)
        else:
            if params.get('num_decoder_layers') is None:
                params.setdefault('num_layers', 1)
            else:
                params['num_layers'] = params['num_decoder_layers']
            params = MultiStepTransformer.get_config(**params)
        return params

    def init_weights(self):
        if self.mod_generate.weight is not self.mod_embed_tok.weight:
            nn.init.orthogonal_(self.mod_generate.weight)

    def prepare_features(self, seq=None, **features):
        mem = features.get('mem_id_seq')
        prev_seq = features.get('prev_seq')
        if seq is not None:
            if seq.dim() == 2:
                features['id_seq'] = seq
                if prev_seq is None:
                    mask_self  = make_attention_mask(seq, seq, self.padding)
                    mask_self *= make_history_mask(seq)
                    features['mask_self'] = mask_self
                else:
                    all_seq = torch.cat([prev_seq, seq], dim=1)
                    mask_self  = make_attention_mask(seq, all_seq, self.padding)
                    features['mask_self'] = mask_self
                    features['all_seq'] = all_seq
        if all(val is not None for val in [seq, mem]):
            features['mask_combine'] = make_attention_mask(seq, mem, self.padding)
        return features

    def forward(self, seq, memory, **features):
        #features = self.prepare_features(y=y_seq, **features)
        if hasattr(self, 'mod_memory2hidden'):
            memory = self.mod_memory2hidden(memory)
        features = self.prepare_features(seq=seq, **features)
        embed_size = self.embed_size
        #seq_emb = self.mod_embed_tok(seq) * (embed_size ** 0.5)
        seq_emb = self.mod_embed_tok(seq)
        if hasattr(self, 'mod_embed2hidden'):
            seq_emb = self.mod_embed2hidden(seq_emb)
        h_dec = self.mod_transform(seq_input=seq_emb, memory=memory, **features)
        if hasattr(self, 'mod_hidden2embed'):
            h_dec = self.mod_hidden2embed(h_dec) # (B, L, H) -> (B, L, E)
        logits = self.mod_generate(h_dec) # (B, L, V)
        logits = logits.transpose(1, 2) # (B, V, L) for loss calculation
        return logits

    #def decode_one(self, y, memory, **features):
    def decode_one(self, seq_inc, memory, **features):
        if seq_inc.dim() == 1:
            # assuming (B,)
            seq_inc = seq_inc[:,None] # (B, 1)
        logits = self(seq_inc, memory, **features) # (B, V, L)
        return logits[:,:,-1] # (B, V)

    def get_state(self):
        self.last_state['transformer_state'] = self.mod_transform.get_state()
        return self.last_state

    def reset_state(self):
        self.last_state = {}
        self.mod_transform.reset_state()

    def set_state(self, state):
        if state is None:
            return self.reset_state()
        self.mod_transform.set_state(state['transformer_state'])
