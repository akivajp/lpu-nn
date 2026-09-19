#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# system
import math

# 3rd
import torch
from torch import nn

# local
from lpu.common import logging
from lpu_nn.common.utils import purge_tensor
from lpu_nn import modeling
from lpu_nn.modeling.activation import get_activator
from lpu_nn.modeling.activation import get_gain

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

def get_attention(name, **params):
    name = name.lower()
    if name == 'dot':
        return DotAttention(**params)
    if name == 'concat':
        return ConcatAttention(**params)
    if name == 'general':
        return GeneralAttention(**params)
    if name == 'mlp':
        return MLPAttention(**params)
    if name == 'none':
        return NoAttention(**params)
    raise ValueError("unknown name for attention layer: {}".format(name))

class NoAttention(modeling.Module):
    def __init__(self, **params):
        super(NoAttention,self).__init__()
    #def forward(self, memory, h_dec, **features):
    def forward(self, h_dec, memory, **features):
        return memory[:,-1]

class AttentionBase(modeling.Module):
    def __init__(self, **params):
        super(AttentionBase,self).__init__()
        self.hidden_size = params.get('hidden_size')
        self.local_attention = params.get('local_attention')
        self.dropout_ratio = params.get('dropout_ratio', 0.1)
        if self.local_attention:
            self.window_size = params.get('window_size', 5)
            self.mod_focus = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
                nn.Dropout(self.dropout_ratio),
                nn.Linear(self.hidden_size, 1),
                nn.Sigmoid(),
            )

    def init_weights(self):
        if self.local_attention:
            nn.init.orthogonal_(self.mod_focus[0].weight, gain=get_gain('tanh'))
            self.mod_focus[0].bias.data.zero_()
            nn.init.orthogonal_(self.mod_focus[3].weight, gain=get_gain('sigmoid'))
            self.mod_focus[3].bias.data.zero_()

    def score(self, memory, h_dec):
        raise NotImplementedError

    def get_local_attention(self, attention, memory_mask, h_dec):
        device = attention.device
        #window_size = 1
        #window_size = 2
        #window_size = 5
        window_size = self.window_size
        eps = 1e-7
        sigma = self.window_size / 2.0
        batch_size, len_mem = memory_mask.shape
        batch_len = memory_mask.sum(dim=1)[:,None].float() # (B,1)
        #pos = torch.arange(len_mem)[None,:].repeat(batch_size, axis=0) # (B, L)
        pos = torch.arange(len_mem)[None,:].float().to(device) # (1, L)
        focus = self.mod_focus(h_dec) # (B,H) -> (B,1)
        focus = batch_len * focus
        local_weight = torch.exp( - (pos - focus)**2 / (2 * sigma ** 2) ) # (B, L)
        local_mask = (focus - window_size <= pos) & (pos <= focus + window_size)
        #local_weight = purge_variables(local_weight, local_mask, 0)
        attention = attention * local_weight[:,:,None] # (B, L, 1)
        attention = purge_tensor(attention, local_mask, 0)
        # rescaling
        attention = attention / (torch.sum(attention, dim=1)[:,None,:] + eps)
        return attention

    #def forward(self, memory, h_dec, **features):
    def forward(self, h_dec, memory, **features):
        # memory : (B, L, H) or (B, L, H*2)
        # h_dec : (B, H)
        mask_mem = features['mask_mem'] # mandatory
        scores = self.score(memory, h_dec)
        if mask_mem is not None:
            scores = purge_tensor(scores, mask_mem, -math.inf)
        attention = torch.softmax(scores, dim=1) # (B, L, 1)
        attention = purge_tensor(attention, mask_mem, 0.0)
        if self.local_attention:
            attention = self.get_local_attention(attention, mask_mem, h_dec)
        context = torch.matmul(memory.transpose(1,2), attention) # (B, H, L) * (B, L, 1) -> (B,H,1)
        return context.squeeze(2) # (B,H)

class DotAttention(AttentionBase):
    def __init__(self, **params):
        super(DotAttention,self).__init__(**params)
        self.bidirectional = params.get('bidirectional_encoder')

    def score(self, memory, h_dec):
        # (B, L, H) * (B, H, 1) => (B, L, 1)
        return torch.matmul(memory, h_dec[:,:,None]) # (B, L, 1)

    #def forward(self, memory, h_dec, **features):
    def forward(self, h_dec, memory, **features):
        #memory_mask = features['mask_x']
        if self.bidirectional:
            # (B, L, H*2) -> (B, L, H)
            memory = torch.mean(torch.stack(memory.split(2, dim=2), dim=2), dim=3)
        #return super(DotAttention,self).forward(memory, h_dec, **features)
        return super(DotAttention,self).forward(h_dec, memory, **features)

class ConcatAttention(AttentionBase):
    def __init__(self, **params):
        self.bidirectional = params.get('bidirectional_encoder')
        self.hidden_size = params.get('hidden_size')
        super(ConcatAttention,self).__init__(**params)
        if self.bidirectional:
            self.mod_linear = nn.Linear(self.hidden_size*3, 1)
        else:
            self.mod_linear = nn.Linear(self.hidden_size*2, 1)

    def init_weights(self):
        super(ConcatAttention,self).init_weights()
        nn.init.orthogonal_(self.mod_linear.weight)
        self.mod_linear.bias.data.zero_()

    def score(self, memory, h_dec):
        # memory : (B, L, H1)
        # h_dec  : (B, H2)
        batch_size, len_mem, hidden_size1 = memory.shape
        h_dec = h_dec[:,None,:].repeat(1, len_mem, 1) # (B, L, H2)
        concat = torch.cat([memory, h_dec], dim=2) # (B, L, H1+H2)
        return self.mod_linear(concat) # (B, L, 1)

class GeneralAttention(AttentionBase):
    def __init__(self, **params):
        self.bidirectional = params.get('bidirectional_encoder')
        self.hidden_size = params.get('hidden_size')
        super(GeneralAttention,self).__init__(**params)
        if self.bidirectional:
            self.mod_linear = nn.Linear(self.hidden_size*2, self.hidden_size)
        else:
            self.mod_linear = nn.Linear(self.hidden_size, self.hidden_size)

    def init_weights(self):
        super(GeneralAttention,self).init_weights()
        nn.init.orthogonal_(self.mod_linear.weight)
        self.mod_linear.bias.data.zero_()

    def score(self, memory, h_dec):
        # (B, nL, H) -> (B, L, H)
        prod = self.mod_linear(memory)
        # (B, L, H) * (B, H, 1) -> (B, L, 1)
        return torch.matmul(prod, h_dec[:,:,None])

class MLPAttention(AttentionBase):
    def __init__(self, **params):
        self.activation = params.get('activation')
        self.bidirectional = params.get('bidirectional_encoder')
        self.hidden_size = params.get('hidden_size')
        super(MLPAttention,self).__init__(**params)
        if self.bidirectional:
            linear1 = nn.Linear(self.hidden_size*3, self.hidden_size)
        else:
            linear1 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.mod_score = nn.Sequential(
            linear1,
            get_activator(self.activation, self.hidden_size),
            nn.Linear(self.hidden_size, 1),
        )

    def init_weights(self):
        super(MLPAttention,self).init_weights()
        nn.init.orthogonal_(self.mod_score[0].weight, gain=get_gain('relu'))
        self.mod_score[0].bias.data.zero_()
        nn.init.orthogonal_(self.mod_score[2].weight)
        self.mod_score[2].bias.data.zero_()

    def score(self, memory, h_dec):
        # memory : (B, L, H)
        # h_dec  : (B, H)
        #h_dec = h_dec[:,None,:].expand_as(memory) # (B, L, H)
        batch_size, len_mem, hidden_size1 = memory.shape
        h_dec = h_dec[:,None,:].repeat(1, len_mem, 1) # (B, L, H2)
        concat = torch.cat([memory, h_dec], dim=2) # (B, L, H1+H2)
        return self.mod_score(concat) # (B, L, 1)

