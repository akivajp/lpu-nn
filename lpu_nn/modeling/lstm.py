#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 3rd
import torch
from torch import nn

# local
from lpu.common import logging
from lpu_nn.common.utils import format_state
from lpu_nn import modeling

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

class StatefulLSTM(modeling.Module):
    def __init__(self, in_size, out_size):
        super(StatefulLSTM,self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.mod_lstm = nn.LSTM(in_size, out_size, batch_first=True)
        self.reset_state()

    def forward(self, x, mask=None):
        if x.dim() == 2:
            # assuming (B, E)
            x = x[:,None,:] # (B, 1, E)
            if mask is not None:
                if mask.dim() == 1:
                    mask = mask[:,None] # (B, 1)
        assert x.dim() == 3
        if mask is None:
            h, self.last_state = self.mod_lstm(x, self.last_state)
            return h
        else:
            batch_size = x.shape[0]
            list_x = x.split(1, dim=1) # List[B, 1, E]
            list_mask = mask.split(1, dim=1) # List[B, 1]
            list_h = []
            if self.last_state is None:
                zeros = torch.zeros(1, batch_size, self.out_size).to(x.device, x.dtype)
                self.last_state = (zeros, zeros)
            for elem_x, elem_mask in zip(list_x, list_mask):
                _, (h, c) = self.mod_lstm(elem_x, self.last_state) # Tuple[1, B, H]
                elem_mask = elem_mask.transpose(0,1).unsqueeze(2) # (1, B, 1)
                #dprint(elem_mask.flatten())
                h = torch.where(elem_mask, h, self.last_state[0])
                c = torch.where(elem_mask, c, self.last_state[1])
                self.last_state = (h, c)
                list_h.append(h.transpose(0,1)) # List[B, 1, H]
            return torch.cat(list_h, dim=1) # (B, L, H)

    def get_state(self):
        #dprint(self.last_state)
        #dprint(format_state(self.last_state))
        if self.last_state is None:
            return None
        else:
            return dict(
                h = self.last_state[0][0], # (B, H)
                c = self.last_state[1][0], # (B, H)
            )

    def reset_state(self):
        self.last_state = None

    def set_state(self, state):
        h = state['h'][None] # (1, B, H)
        c = state['c'][None] # (1, B, H)
        self.last_state = (h, c)

class MultiLayerLSTM(nn.Module):
    #def __init__(self, **params):
    def __init__(self, in_size, out_size, **params):
        params = MultiLayerLSTM.get_config(**params)
        #self.in_size       = params['in_size']
        #self.out_size      = params['out_size']
        self.in_size       = in_size
        self.out_size      = out_size
        self.num_layers    = params['num_layers']
        self.dropout_ratio = params['dropout_ratio']
        self.residual      = params['residual_connection']
        self.normalize     = params['normalize']
        super(MultiLayerLSTM,self).__init__()
        mods_lstm = []
        for i in range(self.num_layers):
            if i == 0:
                mods_lstm.append(StatefulLSTM(self.in_size, self.out_size))
            else:
                mods_lstm.append(StatefulLSTM(self.out_size, self.out_size))
        self.mods_lstm = nn.ModuleList(mods_lstm)
        #self.mods_lstm = models.ModuleArray(mods_lstm)
        if self.normalize:
            mods_norm = []
            for i in range(0, self.num_layers-1):
                mods_norm.append(nn.LayerNorm(self.out_size))
            self.mods_norm = nn.ModuleList(mods_norm)
        #self.mod_dropout = nn.Dropout(self.dropout_ratio)

    @staticmethod
    def get_config(**params):
        params.setdefault('dropout_ratio', 0.1)
        params.setdefault('num_layers', 1)
        if params['num_layers'] >= 2:
            params.setdefault('normalize', True)
            params.setdefault('residual_connection', True)
        else:
            params.setdefault('normalize', False)
            params.setdefault('residual_connection', False)
        return params

    def forward(self, x, mask=None):
        #h = self.mods_lstm[0](x)
        h = self.mods_lstm[0](x, mask)
        for i in range(1, self.num_layers):
            lstm = self.mods_lstm[i]
            if self.residual:
                #dprint("RESIDUAL!")
                #h = lstm(h) + h
                h = lstm(h, mask) + h
            else:
                #h = lstm(h)
                h = lstm(h, mask)
            if self.normalize:
                normalize = self.mods_norm[i-1]
                h = normalize(h)
            h = torch.dropout(h, self.dropout_ratio, self.training)
        return h

    def get_state(self):
        state_list = []
        for i in range(self.num_layers):
            state_list.append(self.mods_lstm[i].get_state())
        return state_list

    def reset_state(self):
        for i in range(self.num_layers):
            self.mods_lstm[i].reset_state()

    def set_state(self, state):
        for i, lstm_state in enumerate(state):
            self.mods_lstm[i].set_state(lstm_state)

