#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 3rd
import torch
from torch import nn

# local
from lpu.common import logging
from lpu_nn.common.utils import format_state
from lpu_nn import modeling
from lpu_nn.modeling import transformer

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

class UniversalTransformer(modeling.Module):
    def __init__(self, **params):
        super(UniversalTransformer, self).__init__()
        # parameters
        params = self.get_config(**params)
        self.act_output         = params['act_output']
        self.dropout_ratio      = params['dropout_ratio']
        self.max_steps          = params['max_steps']
        self.recurrence         = params['recurrence']
        #self.embed_size         = params['embed_size']
        self.hidden_size        = params['hidden_size']
        self.continue_threshold = params.get('continue_threshold')
        self.post = params.get('sublayer_postprocess')
        # modules
        mod_encode_pos = params.get('shared_encode_pos')
        self.mod_transform = transformer.Transformer(**params)
        self.mod_halt = nn.Sequential(
            #nn.Linear(self.embed_size, 1),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        )
        if self.post and 'n' not in self.post:
            #self.mod_norm_output = nn.LayerNorm(self.embed_size)
            self.mod_norm_output = nn.LayerNorm(self.hidden_size)
        if mod_encode_pos is None:
            self.mod_encode_pos = transformer.PositionalEncoder()
        else:
            self.mod_encode_pos = mod_encode_pos

    @classmethod
    def get_config(cls, **params):
        params.setdefault('embed_size', 512)
        params.setdefault('hidden_size', params['embed_size'])
        params.setdefault('act_output', 'accum')
        params.setdefault('dropout_ratio', 0.1)
        params.setdefault('recurrence', 'act')
        params.setdefault('max_steps', 12)
        #params.setdefault('sublayer_preprocess', '')
        #params.setdefault('sublayer_postprocess', 'dan')
        if params['recurrence'] in ['act', 'act-prob', 'act-accum']:
            params.setdefault('continue_threshold', 0.01)
        params = transformer.Transformer.get_config(**params)
        return params

    def init_weights(self):
        nn.init.orthogonal_(self.mod_halt[0].weight)
        self.mod_halt[0].bias.data = torch.tensor([1.0])

    def add_positional_encoding(self, seq, step, **features):
        #batch_size, len_seq, embed_size = seq.shape
        batch_size, len_seq, hidden_size = seq.shape
        if 'all_seq' in features:
            #shape = features['all_seq'].shape + (self.embed_size,)
            shape = features['all_seq'].shape + (self.hidden_size,)
            all_pos_enc = self.mod_encode_pos(shape, step=step)
            pos_enc = all_pos_enc[:,-len_seq:None]
        else:
            pos_enc = self.mod_encode_pos(seq.shape, step=step)
        seq = seq + pos_enc
        seq = torch.dropout(seq, self.dropout_ratio, self.training)
        return seq

    def forward(self, seq_input, memory=None, mask_self=None, mask_combine=None, **features):
        device = self.device
        dtype  = self.dtype
        batch_size, seq_len, hidden_size = seq_input.shape
        seq = seq_input
        max_steps = features.get('max_steps')
        if max_steps is None:
            max_steps = self.max_steps
        new_state = []
        if self.recurrence.lower() in ['act', 'act-prob', 'act-accum']:
            work_len = seq_len
            #zeros = torch.zeros([batch_size, work_len]).to(device, dtype)
            mask_seq = mask_self[:,-work_len:None,0].reshape(batch_size, work_len)
            #step_seq = torch.zeros([batch_size, work_len]).to(device, dtype)
            step_seq = torch.zeros([batch_size, work_len]).to(device, torch.uint8)
            remain = mask_seq.to(dtype)
            b_running = mask_seq
            if self.act_output == "accum":
                #accum_seq = chainer.Variable(self.xp.zeros(seq.shape, dtype=ftype))
                accum_seq = torch.zeros(seq.shape).to(device, dtype)
            for i in range(max_steps):
                prev_seq = seq
                step_seq += b_running.to(torch.uint8)
                seq = self.add_positional_encoding(seq, step=i+1, **features)
                step_state = self.last_state.get(i)
                #if step_state is not None:
                #    dprint(i)
                #    dprint(format_state(step_state))
                self.mod_transform.set_state(step_state)
                seq = self.mod_transform(seq, memory=memory, mask_self=mask_self, mask_combine=mask_combine, step=i+1, **features)
                self.last_state[i] = self.mod_transform.get_state()
                seq = torch.where(b_running[:,:,None], seq, prev_seq)
                # (B, L, E) -> (B, L, 1)
                #halt = feed_seq(self.linear_halt, seq, dropout=self.dropout_ratio)
                halt_prob = self.mod_halt(seq).squeeze(2) # (B, L)
                # (B, L, 1) -> (B, L)
                #halt = halt.reshape(batch_size, work_len)
                #halt_prob = torch.sigmoid(halt)
                if self.recurrence in ['act', 'act-prob']:
                    halt_prob = remain * halt_prob
                if i == max_steps - 1:
                    # final step, all running elements should be halted
                    b_halting = b_running
                    b_running = b_running * False
                elif i == 0:
                    # not halting in the first step, otherwise not stable to learn to ponder
                    b_halting = torch.zeros([batch_size, work_len]).to(device, torch.uint8)
                else:
                    # halting if new remain is under threshold
                    b_halting = b_running & (remain - halt_prob < self.continue_threshold)
                    b_running = b_running & ~b_halting
                remain = torch.where(b_running, remain - halt_prob, remain)
                if self.act_output == "accum":
                    #weight = halt_prob * b_running + remain * b_halting
                    weight = halt_prob * b_running.to(dtype) + remain * b_halting.to(dtype)
                    # (B, L) -> (B, L, E)
                    #weight = broadcast_embed(weight, [batch_size,work_len,embed_size])
                    #accum_seq = F.where(weight.data > 0, accum_seq + seq * weight, accum_seq)
                    weight = weight[:,:,None]
                    #dprint(weight.shape)
                    #dprint(accum_seq.shape)
                    #dprint(seq.shape)
                    accum_seq = torch.where(weight > 0, accum_seq + seq * weight, accum_seq)
                    #dprint(accum_seq.shape)
                #if not self.xp.any(b_running):
                if not b_running.any():
                    break
            for j in range(i+1, max_steps):
                #dprint(i)
                #dprint(j)
                step_state = self.last_state.get(j)
                #dprint(format_state(step_state))
                if step_state is None:
                    step_state = {}
                    step_state['input'] = seq_input
                    step_state['output'] = seq
                    self.last_state[j] = step_state
                else:
                    step_state['input'] = torch.cat([step_state['input'], seq_input], dim=1) # (B,LenPrev+LenNew,H)
                    step_state['output'] = torch.cat([step_state['output'], seq], dim=1) # (B,LenPrev+LenNew,H)
                    #dprint(format_state(step_state))
            #ponder_cost = step_seq + remain
            ponder_cost = step_seq.to(dtype) + remain
            max_ponder = torch.max(step_seq, dim=1)
            self.last_state['ponder_cost'] = ponder_cost
            self.last_state['max_ponder']  = max_ponder
            #dprint(max_steps)
            #dprint(ponder_cost)
            #dprint(max_ponder)
            if self.act_output == "accum":
                seq = accum_seq
            else:
                pass # to return "seq"
        else:
            # straight-forward recurrent stacking
            for i in range(max_steps):
                #dprint(features)
                seq = self.add_positional_encoding(seq, step=i+1, **features)
                seq = self.transform(seq, memory=memory, mask_self=mask_self, mask_combine=mask_combine, step=i+1, **features)
        if hasattr(self, 'mod_norm_output'):
            #logger.debug("--normalize output--")
            #dprint(seq[0,:5,0],)
            #seq = feed_seq(self.mod_norm_output, seq)
            seq = self.mod_norm_output(seq)
            #dprint(seq[0,:5,0],)
        return seq

    def get_state(self):
        return self.last_state

    def reset_state(self):
        self.last_state = {}
        self.mod_transform.reset_state()
        return self

    def set_state(self, state):
        if state is None:
            return self.reset_state()
        self.last_state = state
