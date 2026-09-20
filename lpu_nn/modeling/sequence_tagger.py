#!/usr/bin/env python3

# system
from typing import Any, cast

# 3rd
import pandas as pd
import torch
from torch import nn

# local
from lpu.common import logging
from lpu_nn.common import criteria
from lpu_nn.common import utils
from lpu_nn import modeling
from lpu_nn.modeling import bert
from lpu_nn.modeling import transformer
from lpu_nn.modeling.encoder_decoder import LSTMEncoder

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

class CRF(modeling.Module):
    def __init__(self, idmap: Any, **params: Any) -> None:
        super().__init__()
        num_tags = len(idmap)
        self.num_tags = num_tags
        self.idmap = idmap
        self.padding = idmap.pad
        # modules
        #self.param_first = nn.Parameter(torch.zeros(num_tags))
        self.param_transition = nn.Parameter(torch.zeros(num_tags,num_tags))
        # reset_state() を呼ぶまで last_state が存在せず、構築直後の
        # loss() / decode() が AttributeError になっていた
        self.reset_state()

    #def init_weights(self):
    #    idmap = self.idmap
    #    #nn.init.orthogonal_(self.param_transition)
    #    INVALID_SCORE = -10000
    #    self.param_transition.data[idmap.pad,:] = INVALID_SCORE # any -> <pad>
    #    self.param_transition.data[:,idmap.eos] = INVALID_SCORE # <eos> -> any
    #    self.param_transition.data[idmap.pad,idmap.eos] = 0 # <eos> -> <pad>
    #    self.param_transition.data[:,idmap.pad] = INVALID_SCORE # <pad> -> any
    #    self.param_transition.data[idmap.pad,idmap.pad] = 0 # <pad> -> <pad>
    #    self.param_transition.data[idmap.bos,:] = INVALID_SCORE # any -> <bos>
    #    #self.param_first.data[:] = INVALID_SCORE
    #    #self.param_first.data[idmap.bos] = 0

    def loss(self, logits: torch.Tensor, t: torch.Tensor, reduction: str = 'mean',
             **features: Any) -> torch.Tensor:
        # logits.shape : (B, L, C)
        batch_size, _length, _num_tags = logits.shape
        list_tags = (tags.squeeze(1) for tags in t.split(1, dim=1)) # List[Batch]
        mask_x = features['mask_x']
        #list_tag_scores = []
        list_pred_scores = []
        #dprint("-----")
        #dprint(self.param_first)
        #dprint(self.param_transition)
        #dprint(self.param_first.softmax(0))
        #dprint(self.param_transition.softmax(0).t())
        #for i in range(length):
        # 最初の回で必ず上書きされるが、字句上の定義前参照を避けるために置く
        last_correct_scores = torch.zeros(batch_size, dtype=logits.dtype, device=logits.device)
        last_tags = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        total_scores = logits[:,0]
        for i, tags in enumerate(list_tags):
            #def DPRINT(val):
            #    if i < 3 or length - i < 3:
            #        dprint(val)
            #DPRINT(i)
            # パディング位置は系列のマスクで判別する。タグ側の値は
            # 系列語彙のパディング ID (SentencePiece では -1) であり、
            # タグ表の範囲外になるため、そのまま添字に使えない
            valid = mask_x[:,i] # (B,)
            emit_scores = logits[:,i] # (B,)
            #DPRINT(emit_scores)
            #DPRINT(emit_scores[0].softmax(0))
            if i == 0:
                #total_scores = self.param_first + emit_scores # (B, C)
                total_scores = emit_scores # (B, C)
                pred_scores = total_scores
                #last_correct_scores = pred_scores.gather(1, tags[:,None]).squeeze(1)
            else:
                b_trans = self.param_transition[None,:,:] # (B, C_next, C_prev)
                b_scores = total_scores[:,None,:] # (B, 1, C_prev)
                step_total_scores = (b_scores + b_trans).logsumexp(2) + emit_scores # (B, C_next)
                # パディング位置では前向きスコアを更新しない
                # (更新すると分配関数がパディング分まで積み上がる)
                total_scores = torch.where(valid[:,None], step_total_scores, total_scores)
                #total_scores = (b_scores + b_trans + emit_scores[:,:,None]).logsumexp(2) # (B, C_next)
                pred_scores = last_correct_scores[:,None] + self.param_transition.t()[last_tags] + emit_scores # (B, C_next)
                #pred_scores = last_correct_scores[:,None] + self.param_transition[last_tags] + emit_scores # (B, C_next)
                #pred_scores = torch.where(mask_x[:,i,None], pred_scores, last_pred_scores)
                #dprint(last_correct_scores.shape)
                #dprint(last_correct_scores[:].shape)
                #dprint(self.param_transition[tags,last_tags].shape,)
                #dprint(emit_scores.shape)
                #dprint(emit_scores[tags].shape)
                #emit_score = emit_scores.gather(1, tags[:,None]).squeeze(1)
                #last_correct_scores = last_correct_scores[:] + self.param_transition[tags,last_tags] + emit_score # (B,)
            #list_pred_scores.append(pred_scores)
            list_pred_scores.append(total_scores)
            #dprint(pred_scores.shape)
            #dprint(tags.shape)
            # 有効な添字へ丸めてから参照し、パディング位置では
            # 直前の値を保って寄与させない
            safe_tags = torch.where(valid, tags, torch.zeros_like(tags))
            step_correct_scores = pred_scores.gather(1, safe_tags[:,None]).squeeze(1)
            last_correct_scores = torch.where(valid, step_correct_scores, last_correct_scores)
            last_tags = torch.where(valid, tags, last_tags)
        self.last_state['logits'] = torch.stack(list_pred_scores, dim=1)
        # 正規化はパディングを除いた実長で行う
        # (固定長 length で割るとバッチ内の短い系列が過小評価される)
        normalizer: torch.Tensor = mask_x.sum(1).to(logits.dtype).clamp(min=1.0)
        forward_scores = total_scores.logsumexp(1) / normalizer # (B,)
        gold_scores = last_correct_scores / normalizer
        crf_loss = forward_scores - gold_scores
        self.last_state['gold_score'] = gold_scores
        if reduction == 'mean':
            crf_loss = crf_loss.mean(0)
        return crf_loss

    def decode(self, logits: torch.Tensor, **features: Any) -> torch.Tensor:
        idmap = self.idmap
        list_best_paths = []
        list_best_scores = []
        batch_size, length, _num_tags = logits.shape
        #dprint(self.param_first)
        #dprint(self.param_transition)
        #dprint(self.param_first.softmax(0))
        #dprint(self.param_transition.softmax(0))
        for i in range(logits.shape[1]):
            #def DPRINT(val):
            #    if i < 3 or length - i < 3:
            #        dprint(val,2)
            #def DPRINT(val): pass
            #DPRINT(i)
            emit_scores = logits[:,i]
            #DPRINT(emit_scores[0])
            #DPRINT(emit_scores[0].softmax(0))
            if i == 0:
                #best_scores = self.param_first + emit_scores # (B, C)
                best_scores = emit_scores # (B, C)
                #best_scores = self.param_first # (B, C)
                #DPRINT(best_scores)
            else:
                # (B, 1, C_prev) + (1, C_next, C_prev) + (B, C_next, 1) +  -> (B, C_next, C_prev)
                path_scores = best_scores[:,None,:] + self.param_transition[None,:,:] + emit_scores[:,:,None] # (B, C_next, C_prev)
                best_scores, best_tags = path_scores.max(dim=2) # (B, C_next)
                list_best_paths.append(best_tags)
                list_best_scores.append(best_scores)
                #DPRINT(path_scores[0])
                #DPRINT(path_scores.softmax(2)[0])
                #DPRINT(best_scores[0])
                #DPRINT(best_tags[0])
        list_tags = []
        list_scores = []
        list_best_paths.reverse()
        list_best_scores.reverse()
        mask_x_eos = features['mask_x_eos']
        #dprint(len(list_best_paths))
        #dprint(len(list_best_scores))
        for i, (best_paths, best_scores) in enumerate(zip(list_best_paths, list_best_scores, strict=True)):
            j = length - 2 - i
            #def DPRINT(val):
            #    if i < 3 or j < 3:
            #        dprint(val,2)
            #def DPRINT(val): pass
            #DPRINT(i)
            #DPRINT(best_scores[0])
            #DPRINT(best_paths[0])
            if i == 0:
                #best_score, tags = best_scores.max(1) # (B,)
                #list_scores.append(best_score)
                tags = torch.full([batch_size], idmap.pad).to(self.device, torch.long)
                tags = utils.purge_tensor(tags, ~mask_x_eos[:,-1], idmap.eos)
                list_tags.append(tags)
                #DPRINT(tags[0])
                last_tags = tags
            tags = best_paths.gather(1, last_tags[:,None]).squeeze(1) # (B,)
            best_score = best_scores.gather(1, last_tags[:,None]).squeeze(1) # (B,)
            list_scores.append(best_score)
            #DPRINT(best_score[0])
            #DPRINT(tags[0])
            tags = utils.purge_tensor(tags, ~mask_x_eos[:,j], idmap.eos)
            #DPRINT(tags[0])
            list_tags.append(tags)
            last_tags = tags
        self.last_state['logits'] = torch.stack(list_best_scores, dim=1) # (B, L, C)
        #XXX
        return torch.stack(list_tags[::-1], dim=1)

    def reset_state(self) -> "CRF":
        self.last_state: dict[str, Any] = {}
        return self

class SequenceTagger(modeling.Module):
    def __init__(self, idmaps: "dict[str, Any]", **params: Any) -> None:
        #dprint(idmaps)
        #dprint(idmaps.dict_maps)
        super().__init__()
        # parameters
        params = self.get_config(**params)
        vocab = idmaps['x']
        self.idmaps = idmaps
        self.vocab = vocab
        self.padding = vocab.pad
        #self.vocab_size = len(vocab)
        self.vocab_size = len(vocab)
        self.num_tags = len(idmaps['t'])
        #self.padding = vocab.pad
        self.embed_size = params['embed_size']
        self.hidden_size = params['hidden_size']
        self.encoder_type = params['encoder_type']
        self.decoder_type = params['decoder_type']
        # modules
        #self.mod_encode = LSTMEncoder(vocab, **params)
        classifier_hidden_size = self.hidden_size
        self.mod_encode: (LSTMEncoder | transformer.Encoder | bert.Bert)
        if self.encoder_type == 'lstm':
            if params['bidirectional']:
                classifier_hidden_size = self.hidden_size * 2
            self.mod_encode = LSTMEncoder(idmaps, **params)
        elif self.encoder_type == 'transformer':
            self.mod_encode = transformer.Encoder(idmaps, **params)
            #dprint(self.mod_encode.hidden_size)
            dprint(self.hidden_size)
        elif self.encoder_type == 'bert':
            self.mod_encode = bert.Bert(idmaps, **params)
            self.mod_tune = transformer.Transformer(conditioned=False, **params)
        #dprint(self.mod_encode.embed_size)
        #dprint(self.mod_encode.hidden_size)
        if self.decoder_type == 'crf':
            #self.mod_crf = CRF(self.num_tags, self.num_tags)
            #self.mod_crf = CRF(self.num_tags)
            self.mod_crf = CRF(idmaps['t'])
        self.mod_tag = nn.Linear(classifier_hidden_size, self.num_tags, bias=False)
        # 同上。構築直後から状態を参照できるようにする
        self.reset_state()

    @classmethod
    def get_config(cls, **params: Any) -> dict[str, Any]:
        #params['decoder_type'] = 'lstm'
        #dprint(params)
        params.setdefault('embed_size', 512)
        #params.setdefault('hidden_size', 1024)
        #params.setdefault('dropout_ratio', 0.1)
        #params.setdefault('main_component', 'lstm')
        params.setdefault('encoder_type', 'lstm')
        params.setdefault('num_layers', 1)
        params.setdefault('decoder_type', 'linear')
        #if params['main_component'] == 'lstm':
        if params['encoder_type'] == 'lstm':
            params.setdefault('bidirectional', True)
            params = LSTMEncoder.get_config(**params)
        elif params['encoder_type'] == 'transformer':
            params = transformer.Encoder.get_config(**params)
        elif params['encoder_type'] == 'bert':
            params = bert.Bert.get_config(**params)
        #dprint(params)
        return params

    #def prepare_batch(self, seq):
    def prepare_batch(self, name: str, seq: Any) -> "torch.Tensor | None":
        device = self.device
        if isinstance(seq, torch.Tensor):
            return seq
        elif isinstance(seq, pd.Series):
            #if name in ['segment', 'segment_info'] or self.encoder_type == 'bert':
            if self.encoder_type != 'bert' and name in ['segment', 'segment_info']:
                # セグメント情報を使うのは BERT 符号化器のときだけ
                return None
            # batch を作る行が両方ともコメントアウトされており、
            # BERT 以外の符号化器では UnboundLocalError になっていた。
            # 系列とタグは encode_pair が 1:1 に対応付けて ID 化済みなので、
            # ここで特殊記号を足すと対応が崩れる
            batch = [torch.tensor(idvec) for idvec in seq]
        else:
            raise TypeError(f"unsupported type: {type(seq).__name__}")
        return nn.utils.rnn.pad_sequence(batch, True, self.padding).to(device)

    def prepare_features(self, x: torch.Tensor, **features: Any) -> dict[str, Any]:
        if x.dim() == 2:
            features['x_id_seq'] = x
            features['mask_x'] = (x != self.padding)
            features['mask_x_bos'] = (x == self.vocab.bos)
            features['mask_x_eos'] = (x == self.vocab.eos)
            features['mask_self'] = utils.make_attention_mask(x, x, self.padding)
        # 符号化器ごとに必要な特徴 (LSTM 符号化器が必須とする mask_mem など)
        # は符号化器自身に用意させる。ここで揃えないと forward が KeyError
        # になり、LSTM / Transformer のいずれも動かない
        features = self.mod_encode.prepare_features(x, **features)
        # 系列は位置引数として渡すため、同じものを指す id_seq が残っていると
        # forward() で多重指定になる。この特徴はどこからも読まれていない
        features.pop('id_seq', None)
        return features

    def decode(self, x: torch.Tensor, **features: Any) -> torch.Tensor:
        #features = self.prepare_features(x)
        features = self.prepare_features(x, **features)
        self.reset_state()
        # x : (B, L)
        #encoded = self.mod_encode(x) # (B, L, H)
        #dprint(encoded.shape)
        #h = self.mod_tag(encoded) # (B, L, C)
        # features を渡さないと、呼び出し側が与えた情報 (BERT 符号化器の
        # segment_info など) が落ち、学習時と異なる条件で復号してしまう
        logits = self(x, **features) # (B, L, C)
        if hasattr(self, 'mod_crf'):
            #h = h + self.mod_crf(h)
            #return self.mod_crf.decode(h)
            #return self.mod_crf.decode(h, **features)
            crf_features = dict(features)
            offset = x.shape[1] - logits.shape[1]
            if offset > 0:
                # BERT 符号化器では先頭の <cls> の分だけ logits が短い。
                # x から作ったマスクを同じ長さに揃える
                for key in ('mask_x', 'mask_x_bos', 'mask_x_eos'):
                    if key in crf_features:
                        crf_features[key] = crf_features[key][:, offset:]
            return cast(torch.Tensor, self.mod_crf.decode(logits, **crf_features))
        else:
            return cast(torch.Tensor, logits.argmax(2)) # (B, L)

    def forward(self, x: torch.Tensor, **features: Any) -> torch.Tensor:
        # x : (B, L)
        #encoded = self.mod_encode(x) # (B, L, H)
        features = self.prepare_features(x, **features)
        encoded = self.mod_encode(x, **features) # (B, L, H)
        if self.encoder_type == "bert":
            #dprint(features['mask_self'].shape)
            encoded = self.mod_tune(encoded, **features) # fine tuning
            encoded = encoded[:,1:] # first token <cls> is not needed
        #dprint(encoded.shape)
        #dprint(encoded.flatten()[:5])
        h = cast(torch.Tensor, self.mod_tag(encoded)) # (B, L, C)
        #dprint(h.flatten()[:5])
        #if hasattr(self, 'mod_crf'):
        #    #h = h + self.mod_crf(h)
        #    h = self.mod_crf(h)
        # ここにはタグの自己回帰遷移を加える分岐があったが、
        # mod_transition / mod_embed_tag / mod_transition_tag はいずれも
        # 生成されておらず、参照していた t は forward の引数ですら
        # なかったため、一度も実行されえない未完成のコードだった
        #return h.transpose(1, 2) # (B, C, L)
        return h

    def loss(self, x: torch.Tensor, t: torch.Tensor, reduction: str = 'mean',
             **features: Any) -> torch.Tensor:
        #features = self.prepare_features(x)
        features = self.prepare_features(x, **features)
        #logits = self(x) # (B, L, C)
        logits = self(x, **features) # (B, L, C)
        padding = self.padding
        if hasattr(self, 'mod_crf'):
            #crf_loss = self.mod_crf.loss(logits, t, reduction)
            # BERT 符号化器では x の先頭に <cls> が付くため、mask_x は
            # タグ列より 1 つ長く、そのまま渡すと 1 つずれる。CRF が見る
            # べきはタグ側の有効位置なので、ここで作り直す
            crf_features = dict(features)
            crf_features['mask_x'] = (t != padding)
            crf_loss = self.mod_crf.loss(logits, t, reduction, **crf_features)
            self.last_state['logits'] = self.mod_crf.last_state['logits']
            self.last_state['gold_score'] = self.mod_crf.last_state['gold_score']
            #if True:
            #    logits = self.mod_crf.last_state['logits']
            #    xent_loss = criteria.cross_entropy(logits.transpose(1,2), t, ignore_index=padding, reduction=reduction)
            #    return crf_loss + xent_loss
            return crf_loss
        else:
            self.last_state['logits'] = logits
            return criteria.cross_entropy(logits.transpose(1,2), t, ignore_index=padding, reduction=reduction)

    def get_state(self) -> dict[str, Any]:
        self.last_state['encoder_state'] = self.mod_encode.get_state()
        return self.last_state

    def reset_state(self) -> "SequenceTagger":
        self.last_state: dict[str, Any] = {}
        self.mod_encode.reset_state()
        if hasattr(self, 'mod_crf'):
            self.mod_crf.reset_state()
        if hasattr(self, 'mod_tune'):
            self.mod_tune.reset_state()
        return self

