#!/usr/bin/env python3

# system
import os
from collections import defaultdict
from lpu_nn.common.args import strtobool

# 3rd
import pandas as pd
import torch

# local
from lpu_nn.common import criteria
from lpu.common import logging
from lpu_nn.common import training
from lpu_nn.common import utils
from lpu.common.config import Config
from lpu.common.progress import view as pview
from lpu_nn.common.utils import purge_tensor
from lpu_nn.modeling.sequence_tagger import SequenceTagger
from lpu_nn.commands import train_bert

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

#default = training.default
default = Config(training.default).data
# general
default.model.task = 'seq2tags'
#default.model.main_component = None
default.model.encoder_type = None
default.model.decoder_type = None
default.model.bidirectional_encoder = None
default.model.residual_connection = None
#default.model.max_length = 256
#default.model.max_length = 512
#default.model.max_length = 1024
#default.model.max_length = 2048
default.model.max_length = 4096
default.model.num_token_types = None
#default.train.dropout_ratio = 0.3
default.train.dropout_ratio = 0.1
#default.train.batch_size = 64
default.train.batch_size = 32
default.train.batch_type = 'samples'
default.train.min_batch_size = 1
# optimizers
default.train.optimizer = 'sgd'
#default.train.optimizer = 'lamb'
#default.train.sgd_learning_rate = 0.2
default.train.sgd_learning_rate = 0.1
default.train.adam_alpha = 1e-3
#default.train.adam_alpha = 1e-4
default.train.adam_beta1 = 0.9
#default.train.adam_beta2 = 0.98
#default.train.adam_beta2 = 0.998
default.train.adam_beta2 = 0.999
default.train.adam_eps = 1e-8
default.train.adabound_gamma = 1e-5
default.train.weight_decay_rate = 0.0 # to prevent to reduce transition scores
#default.train.weight_decay_rate = 1e-3
default.train.gradient_clipping = 5.0
#default.train.gradient_clipping = 0.0
# log
default.log.eval_timeout = 60
default.log.fed_src_tokens = 0
default.log.fed_trg_tokens = 0

specific = Config()
sdata = specific.data
sdata.report = ['epoch', 'proc', 'lr', 'acc', 'ntacc', 'ppl', 'ntppl', 'loss', 'gold', 'pcost', 'err', 'samples', 'steps', 'tokens/s', 'elapsed']
sdata.format = {}
sdata.format.input = {}
sdata.format.input.x = 'seq'
#sdata.format.input.x = 'tokens'
sdata.format.output = {}
sdata.format.output.t = 'tags'
#sdata.format.output.t = 'tokens'
#sdata.format.output.t = 'bio'
sdata.extra_symbols = {}
#sdata.extra_symbols.x = {}

class TaggerTrainer(training.Trainer):
    default  = default
    specific = specific
    Model    = SequenceTagger

    #def __init__(self, args):
    #    super(TaggerTrainer, self).__init__(args)

    #def calc_loss(self, report, logits, t, state):
    #    cdata = self.config.data
    #    padding = self.model.padding
    #    time_penalty = cdata.train.time_penalty
    #    ponder_cost = state.get('ponder_cost', 0.0)
    #    if cdata.train.loss is 'smooth':
    #        loss_smooth = criteria.smoothed_cross_entropy(logits, t, smooth=0.1, ignore_index=padding)
    #        loss = loss_smooth + ponder_cost * time_penalty
    #        report['loss_smooth'] = float(loss_smooth)
    #    else:
    #        loss_xent = criteria.cross_entropy(logits, t, ignore_index=padding)
    #        loss = loss_xent + ponder_cost * time_penalty
    #        report['loss_xent'] = float(loss_xent)
    #    report['loss'] = float(loss)
    #    return loss

    def calc_perplexity(self, report, logits, t):
        dtype = logits.dtype
        padding = self.model.padding
        batch_ppl = criteria.perplexity(logits, t, ignore_index=padding, reduction='hmean')
        batch_ppl_safe = purge_tensor(batch_ppl, torch.isfinite(batch_ppl), 0.0)
        batch_ppl_safe_count = torch.isfinite(batch_ppl).sum().to(dtype)
        #report['ppl'] = float(batch_ppl_safe.sum() / batch_ppl_safe_count)
        report['ppl'] = float(batch_ppl_safe.div(batch_ppl_safe_count).sum().detach())
        batch_ppl_safe = purge_tensor(batch_ppl, torch.isfinite(batch_ppl), -1.0)
        list_ppl = batch_ppl_safe.tolist()
        return list_ppl

    def calc_non_trivial_perplexity(self, report, logits, t):
        tagmap = self.model.idmaps['t']
        dtype = logits.dtype
        padding = self.model.padding
        ignore_indices = [padding]
        other = tagmap['O']
        if other != tagmap.unk:
            ignore_indices.append(other)
            #dprint(ignore_indices)
        batch_non_trivial_ppl = criteria.perplexity(logits, t, ignore_index=ignore_indices, reduction='hmean')
        batch_non_trivial_ppl_safe = purge_tensor(batch_non_trivial_ppl, torch.isfinite(batch_non_trivial_ppl), 0.0)
        batch_non_trivial_ppl_safe_count = torch.isfinite(batch_non_trivial_ppl).sum().to(dtype)
        #report['ppl'] = float(batch_ppl_safe.sum() / batch_ppl_safe_count)
        report['ntppl'] = float(batch_non_trivial_ppl_safe.div(batch_non_trivial_ppl_safe_count).sum().detach())
        batch_non_trivial_ppl_safe = purge_tensor(batch_non_trivial_ppl, torch.isfinite(batch_non_trivial_ppl), -1.0)
        list_non_trivial_ppl = batch_non_trivial_ppl_safe.tolist()
        return list_non_trivial_ppl

    def calc_ponder_cost(self, report, x, state):
        dtype = self.model.dtype
        mask_x = (x != self.model.padding)
        transformer_state = state.get('encoder_state',{}).get('transformer_state',{})
        ponder_cost = 0
        if isinstance(transformer_state, dict):
            ponder_cost = transformer_state.get('ponder_cost', 0)
        if isinstance(ponder_cost, torch.Tensor):
            ponder_cost = (ponder_cost.sum(1) / mask_x.to(dtype).sum(1)).mean()
            report['ponder_cost'] = float(ponder_cost)
        state['ponder_cost'] = ponder_cost
        return ponder_cost

    def prepare_sample(self, sample):
        idmaps = self.idmaps
        x, t = idmaps.encode_pair('x', sample.x, 't', sample.t, to='ids')
        x_tokens, t_tokens = idmaps.encode_pair('x', sample.x, 't', sample.t, to='tokens')
        input = pd.Series()
        input.name = sample.name
        input['x'] = x
        input['t'] = t
        input['x_tokens'] = x_tokens
        input['t_tokens'] = t_tokens
        if self.config.data.model.encoder_type == "bert":
            bert_input = self.model.mod_encode.prepare_input(x)
            input['x'] = bert_input.x
            input['segment_info'] = bert_input.segment_info
            #dprint(len(bert_input.x))
            #dprint(len(bert_input.segment_info))
        else:
            input['segment_info'] = None
        #dprint(input)
        return input

    def feed_one_batch(self, batch, fallback=False, df=None):
        padding = self.model.padding
        cdata = self.config.data
        other = self.model.idmaps['t']['O']
        #dprint(other)
        #dprint(self.model.idmaps['t'].unk)

        if self.model.training:
            if cdata.train.schedule_num_steps:
                max_steps = getattr(self.model, 'max_steps', None)
                if max_steps is None:
                    if cdata.model.universal:
                        self.set_max_steps(2)
                    else:
                        self.set_max_steps(1)
                    # training.comm_main (主プロセス判定) は移植時に
                    # 落とされている。単一プロセスなので常に真として扱う
                    dprint(getattr(self.model, 'max_steps', None))

        report = pd.Series()
        samples = pd.DataFrame(self.prepare_sample(sample) for i, sample in batch.iterrows())
        #dprint(input)
        #batch_x = self.model.prepare_batch(batch.x)
        #batch_t = self.model.prepare_batch(batch.t)
        batch_x = self.model.prepare_batch('x', samples.x)
        batch_t = self.model.prepare_batch('t', samples.t)
        batch_segment = self.model.prepare_batch('segment', samples.segment_info)
        #dprint(max(len(x) for x in samples.x))
        #dprint(max(len(s) for s in samples.segment_info))
        #dprint(batch_x.shape)
        #dprint(batch_segment.shape)
        #dprint(batch_x[:,:5])
        #dprint(batch_t[:,:5])
        #batch_x, batch_t = self.model.prepare_batch_pair(batch.x, batch.t)
        mask_x = (batch_x != padding)
        mask_t = (batch_t != padding)

        #trg_id_seq_input  = batch_t[:,0:-1]
        #trg_id_seq_expect = batch_t[:,1:None]
        #logits = self.model(batch_x, trg_id_seq_input)
        #logits = self.model(batch_x)
        #logits = self.model(batch_x, batch_t)
        #pred = self.reset_state().decode(batch_x)
        #loss = self.model.loss(batch_x, batch_t)
        loss = self.model.loss(batch_x, batch_t, segment_info=batch_segment)
        report['loss'] = loss.item()
        model_state = self.model.get_state()
        logits = model_state['logits'].transpose(1, 2)
        if 'gold_score' in model_state:
            report['gold'] = model_state['gold_score'].mean().item()

        #dprint(logits.shape)
        #dprint(batch_t.shape)
        list_ppl = self.calc_perplexity(report, logits, batch_t)
        self.calc_non_trivial_perplexity(report, logits, batch_t)

        accuracy = float( criteria.accuracy(logits, batch_t, ignore_index=padding) )
        report['acc'] = accuracy
        non_trivial_accuracy = float( criteria.accuracy(logits, batch_t, ignore_index=[padding,other]) )
        report['ntacc'] = non_trivial_accuracy

        ponder_cost = self.calc_ponder_cost(report, batch_x, model_state)
        #loss = self.calc_loss(report, logits, batch_t, model_state)

        if self.model.training:
            try:
                if not torch.isfinite(loss).all():
                    dprint(loss)
                else:
                    #torch.autograd.set_detect_anomaly(True)
                    self.update_parameters(loss)
            except Exception as e:
                del loss
                self.update_to_reduce_ponder_cost(ponder_cost, cdata.train.time_penalty)
                raise e
        try:
            #if chainer.config.train:
            if self.model.training:
                self.train_df.loc[batch.index, 'criterion'] = list_ppl
                cdata.log.fed_samples += len(batch)
                num_tokens = int(mask_x.sum())
                cdata.log.fed_tokens += num_tokens
            else:
                self.dev_df.loc[batch.index, 'criterion'] = list_ppl
                pred = logits.argmax(1)
                pred = utils.purge_tensor(pred, mask_t, else_value=padding)
                        #list_pred = [idmap.clean_ids(idvec) for idvec in pred.tolist()]
                #list_pred = [idmap.clean_ids(idvec) for idvec in pred.tolist()]
                list_pred = pred.tolist()
                #list_len  = mask_t.sum(1).tolist()
                #for index, idvec in zip(batch.index, list_pred, strict=True):
                #self.dev_df.loc[batch.index, 'x_tokens'] = samples.x
                #self.dev_df.loc[batch.index, 'x'] = samples.x
                self.dev_df.loc[batch.index, 'x_tokens'] = samples.x_tokens
                #self.dev_df.loc[batch.index, 'ref'] = samples.t
                self.dev_df.loc[batch.index, 'ref_tokens'] = samples.t_tokens
                #for index, idvec, len_t in zip(batch.index, list_pred, list_len):
                for index, idvec in zip(batch.index, list_pred, strict=True):
                    #self.dev_df.at[index, 'pred'] = tuple( self.model.vocab.clean_ids(idvec) )
                    #self.dev_df.at[index, 'pred'] = tuple( idmap.clean_ids(idvec) )
                    #self.dev_df.at[index, 'pred'] = tuple( idvec )
                    #dprint(self.dev_df.loc[index, 'ref'],)
                    #dprint(len(self.dev_df.loc[index, 'ref']),)
                    #dprint(len(self.dev_df.loc[index, 'x_tokens']),)
                    #len_ref = len(self.dev_df.loc[index, 'ref'])
                    len_ref = len(self.dev_df.loc[index, 'ref_tokens'])
                    #dprint(len_ref)
                    self.dev_df.at[index, 'pred'] = tuple( idvec[:len_ref] )
        except Exception as e:
            logger.exception(e)

        if self.model.training:
            if cdata.train.schedule_num_steps:
                if accuracy >= 0.6:
                    max_steps = self.model.max_steps
                    new_max_steps = self.set_max_steps(max_steps+1)
                    if new_max_steps > max_steps:
                        if training.comm_main:
                            dprint(self.model.max_steps)
        return report

    def load_eval_data(self, path):
        df = super().load_eval_data(path)
        #df['ref'] = df.t
        df['x_tokens'] = None
        #df['x_tokens'] = [()] * len(df)
        df['ref'] = None
        #df['ref'] = [()] * len(df)
        df['pred'] = None
        return df

    # 基底の evaluate は feed_batches / report も渡す
    def evaluate(self, tag, df, args, feed_batches=None, report=None):
        cdata = self.config.data
        eval_report = super().evaluate(tag, df, args, feed_batches, report)
        #batch_size = max(cdata.train.min_batch_size, int(cdata.train.batch_size / 2))
        batch_size = max(cdata.train.min_batch_size, int(cdata.train.batch_size / 4))
        batches = list(training.build_batches(df, batch_size, cdata.train.batch_type))
        try:
            idmap = self.model.idmaps['t']
            with torch.no_grad():
                #outpath = os.path.join(args.workdir, 'pred_{}.txt'.format(tag))
                outpath = os.path.join(args.workdir, 'record.latest', f'pred_{tag}.txt')
                # 評価はチェックポイントの書き出しより先に走るため、
                # record.latest がまだ存在しないことがある
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                list_ref_entities = []
                list_pred_entities = []
                with open(outpath, 'w', encoding='utf-8') as fobj:
                    logger.info(f"writing tagging results into: {outpath}")
                    for batch in pview(batches, header=f'evaluating {tag} data'):
                        samples = pd.DataFrame(self.prepare_sample(sample) for i, sample in batch.iterrows())
                        self.model.reset_state()
                        batch_x = self.model.prepare_batch('x', samples.x)
                        batch_segment = self.model.prepare_batch('segment', samples.segment_info)
                        #batch_pred = self.model(batch_x) # (B, V, L)
                        #samples['pred'] = batch_pred.argmax(1).tolist()
                        #dprint(batch_x.shape)
                        #batch_pred = self.model.decode(batch_x) # (B, L)
                        batch_pred = self.model.decode(batch_x, segment_info=batch_segment) # (B, L)
                        #dprint(batch_pred.shape)
                        samples['pred'] = batch_pred.tolist()
                        for _i, sample in samples.iterrows():
                            #x_tokens = vocab.decode(sample.x, remove_symbols=False, as_tokens=True)
                            x_tokens = sample.x_tokens
                            #dprint(len(x_tokens))
                            #dprint(x_tokens)
                            ref_tokens = idmap.decode(sample.t, remove_symbols=False, as_tokens=True)
                            #input_tagged = self.pair2tag(zip(x_tokens, ref_tokens, strict=False))
                            #dprint(input_tagged[:100])
                            #dprint(self.extract_tags(zip(x_tokens, ref_tokens, strict=False)),)
                            list_ref_entities.append(self.extract_tags(zip(x_tokens, ref_tokens, strict=False)))
                            # clean_ids で記号を抜くと列が縮み、zip が短い方で
                            # 打ち切られてトークンとの対応がずれる (文末が落ちる)。
                            # パディング分の余りは zip が切り捨てるので不要
                            pred_tokens = idmap.decode(sample.pred, remove_symbols=False, as_tokens=True)
                            #dprint(sample.pred)
                            #dprint(pred_tokens)
                            #dprint(idmap.clean_ids(sample.pred))
                            #dprint(idmap.decode(sample.pred, remove_symbols=False, as_tokens=True))
                            #dprint(idmap.decode(idmap.clean_ids(sample.pred), remove_symbols=False, as_tokens=True))
                            pred_tagged = self.pair2tag(zip(x_tokens, pred_tokens, strict=False))
                            list_pred_entities.append(self.extract_tags(zip(x_tokens, pred_tokens, strict=False)))
                            #dprint(pred_tagged)
                            #XXX
                            #dprint(self.extract_tags(zip(x_tokens, pred_tokens, strict=False)))
                            #dprint(pred_tagged[:200])
                            #dprint(self.extract_tags(zip(x_tokens, pred_tokens, strict=False)),)
                            fobj.write(pred_tagged.strip())
                            fobj.write("\n")
                    #logger.info("{} bleu: {} [%]".format(tag, bleu_score * 100))
                    #eval_report['bleu'] = bleu_score
                    scores = self.calc_scores(list_ref_entities, list_pred_entities)
                    boundary_scores = self.calc_scores(list_ref_entities, list_pred_entities, match_labels=False)
                    eval_report['precision'] = scores['precision']
                    eval_report['recall'] = scores['recall']
                    eval_report['f1'] = scores['f1']
                    eval_report['boundary_precision'] = boundary_scores['precision']
                    eval_report['boundary_recall'] = boundary_scores['recall']
                    eval_report['boundary_f1'] = boundary_scores['f1']
                    dprint(eval_report)
                return eval_report
        except Exception as e:
            logger.exception(e)
            return eval_report

    def calc_scores(self, list_ref, list_pred, match_labels=True):
        total_true_positive = 0
        total_false_positive = 0
        total_false_negative = 0
        for ref, pred in zip(list_ref, list_pred, strict=True):
            count_ref = defaultdict(int)
            count_pred = defaultdict(int)
            for entity in ref:
                if match_labels:
                    count_ref[entity] += 1
                else:
                    count_ref[entity[1]] += 1
            for entity in pred:
                if match_labels:
                    count_pred[entity] += 1
                else:
                    count_pred[entity[1]] += 1
            for entity, count in count_pred.items():
                num_true_positive = min(count, count_ref[entity])
                total_true_positive += num_true_positive
                total_false_positive += (count - num_true_positive)
            for entity, count in count_ref.items():
                num_true_positive = min(count, count_pred[entity])
                total_false_negative += count - num_true_positive
        if total_true_positive > 0:
            precision = total_true_positive / (total_true_positive + total_false_positive)
            recall = total_true_positive / (total_true_positive + total_false_negative)
            f1 = 2 * precision * recall / (precision + recall)
        else:
            precision = 0
            recall = 0
            f1 = 0
        return {'precision': precision, 'recall': recall, 'f1': f1}

    def normalize_tag(self, tag):
        """Map a tag the BIO scheme has no place for onto 'O'

        BIO 方式に居場所の無いタグを 'O' に潰す。

        タグ表には <pad> / <s> / </s> / <unk> も含まれており、モデルは
        それらも予測しうる。head = tag[0:1], label = tag[2:] の切り出しは
        '<pad>' から '<' と 'ad>' を作ってしまい、出力に "<ad>>" のような
        壊れたタグが現れる。<unk> だけは 'U' として潰されていたが、
        他の記号は取りこぼされていた ('U' と 'O' は head が 'B' でなく
        label が空という点で等価なので、まとめて 'O' とする)。
        """
        if tag in self.model.idmaps['t'].symbols.values():
            return 'O'
        return tag

    def pair2tag(self, pair_seq):
        vocab = self.model.vocab
        tagged = ""
        #last_tag = ""
        last_head = ""
        last_label = ""
        tokens = []
        def flush_tokens():
            flush = tokens.copy()
            tokens.clear()
            return vocab.decode(flush)
        def check_space(token):
            if token[:1] == "▁":
                return " "
            return ""
        for token, tag in pair_seq:
            tag = self.normalize_tag(tag)
            head = tag[0:1]
            label = tag[2:]
            if head == 'B':
                if last_head:
                    #tagged += vocab.decode(flush_tokens())
                    #tagged += "</{}>".format(last_label)
                    #tagged += "{}</{}>".format(flush_tokens(), last_label)
                    tagged += f"{flush_tokens()}</{last_label}>{check_space(token)}"
                #tagged += "<{}>".format(label)
                #tagged += "{} <{}>".format(flush_tokens(), label)
                tagged += f"{flush_tokens()}{check_space(token)}<{label}>"
            #elif last_label and tag != last_tag:
            #elif last_label and label != last_label:
            #    tagged += "</{}>".format(last_label)
            #    if head == "I":
            #        tagged += "<{}>".format(label)
            elif label != last_label:
                if last_label:
                    #tagged += "</{}>".format(last_label)
                    #tagged += "{}</{}>".format(flush_tokens(), last_label)
                    tagged += f"{flush_tokens()}</{last_label}>{check_space(token)}"
                if label:
                    #tagged += "<{}>".format(label)
                    #tagged += "{}<{}>".format(flush_tokens(), label)
                    tagged += f"{flush_tokens()}{check_space(token)}<{label}>"
            #tagged += token
            tokens.append(token)
            #tagged += vocab.decode(token)
            #tagged += vocab.decode([token])
            last_head = tag[0:1]
            last_label = tag[2:]
        if last_label:
            #tagged += "</{}>".format(last_label)
            tagged += f"{flush_tokens()}</{last_label}>"
        tagged += flush_tokens()
        return tagged

    def extract_tags(self, pair_seq):
        tag_list = []
        vocab = self.model.vocab
        last_head = ""
        last_label = ""
        tokens = []
        def flush_tokens():
            flush = tokens.copy()
            tokens.clear()
            return vocab.decode(flush)
        for token, tag in pair_seq:
            tag = self.normalize_tag(tag)
            head = tag[0:1]
            label = tag[2:]
            if head == 'B':
                if last_head:
                    tag_list.append((last_label, flush_tokens()))
                flush_tokens()
            elif label != last_label:
                if last_label:
                    tag_list.append((last_label, flush_tokens()))
                if label:
                    flush_tokens()
            tokens.append(token)
            last_head = tag[0:1]
            last_label = tag[2:]
        if last_label:
            tag_list.append((last_label, flush_tokens()))
        return tag_list

    def setup_model(self, args):
        bert_trainer = None
        cdata = self.config.data
        if cdata.model.encoder_type == "bert":
            sdata.extra_symbols.cls = '<cls>'
            sdata.extra_symbols.sep = '<sep>'
            sdata.extra_symbols.mask = '<mask>'
            #sdata.extra_symbols.x.cls = '<cls>'
            #sdata.extra_symbols.x.sep = '<sep>'
            #sdata.extra_symbols.x.mask = '<mask>'
            if not args.resume:
                if args.pre_trained_model:
                    bert_trainer = train_bert.BertTrainer(args).load_status(args.pre_trained_model)
                    dprint(bert_trainer.config.to_json(indent=2))
                    #self.config['model'] = bert_trainer.config['model']
                    bert_cdata = bert_trainer.config.data
                    cdata.model.embed_size = bert_cdata.model.embed_size
                    cdata.model.num_token_types = bert_cdata.model.num_token_types
                    dprint(self.config.to_json(indent=2))
                    cdata.train.optimizer = bert_cdata.train.optimizer
                    cdata.train.weight_decay_rate = bert_cdata.train.weight_decay_rate
                    #cdata.train.adam_alpha = bert_cdata.train.adam_alpha * 0.1
                    cdata.train.adam_alpha = 5e-5
                    #cdata.train.adam_alpha = 3e-5
                    #cdata.train.adam_alpha = 2e-5
        super().setup_model(args)
        if bert_trainer:
            training.infomain("copying parameters of pre-trained model into fine-tuning model")
            #utils.copy_model(bert_trainer.model, self.model)
            #utils.copy_model(bert_trainer.model.L_bert, self.model.L_bert)
            self.model.mod_encode = bert_trainer.model.mod_bert

    def test_sample(self, sample, msg):
        LIMIT = 1000
        #vocab = self.vocab
        idmap = self.idmaps['t']
        logger.info(msg)
        logger.info(f'  index: {sample.name}')
        #x_tokens = vocab.decode(sample.x, remove_symbols=False, as_tokens=True)
        #dprint(x_tokens)
        x_tokens = sample.x_tokens
        #dprint(sample.x_tokens)
        #ref_tokens = idmap.decode(sample.ref, remove_symbols=False, as_tokens=True)
        ref_tokens = sample.ref_tokens
        input_tagged = self.pair2tag(zip(x_tokens, ref_tokens, strict=False))
        if len(input_tagged) > LIMIT:
            input_tagged = input_tagged[:LIMIT-3] + "..."
        logger.info(f'  input-ref tagged: {input_tagged}')
        dprint(self.extract_tags(zip(x_tokens, ref_tokens, strict=False)),)
        pred_tokens = idmap.decode(idmap.clean_ids(sample.pred), remove_symbols=False, as_tokens=True)
        pred_tagged = self.pair2tag(zip(x_tokens, pred_tokens, strict=False))
        if len(pred_tagged) > LIMIT:
            pred_tagged = pred_tagged[:LIMIT-3] + "..."
        logger.info(f'  input-pred tagged: {pred_tagged}')
        dprint(self.extract_tags(zip(x_tokens, pred_tokens, strict=False)),)
        logger.info(f"  last perplexity: {sample.criterion}")

    def test_model(self):
        if self.dev_df is not None:
            df = self.dev_df
        else:
            df = self.train_df
        df = df[df.criterion >= 0]
        easy_index  = df.criterion.idxmin()
        easy_one    = df.loc[easy_index]
        self.test_sample(easy_one, "testing the most successful one:")
        sampled_one = df.sample(1).iloc[0]
        self.test_sample(sampled_one, "testing randomly sampled one:")
        hard_index  = df.criterion.idxmax()
        hard_one    = df.loc[hard_index]
        self.test_sample(hard_one, "testing difficult one:")

    @classmethod
    def create_parser(cls, model_name, default=None):
        if default is None:
            default = cls.default
        parser = training.Trainer.create_parser(model_name, default)
        group = parser['model']
        #cls.add_argument(group, default.model.main_component, '--main-component', '--main-layer', '--layer-type', '--layer', type=str, choices=['lstm', 'transformer'], help='Main layer component')
        cls.add_argument(group, default.model.encoder_type, '--encoder-type', '--encoder', '--enc', type=str, choices=['lstm', 'transformer', 'bert'], help='Encoder module type')
        cls.add_argument(group, default.model.decoder_type, '--decoder-type', '--decoder', '--dec', type=str, choices=['linear', 'crf'], help='Decoder module type')
        cls.add_argument(group, default.model.bidirectional_encoder, '--bidirectional-encoder', '--bidirectional', '--brnn', type=strtobool, nargs='?', const=True, help='Using bidirectional encoder')
        cls.add_argument(group, default.model.residual_connection, '--residual-connection', '--residual', '--rc', type=strtobool, nargs='?', const=True, help='Using residual connection for RNN encoder/decoder layers')
        # '--pre' と '-P' は基底パーサの --preset が既に使っており、
        # 残したままだと argparse がパーサ構築時に衝突で落ちる
        group.add_argument('--pre-trained-model', '--pre-trained', type=str, help='Path to the pre-trained BERT model to load initially')
        #group.add_argument('--num-token-types', '--num-types', '--num-segments', '--segments', type=int, default=None, help='Number of token types (default: {})'.format(default.model.num_token_types))
        #cls.add_argument(group, default.model.attention_type, '--attention-type', '--global-attention-type', '--attention', '--global-attention', '--att', '--at', '--ga', type=str, choices=['dot', 'concat', 'general', 'mlp', 'none'], help='Decoder attention type for RNN encoder-decoder models')
        #cls.add_argument(group, default.model.local_attention, '--local-attention', '--local', '--la', type=strtobool, nargs='?', const=True, help='Using local attention mechanism for RNN encoder-decoder models')
        #cls.add_argument(group, default.model.input_feeding, '--input-feeding', '--feeding', '--if', type=strtobool, nargs='?', const=True, help='Using input feeding of last decoder output state')
        return parser

def main():
    training.main(TaggerTrainer, 'Sequence Tagger')

if __name__ == '__main__':
    main()

