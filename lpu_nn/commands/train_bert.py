#!/usr/bin/env python3

# system
import math
import pandas as pd
import random
import torch

# local
from lpu_nn.common import criteria
from lpu.common import logging
from lpu_nn.common import training
from lpu_nn.common import utils
from lpu.common.config import Config
from lpu_nn.modeling import bert

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

#default = training.default
default = Config(training.default).data
default.model.task = 'bert_lm'
#default.model.num_segments = 2
default.model.num_token_types = 2
#default.model.activation = 'gelu'
default.model.max_length = 512
#default.model.max_length = 256
#default.model.embed_positions = True
#default.model.share_embedding = True
#default.model.share_embedding = None
# train general
#default.train.optimizer = 'adam'
#default.train.optimizer = 'adabound'
default.train.optimizer = 'lamb'
default.train.batch_size = 32
#default.train.min_batch_size = 16
default.train.min_batch_size = 8
#default.train.min_batch_size = 4
default.train.max_batches = 5000
default.train.dropout_ratio = 0.1 # BERT
# curriculum
default.train.curriculum = 'none'
# warming up
default.train.warmup_steps = 10000 # BERT
#default.train.warmup_factor = None
#default.train.weight_decay_warmup_steps = 10**6
#default.train.weight_decay_warmup_steps = 10**7
# adam
#default.train.adam_alpha = 5e-5
default.train.adam_alpha = 1e-4 # BERT
#default.train.adam_alpha = 1e-3
#default.train.adam_alpha = 0.001
default.train.adam_beta1 = 0.9
#default.train.adam_beta2 = 0.999 # BERT
default.train.adam_beta2 = 0.98 # RoBERTa
#default.train.adam_eps = 1e-8
#default.train.adam_eps = 1e-7
default.train.adam_eps = 1e-6 # BERT
#default.train.gamma = 1e-3
#default.train.gamma = 1e-4
#default.train.gamma = 1e-5
default.train.adabound_gamma = 1e-5
default.train.weight_decay_rate = 0.01 # BERT, hard to learn ...
#default.train.weight_decay_rate = 0.001 # hard to learn, yet
#default.train.weight_decay_rate = 0.0001
#default.train.weight_decay_rate = 0
#default.train.gradient_clipping = 0
# multi-step
#default.train.schedule_num_steps = True
default.train.schedule_num_steps = False
# reviewing
default.train.review_rate = 0.05

specific = Config()
sdata = specific.data
sdata.report = ['epoch', 'proc', 'lr', 'acc', 'restore', 'cont_acc', 'ppl', 'loss', 'pcost', 'err', 'samples', 'tokens/sec', 'steps', 'elapsed']
sdata.format = {}
sdata.format.input = {}
sdata.format.input.s1 = 'seq'
sdata.format.input.s2 = 'seq'
sdata.extra_symbols = {}
sdata.extra_symbols.cls = '<cls>'
sdata.extra_symbols.sep = '<sep>'
sdata.extra_symbols.mask = '<mask>'

class BertTrainer(training.Trainer):
    default = default
    specific = specific
    Model = bert.BertLanguageModel
    #def __init__(self):
    #    super(BertTrainer, self).__init__()

    def make_noisy_seq(self, seq, prob=0.15, modify=0.1):
        #vocab = self.vocab
        vocab = self.idmaps['seq']
        indices = [i for i in range(len(seq)) if seq[i] not in [vocab.cls]]
        #indices = [i for i in range(len(seq)) if seq[i] not in [vocab.cls, vocab.sep]]
        random.shuffle(indices)
        #take_upto = int( len(indices) * prob )
        take_upto = math.ceil( len(indices) * prob )
        noisy_indices = indices[:take_upto]
        def random_mask(i, v):
            if i in noisy_indices:
                r = random.random()
                if r < modify:
                    #return vocab.sample(additions=[vocab.sep])
                    return vocab.sample(exclude_symbols=False)
                    #return vocab.sample(exclude_symbols=True)
                elif r < (1 - modify):
                    # modify <= r < 1-modify
                    return vocab.mask
            # as is
            return v
        return tuple(random_mask(i, v) for i, v in enumerate(seq))

    def prepare_sample(self, sample, continuity=1):
        #vocab = self.vocab
        #vocab = self.model.idmaps['x']
        vocab = self.idmaps['seq']
        sample1 = sample
        max_length = self.config.data.model.max_length
        if continuity:
            sample2 = sample
        else:
            while True:
                sample2 = self.train_df.sample(1).iloc[0]
                if sample1.s2 != sample2.s2:
                    break
        input = pd.Series()
        #input['t'] = (continuity,) + tuple(sample1.s1) + (vocab.sep,) + tuple(sample2.s2) + (vocab.sep,)
        #input['t'] = (continuity,) + sample1.s1 + (vocab.sep,) + sample2.s2 + (vocab.sep,)
        input.name = sample.name
        input['raw1'] = sample1.s1
        input['raw2'] = sample2.s2
        input['continuity'] = continuity
        #s1 = vocab.encode(sample1.s1)
        #s2 = vocab.encode(sample2.s2)
        s1 = vocab.encode(str(sample1.s1))
        s2 = vocab.encode(str(sample2.s2))
        #input['t'] = [continuity] + list(sample1.s1) + [vocab.sep] + list(sample2.s2) + [vocab.sep]
        #input['segment'] = (0,) * (len(sample1.s1)+2) + (1,) * (len(sample2.s2)+1)
        input['t'] = [continuity, *s1, vocab.sep, *s2, vocab.sep]
        #input['segment'] = (0,) * (len(s1)+2) + (1,) * (len(s2)+1)
        input['segment'] = (1,) * (len(s1)+2) + (2,) * (len(s2)+1)
        #input['token_types'] = (0,) * (len(sample1.s1)+2) + (1,) * (len(sample2.s2)+1)
        if len(input.t) > max_length:
            input.t = input.t[:max_length]
            input.segment = input.segment[:max_length]
            #input.token_types = input.token_types[:max_length]
        input['x'] = (vocab.cls, *self.make_noisy_seq(input.t[1:]))
        return input

    def feed_one_batch(self, batch, fallback=False, df=None):
        # using
        padding = self.model.padding
        cdata = self.config.data
        #vocab = self.vocab
        vocab = self.idmaps['seq']
        dtype = self.model.dtype

        #max_steps = int(self.config.data.log.train_step / 100) + 1
        #max_steps = int(self.config.data.log.train_step / 1000) + 1
        #max_steps = int(self.config.data.log.train_step * 10 / cdata.train.warmup_steps) + 1
        #self.set_max_steps(max_steps)
        #self.model.max_steps = None

        if self.model.training:
            if cdata.train.schedule_num_steps:
                max_steps = getattr(self.model, 'max_steps', None)
                if max_steps is None:
                    if cdata.model.universal:
                        self.set_max_steps(2)
                    else:
                        self.set_max_steps(1)
                    # training.comm_main (主プロセス判定) は移植時に
                    # 落とされている。単一プロセスなので常に真として扱う。
                    # set_max_steps は設定に model.max_steps がある場合のみ
                    # 値を代入するため、直接参照すると AttributeError になる
                    dprint(getattr(self.model, 'max_steps', None))

        #report = pd.Series()
        reports = []

        if fallback:
            continuity_list = [1]
        else:
            continuity_list = [0,1]
        #for continuity in (0,1):
        for continuity in continuity_list:
            report = pd.Series()
            if all((field in batch) for field in ['x', 't', 'segment']):
            #if all((field in batch) for field in ['x', 't', 'token_types']):
                # dev
                samples = batch
            else:
                sample_list = [self.prepare_sample(s, continuity) for i, s in batch.iterrows()]
                samples = pd.DataFrame(sample_list)
            batch_x = self.model.prepare_batch(samples.x)
            batch_segment = self.model.prepare_batch(samples.segment)
            #batch_token_types = self.prepare_batch(input.token_types)
            batch_t = self.model.prepare_batch(samples.t)
            #dprint(continuity)
            #dprint(batch_t[:,0])
            #dprint(vocab.decode(input.x.iloc[0], remove_symbols=False, as_tokens=True),)

            mask_x = (batch_x != padding)
            self.model.reset_state()
            h = self.model(batch_x, type_id_seq=batch_segment)
            logits_cont    = self.model.predict_next_sentence(h)
            logits_restore = self.model.decode(h)
            model_state = self.model.get_state()

            #t_sep_mask = (batch_t.data != vocab.sep)
            #batch_t = utils.purge_variables(batch_t, t_sep_mask, -1) # to prevent overfitting to generate <sep> always
            #loss_lm = F.softmax_cross_entropy(h_out_restore_trans, batch_t[:,1:], ignore_label=padding)
            #loss_lm = F.softmax_cross_entropy(h_out_restore, batch_t[:,1:], ignore_label=padding)
            #loss_lm = F.softmax_cross_entropy(logits_restore, batch_t[:,1:], ignore_label=padding)
            loss_lm = criteria.cross_entropy(logits_restore, batch_t[:,1:], ignore_index=padding)
            report['loss_lm'] = float(loss_lm)
            #loss_cont = F.softmax_cross_entropy(h_out_cont, batch_t[:,0])
            #loss_cont = F.softmax_cross_entropy(logits_cont, batch_t[:,0])
            loss_cont = criteria.cross_entropy(logits_cont, batch_t[:,0])
            report['loss_cont'] = float(loss_cont)

            #batch_ppl = criteria.calc_perplexity(h_out_restore_trans, batch_t[:,1:], ignore_label=padding)
            #batch_ppl = criteria.calc_perplexity(h_out_restore, batch_t[:,1:], ignore_label=padding)
            #batch_ppl = criteria.perplexity(logits_restore, batch_t[:, 1:], ignore_label=padding)
            batch_ppl = criteria.perplexity(logits_restore, batch_t[:, 1:], ignore_index=padding)
            #batch_ppl = batch_ppl.data
            #ppl = float(xp.mean(batch_ppl))
            #report['ppl'] = ppl
            list_ppl = batch_ppl.tolist()
            report['ppl'] = float(batch_ppl.mean())
            del batch_ppl

            #accuracy = F.accuracy(h_out_restore_trans, batch_t[:,1:], ignore_label=padding)
            #accuracy = F.accuracy(h_out_restore, batch_t[:,1:], ignore_label=padding)
            #accuracy = F.accuracy(logits_restore, batch_t[:,1:], ignore_label=padding)
            accuracy = float( criteria.accuracy(logits_restore, batch_t[:,1:], ignore_index=padding) )
            report['acc'] = accuracy

            #continuity_predict = F.argmax(h_out_cont, axis=1)
            #continuity_predict = F.argmax(logits_cont, axis=1)
            continuity_predict = logits_cont.argmax(dim=1)
            #continuity_correct_count = xp.sum(continuity_predict.data == batch_t[:,0].data)
            continuity_correct_count = (continuity_predict == batch_t[:,0]).sum()
            continuity_accuracy = float(continuity_correct_count) / batch_t.shape[0]
            report['cont_acc'] = float(continuity_accuracy)
            #del continuity_predict
            del continuity_correct_count

            #noise_mask = (batch_x[:,1:].data != batch_t[:,1:].data)
            noise_mask = (batch_x[:,1:] != batch_t[:,1:])
            #noise_count = xp.sum(noise_mask)
            noise_count = ( noise_mask.sum(dim=1) ).to(self.model.dtype)
            #pred = F.argmax(logits_restore, axis=1) # (B, L)
            pred = logits_restore.argmax(dim=1) # (B, L)
            #restore_correct = (restore_pred.data == batch_t[:,1:].data) * noise_mask
            #restore_correct = (pred.data == batch_t[:,1:].data) * noise_mask
            #correct = (pred.data == batch_t[:,1:].data) * noise_mask
            correct = ((pred == batch_t[:,1:]) * noise_mask).to(self.model.dtype)
            #if noise_count > 0:
            #    #restore_accuracy = float(xp.sum(restore_correct)) / noise_count
            #    restore_accuracy = float(xp.sum(correct)) / noise_count
            #    report['rest_acc'] = float(restore_accuracy)
            #else:
            #    report['rest_acc'] = 0
            #restore_accuracy = correct.sum(dim=1) / noise_count
            #restore_accuracy = torch.where(noise_count > 0, correct.sum(dim=1) / noise_count, torch.tensor(0.0))
            restore_accuracy = utils.purge_tensor(correct.sum(dim=1) / noise_count, noise_count > 0)
            report['rest_acc'] = float(restore_accuracy.mean())
            del noise_mask
            del noise_count
            #del correct
            del restore_accuracy

            #dprint(utils.format_state(model_state))
            transformer_state = model_state.get('bert_state',{}).get('transformer_state',{})
            if isinstance(transformer_state, dict):
                ponder_cost = transformer_state.get('ponder_cost', 0)
                if isinstance(ponder_cost, torch.Tensor):
                    ponder_cost = ponder_cost.sum(dim=1).div(mask_x.to(dtype).sum(dim=1)).mean(0)
            else:
                ponder_cost = 0
            time_penalty = cdata.train.time_penalty
            report['ponder_cost'] = float(ponder_cost)

            if fallback:
                # ignoring correctness of continuity
                loss = loss_lm + ponder_cost * time_penalty
            else:
                loss = loss_lm + loss_cont + ponder_cost * time_penalty
            del loss_lm
            del loss_cont
            report['loss'] = float(loss.data)

            #if chainer.config.train:
            if self.model.training:
                try:
                    self.update_parameters(loss)
                except Exception as e:
                    del loss
                    self.update_to_reduce_ponder_cost(ponder_cost, time_penalty)
                    raise e
            if continuity == 1:
                try:
                    #if chainer.config.train:
                    if self.model.training:
                        self.train_df.loc[batch.index, 'criterion'] = list_ppl
                        cdata.log.fed_samples += len(batch)
                        cdata.log.fed_tokens += int(batch.len_s1.sum() + batch.len_s2.sum())
                    else:
                        # should be dev dataset
                        self.dev_df.loc[batch.index, 'criterion'] = list_ppl
                        #pred = F.argmax(h_out_restore, axis=2)
                        #pred = F.argmax(h_out_restore, axis=1)
                        #pred = utils.purge_variables(pred, x_mask[:,1:], else_value=padding)
                        pred = utils.purge_tensor(pred, mask_x[:, 1:], else_value=padding)
                        list_pred = [vocab.clean_ids(idvec) for idvec in pred.tolist()]
                        #dprint(repr(pred)[:50])
                        #self.dev_df.loc[batch.index, 'pred'] = pred
                        self.dev_df.loc[batch.index, 'cls'] = continuity_predict.tolist()
                        for index, idvec in zip(batch.index, list_pred, strict=True):
                            #self.dev_data.at[index, 'last_pred'] = tuple(idvec)
                            self.dev_df.at[index, 'pred'] = tuple( self.model.vocab.clean_ids(idvec) )
                except Exception as e:
                    dprint(batch.index)
                    #dprint(pred)
                    dprint(len(batch.index))
                    dprint(len(pred))
                    logger.exception(e)
            reports.append(report)
        #for param in self.model.params():
        #    dprint(param.shape)
        #    #dprint(param.update_rule)
        #    dprint(param.update_rule.hyperparam.weight_decay_rate)

        #if chainer.config.train:
        #    if cdata.train.schedule_num_steps:
        #        #if accuracy >= 0.9:
        #        if accuracy >= 0.8:
        #            max_steps = self.model.max_steps
        #            if max_steps < cdata.model.num_layers:
        #                self.set_max_steps(max_steps+1)
        #                if training.comm_main:
        #                    dprint(self.model.max_steps)
        #if chainer.config.train:
        if self.model.training:
            if cdata.train.schedule_num_steps:
                #if accuracy >= 0.9:
                if accuracy >= 0.8:
                    # 設定に model.max_steps が無ければ段数は動かせない
                    max_steps = getattr(self.model, 'max_steps', None)
                    if max_steps is not None:
                        new_max_steps = self.set_max_steps(max_steps+1)
                        if new_max_steps > max_steps:
                            dprint(self.model.max_steps)
                #elif accuracy < 0.5:
                #    max_steps = self.model.max_steps
                #    new_max_steps = self.set_max_steps(max_steps-1)
                #    if new_max_steps < max_steps:
                #        if training.comm_main:
                #            dprint(self.model.max_steps)
        report = pd.DataFrame(reports).mean()
        return report

    def load_eval_data(self, path):
        df = super().load_eval_data(path)
        input_list = [self.prepare_sample(s) for i, s in df.iterrows()]
        input = pd.DataFrame(input_list)
        df['x'] = input.x
        df['segment'] = input.segment
        #df['token_types'] = input.token_types
        df['t'] = input.t
        df['ref'] = df.t.apply(lambda t: t[1:])
        df['pred'] = None
        df['cls'] = None
        return df

    def test_sample(self, sample, msg):
        #vocab = self.vocab
        vocab = self.idmaps['seq']
        logger.info(msg)
        #logger.info('  index: {}'.format(sample.index))
        logger.info(f'  index: {sample.name}')
        #logger.info('  input:\t{}'.format(sample.x))
        logger.info(f'  input:\t{vocab.decode(sample.x, remove_symbols=False)}')
        #logger.info('  original: {}'.format(sample.t))
        logger.info(f'  original: <{sample.t[0]}>\t{vocab.decode(sample.ref, remove_symbols=False)}')
        #cls, pred = self.model.predict(sample.x, sample.segment)
        cls, pred = sample.cls, sample.pred
        logger.info(f'  predicted: <{cls}>\t{vocab.decode(pred, remove_symbols=False)}')
        logger.info(f"  last perplexity: {sample.criterion}")

    def test_model(self):
        if self.dev_df is not None:
            df = self.dev_df
        else:
            #df = self.train_df
            return False
        easy_index  = df.criterion.idxmin()
        easy_one    = df.loc[easy_index]
        self.test_sample(easy_one, "testing the most successful one:")
        sampled_one = df.sample(1).iloc[0]
        self.test_sample(sampled_one, "testing randomly sampled one:")
        hard_index  = df.criterion.idxmax()
        hard_one    = df.loc[hard_index]
        self.test_sample(hard_one, "testing difficult one:")

    #@staticmethod
    @classmethod
    #def create_parser(model_name):
    def create_parser(cls, model_name, default=None):
        if default is None:
            default = cls.default
        #parser = training.Trainer.create_parser(model_name)
        parser = training.Trainer.create_parser(model_name, default)
        group = parser['model']
        #group.add_argument('--num_segments', '--segments', type=int, default=None, help='Number of segments (default: {})'.format(default.model.num_segments))
        group.add_argument('--num-token-types', '--num-types', '--num-segments', '--segments', type=int, default=None, help=f'Number of token types (default: {default.model.num_token_types})')
        #group.add_argument('--num-heads', '--heads', '--head', type=int, default=None, help='Number of ensembles for multi-head attention mechanism (default: {})'.format(default.model.num_heads))
        #group.add_argument('--num-layers', '--layers', '-L', type=int, default=None, help='Number layers for standard transformer (default: {})'.format(default.model.num_layers))
        #group.add_argument('--relative-attention', '--relative', '--rel', type=strtobool, default=None, nargs='?', const=True, help='Using relative position representations for self-attention (default: {})'.format(default.model.relative_attention))
        #group.add_argument('--clip-distance', '--distance', '--dist', '-c', type=int, default=None, help='Threshold of relative distance for clipping in relative-positional embedding (default: {})'.format(default.model.clip_distance))
        #group.add_argument('--universal', '-U', type=strtobool, default=None, nargs='?', const=True, help='Using universal transformer model (default: {})'.format(default.model.universal))
        #group.add_argument('--embed-positions', '--embed-pos', '--emb-pos', '--ep', type=strtobool, default=None, nargs='?', const=True, help='Using learnable position embeddings (default: {})'.format(default.model.embed_positions))
        #group.add_argument('--share-embedding', '-S', type=strtobool, default=None, nargs='?', const=True, help='Using single shared embedding weight for bert input/output (default: {})'.format(default.model.share_embedding))
        return parser

def main():
    training.main(BertTrainer, 'BERT')

if __name__ == '__main__':
    main()

