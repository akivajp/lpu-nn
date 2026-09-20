#!/usr/bin/env python3

# system

# 3rd
import pandas as pd
import torch

# local
from lpu.common import logging
from lpu_nn.common import training
from lpu.common.config import Config
from lpu_nn.modeling.sequence_matcher import SequenceMatcher
from lpu_nn.commands import run_match_ranker

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

default = Config(training.default).data
default.model.task = 'match_rank'
#default.model.num_segments = 2
default.model.num_encoder_layers = None
default.model.match_pooler_type = None
default.model.sequence_pooling = None
default.model.ngram_orders = None
default.model.num_encoder_layers = None
default.model.num_blocks = None
#default.model.embed_size = 512
#default.model.hidden_size = 1024
#default.model.hidden_size = 150 # following [Wang+ 2017]
default.model.num_classes = None
default.train.curriculum = 'none'
#default.train.curriculum = 'crit'
#default.train.loss = 'point'
default.train.loss_method = 'point'
#default.train.loss = 'pair-wise'
#default.model.max_length = 256
#default.model.max_length = 512
#default.model.max_length = 1024
default.model.max_length = 2048
#default.train.batch_size = 32
default.train.batch_size = 128
#default.train.min_batch_size = 16
#default.train.optimizer = 'adam'
default.train.optimizer = 'lamb'
default.train.dropout_ratio = 0.2
#default.train.optimizer = 'adamax' # following [Wang+ 2017]
#default.train.adam_alpha = 0.002 # following [Wang+ 2017]
default.train.adam_alpha = 0.001
default.train.adam_beta1 = 0.9 # following [Wang+ 2017]
default.train.adam_beta2 = 0.999 # following [Wang+ 2017]
default.train.weight_decay_rate = 0 # following [Wang+ 2017]
default.train.warmup_steps = 0
default.train.review_rate = 0
default.train.challenge_rate = 0

#REPORT_CLASSIFY = ['epoch', 'proc', 'lr', 'acc', 'loss', 'ppl', 'err', 'samples', 'tokens/sec', 'steps', 'elapsed']
REPORT_CLASSIFY = ['epoch', 'proc', 'lr', 'acc', 'loss', 'ppl', 'gnorm', 'clip', 'err', 'samples', 'tokens/sec', 'steps', 'elapsed']
REPORT_POINT    = ['epoch', 'proc', 'lr', 'loss', 'err', 'samples', 'tokens/sec', 'steps', 'elapsed']
REPORT_PAIR     = ['epoch', 'proc', 'lr', 'acc', 'loss', 'err', 'samples', 'tokens/sec', 'steps', 'elapsed']

specific = Config()
sdata = specific.data
sdata.report = REPORT_POINT
sdata.format = {}
sdata.format.input = {}
sdata.format.input.s1 = 'seq'
sdata.format.input.s2 = 'seq'
sdata.format.output = {}
sdata.format.output.t = 'score'
#sdata.extra_symbols = {}

class RankerTrainer(training.Trainer):
    default  = default
    specific = specific
    #Model    = CompareAggregate
    #Model    = RE2
    Model    = SequenceMatcher

    def setup_model(self, args):
        cdata = self.config.data
        sdata = self.specific.data
        loss_method = cdata.train.loss_method
        if loss_method.startswith('class'):
            sdata.report = REPORT_CLASSIFY
            sdata.format.output.t = 'label'
            cdata.train.loss_method = 'classify'
        elif loss_method.startswith('point'):
            sdata.report = REPORT_POINT
            cdata.train.loss_method = 'point-wise'
        elif loss_method.startswith('pair'):
            sdata.report = REPORT_PAIR
            cdata.train.loss_method = 'pair-wise'
        else:
            raise ValueError(f"Unknown loss method: {loss_method}")
        self.update_main_fields()
        return super().setup_model(args)

    def prepare_sample(self, sample, resample=False):
        sample1 = sample
        #if correct:
        if not resample:
            sample2 = sample
        else:
            if self.model.training:
                if getattr(self, 'train_df', None) is None:
                    self.load_train_data()
                df = self.train_df
            else:
                df = self.dev_df
            df = df[df.s1 == sample1.s1]
            df_diff = df[df.t != sample1.t]
            if len(df_diff) > 0:
                sample2 = df_diff.sample(1).iloc[0]
            elif len(df) > 0:
                sample2 = df.sample(1).iloc[0]
            else:
                dprint(self.model.training)
                dprint(sample1.s1)
                dprint(sample1.s2)
                dprint(df.t != sample1.t)
                return sample1
        input = pd.Series()
        input['s1'] = sample1.s1
        input['s2'] = sample2.s2
        input['t'] = float(sample2.t)
        return input

    def feed_one_batch(self, batch, fallback=False, df=None):
        cdata = self.config.data
        loss_method = cdata.train.loss_method
        if loss_method.startswith('class'):
            return self.feed_one_batch_class(batch, fallback)
        elif loss_method.startswith('point'):
            return self.feed_one_batch_point(batch, fallback)
        elif loss_method.startswith('pair'):
            return self.feed_one_batch_pair(batch, fallback)
        else:
            raise ValueError(f'Unknown loss method: {loss_method}')

    def feed_one_batch_class(self, batch, fallback=False, df=None):
        cdata = self.config.data
        vocab = self.model.vocab
        report = pd.Series()

        queries  = self.model.prepare_batch('s1', batch.s1)
        replies  = self.model.prepare_batch('s2', batch.s2)
        expected = self.model.prepare_batch('t', batch.t)
        #dprint(expected)

        logits = self.model(queries, replies) # (B, C)
        #xent = (-expected * logits.log_softmax(1)).sum(1)
        xent = torch.nn.functional.kl_div(logits.log_softmax(1), expected, reduction='none').sum(1)
        loss = torch.mean(xent)
        #e = expected.argmax(1)
        #loss = torch.nn.functional.cross_entropy(logits, e)
        loss_list = xent.flatten().tolist()
        report['loss'] = float(loss)
        accuracy = float((expected * logits.softmax(1)).sum(1).mean())
        report['acc'] = accuracy
        perplexity = float( xent.exp().mean() )
        report['ppl'] = perplexity
        #perplexity = float( criteria.perplexity(logits, expected) )

        if self.model.training:
            del logits
            self.update_parameters(loss, report)
        try:
            if self.model.training:
                df = self.train_df
                cdata.log.fed_samples += len(batch)
                cdata.log.fed_tokens += int((queries != vocab.pad).sum())
                cdata.log.fed_tokens += int((replies != vocab.pad).sum())
            else:
                df = self.dev_df
                pred_scores = self.model.logit_to_score(logits)
                df.loc[batch.index, 'last_pred'] = pred_scores.flatten().tolist()
                #dprint(logits.shape)
                for index, logit in zip(batch.index, logits, strict=True):
                    dist = logit.softmax(0)
                    df.at[index, 'last_logit'] = tuple( logit.tolist() )
                    df.at[index, 'last_dist'] = tuple( dist.tolist() )
            df.loc[batch.index, 'criterion'] = loss_list
        except Exception as e:
            logger.exception(e)
        return report

    def feed_one_batch_point(self, batch, fallback=False, df=None):
        cdata = self.config.data
        vocab = self.model.vocab
        report = pd.Series()

        queries  = self.model.prepare_batch('s1', batch.s1)
        replies  = self.model.prepare_batch('s2', batch.s2)
        correct_scores = self.model.prepare_batch('t', batch.t)

        pred_scores = self.model(queries, replies) # (B)
        loss = (pred_scores - correct_scores) ** 2
        loss_list = loss.flatten().tolist()
        loss = torch.mean(loss)
        report['loss'] = float(loss)

        if self.model.training:
            self.update_parameters(loss, report)
        try:
            if self.model.training:
                df = self.train_df
                cdata.log.fed_samples += len(batch)
                cdata.log.fed_tokens += int((queries != vocab.pad).sum())
                cdata.log.fed_tokens += int((replies != vocab.pad).sum())
            else:
                df = self.dev_df
                df.loc[batch.index, 'last_pred'] = pred_scores.flatten().tolist()
            df.loc[batch.index, 'criterion'] = loss_list
        except Exception as e:
            logger.exception(e)
        return report

    def feed_one_batch_pair(self, batch, fallback=False, df=None):
        cdata = self.config.data
        vocab = self.model.vocab
        dtype = self.model.dtype
        device = self.model.device
        report = pd.Series()

        batch1 = pd.DataFrame([self.prepare_sample(s, False) for i, s in batch.iterrows()])
        batch2 = pd.DataFrame([self.prepare_sample(s, True)  for i, s in batch.iterrows()])
        queries  = self.model.prepare_batch('s1', batch1.s1)
        replies1 = self.model.prepare_batch('s2', batch1.s2)
        replies2 = self.model.prepare_batch('s2', batch2.s2)
        t_diff = self.model.prepare_batch('t', batch1.t - batch2.t)
        t_sign = torch.sign(t_diff)

        scores1 = self.model(queries, replies1).flatten() # (B, 1) -> (B)
        scores2 = self.model(queries, replies2).flatten() # (B, 1) -> (B)

        SCALE = 1
        scores_diff = scores1 - scores2
        scores_diff_scaled = SCALE * scores_diff
        p_12 = 1 / (1 + torch.exp(-scores_diff_scaled))
        loss = - (t_sign + 1) / 2 * torch.log(p_12) - (1 - t_sign) / 2 * torch.log(1 - p_12)
        loss_list = loss.flatten().tolist()
        loss = torch.mean(loss)
        report['loss'] = float(loss)

        correct = torch.zeros(t_sign.shape, dtype=dtype, device=device)
        correct = correct + ((t_sign > 0) & (scores_diff > 0) | (t_sign < 0) & (scores_diff < 0)).to(dtype)
        correct = correct + (t_sign == 0).to(dtype) * 0.5
        report['acc'] = float(correct.mean())

        if self.model.training:
            self.update_parameters(loss, report)
        if torch.all( torch.isfinite(loss) ):
            try:
                if self.model.training:
                    df = self.train_df
                    cdata.log.fed_samples += len(batch)
                    cdata.log.fed_tokens += int((queries != vocab.pad).sum() * 2)
                    cdata.log.fed_tokens += int((replies1 != vocab.pad).sum())
                    cdata.log.fed_tokens += int((replies2 != vocab.pad).sum())
                else:
                    df = self.dev_df
                    df.loc[batch.index, 'last_pred1'] = scores1.flatten().tolist()
                    df.loc[batch.index, 'last_pred2'] = scores2.flatten().tolist()
                    df.loc[batch.index, 'last_t2'] = list(batch2.t)
                    for index, idvec in zip(batch.index, batch2.s2, strict=True):
                        df.at[index, 'last_reply2'] = idvec
                df.loc[batch.index, 'criterion'] = loss_list
            except Exception as e:
                logger.exception(e)
        return report

    def evaluate(self, tag, df, args, feed_batches=None, report=None):
        cdata = self.config.data
        #eval_report = super(RankerTrainer,self).evaluate(tag, df, args)
        eval_report = super().evaluate(tag, df, args, feed_batches, report)
        batch_size = max(cdata.train.min_batch_size, int(cdata.train.batch_size / 2))
        try:
            with torch.no_grad():
                progress = True
                timeout = cdata.log.eval_timeout
                loss_method = cdata.train.loss_method
                if loss_method.startswith('class'):
                    idmap = self.model.idmaps['t']
                    expected = [idmap.str2score(s) for s in df.t]
                else:
                    expected = df.t
                #results = run_match_ranker.evaluate(self.model, df.s1, df.s2, df.t, batch_size, progress, timeout)
                results = run_match_ranker.evaluate(self.model, df.s1, df.s2, expected, batch_size, progress, timeout)
                for key, val in results.items():
                    eval_report[key] = val
                #dprint(eval_report)
                str_report = str(eval_report)
                logger.info(f"{tag} evaluation result (following lines):\n{str_report}")
                return eval_report
        except Exception as e:
            logger.exception(e)
            return eval_report

    def load_eval_data(self, path):
        df = training.Trainer.load_eval_data(self, path)
        df['last_pred'] = None
        df['last_pred1'] = None
        df['last_pred2'] = None
        df['last_reply2'] = None
        df['last_t2'] = None
        cdata = self.config.data
        loss_method = cdata.train.loss_method
        #dprint(loss_method)
        if loss_method:
            df['last_logit'] = pd.Series(None, dtype=object)
            df['last_dist'] = pd.Series(None, dtype=object)
        return df

    def test_sample(self, sample, msg):
        cdata = self.config.data
        loss_method = cdata.train.loss_method
        logger.info(msg)
        logger.info(f'  index: {sample.name}')
        logger.info(f'  query: {sample.s1}')
        if loss_method.startswith('class'):
            logger.info(f'  reply: {sample.s2}')
            logger.info(f'  expected score: {sample.t}')
            logger.info(f'  pred score: {sample.last_pred}')
            logger.info(f'  loss: {sample.criterion}')
            logger.info(f'  logit: {sample.last_logit}')
            logger.info(f'  dist: {sample.last_dist}')
        elif loss_method.startswith('point'):
            logger.info(f'  reply: {sample.s2}')
            logger.info(f'  expected score: {sample.t}')
            logger.info(f'  pred score: {sample.last_pred}')
            logger.info(f'  loss: {sample.criterion}')
        elif loss_method.startswith('pair'):
            logger.info(f'  reply 1: {sample.s2}')
            logger.info(f"    target score: {sample.t}")
            logger.info(f"    pred score: {sample.last_pred1}")
            logger.info(f'  reply2: {sample.last_reply2}')
            logger.info(f"    target score: {sample.last_t2}")
            logger.info(f"    pred score: {sample.last_pred2}")
        else:
            raise ValueError(f"Unknown loss method: {loss_method}")

    def test_model(self):
        if self.dev_df is not None:
            df = self.dev_df
        else:
            return False
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
        cls.add_argument(group, default.model.num_encoder_layers, '--num-encoder-layers', '--encoder-layers', '--el', type=int, help='Number of encoder layers')
        cls.add_argument(group, default.model.num_blocks, '--num-blocks', '--blocks', '--nb', type=int, help='Number of main component blocks')
        cls.add_argument(group, default.model.match_pooler_type, '--match-pooler-type', '--match', '--pooler-type', '--pooler', type=str, choices=['re2', 'compare-aggregate', 'ca'], help='Pooler module type')
        cls.add_argument(group, default.model.sequence_pooling, '--sequence-pooling', '--pooling', '--pool', type=str, choices=['max', 'average', 'attention'], help='Sequence pooling method')
        cls.add_argument(group, default.model.ngram_orders, '--ngram-orders', '--ngram', '--orders', type=int, help='N-gram orders (kernel sizes) for sequence convolution', nargs='+')
        group = parser['training']
        cls.add_argument(group, default.train.loss_method, '--loss-method', '--lossmethod', '--loss', type=str, choices=['classify', 'class', 'point-wise', 'point', 'pair-wise', 'pair'], help='Using input feeding of last decoder output state')
        group.add_argument('--eval-timeout', '--evaluation-timeoout', '--test-timeout', type=float, default=None, help=f'Timeout duration for each evaluation batch in seconds (default: {default.log.eval_timeout})')
        return parser

def main():
    training.main(RankerTrainer, 'Ranker')

if __name__ == '__main__':
    main()
