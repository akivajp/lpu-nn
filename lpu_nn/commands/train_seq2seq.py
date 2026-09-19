#!/usr/bin/env python3

# system
import os
from lpu_nn.common.args import strtobool

# 3rd party
import pandas as pd
import torch
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

# local
from lpu_nn.common import criteria
from lpu.common import logging
from lpu_nn.common import training
from lpu.common.config import Config
from lpu.common.files import safe_remove
from lpu.common.progress import view as pview
from lpu_nn.common.utils import purge_tensor
from lpu_nn.modeling.encoder_decoder import EncoderDecoder

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

default = Config(training.default).data
# general
default.model.task = 'seq2seq'
default.model.architecture = 'encoder-decoder'
#default.model.main_component = None
default.model.encoder_type = None
default.model.decoder_type = None
default.model.num_layers = None
default.model.bidirectional_encoder = None
default.model.residual_connection = None
default.model.attention_type = None
default.model.local_attention = None
default.model.input_feeding = None
#default.model.share_embedding = None
#default.model.activation = None
#default.model.max_length = 256
default.model.max_length = 512
#default.train.dropout_ratio = 0.3
default.train.dropout_ratio = 0.1
#default.train.batch_size = 64
default.train.batch_size = 32
#default.train.batch_size = 8192
default.train.batch_type = 'samples'
#default.train.batch_type = 'tokens'
default.train.min_batch_size = 16
# warming up
#default.train.factor = 1
#default.train.factor = 2
#default.train.warmup_steps = 4000
default.train.warmup_steps = 16000
#default.train.warmup_steps = 32000
# optimizers
#default.train.optimizer = 'adabound'
default.train.optimizer = 'lamb'
#default.train.sgd_learning_rate = 0.2
default.train.sgd_learning_rate = 0.1
#default.train.sgd_learning_rate = 0.05
#default.train.adam_alpha = 5e-5
#default.train.adam_alpha = 2e-4
#default.train.adam_alpha = 0.001
#default.train.adam_alpha = 1e-4
default.train.adam_alpha = 1e-3
default.train.adam_beta1 = 0.9
#default.train.adam_beta2 = 0.98
#default.train.adam_beta2 = 0.998
default.train.adam_beta2 = 0.999
#default.train.adam_eps = 1e-9
default.train.adam_eps = 1e-8
#default.train.final_learning_rate = 1.0
#default.train.gamma = 1e-3
#default.train.gamma = 1e-4
#default.train.gamma = 1e-5
default.train.adabound_gamma = 1e-5
#default.train.weight_decay_rate = 1e-2
default.train.weight_decay_rate = 1e-3
#default.train.weight_decay_rate = 1e-4
#default.train.weight_decay_rate = 1e-5
#default.train.weight_decay_rate = 0
# loss func
default.train.loss = 'smooth'
# multi-step
#default.train.schedule_num_steps = True
default.train.schedule_num_steps = False
# log
default.log.eval_timeout = 60
default.log.fed_src_tokens = 0
default.log.fed_trg_tokens = 0

specific = Config()
sdata = specific.data
sdata.report = ['epoch', 'proc', 'lr', 'acc', 'acc_seq', 'ppl', 'loss', 'pcost', 'err', 'samples', 'steps', 'tokens/s', 'elapsed']
sdata.format = {}
sdata.format.input = {}
sdata.format.input.x = 'seq'
sdata.format.output = {}
sdata.format.output.t = 'seq'

class Seq2SeqTrainer(training.Trainer):
    default  = default
    specific = specific
    Model    = EncoderDecoder

    #def __init__(self, args):
    #    super(Seq2SeqTrainer, self).__init__(args)

    def calc_loss(self, report, logits, t, state):
        cdata = self.config.data
        padding = self.model.padding
        expected = t[:,1:None]
        time_penalty = cdata.train.time_penalty
        ponder_cost = state.get('ponder_cost', 0.0)
        if cdata.train.loss == 'smooth':
            loss_smooth = criteria.smoothed_cross_entropy(logits, expected, smooth=0.1, ignore_index=padding)
            loss = loss_smooth + ponder_cost * time_penalty
            report['loss_smooth'] = float(loss_smooth.detach())
        else:
            loss_xent = criteria.cross_entropy(logits, expected, ignore_index=padding)
            loss = loss_xent + ponder_cost * time_penalty
            report['loss_xent'] = float(loss_xent.detach())
        report['loss'] = float(loss.detach())
        return loss

    def calc_perplexity(self, report, logits, t):
        dtype = logits.dtype
        expected = t[:,1:None]
        padding = self.model.padding
        batch_ppl = criteria.perplexity(logits, expected, ignore_index=padding, reduction='hmean')
        batch_ppl_safe = purge_tensor(batch_ppl, torch.isfinite(batch_ppl), 0.0)
        batch_ppl_safe_count = torch.isfinite(batch_ppl).sum().to(dtype)
        #report['ppl'] = float(batch_ppl_safe.sum() / batch_ppl_safe_count)
        # 報告用の値なので勾配は不要。detach しないと torch が警告する
        report['ppl'] = float(batch_ppl_safe.div(batch_ppl_safe_count).sum().detach())
        batch_ppl_safe = purge_tensor(batch_ppl, torch.isfinite(batch_ppl), -1.0)
        list_ppl = batch_ppl_safe.tolist()
        return list_ppl

    def calc_ponder_cost(self, report, x, t, state):
        dtype = self.model.dtype
        mask_x = (x != self.model.padding)
        mask_t = (t != self.model.padding)
        #dprint(format_state(model_state))
        encoder_transformer_state = state.get('encoder_state',{}).get('transformer_state',{})
        decoder_transformer_state = state.get('decoder_state',{}).get('transformer_state',{})
        if isinstance(encoder_transformer_state, dict):
            encoder_ponder_cost = encoder_transformer_state.get('ponder_cost', 0)
        else:
            encoder_ponder_cost = 0
        if isinstance(decoder_transformer_state, dict):
            decoder_ponder_cost = decoder_transformer_state.get('ponder_cost', 0)
        else:
            decoder_ponder_cost = 0
        if isinstance(encoder_ponder_cost, torch.Tensor):
            encoder_ponder_cost = (encoder_ponder_cost.sum(1) / mask_x.to(dtype).sum(1)).mean()
        if isinstance(decoder_ponder_cost, torch.Tensor):
            decoder_ponder_cost = (decoder_ponder_cost.sum(1) / mask_t.to(dtype).sum(1)).mean()
        ponder_cost = encoder_ponder_cost + decoder_ponder_cost
        # 報告用の値なので勾配は不要。detach しないと torch が警告する
        if isinstance(encoder_ponder_cost, torch.Tensor):
            report['encoder_ponder_cost'] = float(encoder_ponder_cost.detach())
        if isinstance(decoder_ponder_cost, torch.Tensor):
            report['decoder_ponder_cost'] = float(decoder_ponder_cost.detach())
        if isinstance(ponder_cost, torch.Tensor):
            report['ponder_cost'] = float(ponder_cost.detach())
        state['ponder_cost'] = ponder_cost
        return ponder_cost

    def feed_one_batch(self, batch, fallback=False, df=None):
        padding = self.model.padding
        cdata = self.config.data
        dtype = self.model.dtype

        if self.model.training:
            if cdata.model.max_steps is not None:
                if cdata.train.schedule_num_steps:
                    max_steps = getattr(self.model, 'max_steps', None)
                    if max_steps is None:
                        if cdata.model.universal:
                            self.set_max_steps(2)
                        else:
                            self.set_max_steps(1)
                        dprint(self.model.max_steps)

        report = pd.Series()
        batch_x = self.model.prepare_batch(batch.x)
        batch_t = self.model.prepare_batch(batch.t)
        mask_x = (batch_x != padding)
        mask_t = (batch_t != padding)

        trg_id_seq_input  = batch_t[:,0:-1]
        trg_id_seq_expect = batch_t[:,1:None]
        logits = self.model(batch_x, trg_id_seq_input)
        model_state = self.model.get_state()

        #batch_ppl = criteria.perplexity(logits, trg_id_seq_expect, ignore_index=padding, reduction='hmean')
        #batch_ppl_safe = purge_tensor(batch_ppl, torch.isfinite(batch_ppl), 0.0)
        #batch_ppl_safe_count = torch.isfinite(batch_ppl).sum().to(dtype)
        ##report['ppl'] = float(batch_ppl.mean())
        #report['ppl'] = float(batch_ppl_safe.sum() / batch_ppl_safe_count)
        #list_ppl = batch_ppl.tolist()
        #del batch_ppl
        list_ppl = self.calc_perplexity(report, logits, batch_t)

        accuracy = float( criteria.accuracy(logits, trg_id_seq_expect, ignore_index=padding) )
        report['acc'] = accuracy
        report['acc_seq'] = criteria.sequence_accuracy(logits, trg_id_seq_expect, ignore_index=padding).item()

        ponder_cost = self.calc_ponder_cost(report, batch_x, batch_t, model_state)
        loss = self.calc_loss(report, logits, batch_t, model_state)

        if self.model.training:
            try:
                self.update_parameters(loss, report=report)
            except Exception as e:
                del loss
                #self.update_to_reduce_ponder_cost(ponder_cost, time_penalty)
                self.update_to_reduce_ponder_cost(ponder_cost, cdata.train.time_penalty)
                raise e
        try:
            #if chainer.config.train:
            if self.model.training:
                #self.train_df.loc[batch.index, 'criterion'] = list_ppl
                cdata.log.fed_samples += len(batch)
                num_src_tokens = int(mask_x.sum())
                num_trg_tokens = int(mask_t.sum())
                cdata.log.fed_src_tokens += num_src_tokens
                cdata.log.fed_trg_tokens += num_trg_tokens
                cdata.log.fed_tokens = cdata.log.fed_src_tokens + cdata.log.fed_trg_tokens
            #else:
            #    #dprint(batch.index)
            #    #dprint(list_ppl)
            #    self.dev_df.loc[batch.index, 'criterion'] = list_ppl
            if df is not None:
                df.loc[batch.index, 'criterion'] = list_ppl
        except Exception as e:
            logger.exception(e)

        if self.model.training:
            if cdata.model.max_steps is not None:
                if cdata.train.schedule_num_steps:
                    if accuracy >= 0.6:
                        max_steps = self.model.max_steps
                        new_max_steps = self.set_max_steps(max_steps+1)
                        if new_max_steps > max_steps:
                            dprint(self.model.max_steps)
        return report

    def load_eval_data(self, path):
        df = super().load_eval_data(path)
        df['ref'] = df.t
        return df

    def evaluate(self, tag, df, args, feed_batches=None, report=None):
        cdata = self.config.data
        if tag == 'test':
            if feed_batches is None:
                feed_batches = True
        eval_report = super().evaluate(tag, df, args, feed_batches, report)
        batch_size = max(cdata.train.min_batch_size, int(cdata.train.batch_size / 2))
        batches = list(training.build_batches(df, batch_size, cdata.train.batch_type))
        vocab = self.model.vocab
        try:
            with torch.no_grad():
                eval_bleu = False
                if 'ref' in df:
                    #ref = [[r] for r in df.ref]
                    ref = [[vocab.convert(r, to='tokens')] for r in df.ref]
                    eval_bleu = True
                if eval_bleu:
                    result = []
                    if 'pred' in df:
                        result = df.pred.tolist()
                    else:
                        for batch in pview(batches, header=f'evaluating {tag} data'):
                            #max_length = batch.len_x.max() * 1 + cdata.log.epoch
                            #batch_result = self.model.generate(batch.x.tolist(), max_length=max_length)
                            #batch_result = self.model.generate(self.convert_to_batch(batch.x), max_length=max_length)
                            if args.eval_timeout is not None:
                                #batch_result = self.model.generate(batch.x, max_length=max_length, timeout=args.eval_timeout)
                                batch_result = self.model.generate(batch.x, timeout=args.eval_timeout)
                            else:
                                #batch_result = self.model.generate(batch.x, max_length=max_length)
                                batch_result = self.model.generate(batch.x)
                            #result = result + self.restore_batch(batch_result, to=list)
                            result = result + self.model.restore_batch(batch_result, to=list)
                        result = [vocab.clean_ids(idvec) for idvec in result]
                        result = [vocab.convert(idvec, to='tokens') for idvec in result]
                        result = [vocab.remove_unk(tokens) for tokens in result]
                    if result:
                        bleu_score = corpus_bleu(ref, result, smoothing_function=SmoothingFunction().method1)
                        outpath = os.path.join(args.workdir, 'record.tmp', f'pred_{tag}.txt')
                        safe_remove(outpath, log=False)
                        with open(outpath, 'w', encoding='utf-8', errors='backslashreplace') as fobj:
                            logger.info(f"writing generation results into: {outpath}")
                            for sent in result:
                                fobj.write(vocab.convert(sent, str))
                                fobj.write("\n")
                        logger.info(f"{tag} bleu: {bleu_score * 100} [%]")
                        eval_report['bleu'] = bleu_score
                logger.info(f"{tag} evaluation result (following lines):\n{eval_report!s}")
                return eval_report
        except Exception as e:
            logger.exception(e)
            return eval_report

    def test_sample(self, sample, msg):
        vocab = self.model.vocab
        cdata = self.config.data
        logger.info(msg)
        #logger.info('  index: {}'.format(sample.index))
        logger.info(f'  index: {sample.name}')
        #logger.info('  input: {}'.format(vocab.decode(sample.x)))
        logger.info(f'  input: {sample.x}')
        #logger.info('  ref: {}'.format(vocab.decode(sample.t)))
        logger.info(f'  ref: {sample.t}')
        #pred = self.model.generate(sample.x, max_length=sample.len_x+cdata.log.epoch)
        #pred = self.model.generate(self.convert_to_batch(sample.x), max_length=sample.len_x+cdata.log.epoch)
        pred = self.model.generate(sample.x, max_length=sample.len_x+cdata.log.epoch)
        #logger.info("  pred: {}".format(vocab.decode(pred)))
        #logger.info("  pred: {}".format(self.restore_batch(pred, str)))
        logger.info(f"  pred: {self.model.restore_batch(pred, str)}")
        logger.info(f"  last perplexity: {sample.criterion}")

    def test_model(self):
        if self.dev_df is not None:
            df = self.dev_df
        else:
            df = self.train_df
        df = df[df.criterion >= 0]
        # criterion がまだ書き込まれていない (全て NaN の) 段階では、
        # 絞り込んだ結果が空になり idxmin/idxmax が例外を送出する
        if df.empty:
            logger.debug("no sample has a computed criterion yet, skipping the test")
            return
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
        cls.add_argument(group, default.model.architecture, '--architecture', '--arch', type=str, choices=['encoder-decoder'], help='Neural network architecture')
        cls.add_argument(group, None, '--main-component', '--main-layer', '--layer-type', '--layer', type=str, choices=['lstm', 'rnn', 'transformer', 'trans'], help='Main layer component')
        cls.add_argument(group, default.model.num_layers, '--num-layers', '--layers', '-L', type=int, help='Number of RNN/Transformer layers')
        cls.add_argument(group, default.model.encoder_type, '--encoder-type', '--encoder', '--enc', type=str, choices=['lstm', 'rnn', 'transformer', 'trans', 'bert'], help='Encoder module type')
        cls.add_argument(group, default.model.decoder_type, '--decoder-type', '--decoder', '--dec', type=str, choices=['lstm', 'rnn', 'transformer', 'trans'], help='Decoder module type')
        cls.add_argument(group, default.model.bidirectional_encoder, '--bidirectional-encoder', '--bidirectional', '--brnn', type=strtobool, nargs='?', const=True, help='Using bidirectional encoder')
        cls.add_argument(group, default.model.residual_connection, '--residual-connection', '--residual', '--rc', type=strtobool, nargs='?', const=True, help='Using residual connection for RNN encoder/decoder layers')
        cls.add_argument(group, default.model.attention_type, '--attention-type', '--global-attention-type', '--attention', '--global-attention', '--att', '--at', '--ga', type=str, choices=['dot', 'concat', 'general', 'mlp', 'none'], help='Decoder attention type for RNN encoder-decoder models')
        cls.add_argument(group, default.model.local_attention, '--local-attention', '--local', '--la', type=strtobool, nargs='?', const=True, help='Using local attention mechanism for RNN encoder-decoder models')
        cls.add_argument(group, default.model.input_feeding, '--input-feeding', '--feeding', '--if', type=strtobool, nargs='?', const=True, help='Using input feeding of last decoder output state')
        group = parser['training']
        group.add_argument('--loss-function', '--lossfunc', '--loss', type=str, default=None, choices=['xent', 'smooth'], help=f'Loss function (default: {default.train.loss})')
        group.add_argument('--eval-timeout', '--evaluation-timeoout', '--test-timeout', type=float, default=None, help=f'Timeout duration for each evaluation batch in seconds (default: {default.log.eval_timeout})')
        return parser

def main():
    training.main(Seq2SeqTrainer, 'Sequence-to-Sequence Model')

if __name__ == '__main__':
    main()

