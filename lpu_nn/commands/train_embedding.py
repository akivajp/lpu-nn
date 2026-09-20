#!/usr/bin/env python3

# system
from lpu_nn.common.args import strtobool

# 3rd
import pandas as pd
from torch import nn

# local
from lpu_nn.common import criteria
from lpu.common import logging
from lpu_nn.common import training
from lpu.common.config import Config

from lpu_nn import modeling
from lpu_nn.modeling.embeddings import ContextualStringEmbedding
from lpu_nn.modeling.language_models import LanguageModel

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

#default = training.default
default = Config(training.default).data
# general
default.model.task = 'gen_seq'
#default.model.architecture = 'encoder-decoder'
#default.model.main_component = None
#default.model.encoder_type = None
#default.model.decoder_type = None
#default.model.bidirectional_encoder = None
default.model.residual_connection = None
#default.model.attention_type = None
#default.model.local_attention = None
#default.model.input_feeding = None
#default.model.share_embedding = None
#default.model.activation = None
default.model.max_length = 256
#default.model.character_level = True
#default.train.dropout_ratio = 0.3
default.train.dropout_ratio = 0.1
default.train.batch_size = 64
#default.train.batch_size = 8192
default.train.batch_type = 'samples'
#default.train.batch_type = 'tokens'
default.train.min_batch_size = 16
# warming up
#default.train.factor = 1
#default.train.factor = 2
default.train.warmup_steps = None
#default.train.warmup_steps = 16000
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
sdata.report = ['epoch', 'proc', 'lr', 'acc', 'ppl', 'loss', 'pcost', 'err', 'samples', 'steps', 'tokens/s', 'elapsed']
sdata.format = {}
sdata.format.input = {}
sdata.format.input.x = 'seq'
#sdata.format.output = {}
#sdata.format.output.t = 'seq'

class CharacterLanguageModelLoss(modeling.Module):
    def __init__(self, **params):
        # parameters
        hparams = self.get_config(**params)
        self.padding = hparams['padding']
        self.char_embed_size = hparams['char_embed_size']
        self.hidden_size = hparams['hidden_size']
        self.out_size = hparams['out_size']
        self.share_embedding = hparams['share_embedding']
        self.share_generator = hparams['share_generator']
        # modules
        self.mod_embed_string = ContextualStringEmbedding(**params)
        self.mod_generate_forward = nn.Linear(self.out_size, 256, bias=False)
        self.mod_generate_backward = nn.Linear(self.out_size, 256, bias=False)
        if self.share_embedding:
            self.mod_generate_forward.weight = self.mod_embed_string.embed_char.weight
            self.mod_generate_backward.weight = self.mod_embed_string.embed_char.weight
        if self.share_generator:
            self.mod_generate_backward.weight = self.mod_generate_forward.weight

    @classmethod
    def get_config(cls, **params):
        params.setdefault('padding', 1)
        params.setdefault('char_embed_size', 256)
        params.setdefault('hidden_size', 256)
        params['out_size'] = params['hidden_size'] * 2
        params.setdefault('share_embedding', True)
        params.setdefault('share_generator', True)
        return params

    def forward(self, seq_ids):
        # seq_ids.shape : (B, L)
        seq_mask = seq_ids != self.padding
        seq_emb = self.embed_char(seq_ids)
        for forward in [True, False]:
            if forward:
                rnn = self.mod_embed_string.mod_rnn_forward
                generate = self.mod_generate_forward
                seq_input = seq_emb[:,0:-1]
                mask_input = seq_mask[:,0:-1]
                expected = seq_ids[:,1:None]
            else:
                rnn = self.mod_embed_string.mod_rnn_backward
                generate = self.mod_generate_backward
                seq_input = seq_emb.flip()[:,0:-1]
                mask_input = seq_mask.flip()[:,0:-1]
                expected = seq_ids.flip()[:,1:None]
            h = rnn(seq_input, mask_input) # (B, L, H)
            logits = generate(h) # (B, L, C)
            if forward:
                loss_forward = criteria.cross_entropy(logits, expected, ignore_index=self.padding)
                ppl_forward = criteria.perplexity(logits, expected, ignore_index=self.padding, reduction='hmean')
                accuracy_forward = criteria.accuracy(logits, expected, ignore_index=self.padding, reduction='hmean')
                self.last_state['loss_forward'] = loss_forward
                self.last_state['ppl_forward'] = ppl_forward
                self.last_state['accuracy_forward'] = accuracy_forward
            else:
                loss_backward = criteria.cross_entropy(logits, expected, ignore_index=self.padding)
                ppl_backward = criteria.perplexity(logits, expected, ignore_index=self.padding, reduction='hmean')
                accuracy_backward = criteria.accuracy(logits, expected, ignore_index=self.padding, reduction='hmean')
                self.last_state['loss_backward'] = loss_backward
                self.last_state['ppl_backward'] = ppl_backward
                self.last_state['accuracy_backward'] = accuracy_backward
        self.last_state['ppl'] = (ppl_forward + ppl_backward) / 2.0
        self.last_state['accuracy'] = (accuracy_forward + accuracy_backward) / 2.0
        return loss_forward + loss_backward

    def prepare_batch(self, seq):
        return self.mod_embed_string.prepare_batch(seq)

    def reset_state(self):
        self.mod_embed_string.reset_state()
        return self

class EmbeddingTrainer(training.Trainer):
    default  = default
    specific = specific
    #Model    = CharacterLanguageModelLoss
    Model    = LanguageModel

    def feed_one_batch(self, batch, fallback=False, df=None):
        padding = self.model.padding
        cdata = self.config.data

        report = pd.Series()
        batch_x = self.model.prepare_batch(batch.x)
        mask_x = (batch_x != padding)
        input = batch_x[:,0:-1]
        expected = batch_x[:,1:None]

        #loss = self.model(batch_x)
        #logits = self.model(batch_x)
        logits = self.model(input) # (B, V, L)

        loss_forward = criteria.cross_entropy(logits, expected, ignore_index=padding)
        ppl_forward = criteria.perplexity(logits, expected, ignore_index=padding, reduction='hmean')
        accuracy_forward = criteria.accuracy(logits, expected, ignore_index=padding, reduction='hmean')

        loss = loss_forward
        ppl = ppl_forward
        accuracy = accuracy_forward

        #for forward in [True, False]:
        #    if forward:
        #        rnn = self.mod_embed_string.mod_rnn_forward
        #        generate = self.mod_generate_forward
        #        seq_input = seq_emb[:,0:-1]
        #        mask_input = seq_mask[:,0:-1]
        #        expected = seq_ids[:,1:None]
        #    else:
        #        rnn = self.mod_embed_string.mod_rnn_backward
        #        generate = self.mod_generate_backward
        #        seq_input = seq_emb.flip()[:,0:-1]
        #        mask_input = seq_mask.flip()[:,0:-1]
        #        expected = seq_ids.flip()[:,1:None]
        #    h = rnn(seq_input, mask_input) # (B, L, H)
        #    logits = generate(h) # (B, L, C)
        #    if forward:
        #        loss_forward = criteria.cross_entropy(logits, expected, ignore_index=self.padding)
        #        ppl_forward = criteria.perplexity(logits, expected, ignore_index=self.padding, reduction='hmean')
        #        accuracy_forward = criteria.accuracy(logits, expected, ignore_index=self.padding, reduction='hmean')
        #        self.last_state['loss_forward'] = loss_forward
        #        self.last_state['ppl_forward'] = ppl_forward
        #        self.last_state['accuracy_forward'] = accuracy_forward
        #    else:
        #        loss_backward = criteria.cross_entropy(logits, expected, ignore_index=self.padding)
        #        ppl_backward = criteria.perplexity(logits, expected, ignore_index=self.padding, reduction='hmean')
        #        accuracy_backward = criteria.accuracy(logits, expected, ignore_index=self.padding, reduction='hmean')
        #        self.last_state['loss_backward'] = loss_backward
        #        self.last_state['ppl_backward'] = ppl_backward
        #        self.last_state['accuracy_backward'] = accuracy_backward
        #self.last_state['ppl'] = (ppl_forward + ppl_backward) / 2.0
        #self.last_state['accuracy'] = (accuracy_forward + accuracy_backward) / 2.0
        #return loss_forward + loss_backward

        #list_ppl = model_state['ppl'].tolist()
        #report['ppl'] = model_state['ppl'].mean()
        #report['acc'] = model_state['accuracy'].mean()
        list_ppl = ppl.tolist()
        report['loss'] = float(loss.mean())
        report['ppl'] = float(ppl.mean())
        report['acc'] = float(accuracy.mean())

        if self.model.training:
            try:
                self.update_parameters(loss)
            except Exception as e:
                del loss
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
        except Exception as e:
            logger.exception(e)

        return report

    #def load_eval_data(self, path):
    #    df = super(Seq2SeqTrainer, self).load_eval_data(path)
    #    df['ref'] = df.t
    #    return df

    #def evaluate(self, tag, df, args):
    #    cdata = self.config.data
    #    eval_report = super(Seq2SeqTrainer,self).evaluate(tag, df, args)
    #    batch_size = max(cdata.train.min_batch_size, int(cdata.train.batch_size / 2))
    #    batches = list(training.build_batches(df, batch_size, cdata.train.batch_type))
    #    vocab = self.model.vocab
    #    try:
    #        with torch.no_grad():
    #            eval_bleu = False
    #            if 'ref' in df:
    #                #ref = [[r] for r in df.ref]
    #                ref = [[vocab.convert(r, to='tokens')] for r in df.ref]
    #                eval_bleu = True
    #            if eval_bleu:
    #                outpath = os.path.join(args.workdir, 'pred_{}.txt'.format(tag))
    #                result = []
    #                if 'pred' in df:
    #                    result = df.pred.tolist()
    #                else:
    #                    for batch in pview(batches, header='evaluating {} data'.format(tag)):
    #                        #max_length = batch.len_x.max() * 1 + cdata.log.epoch
    #                        #batch_result = self.model.generate(batch.x.tolist(), max_length=max_length)
    #                        #batch_result = self.model.generate(self.convert_to_batch(batch.x), max_length=max_length)
    #                        if args.eval_timeout is not None:
    #                            #batch_result = self.model.generate(batch.x, max_length=max_length, timeout=args.eval_timeout)
    #                            batch_result = self.model.generate(batch.x, timeout=args.eval_timeout)
    #                        else:
    #                            #batch_result = self.model.generate(batch.x, max_length=max_length)
    #                            batch_result = self.model.generate(batch.x)
    #                        #result = result + self.restore_batch(batch_result, to=list)
    #                        result = result + self.model.restore_batch(batch_result, to=list)
    #                    result = [vocab.clean_ids(idvec) for idvec in result]
    #                    result = [vocab.convert(idvec, to='tokens') for idvec in result]
    #                    result = [vocab.remove_unk(tokens) for tokens in result]
    #                if result:
    #                    #dprint(ref[:1])
    #                    #dprint(result[:1])
    #                    bleu_score = corpus_bleu(ref, result, smoothing_function=SmoothingFunction().method1)
    #                    safe_remove(outpath, log=False)
    #                    with open(outpath, 'w', encoding='utf-8', errors='backslashreplace') as fobj:
    #                        logger.info("writing generation results into: {}".format(outpath))
    #                        for sent in result:
    #                            fobj.write(vocab.convert(sent, str))
    #                            fobj.write("\n")
    #                    logger.info("{} bleu: {} [%]".format(tag, bleu_score * 100))
    #                    eval_report['bleu'] = bleu_score
    #            return eval_report
    #    except Exception as e:
    #        logger.exception(e)
    #        return eval_report

    def test_sample(self, sample, msg):
        logger.info(msg)
        #logger.info('  index: {}'.format(sample.index))
        logger.info(f'  index: {sample.name}')
        #logger.info('  input: {}'.format(vocab.decode(sample.x)))
        logger.info(f'  input: {sample.x}')
        #logger.info('  ref: {}'.format(vocab.decode(sample.t)))
        #logger.info('  ref: {}'.format(sample.t))
        #pred = self.model.generate(sample.x, max_length=sample.len_x+cdata.log.epoch)
        #pred = self.model.generate(self.convert_to_batch(sample.x), max_length=sample.len_x+cdata.log.epoch)
        #pred = self.model.generate(sample.x, max_length=sample.len_x+cdata.log.epoch)
        #logger.info("  pred: {}".format(vocab.decode(pred)))
        #logger.info("  pred: {}".format(self.restore_batch(pred, str)))
        #logger.info("  pred: {}".format(self.model.restore_batch(pred, str)))
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
        #cls.add_argument(group, default.model.architecture, '--architecture', '--arch', type=str, choices=['encoder-decoder'], help='Neural network architecture')
        #cls.add_argument(group, default.model.main_component, '--main-component', '--main-layer', '--layer-type', '--layer', type=str, choices=['lstm', 'transformer'], help='Main layer component')
        #cls.add_argument(group, default.model.bidirectional_encoder, '--bidirectional-encoder', '--bidirectional', '--brnn', type=strtobool, nargs='?', const=True, help='Using bidirectional encoder')
        cls.add_argument(group, default.model.residual_connection, '--residual-connection', '--residual', '--rc', type=strtobool, nargs='?', const=True, help='Using residual connection for RNN encoder/decoder layers')
        #cls.add_argument(group, default.model.attention_type, '--attention-type', '--global-attention-type', '--attention', '--global-attention', '--att', '--at', '--ga', type=str, choices=['dot', 'concat', 'general', 'mlp', 'none'], help='Decoder attention type for RNN encoder-decoder models')
        #cls.add_argument(group, default.model.local_attention, '--local-attention', '--local', '--la', type=strtobool, nargs='?', const=True, help='Using local attention mechanism for RNN encoder-decoder models')
        #cls.add_argument(group, default.model.input_feeding, '--input-feeding', '--feeding', '--if', type=strtobool, nargs='?', const=True, help='Using input feeding of last decoder output state')
        cls.add_argument(group, default.model.character_level, '--character-level', '--char', '--cl', type=strtobool, nargs='?', const=True, help='Using character-based model')
        group = parser['training']
        #group.add_argument('--loss-function', '--lossfunc', '--loss', type=str, default=None, choices=['xent', 'smooth'], help='Loss function (default: {})'.format(default.train.loss))
        group.add_argument('--eval-timeout', '--evaluation-timeoout', '--test-timeout', type=float, default=None, help=f'Timeout duration for each evaluation batch in seconds (default: {default.log.eval_timeout})')
        return parser

def main():
    training.main(EmbeddingTrainer, 'Language Model')

if __name__ == '__main__':
    main()

