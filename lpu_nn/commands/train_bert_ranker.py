#!/usr/bin/env python3

# system
import time

# 3rd
import pandas as pd
import torch

# local
from lpu.common import logging
from lpu.metrics import ranking
from lpu_nn.common import training
from lpu.common.config import Config
from lpu.common.progress import view as pview
from lpu_nn.modeling.bert_ranker import BertRanker
from lpu_nn.commands import train_bert
from lpu_nn.commands import run_bert_ranker

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

#default = train_bert.default
default = Config(train_bert.default).data
#default = training.default
#default.model.num_segments = 2
#default.model.max_length = 256
#default.train.min_batch_size = 16
default.train.batch_size = 32
default.train.optimizer = 'adam'
default.train.adam_alpha = 5e-5
default.train.warmup_steps = 0
default.train.weight_decay_rate = 1e-2
#default.train.weight_decay_rate = 1e-5
default.train.schedule_num_steps = False
default.log.eval_timeout = 600

#specific = train_bert.specific
specific = Config(train_bert.specific)
sdata = specific.data
sdata.report = ['epoch', 'proc', 'lr', 'acc', 'loss', 'err', 'samples', 'tokens/sec', 'steps', 'elapsed']

class BertRankerTrainer(training.Trainer):
    default = default
    specific = specific
    Model = BertRanker
    def prepare_sample(self, sample, correct=1):
        if getattr(self, 'train_df', None) is None:
            self.load_train_data()
        sample1 = sample
        if correct:
            sample2 = sample
        else:
            while True:
                sample2 = self.train_df.sample(1).iloc[0]
                if sample1.s2 != sample2.s2:
                    break
        raw1 = sample1.s1
        raw2 = sample2.s2
        sample = self.model.prepare_input(raw1, raw2)
        sample['raw1'] = raw1
        sample['raw2'] = raw2
        sample['t'] = correct
        return sample

    def feed_one_batch(self, batch, fallback=False, df=None):
        # using
        cdata = self.config.data
        dtype = self.model.dtype
        report = pd.Series()

        df_correct   = pd.DataFrame([self.prepare_sample(s, 1) for i, s in batch.iterrows()])
        df_incorrect = pd.DataFrame([self.prepare_sample(s, 0) for i, s in batch.iterrows()])

        #correct_seq = self.prepare_batch(input_correct.x)
        correct_seq = self.model.prepare_batch(df_correct.x)
        #correct_segment_seq = self.prepare_batch(input_correct.segment)
        correct_segment_info = self.model.prepare_batch(df_correct.segment_info)
        #incorrect_seq = self.prepare_batch(input_incorrect.x)
        incorrect_seq = self.model.prepare_batch(df_incorrect.x)
        #incorrect_segment_seq = self.prepare_batch(input_incorrect.segment)
        incorrect_segment_info = self.model.prepare_batch(df_incorrect.segment_info)

        correct_scores   = self.model(correct_seq,   segment_info=correct_segment_info)
        incorrect_scores = self.model(incorrect_seq, segment_info=incorrect_segment_info)

        bool_correct = correct_scores > incorrect_scores
        #accuracy = bool_correct.sum() / len(bool_correct)
        #report['acc'] = float(accuracy)
        report['acc'] = float(bool_correct.to(dtype).mean())
        #dprint(bool_correct)

        scores_diff = correct_scores - incorrect_scores
        #loss = F.log( 1 + F.exp(-scores_diff) )
        loss = torch.log( 1 + torch.exp(-scores_diff) )
        #loss_list = loss.data.ravel().tolist()
        loss_list = loss.data.tolist()
        #loss = F.mean(loss)
        loss = loss.mean()
        report['loss'] = float(loss)

        #if not chainer.config.debug:
        #    if not xp.all( xp.isfinite(loss.data) ):
        #        dprint(loss.data)
        #        raise ValueError("loss is NaN")
        #self.model.cleargrads()
        #try:
        #    loss.backward()
        #except Exception as e:
        #    raise e
        #loss.unchain_backward()
        #if chainer.config.train:
        if self.model.training:
            #self.optimizer.update()
            self.update_parameters(loss)
        try:
            #if chainer.config.train:
            if self.model.training:
                df = self.train_df
                cdata.log.fed_samples += len(batch)
                cdata.log.fed_tokens += int(batch.len_s1.sum() + batch.len_s2.sum())
            else:
                df = self.dev_df
                #df.loc[batch.index, 'last_score'] = correct_scores.data.ravel().tolist()
                df.loc[batch.index, 'last_score'] = correct_scores.tolist()
                #df.loc[batch.index, 'last_incorrect_score'] = incorrect_scores.data.ravel().tolist()
                df.loc[batch.index, 'last_incorrect_score'] = incorrect_scores.tolist()
                for index, idvec in zip(batch.index, df_incorrect.s2, strict=True):
                    #dprint(idvec)
                    #dprint(type(idvec))
                    #dprint(idvec[0])
                    #dprint(type(idvec[0]))
                    df.at[index, 'last_incorrect'] = idvec
                    #dprint(df.loc[index, 'last_incorrect'])
                    #dprint(type(df.loc[index, 'last_incorrect']))
                    #dprint(df.loc[index, 'last_incorrect'][0])
                    #dprint(type(df.loc[index, 'last_incorrect'][0]))
            df.loc[batch.index, 'criterion'] = loss_list
        except Exception as e:
            logger.exception(e)
        return report

    # 基底の evaluate は feed_batches / report を受け取るようになっており、
    # 受け流さないと訓練ループからの呼び出しが TypeError になる
    def evaluate(self, tag, df, args, feed_batches=None, report=None):
        cdata = self.config.data
        #batch_size = max(cdata.train.min_batch_size, int(cdata.train.batch_size / 2))
        #batches = list(chainer.iterators.SerialIterator(df, batch_size, repeat=False, shuffle=False))
        eval_report = super().evaluate(tag, df, args, feed_batches, report)
        #eval_report = pd.Series()
        try:
            with torch.no_grad():
            #with chainer.using_config('train', False), chainer.no_backprop_mode():
                #if tag is 'dev':
                #    eval_report = self.feed_batches(batches, report=False)
                #    loss = eval_report.get('loss', math.nan)
                #    acc  = eval_report.get('acc',  math.nan)
                #    #ppl  = eval_report.get('ppl',  math.nan)
                #    #str_msg = "{} average loss: {}, accuracy: {}, ppl: {}".format(tag, loss, acc, ppl)
                #    str_msg = "{} average loss: {}, accuracy: {}".format(tag, loss, acc)
                #    logger.info(str_msg)
                correct_ranks = []
                replies = set()
                replies.update(self.train_df.s2)
                replies.update(df.s2)
                replies = list(replies)
                #for i, row in df.iterrows():
                start = time.time()
                for _index, row in pview(list(df.iterrows()), header='evaluating'):
                    if cdata.log.eval_timeout is not None:
                        if time.time() - start > cdata.log.eval_timeout:
                            break
                    _ranked_replies, _ranked_scores, correct_rank = run_bert_ranker.rank(self.model, row.s1, replies, correct=row.s2, batch_size=cdata.train.batch_size, progress=False)
                    correct_ranks.append(correct_rank)
                for k in (1, 5, 20, 50, 100):
                    p_at_k = ranking.calc_precision_at_k(correct_ranks, k)
                    #logger.info("P@{}: {}".format(k, p_at_k))
                    eval_report[f'p_at_{k}'] = p_at_k
                mrr = ranking.calc_mean_reciprocal_rank(correct_ranks)
                eval_report['mrr'] = mrr
                mean = float(sum(correct_ranks)) / len(correct_ranks)
                eval_report['mr'] = mean
                dprint(eval_report)
                return eval_report
        except Exception as e:
            logger.exception(e)
            return eval_report

    def load_eval_data(self, path):
        #dprint(self.main_fields)
        df = training.Trainer.load_eval_data(self, path)
        #input_correct = pd.DataFrame([self.prepare_sample(s, 1) for i, s in df.iterrows()])
        #input_incorrect = pd.DataFrame([self.prepare_sample(s, 0) for i, s in df.iterrows()])
        #df['correct_seq'] = input_correct.x
        #df['correct_segment'] = input_correct.segment
        #df['incorrect'] = input_incorrect.s2
        #df['incorrect_segment'] = input_incorrect.segment
        df['last_score'] = None
        df['last_incorrect_score'] = None
        #df['last_incorrect'] = input_incorrect.s2
        df['last_incorrect'] = None
        return df

    def setup_model(self, args):
        bert_trainer = None
        if not args.resume:
            if args.pre_trained_model:
                bert_trainer = train_bert.BertTrainer().load_status(args.pre_trained_model)
                dprint(bert_trainer.config.to_json(indent=2))
                self.config['model'] = bert_trainer.config['model']
        super().setup_model(args)
        if bert_trainer:
            # 事前学習モデルの mod_bert をそのまま差し替えるため、語彙が
            # 一致していないと埋め込みの形と設定上の語彙サイズが食い違い、
            # 保存したチェックポイントを読み直せなくなる。
            # 作業ディレクトリごとにトークナイザを学習するので、既定では
            # まず一致しない
            pre_vocab_size = bert_trainer.model.mod_bert.vocab_size
            own_vocab_size = self.model.mod_bert.vocab_size
            if pre_vocab_size != own_vocab_size:
                raise ValueError(
                    "vocabulary size differs between the pre-trained model "
                    f"({pre_vocab_size}) and this one ({own_vocab_size}); "
                    "pass --sentencepiece <pre-trained workdir>/sp.model "
                    "to reuse the pre-trained tokenizer")
            # training.infomain は複数プロセス学習用の補助で、
            # 移植時に落とされており存在しない
            logger.info("copying parameters of pre-trained model into fine-tuning model")
            #utils.copy_model(bert_trainer.model, self.model)
            #utils.copy_model(bert_trainer.model.L_bert, self.model.L_bert)
            self.model.mod_bert = bert_trainer.model.mod_bert

    def test_sample(self, sample, msg):
        # self.vocab は廃止され、idmaps 経由で引くようになった
        vocab = self.idmaps['seq']
        logger.info(msg)
        logger.info(f'  index: {sample.name}')
        logger.info(f'  query: {vocab.decode(sample.s1)}')
        logger.info(f'  correct: {vocab.decode(sample.s2)}')
        #logger.info('  incorrect: {}'.format(vocab.decode_ids(sample.incorrect)))
        #dprint(sample.last_incorrect)
        #dprint(sample.s2)
        #dprint(vocab.decode(sample.s2))
        logger.info(f'  incorrect: {vocab.decode(sample.last_incorrect)}')
        logger.info(f"  last score: {sample.last_score}")
        logger.info(f"  last incorrect score: {sample.last_incorrect_score}")
        #logger.info('  original: <{}>\t{}'.format(sample.t[0], vocab.decode_ids(sample.ref)))
        #cls, pred = self.model.predict(sample.x, sample.segment)
        #cls, pred = sample.cls, sample.pred
        #logger.info('  predicted: <{}>\t{}'.format(cls, vocab.decode_ids(pred)))
        #logger.info("  last perplexity: {}".format(sample.criterion))

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

    # 基底と train_bert は classmethod で default も受け取る。
    # --help はモデル固有の既定値を反映するために 2 引数で呼び直すため、
    # staticmethod のままだと --help が TypeError になる
    @classmethod
    def create_parser(cls, model_name, default=None):
        if default is None:
            default = cls.default
        parser = train_bert.BertTrainer.create_parser(model_name, default)
        #group = parser['files']
        group = parser['model']
        # '--pre' と '-P' は基底パーサの --preset が既に使っており、
        # 残したままだと argparse がパーサ構築時に衝突で落ちる
        group.add_argument('--pre-trained-model', '--pre-trained', type=str, help='Path to the pre-trained BERT model to load initially')
        group.add_argument('--eval-timeout', '--evaluation-timeoout', '--test-timeout', type=float, default=None, help=f'Timeout duration for evaluation in seconds (default: {default.log.eval_timeout})')
        return parser

def main():
    training.main(BertRankerTrainer, 'BERT Ranker')

if __name__ == '__main__':
    main()

