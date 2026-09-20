#!/usr/bin/env python3

# system

# 3rd
import pandas as pd
import torch

# local
from lpu_nn.common import criteria
from lpu.common import logging
from lpu.metrics import ranking
from lpu_nn.common import training
from lpu.common.config import Config
from lpu_nn.modeling.bert_classifier import BertClassifier
from lpu_nn.commands import train_bert
from lpu_nn.commands import run_bert_classifier

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

#default = train_bert.default
default = Config(train_bert.default).data
#default = training.default
#default.model.max_length = 256
default.train.batch_size = 32
default.train.optimizer = 'adam'
default.train.adam_alpha = 5e-5
default.train.warmup_steps = 0
default.train.weight_decay_rate = 1e-5
default.train.schedule_num_steps = False

#specific = train_bert.specific
specific = Config(train_bert.specific)
sdata = specific.data
sdata.report = ['epoch', 'proc', 'lr', 'acc', 'ppl', 'loss', 'pcost', 'err', 'samples', 'tokens/sec', 'steps', 'elapsed']

class BertClassifierTrainer(training.Trainer):
    default = default
    specific = specific
    Model = BertClassifier

    def prepare_sample(self, sample):
        if getattr(self, 'train_df', None) is None:
            self.load_train_data()
        #sample = pd.Series()
        raw1 = sample.s1
        raw2 = sample.s2
        sample = self.model.prepare_input(raw1)
        sample['raw1'] = raw1
        sample['raw2'] = raw2
        #input['x'] = [vocab.cls] + list(input.s1) + [vocab.sep] + list(input.s2) + [vocab.sep]
        #sample['x'] = [vocab.cls] + list(sample.s1) + [vocab.sep]
        #input['segment'] = (0,) * (len(sample.s1)+2) + (1,) * (len(sample.s2)+1)
        #sample['segment'] = (0,) * (len(sample.s1)+2)
        #sample['t'] = self.label2id[sample.s2]
        sample['t'] = self.label2id[raw2]
        #dprint(sample)
        return sample

    def feed_one_batch(self, batch, fallback=False, df=None):
        # using
        cdata = self.config.data
        report = pd.Series()

        samples = pd.DataFrame([self.prepare_sample(s) for i, s in batch.iterrows()])
        #dprint(samples.x)
        #dprint(samples.segment_info)
        #dprint(samples.t)
        x = self.model.prepare_batch(samples.x)
        segment_info = self.model.prepare_batch(samples.segment_info)
        #t = self.model.prepare_batch(samples.t)
        t = torch.tensor(samples.t).to(self.device)
        h_out = self.model(x, segment_info=segment_info)
        #loss = F.softmax_cross_entropy(h_out, t)
        loss = criteria.cross_entropy(h_out, t)
        report['loss'] = float(loss)

        batch_ppl = criteria.perplexity(h_out, t, reduction='none')
        #ppl = float(xp.mean(batch_ppl))
        report['ppl'] = float( batch_ppl.mean() )
        list_ppl = batch_ppl.tolist()
        del batch_ppl

        #accuracy = F.accuracy(h_out, t)
        accuracy = float( criteria.accuracy(h_out, t) )
        report['acc'] = accuracy

        if self.model.training:
            self.update_parameters(loss)
        try:
            if self.model.training:
                df = self.train_df
                cdata.log.fed_samples += len(batch)
                cdata.log.fed_tokens += int(batch.len_s1.sum() + batch.len_s2.sum())
            else:
                df = self.dev_df
                #for index, idvec in zip(batch.index, incorrect_seq.data.tolist()):
                #    df.at[index, 'last_incorrect'] = self.vocab.clean_ids(idvec)
                #pred = F.argmax(h_out, axis=1)
                pred = h_out.argmax(1)
                #df.at[batch.index, 'pred'] = pred.data.ravel().tolist()
                # .at は単一ラベル専用のため、複数行への代入は .loc を使う
                df.loc[batch.index, 'pred'] = pred.tolist()
                for i, index in enumerate(batch.index):
                    scores = h_out[i].tolist()
                    sort_index_score = sorted(enumerate(scores), key = lambda v: -v[1])
                    sort_rank_index_score = sorted(enumerate(sort_index_score), key = lambda v: v[1][0])
                    t_i = int(t[i].data)
                    correct_rank = sort_rank_index_score[t_i][0]
                    df.at[index, 'correct_rank'] = int(correct_rank) + 1
            df.loc[batch.index, 'criterion'] = list_ppl
        except Exception as e:
            logger.exception(e)
        return report

    def load_eval_data(self, path):
        df = training.Trainer.load_eval_data(self, path)
        #input_correct = pd.DataFrame([self.prepare_sample(s, 1) for i, s in df.iterrows()])
        #input_incorrect = pd.DataFrame([self.prepare_sample(s, 0) for i, s in df.iterrows()])
        #df['correct_seq'] = input_correct.x
        #df['correct_segment'] = input_correct.segment
        #df['incorrect'] = input_incorrect.s2
        #df['incorrect_segment'] = input_incorrect.segment
        #df['last_score'] = None
        df['pred'] = None
        #df['last_incorrect_score'] = None
        #df['last_incorrect'] = input_incorrect.s2
        #df['last_incorrect'] = None
        return df

    def setup_model(self, args):
        bert_trainer = None
        #dprint(args)
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
            self.model.mod_bert = bert_trainer.model.mod_bert

    def test_sample(self, sample, msg):
        # self.vocab は廃止され、idmaps 経由で引くようになった
        vocab = self.idmaps['seq']
        logger.info(msg)
        logger.info(f'  index: {sample.name}')
        logger.info(f'  query: {vocab.decode(sample.s1)}')
        logger.info(f'  correct: {vocab.decode(sample.s2)}')
        pred = self.labels[sample.pred]
        logger.info(f'  predicted: {vocab.decode(pred)}')
        logger.info(f'  perplexity: {sample.criterion}')
        logger.info(f'  correct rank: {sample.correct_rank}')
        #logger.info('  incorrect: {}'.format(vocab.decode_ids(sample.incorrect)))
        #dprint(sample.last_incorrect)
        #logger.info('  incorrect: {}'.format(vocab.decode_ids(sample.last_incorrect)))
        #logger.info("  last score: {}".format(sample.last_score))
        #logger.info("  last incorrect score: {}".format(sample.last_incorrect_score))
        #logger.info('  original: <{}>\t{}'.format(sample.t[0], vocab.decode_ids(sample.ref)))
        #cls, pred = self.model.predict(sample.x, sample.segment)
        #cls, pred = sample.cls, sample.pred
        #logger.info('  predicted: <{}>\t{}'.format(cls, vocab.decode_ids(pred)))
        #logger.info("  last perplexity: {}".format(sample.criterion))

    # 基底の evaluate は feed_batches / report を受け取るようになっており、
    # 受け流さないと訓練ループからの呼び出しが TypeError になる
    def evaluate(self, tag, df, args, feed_batches=None, report=None):
        cdata = self.config.data
        eval_report = super().evaluate(tag, df, args, feed_batches, report)
        try:
            with torch.no_grad():
                #correct_ranks = []
                replies = set()
                replies.update(self.train_df.s2)
                replies.update(df.s2)
                replies = list(replies)
                _ranked_replies, _ranked_scores, correct_ranks = run_bert_classifier.rank(self, df.s1, replies, correct_list=df.s2, batch_size=cdata.train.batch_size)
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
        group = parser['model']
        # '--pre' と '-P' は基底パーサの --preset が既に使っており、
        # 残したままだと argparse がパーサ構築時に衝突で落ちる
        group.add_argument('--pre-trained-model', '--pre-trained', type=str, help='Path to the pre-trained BERT model to load initially')
        group = parser['files']
        group.add_argument('--labels', type=str, default=None, help='path to the list of labels to classify (if not given, new model is automatically trained with {train_files})')
        return parser

def main():
    training.main(BertClassifierTrainer, 'Classifier')

if __name__ == '__main__':
    main()

