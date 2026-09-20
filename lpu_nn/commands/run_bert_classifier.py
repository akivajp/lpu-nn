#!/usr/bin/env python3

# system
import argparse
import os
import sys
import time
import traceback

# 3rd
import torch

# local
from lpu.common import logging
from lpu.metrics import ranking
from lpu_nn.commands import train_bert_classifier

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

def score(trainer, query_list, replies, batch_size=1):
    #batch_s1 = trainer.prepare_batch([query_list])
    scores = []
    for i in range(0, len(query_list), batch_size):
        #batch_s1 = trainer.prepare_batch(query_list[i:i+batch_size])
        batch_s1 = trainer.model.prepare_batch(query_list[i:i+batch_size])
        #scores += trainer.model(batch_s1).data.tolist()
        #score, features = trainer.model(batch_s1)
        score = trainer.model(batch_s1)
        scores += score.tolist()
    return scores

def rank(trainer, query_list, replies, correct_list=None, batch_size=1):
    scores = score(trainer, query_list, replies, batch_size)
    ranked_scores = []
    ranked_replies = []
    correct_ranks = []
    for i, score_list in enumerate(scores):
        #ranked_scores_replies = sorted(zip(score_list, replies))[::-1]
        ranked_scores_replies = sorted(zip(score_list, trainer.labels, strict=True))[::-1]
        ranked_score_list  = [r[0] for r in ranked_scores_replies]
        ranked_reply_list = [r[1] for r in ranked_scores_replies]
        ranked_scores.append(ranked_score_list)
        ranked_replies.append(ranked_reply_list)
        if correct_list is not None:
            correct = correct_list[i]
            #if correct in replies:
            if correct in ranked_reply_list:
                dprint(correct)
                correct_rank = ranked_reply_list.index(correct) + 1
                correct_ranks.append(correct_rank)
    if correct_list is not None:
        return ranked_replies, ranked_scores, correct_ranks
    else:
        return ranked_replies, ranked_scores

def load_labels_for(trainer, model_path):
    """Load the label list that training wrote beside the work directory

    学習時に書き出したラベル一覧を読み込む。
    labels.txt はチェックポイントの中ではなく作業ディレクトリ直下に
    置かれるため、チェックポイントの親も探索する。
    """
    candidates = [
        os.path.join(model_path, 'labels.txt'),
        os.path.join(os.path.dirname(os.path.normpath(model_path)), 'labels.txt'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            logger.info(f"loading labels from '{path}'")
            return trainer.load_labels(path)
    raise RuntimeError("not found labels file: {}".format(
        str.join(', ', candidates)))

def main():
    parser = argparse.ArgumentParser(description = 'Class Estimator')
    parser.add_argument('model', help='path to read the trained model (directory or config path)')
    parser.add_argument('--gpu', '-G', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--ideep', '-I', type=bool, default=False, nargs='?', const=True, help='Using iDeep64 (ignoreing gpu setting, enabled if possible')
    parser.add_argument('--debug', '-D', action='store_true', help='Debug mode')
    parser.add_argument('--logging', '--log', type=str, default=None, help='Path of file to log (default: %(default)s')
    parser.add_argument('--batch_size', '-B', type=int, default=1, help='Batch Size (more than 1 enable batch decode and disable beam decode)')
    #parser.add_argument('--replies', '--reply', '-r', type=str, default=None, help='Path to the file path containing all the candidates of replies')
    parser.add_argument('--ranking', '--rank', action='store_true', help='ranking mode')

    args = parser.parse_args()
    # using_config は context manager であり、戻り値を捨てるだけでは
    # 設定が適用されない (--debug が効かなかった)。対象ロガーも
    # 移植後のパッケージ名を指す必要がある
    loggers = ['__main__', 'lpu_nn', 'lpu']
    with logging.using_config(loggers, debug=args.debug):
        dprint(args)
        run_classifier(args)

def run_classifier(args):
    # load_status は model_path を取らなくなっており、
    # チェックポイントのディレクトリをそのまま受け取る
    trainer = train_bert_classifier.BertClassifierTrainer().load_status(args.model)
    # load_status はラベル一覧を読まないため、ここで補う
    load_labels_for(trainer, args.model)
    model = trainer.model
    if args.gpu >= 0:
        model.to(args.gpu)
        logger.info(f"using gpu: {args.gpu}")
    logger.info("loaded")

    if args.ranking:
        #replies = []
        replies = trainer.labels
        #for line in progress.view(args.replies, 'loading replies: '):
        #    #reply_toks = model.vocab.sent2idvec(line.strip(), growth=False, add_sep=True)
        #    reply_toks = model.vocab.encode_ids(line.strip())
        #    replies.append(reply_toks)
        correct_ranks = []
        query_list = []
        correct_list = []
        start = time.time()
        for i, line in enumerate(sys.stdin):
            try:
                sent_number = i+1
                line = line.strip()
                dprint(sent_number)
                dprint(line)
                query, correct = line.split('\t')
                dprint(query)
                dprint(correct)
                #query   = model.vocab.sent2idvec(query, growth=False, add_sep=True)
                #correct = model.vocab.sent2idvec(correct, growth=False, add_sep=True)
                query   = tuple(model.vocab.encode(query))
                # 正解は rank() 内で trainer.labels (ラベル文字列の一覧)
                # と突き合わせる。ID 列に符号化すると決して一致せず、
                # correct_ranks が空のまま指標計算に進んでいた
                query_list.append(query)
                correct_list.append(correct)
                #ranked_replies, ranked_scores, correct_rank = rank(trainer, query, replies, correct, batch_size=args.batch_size)
                #dprint(correct_rank)
                #correct_ranks.append(correct_rank)
            except Exception as e:
                logger.exception(e)
        if not query_list:
            logger.error("no query was given")
            return
        _ranked_replies, _ranked_scores, correct_ranks = rank(trainer, query_list, replies, correct_list, batch_size=args.batch_size)
        if not correct_ranks:
            logger.error("no correct label was found among the known labels")
            return
        dprint(len(replies))
        dprint(len(replies) * len(query_list))
        for k in (1, 5, 20, 50, 100):
            p_at_k = ranking.calc_precision_at_k(correct_ranks, k)
            logger.info(f"P@{k}: {p_at_k}")
        mrr = ranking.calc_mean_reciprocal_rank(correct_ranks)
        logger.info(f"MRR: {mrr}")
        mean = float(sum(correct_ranks)) / len(correct_ranks)
        logger.info(f"Mean Rank: {mean} / {len(correct_ranks)}")
        dprint(time.time() - start)
    else:
        # just scoring
        dprint(sys.stdin.isatty())
        if sys.stdin.isatty():
            args.batch_size = 1
        sent_number = 0
        while True:
            sent_number += 1
            #list_seq = []
            lines = []
            last = False
            for _ in range(args.batch_size):
                line = sys.stdin.readline()
                if not line:
                    last = True
                line = line.strip()
                if line:
                    lines.append(line)
                    try:
                        dprint(sent_number)
                        dprint(line)
                    except Exception:
                        logger.exception(traceback.format_exc())
                        continue
            if not lines:
                logger.debug("empty")
            else:
                try:
                    time_start = time.time()
                    #batch_seq   = chainer.Variable(xp.array(seq, dtype=np.int32))
                    #batch_seq = F.pad_sequence(list_seq, padding=-1)
                    # trainer ではなく model が prepare_batch を持つ。
                    # F は chainer.functions の残骸で存在しない
                    with torch.no_grad():
                        batch_s1 = model.prepare_batch(lines)
                        scores = model(batch_s1)
                    dprint(batch_s1)
                    for row in scores:
                        dprint(row.tolist())
                        # 最尤クラスのラベルを 1 行 1 件で書き出す
                        # (従来は dprint のみで標準出力に何も出なかった)
                        print(trainer.labels[int(row.argmax())])
                    dprint(time.time() - time_start)
                except Exception as e:
                    logger.exception(e)
                    sys.stdout.write("<err>\n")
                sys.stdout.flush()
            if last:
                break

if __name__ == '__main__':
    main()

