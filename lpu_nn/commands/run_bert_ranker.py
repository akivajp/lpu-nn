#!/usr/bin/env python3

# system
import argparse
import pandas as pd
import sys
import time
import traceback

# 3rd
import torch

# local
from lpu.common import logging
from lpu.common.progress import view as pview
from lpu.metrics import ranking

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

from lpu_nn.common import training

from lpu_nn.commands import train_bert_ranker

def score(model, query, replies, batch_size=1, progress=True):
    #list_seq = []
    #list_input = []
    #pass
    #for reply in replies:
    #    #seq = np.array( (model.vocab.cls,) + query + reply, dtype=np.int32 )
    #    #seq = (model.vocab.cls,) + query + reply
    #    #list_seq.append(seq)
    #    seq, segment_info = model.prepare_input(query, reply)
    df = pd.DataFrame(model.prepare_input(query, reply) for reply in replies)
    #batches = list( chainer.iterators.SerialIterator(list_seq, batch_size, False, False) )
    #batches = training.build_batches(list_seq, batch_size)
    batches = training.build_batches(df, batch_size)
    scores = []
    #for batch_seq in batch_iter:
    #logger.setLevel('DEBUG')
    if progress:
        iter = pview(batches, 'evaluating')
    else:
        iter = batches
    for batch in iter:
        #dprint(batch_seq)
        dprint(batch)
        #batch_seq = [xp.array(seq, dtype=np.int32) for seq in batch_seq]
        #batch_seq = [torch.tensor(seq) for seq in batch_seq]
        #batch_seq = F.pad_sequence(batch_seq, padding=-1)
        #batch_seq = model.prepare_batch(batch_seq)
        #batch_seq = model.prepare_batch(batch.x, batch.segment_info)
        batch_x            = model.prepare_batch(batch.x)
        batch_segment_info = model.prepare_batch(batch.segment_info)
        #model.L_bert.L_transform.reset_state()
        #batch_scores = model(batch_seq)
        #batch_scores, features = model(batch_seq)
        # 採点に自動微分グラフは不要
        with torch.no_grad():
            batch_scores = model(batch_x, batch_segment_info)
        #scores += list(batch_scores.data)
        scores += [float(s) for s in batch_scores]
        #dprint(scores)
    return scores

def rank(model, query, replies, correct=None, batch_size=1, progress=True):
    scores = score(model, query, replies, batch_size, progress)
    ranked_scores_replies = sorted(zip(scores, replies, strict=True))[::-1]
    ranked_scores  = [r[0] for r in ranked_scores_replies]
    ranked_replies = [r[1] for r in ranked_scores_replies]
    if correct:
        if correct in replies:
            correct_rank = ranked_replies.index(correct) + 1
        else:
            correct_rank = None
        return ranked_replies, ranked_scores, correct_rank
    else:
        return ranked_replies, ranked_scores

def main():
    parser = argparse.ArgumentParser(description = 'BERT Rank Scorer')
    parser.add_argument('model', help='path to read the trained model (directory or config path)')
    parser.add_argument('--gpu', '-G', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--ideep', '-I', type=bool, default=False, nargs='?', const=True, help='Using iDeep64 (ignoreing gpu setting, enabled if possible')
    parser.add_argument('--debug', '-D', action='store_true', help='Debug mode')
    parser.add_argument('--logging', '--log', type=str, default=None, help='Path of file to log (default: %(default)s')
    parser.add_argument('--batch_size', '-B', type=int, default=1, help='Batch Size (more than 1 enable batch decode and disable beam decode)')
    parser.add_argument('--beam_width', '--beam', '-n', type=int, default=10, help='Beam width')
    parser.add_argument('--incomplete_cost', '--incomp', '-c', type=int, default=100, help='Cost of end-of-sentence symbold of incomplete sentence')
    parser.add_argument('--replies', '--reply', '-r', type=str, default=None, help='Path to the file path containing all the candidates of replies')

    args = parser.parse_args()
    # using_config は context manager であり、戻り値を捨てるだけでは
    # 設定が適用されない (--debug が効かなかった)
    loggers = ['__main__', 'lpu_nn', 'lpu']
    with logging.using_config(loggers, debug=args.debug):
        dprint(args)
        run_ranker(args)

def run_ranker(args):
    # 基底クラスの Trainer を未束縛で呼んでおり、path が self として
    # 渡っていた。load_status は model_path も取らなくなっている
    trainer = train_bert_ranker.BertRankerTrainer().load_status(args.model)
    model = trainer.model
    if args.gpu >= 0:
        model.to(args.gpu)
        logger.info(f"using gpu: {args.gpu}")
    logger.info("loaded")

    if args.replies:
        # sent2idvec は廃止された旧 API。prepare_input が生文字列を
        # 受け付けるため、ここでは符号化しない (符号化すると下の
        # `correct in replies` が決して成立しない)
        replies = []
        for line in pview(args.replies, 'loading replies: '):
            replies.append(line.strip())
        correct_ranks = []
        try:
            for i, line in enumerate(sys.stdin):
                sent_number = i+1
                line = line.strip()
                dprint(sent_number)
                dprint(line)
                query, correct = line.split('|||')
                query = query.strip()
                correct = correct.strip()
                dprint(query)
                dprint(correct)
                _ranked_replies, _ranked_scores, correct_rank = rank(model, query, replies, correct, batch_size=args.batch_size)
                dprint(correct_rank)
                # 見つからなければ None (未検出) のまま指標へ渡す
                correct_ranks.append(correct_rank)
        except Exception as e:
            logger.exception(e)
        if not correct_ranks:
            logger.error("no query/reply pair could be scored")
            return
        dprint(len(replies))
        dprint(len(replies) * len(correct_ranks))
        for k in (1, 5, 20, 50, 100):
            p_at_k = ranking.calc_precision_at_k(correct_ranks, k)
            logger.info(f"P@{k}: {p_at_k}")
        mrr = ranking.calc_mean_reciprocal_rank(correct_ranks)
        logger.info(f"MRR: {mrr}")
        # 平均順位は未検出を除いて算出する (順位が定義できないため)
        found = [r for r in correct_ranks if r is not None]
        mean = float(sum(found)) / len(found) if found else float('nan')
        logger.info(f"Mean Rank: {mean} / {len(correct_ranks)}")
    else:
        # just scoring
        dprint(sys.stdin.isatty())
        if sys.stdin.isatty():
            args.batch_size = 1
        sent_number = 0
        while True:
            sent_number += 1
            list_input = []
            lines = []
            for _ in range(args.batch_size):
                line = sys.stdin.readline().strip()
                if line:
                    lines.append(line)
                    try:
                        dprint(sent_number)
                        dprint(line)
                        sent1, sent2 = line.split('|||')
                        sent1 = sent1.strip()
                        sent2 = sent2.strip()
                        dprint(sent1)
                        dprint(sent2)
                    except Exception:
                        logger.exception(traceback.format_exc())
                        continue
                    # <cls> の付与とセグメント ID の割り当ては
                    # prepare_input が行う
                    list_input.append(model.prepare_input(sent1, sent2))
            if not lines:
                break
            elif not list_input:
                continue
            try:
                time_start = time.time()
                #batch_seq   = chainer.Variable(xp.array(seq, dtype=np.int32))
                df = pd.DataFrame(list_input)
                with torch.no_grad():
                    batch_x = model.prepare_batch(df.x)
                    batch_segment_info = model.prepare_batch(df.segment_info)
                    scores = model(batch_x, batch_segment_info)
                for s in scores:
                    print(float(s))
                dprint(time.time() - time_start)
            except Exception as e:
                logger.exception(e)
                sys.stdout.write("<err>\n")
            sys.stdout.flush()

if __name__ == '__main__':
    main()

