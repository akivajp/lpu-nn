#!/usr/bin/env python3

# system
import argparse
from collections import defaultdict
from collections import OrderedDict
import sys
import time
import traceback

# 3rd
import pandas as pd
import torch

# local
from lpu.common import logging
from lpu.common.progress import view as pview
from lpu.metrics import ranking
from lpu_nn.common import training
from lpu_nn.commands import train_match_ranker

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

def score(model, queries, replies, batch_size=1, progress=True, timeout=None):
    df = pd.DataFrame({'s1': queries, 's2': replies})
    batches = training.build_batches(df, batch_size)
    if progress:
        #gen_batches = pview(batches, 'scoring')
        gen_batches = pview(list(batches), 'scoring')
        #gen_batches = pview(batches, 'scoring', max_count=len(batches))
    else:
        gen_batches = batches
    start = time.time()
    list_query_reply_score = []
    # 採点に自動微分グラフは不要。候補数が多いほど無駄が効く
    # (組んだままだと float() 変換時に UserWarning も出る)
    with torch.no_grad():
        for batch in gen_batches:
            #dprint(batch.s1)
            if timeout is not None:
                if time.time() - start > timeout:
                    logger.warning(f"scoring timed out after {timeout} sec")
                    break
            #dprint(batch)
            #batch_queries = model.prepare_batch(batch.s1)
            #batch_replies = model.prepare_batch(batch.s2)
            batch_queries = model.prepare_batch('s1', batch.s1)
            batch_replies = model.prepare_batch('s2', batch.s2)
            #batch_pred_scores = model(batch_queries, batch_replies)
            batch_pred_scores = model.score(batch_queries, batch_replies)
            for query, reply, pred_score in zip(batch.s1, batch.s2, batch_pred_scores, strict=True):
                #pred_score = float(pred_score[0])
                pred_score = float(pred_score)
                list_query_reply_score.append( (query, reply, pred_score) )
    return list_query_reply_score

def rank(model, queries, replies, expect=None, batch_size=1, progress=True, timeout=None):
    scores = score(model, queries, replies, batch_size, progress, timeout)
    dict_query_reply_scores = defaultdict(set)
    for i, (query, reply, pred_score) in enumerate(scores):
        #if scores is not None:
        if expect is not None:
            if isinstance(expect, pd.Series):
                t = (reply, pred_score, expect.iloc[i])
            else:
                #if expect[i] > 0:
                #    dprint( (query,reply, pred_score, expect[i]) )
                t = (reply, pred_score, expect[i])
        else:
            t = (reply, pred_score, None)
        #dprint(query)
        #dprint(t)
        #dprint(i)
        #dprint(query)
        #if query not in dict_query_reply_scores:
        #    dprint(query)
        #    dprint(len(dict_query_reply_scores))
        dict_query_reply_scores[query].add(t)
    dict_query_rank_reply_scores = {}
    for query, set_reply_scores in dict_query_reply_scores.items():
        list_rank_reply_scores = []
        # スコア降順。同点は応答文字列の昇順で決定的に並べる
        # (set の反復順に依存すると評価指標が実行毎に揺れるため)
        for reply, pred, expect in sorted(set_reply_scores, key=lambda r: (-r[1], str(r[0]))):
            list_rank_reply_scores.append( (reply, pred, expect) )
        #dprint(query)
        #dprint(list_rank_reply_scores)
        dict_query_rank_reply_scores[query] = list_rank_reply_scores
    return dict_query_rank_reply_scores

def mean(numbers):
    '''Arithmetic mean, or NaN for an empty sequence

    空列は ZeroDivisionError ではなく NaN を返す
    (正解を含む問い合わせが 1 件も無い評価で落ちないようにするため)。
    '''
    numbers = list(numbers)
    if not numbers:
        return float('nan')
    return float(sum(numbers)) / len(numbers)

LIST_K = training.LIST_K_FOR_RECALL
def evaluate(model, queries, replies, scores, batch_size=1, progress=True, timeout=None, list_k=LIST_K):
    results = OrderedDict()
    list_best_rank = []
    list_reciprocal_rank = []
    list_average_precision = []
    dict_list_recall_at_k = defaultdict(list)
    dict_query_rank_reply_scores = rank(model, queries, replies, scores, batch_size, progress, timeout)
    num_pairs = 0
    for list_rank_reply_scores in dict_query_rank_reply_scores.values():
        #dprint(query)
        #dprint(list_rank_reply_scores)
        num_correct = 0
        dict_num_hit = defaultdict(int)
        num_replies = len(list_rank_reply_scores)
        list_precision_on_correct = []
        for i, (_reply, _pred, expect) in enumerate(list_rank_reply_scores):
            num_pairs += 1
            r = i + 1
            if expect is not None:
                #dprint(expect)
                expect = float(expect)
                #if expect > 0:
                if expect > 0.5:
                    num_correct += 1
                    p_at_r = num_correct / float(r)
                    list_precision_on_correct.append(p_at_r)
                    #for k in list_k:
                    #    if r <= k:
                    #        dict_num_hit[k] += 1
                    if num_correct == 1:
                        # rank for best hit
                        list_best_rank.append(float(r))
                        list_reciprocal_rank.append(1 / float(r))
            dict_num_hit[r] = num_correct
        if num_correct > 0:
            for k in list_k:
                if num_replies > k:
                    dict_list_recall_at_k[k].append(float(dict_num_hit[k]) / num_correct)
            list_average_precision.append(mean(list_precision_on_correct))
    results['num_queries'] = len(dict_query_rank_reply_scores)
    results['num_queries_with_correct_answers'] = len(list_best_rank)
    results['num_pairs'] = num_pairs
    results['mean_num_replies'] = float(num_pairs) / len(dict_query_rank_reply_scores)
    for k in list_k:
        list_recall_at_k = dict_list_recall_at_k[k]
        if list_recall_at_k:
            results[f'r_at_{k}'] = mean(list_recall_at_k)
    #dprint(len(list_reciprocal_rank))
    #dprint(len(list_correct_rank))
    #dprint(len(list_reciprocal_rank))
    #dprint(len(list_average_precision))
    #dprint(len(list_best_rank))
    results['mrr'] = mean(list_reciprocal_rank)
    results['map'] = mean(list_average_precision)
    results['mr'] = mean(list_best_rank)
    return results

def eval_ranker(args):
    #trainer = RankerTrainer().load_status(path, model_path=args.model)
    #trainer = train_ranker.RankerTrainer().load_status(path, model_path=args.model)
    #trainer = train_ranker.RankerTrainer().load_status(args.model)
    trainer = train_match_ranker.RankerTrainer().load_status(args.model)
    model = trainer.model
    if args.gpu >= 0:
        model.to(args.gpu)
    logger.debug("loaded")

    if args.evaluate:
        test_df = trainer.load_eval_data(args.evaluate)
        return trainer.evaluate('test', test_df, args)

    if args.replies:
        # rank() / prepare_batch() は生文字列をそのまま扱えるため、
        # ここで id 列へ変換しない (変換すると dict のキーにできない)
        replies = []
        for line in pview(args.replies, 'loading replies: '):
            replies.append(line.strip())
        correct_ranks = []
        pair_number = 0
        for i, line in enumerate(sys.stdin):
            try:
                pair_number = i+1
                line = line.strip()
                dprint(pair_number)
                dprint(line)
                query, correct = line.split('\t')
                dprint(query)
                dprint(correct)
                # 正解応答を候補集合に加えた上で全候補を採点する
                candidates = [correct] + [r for r in replies if r != correct]
                ranked = rank(
                    model, [query] * len(candidates), candidates,
                    batch_size=args.batch_size, progress=False,
                )[query]
                # 正解応答の順位 (1 始まり)。見つからなければ None = 未検出
                correct_rank = next(
                    (r + 1 for r, (reply, _pred, _expect) in enumerate(ranked)
                     if reply == correct),
                    None,
                )
                dprint(correct_rank)
                correct_ranks.append(correct_rank)
            except Exception as e:
                logger.exception(e)
        if not correct_ranks:
            logger.error("no query/reply pair could be scored")
            return
        dprint(len(replies))
        dprint(len(replies) * pair_number)
        for k in (1, 5, 20, 50, 100):
            p_at_k = ranking.calc_precision_at_k(correct_ranks, k)
            logger.info(f"P@{k}: {p_at_k}")
        mrr = ranking.calc_mean_reciprocal_rank(correct_ranks)
        logger.info(f"MRR: {mrr}")
        # 平均順位は未検出を除いた上で算出する (順位が定義できないため)
        found = [r for r in correct_ranks if r is not None]
        logger.info(f"Mean Rank: {mean(found)} / {len(correct_ranks)}")
    else:
        # just scoring
        dprint(sys.stdin.isatty())
        if sys.stdin.isatty():
            args.batch_size = 1
        pair_number = 0
        time_start = time.time()
        while True:
            #list_seq = []
            list_s1 = []
            list_s2 = []
            lines = []
            last = False
            for _ in range(args.batch_size):
                pair_number += 1
                line = sys.stdin.readline().strip()
                #dprint(line)
                if not line:
                    #logger.debug("not line")
                    last = True
                if line:
                    lines.append(line)
                    try:
                        #dprint(pair_number)
                        #dprint(line)
                        #sent1, sent2 = line.split('\t')
                        sent1, sent2 = line.split('\t')[:2] # ignoring after 2nd column (may containing correct scores)
                        sent1 = sent1.strip()
                        sent2 = sent2.strip()
                    except Exception:
                        logger.error(traceback.format_exc())
                        #continue
                        sys.exit(1)
                    list_s1.append(sent1)
                    list_s2.append(sent2)
            if not list_s1 or not list_s2:
                logger.debug("empty")
            else:
                try:
                    with torch.no_grad():
                        batch_s1 = model.prepare_batch('s1', list_s1)
                        batch_s2 = model.prepare_batch('s2', list_s2)
                        pred_scores = model.score(batch_s1, batch_s2)
                    for s in pred_scores:
                        print(float(s))
                except Exception as e:
                    logger.exception(e)
                    sys.stdout.write("<err>\n")
                    break
                sys.stdout.flush()
            if last:
                #logger.debug("last")
                break
        logger.debug("done")
        dprint(time.time() - time_start)
        dprint(time.time() - time_start)

def main():
    parser = argparse.ArgumentParser(description = 'Rank Scorer')
    parser.add_argument('model', help='path to read the trained model (directory or config path)')
    parser.add_argument('--gpu', '-G', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--debug', '-D', action='store_true', help='Debug mode')
    parser.add_argument('--logging', '--log', type=str, default=None, help='Path of file to log (default: %(default)s')
    parser.add_argument('--batch_size', '-B', type=int, default=1, help='Batch Size (more than 1 enable batch decode and disable beam decode)')
    parser.add_argument('--replies', '--reply', '-r', type=str, default=None, help='Path to the file path containing all the candidates of replies')
    parser.add_argument('--evaluate', '--eval', '-E', type=str, default=None)

    args = parser.parse_args()
    loggers = ['__main__', 'lpu_nn', 'lpu']
    with logging.using_config(loggers, debug=args.debug):
        dprint(args)
        eval_ranker(args)

if __name__ == '__main__':
    main()

