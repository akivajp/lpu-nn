'''Tests for the match-ranker commands

マッチングランカーのコマンド (lpu_nn.commands.*_match_ranker) のテスト。

採点とランキングはモデル本体から切り離して確かめられるので、
ここでは代用モデルを渡し、順位付け・打ち切り・指標の集計だけを固定する。
学習ループ全体の疎通は CI のスモークテストが担当する。
'''

import time

import pytest
import torch

from lpu_nn.commands import run_match_ranker, train_match_ranker


class StubModel:
    '''Scores a pair by how many tokens the two sides share

    共有トークン数でペアを採点する代用モデル。
    順位付けの検証には、決定的で人手で追えるスコアがあれば足りる。
    '''

    dtype = torch.float32

    def __init__(self, delay=0.0):
        self.delay = delay
        self.num_calls = 0

    def prepare_batch(self, name, iterable):
        # 採点側は文字列のまま扱えるので、ここでは素通しする
        return list(iterable)

    def score(self, queries, replies):
        self.num_calls += 1
        if self.delay:
            time.sleep(self.delay)
        values = []
        for query, reply in zip(queries, replies, strict=True):
            shared = set(query.split()) & set(reply.split())
            values.append(float(len(shared)))
        return torch.tensor(values)


QUERIES = ['a b c', 'a b c', 'a b c']
REPLIES = ['a b c', 'a b', 'z']
EXPECTED = [1, 0, 0]


class TestScore:
    def test_every_pair_is_scored(self):
        result = run_match_ranker.score(StubModel(), QUERIES, REPLIES,
                                        batch_size=2, progress=False)
        assert [reply for _query, reply, _pred in result] == REPLIES
        assert [pred for _q, _r, pred in result] == [3.0, 2.0, 0.0]

    def test_scoring_does_not_build_a_gradient_graph(self):
        '''Scoring runs under no_grad, so nothing keeps an autograd graph

        採点に自動微分グラフは要らない。候補数が多いほど無駄が効くため
        no_grad で囲んである。
        '''
        seen = []

        class GradCheckingModel(StubModel):
            def score(self, queries, replies):
                seen.append(torch.is_grad_enabled())
                return super().score(queries, replies)

        run_match_ranker.score(GradCheckingModel(), QUERIES, REPLIES,
                               batch_size=1, progress=False)
        assert seen == [False, False, False]

    def test_a_timeout_stops_the_scoring_loop(self):
        '''The comparison used to be inverted, so any timeout scored nothing

        0.1.0.dev0 は `timeout > 経過時間` で break していたため、
        --eval-timeout を指定すると 1 バッチ目で必ず打ち切られ、
        採点結果が 1 件も得られなかった。
        '''
        model = StubModel(delay=0.05)
        result = run_match_ranker.score(model, QUERIES, REPLIES, batch_size=1,
                                        progress=False, timeout=0.06)
        # 制限内のバッチは採点され、超えたところで打ち切られること
        assert 0 < len(result) < len(QUERIES)

    def test_a_generous_timeout_scores_everything(self):
        result = run_match_ranker.score(StubModel(), QUERIES, REPLIES,
                                        batch_size=1, progress=False, timeout=60)
        assert len(result) == len(QUERIES)


class TestRank:
    def test_replies_are_ordered_by_descending_score(self):
        ranked = run_match_ranker.rank(StubModel(), QUERIES, REPLIES, EXPECTED,
                                       batch_size=3, progress=False)
        assert list(ranked) == ['a b c']
        assert [reply for reply, _pred, _expect in ranked['a b c']] == \
            ['a b c', 'a b', 'z']

    def test_the_expected_value_is_carried_through(self):
        ranked = run_match_ranker.rank(StubModel(), QUERIES, REPLIES, EXPECTED,
                                       batch_size=3, progress=False)
        assert [expect for _r, _p, expect in ranked['a b c']] == [1, 0, 0]

    def test_without_expectations_every_entry_is_none(self):
        ranked = run_match_ranker.rank(StubModel(), QUERIES, REPLIES,
                                       batch_size=3, progress=False)
        assert all(expect is None for _r, _p, expect in ranked['a b c'])

    def test_ties_are_broken_deterministically(self):
        '''Equal scores must not depend on set iteration order

        同点の並びが set の反復順に依存すると評価指標が実行毎に揺れる。
        応答文字列を第 2 キーにして順序を固定してある。
        '''
        queries = ['q'] * 4
        # どれも共有トークン 0 で同点になる
        replies = ['dd', 'bb', 'cc', 'aa']
        orders = set()
        for _ in range(5):
            ranked = run_match_ranker.rank(StubModel(), queries, replies,
                                           batch_size=4, progress=False)
            orders.add(tuple(reply for reply, _p, _e in ranked['q']))
        assert orders == {('aa', 'bb', 'cc', 'dd')}

    def test_several_queries_are_kept_apart(self):
        queries = ['a b', 'a b', 'x y', 'x y']
        replies = ['a b', 'z', 'x y', 'z']
        ranked = run_match_ranker.rank(StubModel(), queries, replies,
                                       batch_size=2, progress=False)
        assert set(ranked) == {'a b', 'x y'}
        assert ranked['a b'][0][0] == 'a b'
        assert ranked['x y'][0][0] == 'x y'


class TestMean:
    def test_the_mean_of_an_empty_sequence_is_nan(self):
        '''An evaluation with no correct answer must not raise

        正解を含む問い合わせが 1 件も無いと ZeroDivisionError になっていた。
        '''
        import math
        assert math.isnan(run_match_ranker.mean([]))

    def test_the_mean_is_the_arithmetic_mean(self):
        assert run_match_ranker.mean([1, 2, 6]) == 3.0


class TestEvaluate:
    def test_a_perfect_ranking_scores_one(self):
        # 正解が 1 位に来るので MRR も MAP も 1.0
        results = run_match_ranker.evaluate(
            StubModel(), QUERIES, REPLIES, EXPECTED, batch_size=3,
            progress=False, list_k=[1])
        assert results['mrr'] == 1.0
        assert results['map'] == 1.0
        assert results['mr'] == 1.0

    def test_the_reciprocal_rank_follows_the_position_of_the_correct_reply(self):
        # 正解 'a b' は共有トークン 2 で 2 位になる
        results = run_match_ranker.evaluate(
            StubModel(), QUERIES, REPLIES, [0, 1, 0], batch_size=3,
            progress=False, list_k=[1])
        assert results['mrr'] == 0.5
        assert results['mr'] == 2.0

    def test_the_pair_and_query_counts_are_reported(self):
        results = run_match_ranker.evaluate(
            StubModel(), QUERIES, REPLIES, EXPECTED, batch_size=3,
            progress=False, list_k=[1])
        assert results['num_queries'] == 1
        assert results['num_pairs'] == 3
        assert results['mean_num_replies'] == 3.0

    def test_a_score_above_the_half_counts_as_correct(self):
        # 閾値は 0.5。0.4 は不正解、0.6 は正解として扱われること
        low = run_match_ranker.evaluate(
            StubModel(), QUERIES, REPLIES, [0.4, 0.4, 0.4], batch_size=3,
            progress=False, list_k=[1])
        assert low['num_queries_with_correct_answers'] == 0
        high = run_match_ranker.evaluate(
            StubModel(), QUERIES, REPLIES, [0.6, 0.0, 0.0], batch_size=3,
            progress=False, list_k=[1])
        assert high['num_queries_with_correct_answers'] == 1

    def test_recall_at_k_is_skipped_when_there_are_too_few_replies(self):
        # 候補が k 件以下の問い合わせは R@k の母数に入れない
        results = run_match_ranker.evaluate(
            StubModel(), QUERIES, REPLIES, EXPECTED, batch_size=3,
            progress=False, list_k=[10])
        assert 'r_at_10' not in results


class TestTrainerConfiguration:
    def test_the_ranker_reads_two_sequences_and_one_score(self):
        fields = train_match_ranker.specific.data.format
        assert fields.input.s1 == 'seq'
        assert fields.input.s2 == 'seq'
        assert fields.output.t == 'score'

    def test_the_default_task_is_match_ranking(self):
        assert train_match_ranker.default.model.task == 'match_rank'

    @pytest.mark.parametrize('loss_method, prefix', [
        ('classify', 'class'),
        ('class', 'class'),
        ('point', 'point'),
        ('point-wise', 'point'),
        ('pair', 'pair'),
        ('pair-wise', 'pair'),
    ])
    def test_every_loss_alias_matches_exactly_one_branch(self, loss_method, prefix):
        '''setup_model and feed_one_batch both branch on startswith

        CLI が受け付ける短縮形が、振り分けに使う 3 つの接頭辞の
        ちょうど 1 つに当たること。'pair' と 'point' は先頭 2 文字が
        同じなので、接頭辞の取り違えがあるとここで露見する。
        '''
        prefixes = ['class', 'point', 'pair']
        matched = [p for p in prefixes if loss_method.startswith(p)]
        assert matched == [prefix]

    def test_the_report_columns_differ_per_loss_method(self):
        # 回帰 (point) には正解率が無く、分類 / ペアにはある
        assert 'acc' not in train_match_ranker.REPORT_POINT
        assert 'acc' in train_match_ranker.REPORT_CLASSIFY
        assert 'acc' in train_match_ranker.REPORT_PAIR


class TestLoggerTargets:
    def test_the_scorer_logs_the_ported_package_names(self):
        '''The logger names had not been renamed on the port

        0.1.0.dev0 は移植前の 'common' / 'models' を対象にしており、
        --debug や --logging を付けても lpu_nn / lpu 配下のログが
        1 行も出力されなかった。
        '''
        import inspect

        source = inspect.getsource(run_match_ranker.main)
        assert "'lpu_nn'" in source
        assert "'lpu'" in source
        assert "'models'" not in source
        assert "'common'" not in source
