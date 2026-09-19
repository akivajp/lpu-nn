'''Tests for the configuration assembly of lpu_nn.common.training.Trainer

lpu_nn.common.training.Trainer の設定組み立てのテスト。

設定は「コマンドライン引数」「保存済みの設定」「既定値」の 3 者から畳まれる。
優先順位を取り違えても実行は成功し、指定したはずの値が効かないという形で
静かに現れるため、ここで固定する。
'''

import json
import os
import subprocess
import sys

import pytest

from lpu_nn.common import training

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(REPO_ROOT, 'tests', 'data', 'train.tsv')

BASE_ARGS = [
    '--num-epochs', '1', '--batch-size', '32', '--vocab-size', '100',
    '--embed-size', '32', '--hidden-size', '32', '--num-heads', '4',
    '--gpu', '-1',
]


def train(workdir, *extra):
    '''Run one epoch of training with the given extra arguments

    追加引数を与えて 1 エポック訓練する。
    '''
    result = subprocess.run([
        sys.executable, '-m', 'lpu_nn.commands.train_seq2seq',
        str(workdir), CORPUS, *BASE_ARGS, *extra,
    ], capture_output=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr.decode()[-2000:]
    return result


def saved_config(workdir):
    '''Read the configuration of the latest record

    最新の記録の設定を読む。
    '''
    with open(workdir / 'record.latest' / 'config.json', encoding='utf-8') as fobj:
        return json.load(fobj)


class TestGetDefault:
    def test_a_given_value_wins(self):
        assert training.Trainer.get_default('train.batch_size', 7) == 7

    def test_none_falls_back_to_the_default(self):
        fallback = training.Trainer.get_default('train.batch_size', None)
        assert fallback == training.default['train.batch_size']

    def test_a_falsy_value_is_still_a_value(self):
        # 0 や False を「未指定」と取り違えないこと
        assert training.Trainer.get_default('train.batch_size', 0) == 0


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    '''A run whose dropout ratio differs from the default

    既定と異なる dropout 比で 1 エポック訓練した結果。
    '''
    workdir = tmp_path_factory.mktemp('config') / 'work'
    train(workdir, '--dropout-ratio', '0.3')
    return workdir


class TestCommandLinePrecedence:
    def test_a_command_line_value_reaches_the_saved_config(self, trained):
        assert saved_config(trained)['train']['dropout_ratio'] == 0.3

    def test_an_unspecified_value_keeps_the_default(self, trained):
        config = saved_config(trained)
        assert config['model']['hidden_size'] == 32

    def test_resuming_without_the_flag_keeps_the_saved_value(self, trained):
        # 保存済みの値は、指定し直さなくても引き継がれること
        train(trained, '--resume', 'latest')
        assert saved_config(trained)['train']['dropout_ratio'] == 0.3

    def test_resuming_keeps_the_saved_value_over_the_command_line(self, trained):
        """The saved configuration wins on a resume, even against an explicit flag

        再開時は、明示的に指定した引数よりも保存済みの設定が優先される。

        この振る舞いは意図的とも読めるが (再開したモデルの設定を勝手に
        変えない)、`--dropout-ratio` のような訓練側の値まで変更できない
        ことになる。現状を固定し、変えるかどうかは別途の判断とする。
        """
        train(trained, '--resume', 'latest', '--dropout-ratio', '0.05')
        assert saved_config(trained)['train']['dropout_ratio'] == 0.3


class TestFloat16:
    def test_the_precision_survives_a_resume(self, tmp_path):
        # float16 で学習したモデルが、再開時に float32 に戻らないこと
        workdir = tmp_path / 'half'
        train(workdir, '--float16')
        assert saved_config(workdir)['model']['dtype'] == 'float16'
        train(workdir, '--resume', 'latest')
        assert saved_config(workdir)['model']['dtype'] == 'float16'


class TestIgnoredArguments:
    def test_the_run_specific_arguments_are_not_stored(self):
        # 作業ディレクトリや GPU 番号は、そのモデルの設定ではない
        source = training.Trainer.update_config.__func__.__code__.co_consts
        ignored = {value for value in source if isinstance(value, str)}
        for key in ['workdir', 'gpu', 'debug', 'num_epochs', 'move_optimizer']:
            assert key in ignored

    def test_every_store_true_flag_is_ignored(self):
        # store_true の既定は False であり None ではないため、無指定でも
        # 「明示的な False」として保存済みの設定を上書きしてしまう。
        # そうならないよう、これらは設定へ畳み込まない
        import inspect
        import re
        sources = [
            inspect.getsource(training.Trainer.create_parser),
            inspect.getsource(training.Trainer.update_config),
        ]
        # argparse は最初の長い綴りを属性名にするため、別名は数えない
        flags = set()
        for call in re.findall(r"add_argument\((.*?)\)\n", sources[0], re.S):
            if "action='store_true'" not in call:
                continue
            first = re.search(r"'--([a-z0-9-]+)'", call)
            if first:
                flags.add(first.group(1))
        ignored = set(re.findall(r"ignore\.append\('([a-z_]+)'\)", sources[1]))
        for flag in flags:
            assert flag.replace('-', '_') in ignored, f'--{flag} is not ignored'
