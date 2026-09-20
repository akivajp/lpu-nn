'''Tests for the configuration assembly of lpu_nn.common.training.Trainer

lpu_nn.common.training.Trainer の設定組み立てのテスト。

設定は「コマンドライン引数」「保存済みの設定」「既定値」の 3 者から畳まれる。
優先順位を取り違えても実行は成功し、指定したはずの値が効かないという形で
静かに現れるため、ここで固定する。
'''

import json
import os
import shutil
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


def train(workdir, *extra, check=True):
    '''Run one epoch of training with the given extra arguments

    追加引数を与えて 1 エポック訓練する。
    '''
    result = subprocess.run([
        sys.executable, '-m', 'lpu_nn.commands.train_seq2seq',
        str(workdir), CORPUS, *BASE_ARGS, *extra,
    ], capture_output=True, cwd=REPO_ROOT)
    if check:
        assert result.returncode == 0, result.stderr.decode()[-2000:]
    return result


def resume(workdir, *extra, check=True):
    '''Resume the run for one more epoch

    さらに 1 エポック分だけ再開する。

    エポック数を増やさずに再開すると、既に到達済みとして学習も記録の
    書き出しも行われない。その状態で設定ファイルを読むと、再開前の内容を
    見て「指定が効かなかった」と誤読することになる。
    '''
    return train(workdir, '--resume', 'latest', '--num-epochs', '2',
                 *extra, check=check)


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

    def test_resuming_without_the_flag_keeps_the_saved_value(self, resumable):
        # 保存済みの値は、指定し直さなくても引き継がれること
        resume(resumable)
        assert saved_config(resumable)['train']['dropout_ratio'] == 0.3


@pytest.fixture(scope='module')
def _trained_once(tmp_path_factory):
    '''One finished epoch, kept to be copied per test

    1 エポック分の結果。テストごとに複製して使う。
    '''
    workdir = tmp_path_factory.mktemp('resume-base') / 'work'
    train(workdir, '--dropout-ratio', '0.3')
    return workdir


@pytest.fixture
def resumable(_trained_once, tmp_path):
    '''A fresh copy of that run, so each test resumes independently

    その結果の複製。テストごとに独立して再開できるようにする。
    '''
    workdir = tmp_path / 'work'
    shutil.copytree(_trained_once, workdir)
    return workdir


class TestResume:
    """What a resume takes from the command line and what it keeps

    再開時に、コマンドラインから受け取る設定と、チェックポイントの値を
    保つ設定の区別。

    モデルは保存済みの設定から組み直したうえで重みを読み込むため、構造に
    関わる値が変わると形が合わずに読み込みが失敗する。訓練側の設定には
    その制約が無く、継続学習では差し替えられる必要がある。
    """

    def test_a_training_parameter_follows_the_command_line(self, resumable):
        # 継続学習で学習条件を変えられること
        resume(resumable, '--dropout-ratio', '0.05')
        assert saved_config(resumable)['train']['dropout_ratio'] == 0.05

    def test_the_optimizer_can_be_replaced(self, resumable):
        resume(resumable, '--optimizer', 'adam')
        assert saved_config(resumable)['train']['optimizer'] == 'adam'

    def test_a_model_parameter_keeps_the_checkpoint_value(self, resumable):
        """Rebuilding the model at another size cannot load the weights

        0.1.0.dev0 は指定どおりの大きさでモデルを組み直してから重みを
        読み込もうとし、形状不一致で異常終了していた。
        """
        resume(resumable, '--hidden-size', '64')
        assert saved_config(resumable)['model']['hidden_size'] == 32

    def test_changing_a_model_parameter_no_longer_aborts(self, resumable):
        # 以前はここで終了コード 1 になっていた
        result = resume(resumable, '--hidden-size', '64', check=False)
        assert result.returncode == 0

    def test_the_ignored_parameter_is_reported(self, resumable):
        '''Dropping a value silently would be worse than not applying it

        黙って捨てず、上書きしたい場合の手段まで伝えること。
        '''
        result = resume(resumable, '--hidden-size', '64')
        message = (result.stdout + result.stderr).decode()
        assert 'hidden_size' in message
        assert '--override-model-params' in message

    def test_the_flag_applies_a_compatible_model_parameter(self, resumable):
        # 構造に影響しない値は、フラグを付ければ変更できること
        resume(resumable, '--max-length', '128', '--override-model-params')
        assert saved_config(resumable)['model']['max_length'] == 128

    def test_the_flag_is_not_folded_into_the_configuration(self, resumable):
        # 1 回の実行限りの指定であり、設定に残らないこと
        resume(resumable, '--override-model-params')
        config = saved_config(resumable)
        assert 'override_model_params' not in config.get('model', {})
        assert 'override_model_params' not in config.get('train', {})
        assert 'override_model_params' not in config.get('log', {})


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
