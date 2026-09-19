'''Tests for the save and load paths of lpu_nn.common.training.Trainer

lpu_nn.common.training.Trainer の保存・復元経路のテスト。

チェックポイントは訓練結果を取り出す唯一の経路であり、`run_seq2seq` が
直接叩く。ここが壊れると学習そのものは成功しても成果が取り出せないため、
往復と、候補の切り替えを確認する。
'''

import json
import os
import shutil

import pytest
import torch

from lpu_nn.commands.train_seq2seq import Seq2SeqTrainer
from lpu_nn.common import training

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(REPO_ROOT, 'tests', 'data', 'train.tsv')


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    '''Train for one epoch and return the working directory

    1 エポックだけ訓練し、その作業ディレクトリを返す。
    '''
    import subprocess
    import sys
    workdir = tmp_path_factory.mktemp('status') / 'work'
    result = subprocess.run([
        sys.executable, '-m', 'lpu_nn.commands.train_seq2seq',
        str(workdir), CORPUS,
        '--num-epochs', '1', '--batch-size', '32', '--vocab-size', '100',
        '--embed-size', '32', '--hidden-size', '32', '--num-heads', '4',
        '--gpu', '-1',
    ], capture_output=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr.decode()[-2000:]
    return workdir


class TestSavedLayout:
    def test_the_latest_record_holds_a_model_and_a_config(self, trained):
        latest = trained / 'record.latest'
        assert (latest / 'model.pt').is_file()
        assert (latest / 'config.json').is_file()
        assert (latest / 'optimizer.pt').is_file()

    def test_the_configuration_is_valid_json(self, trained):
        config = json.loads((trained / 'record.latest' / 'config.json').read_text(
            encoding='utf-8'))
        assert config['model']['hidden_size'] == 32

    def test_the_checkpoint_carries_the_vocabulary_state(self, trained):
        state = torch.load(
            trained / 'record.latest' / 'model.pt', 'cpu', weights_only=False)
        assert sorted(state) == ['config', 'idmaps', 'model']


class TestLoadStatus:
    def test_a_directory_is_resolved_to_its_model(self, trained):
        trainer = Seq2SeqTrainer().load_status(str(trained / 'record.latest'))
        assert trainer.model is not None
        assert len(trainer.vocab) > 0

    def test_a_record_name_is_resolved_under_the_working_directory(self, trained):
        trainer = Seq2SeqTrainer().load_status(str(trained), record='latest')
        assert trainer.model is not None

    def test_a_file_path_is_taken_as_is(self, trained):
        trainer = Seq2SeqTrainer().load_status(
            str(trained / 'record.latest' / 'model.pt'))
        assert trainer.model is not None

    def test_a_missing_record_is_reported_with_the_paths_tried(self, trained):
        with pytest.raises(RuntimeError, match='not found files'):
            Seq2SeqTrainer().load_status(str(trained), record='absent')

    def test_the_optimizer_is_restored_on_request(self, trained):
        trainer = Seq2SeqTrainer().load_status(
            str(trained / 'record.latest'), load_optimizer=True)
        assert trainer.optimizer is not None

    def test_reusing_the_dataset_does_not_need_the_optimizer(self, trained, tmp_path):
        # record_dir が load_optimizer の分岐内でしか代入されておらず、
        # この組み合わせは UnboundLocalError になっていた
        workdir = tmp_path / 'copy'
        shutil.copytree(trained, workdir)
        trainer = Seq2SeqTrainer().load_status(
            str(workdir / 'record.latest'), load_optimizer=False, reuse_dataset=True)
        assert trainer.model is not None

    def test_a_checkpoint_without_a_vocabulary_is_reported(self, trained, tmp_path):
        # 以前は UnboundLocalError になっていた
        state = torch.load(
            trained / 'record.latest' / 'model.pt', 'cpu', weights_only=False)
        del state['idmaps']
        broken = tmp_path / 'broken.pt'
        torch.save(state, broken)
        with pytest.raises(ValueError, match='idmaps'):
            Seq2SeqTrainer().load_status(str(broken))

    def test_the_restored_model_produces_the_same_logits(self, trained):
        # 復元がパラメータまで含めて一致すること
        first = Seq2SeqTrainer().load_status(str(trained / 'record.latest'))
        second = Seq2SeqTrainer().load_status(str(trained / 'record.latest'))
        source = torch.randint(4, 20, (1, 4))
        target = torch.randint(4, 20, (1, 3))
        first.model.eval()
        second.model.eval()
        with torch.no_grad():
            assert torch.allclose(first.model(source, target), second.model(source, target))


class TestTryLoading:
    def test_it_falls_back_to_a_later_record(self, trained, tmp_path):
        # --resume が一覧を取るのは、読み込めない記録を後続で代替するため。
        # 以前は最初の失敗でそのまま送出しており、1 件目しか試されなかった
        workdir = tmp_path / 'fallback'
        shutil.copytree(trained, workdir)
        trainer = Seq2SeqTrainer()
        trainer.try_loading(str(workdir), ['absent', 'latest'])
        assert trainer.model is not None

    def test_the_first_loadable_record_wins(self, trained, tmp_path):
        workdir = tmp_path / 'first'
        shutil.copytree(trained, workdir)
        trainer = Seq2SeqTrainer()
        trainer.try_loading(str(workdir), ['latest', 'absent'])
        assert trainer.model is not None

    def test_every_record_failing_still_raises(self, trained, tmp_path):
        # 読み込めないまま進むと、黙って最初から学習し直すことになる
        workdir = tmp_path / 'none'
        shutil.copytree(trained, workdir)
        # 最後に起きたエラーをそのまま送出する
        with pytest.raises(RuntimeError, match='not found files'):
            Seq2SeqTrainer().try_loading(str(workdir), ['absent', 'missing'])

    def test_an_empty_list_is_accepted(self, trained):
        trainer = Seq2SeqTrainer()
        assert trainer.try_loading(str(trained), []) is trainer


class TestSaveConfig:
    def test_it_writes_utf8(self, trained, tmp_path):
        # 既定エンコーディングのままでは Windows で非 ASCII が壊れる
        trainer = Seq2SeqTrainer().load_status(str(trained / 'record.latest'))
        trainer.save_config(str(tmp_path))
        written = (tmp_path / 'config.json').read_bytes()
        assert json.loads(written.decode('utf-8'))

    def test_a_record_name_places_it_in_the_record_directory(self, trained, tmp_path):
        trainer = Seq2SeqTrainer().load_status(str(trained / 'record.latest'))
        trainer.save_config(str(tmp_path), record='mine')
        assert (tmp_path / 'record.mine' / 'config.json').is_file()


class TestSetLogfileHandler:
    def test_the_previous_handler_is_closed_on_replacement(self, tmp_path):
        first = tmp_path / 'first.log'
        second = tmp_path / 'second.log'
        training.set_logfile_handler(str(first))
        previous = training.logfile_handler
        stream = previous.stream
        training.set_logfile_handler(str(second))
        # StreamHandler.close() は渡されたストリームを閉じないため、
        # 自分でファイルを開く FileHandler に変えてある
        assert stream.closed
        assert training.logfile_handler is not previous
