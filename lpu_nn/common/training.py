#!/usr/bin/env python3

'''
    General purpose trainer class
'''

# system
import argparse
import datetime
import gc
import glob
import math
import os
import re
import random
import sys
import time
from collections import defaultdict
from collections import OrderedDict
from lpu_nn.common.args import strtobool
from gettext import gettext as _
from collections.abc import Iterator
from typing import Any

# 3rd party
import numpy as np
import pandas as pd
import torch

# local
from lpu.common import logging
from lpu.common.config import Config
from lpu.common.config import ConfigData
from lpu_nn.common.dataset import Dataset
from lpu_nn.common.dataset import build_train_data
from lpu_nn.common.dataset import load_eval_data
from lpu.common.dialog import ask_continue_if_exist
from lpu.common.files import load_to_temp
from lpu.common.files import safe_copy
from lpu.common.files import safe_link
from lpu.common.files import safeMakeDirs
from lpu.common.files import safe_remove
from lpu.common.files import safe_rename
from lpu.common.progress import view as pview
from lpu_nn.common.files import PastedFile
from lpu_nn.common.initialization import apply_init_weights
from lpu_nn.common import vocab
from lpu_nn.common.vocab import FieldMap
from lpu_nn import optimizers

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

PRESET_CHOICES = None

default = ConfigData()
### Model parameters
default.model = {}
default.model['//'] = 'model parameters'
default.model.task = None
# general
default.model.dtype = 'float32'
default.model.architecture = None
default.model.universal = None
default.model.embed_size = None
default.model.hidden_size = None
default.model.inner_size = None
default.model.vocab_size = 16000
default.model.share_embedding = None
default.model.activation = None
#default.model.embed_positions = False
default.model.embed_positions = None
default.model.max_length = 512
# repeating layers
#default.model.max_steps = 8
#default.model.max_steps = 12
default.model.max_steps = None
#default.model.num_layers = 8
#default.model.num_layers = 6
#default.model.num_layers = None
#default.model.num_blocks = None
#default.model.recurrence = 'act'
default.model.recurrence = None
# self attentions
#default.model.num_heads = 16
default.model.num_heads = None
#default.model.relative_attention = False
default.model.relative_attention = None
#default.model.clip_distance = 8
default.model.clip_distance = None
default.model.character_level = None
### Training setting
default.train = {}
default.train['//'] = 'training settings'
# general
## mini-batches
default.train.batch_size = 64
default.train.batch_type = 'samples'
#default.train.min_batch_size = 1
default.train.min_batch_size = 32
default.train.dropout_ratio = 0.1
default.train.max_batches = 2500
default.train.timeout = 3600 * 2
default.train.max_samples_per_epoch = 10 ** 6
# vector freezing
default.train.fix_imported_vectors = None
# loss func (e.g. smoothed cross entropy)
default.train.loss = None
## curriculum
default.train.curriculum = 'crit-len'
## optimizer
default.train.optimizer = 'adam'
default.train.learning_rate = None
#default.train.optimizer = 'amsgrad'
### adam specific
#default.train.adam_alpha = 5e-5
default.train.adam_alpha = 0.001
default.train.adam_beta1 = 0.9
#default.train.adam_beta2 = 0.997
#default.train.adam_beta2 = 0.98
default.train.adam_beta2 = 0.999
default.train.adam_eps = 1e-8
#default.train.adam_eps = 1e-7
#default.train.adam_amsgrad = True
### sgd specific
#default.train.sgd_learning_rate = 0.01
default.train.sgd_learning_rate = 0.1
default.train.adabound_gamma = 1e-3
#default.train.gamma = 1.0 / 16000
## warm-up steps
#default.train.warmup_steps = 4000
default.train.warmup_steps = 16000
default.train.warmup_factor = 1
#default.train.weight_decay_warmup_steps = 10**5
default.train.weight_decay_warmup_steps = None
#default.train.weight_decay_rate = 5e-6
default.train.weight_decay_rate = 1e-5
#default.train.weight_decay_rate = 0.01
default.train.gradient_clipping = 5.0
# universal-transformer specific
default.train.time_penalty = 0.01
# multi-step
default.train.schedule_num_steps = False
# reviewing
default.train.review_rate = 0.5
# challenging (review diffcult samples)
default.train.challenge_rate = 0.01
# initialization
default.train.initializer = None
# random generation
default.train.random_seed = 1
### Logging status
default.log = {}
default.log['//'] = 'logging settings and reports'
# training status
default.log.epoch = 0

default.log.interval = 60
default.log.eval_timeout = None
#default.log.interval = 10
default.log.train_step = 0
default.log.fed_samples = 0
default.log.fed_tokens = 0
#default.log.fed_src_tokens = 0
#default.log.fed_trg_tokens = 0
#default.log.fed_tokens = 0
default.log.elapsed = 0
default.log.elapsed_hours = 0
default.log.elapsed_days = 0

logfile_handler = None

# Logger names are dotted, so a name only reaches the loggers beneath it.
# These were the top-level module names before the port: the modules are
# `lpu_nn.common.training` and the like now, and `common` is not an ancestor
# of `lpu_nn.common`, so neither the --logging file handler nor --debug
# reached any of them. The log file held only the __main__ lines.
# (ロガー名はドット区切りで、その名前の配下にしか効かない。これらは移植前の
#  トップレベル名であり、現在のモジュールは `lpu_nn.common.training` 等で、
#  `common` は `lpu_nn.common` の祖先ではない。そのため --logging の
#  ファイルハンドラも --debug もこれらに届いておらず、ログファイルには
#  __main__ の行しか残っていなかった)
target_loggers = ['__main__', 'lpu_nn', 'lpu']

LIST_K_FOR_RECALL = [1, 5, 10, 20, 50, 100]

def set_logfile_handler(logpath: str) -> None:
    """Send the target loggers to the given file

    対象のロガーの出力を、指定したファイルへ流す。
    """
    global logfile_handler
    if logfile_handler is not None:
        for name in target_loggers:
            l = logging.getLogger(name)
            l.removeHandler(logfile_handler)
        # 外したハンドラと、その先のファイルを閉じる (以前は開いたままだった)
        logfile_handler.close()
    # StreamHandler.close() は渡されたストリームを閉じない。FileHandler は
    # 自分で開いたファイルを閉じるため、差し替えで取り残しが出ない
    logfile_handler = logging.FileHandler(
        logpath, mode='a', encoding='utf-8', errors='backslashreplace')
    for name in target_loggers:
        l = logging.getLogger(name)
        l.addHandler(logfile_handler)
        logging.colorize(l)

def format_time(seconds: float) -> str:
    """Render a duration as a compact string

    経過時間を短い文字列に整形する。

    1 日を超える場合は秒を付けない。ちょうど 60 秒は "1M" ではなく
    "60S" になるが、表示のみに用いるためそのままにしている。
    """
    s = ""
    remain = seconds
    if remain > 60 * 60 * 24:
        s = f"{math.floor(remain / (60 * 60 * 24)):d}D"
        remain = remain % (60 * 60 * 24)
    if remain > 60 * 60:
        s += f"{math.floor(remain / (60 * 60)):d}H"
        remain = remain % (60 * 60)
    if remain > 60:
        s += f"{math.floor(remain / 60):d}M"
        remain = remain % 60
    if seconds < 60:
        s += f"{remain:.2f}S"
    elif seconds < 60 * 60 * 24:
        s += f"{math.floor(remain):d}S"
    return s

def setup_optimizer(config: Any, model: Any) -> Any:
    cdata = config.data
    optimizer_name = cdata.train.optimizer
    cdata.train.optimizer = optimizer_name = str(optimizer_name).lower()
    weight_decay_rate = cdata.train.weight_decay_rate
    logger.info("building optimizer")
    biases = []
    others = []
    showed = set()
    for name, param in model.named_parameters():
        if param.dim() == 1 or name.endswith('bias'):
            if weight_decay_rate > 0:
                replaced = re.sub(r'\.\d+', '[:]', name)
                if replaced not in showed:
                    logger.debug(f"setting weight_decay_rate = 0 for parameter {replaced}")
                    showed.add(replaced)
            biases.append(param)
        else:
            others.append(param)
    if weight_decay_rate > 0:
        logger.debug(f"setting weight_decay_rate = {weight_decay_rate} for the other parameters")
    param_groups = [
        {'params': others, 'weight_decay': weight_decay_rate},
        {'params': biases, 'weight_decay': 0},
    ]
    if optimizer_name in ['adam', 'amsgrad']:
        alpha   = cdata.train.adam_alpha
        beta1   = cdata.train.adam_beta1
        beta2   = cdata.train.adam_beta2
        eps     = cdata.train.adam_eps
        amsgrad = (optimizer_name == 'amsgrad')
        optimizer = optimizers.AdaBoundW(param_groups, lr=alpha, final_lr=None, betas=(beta1,beta2), eps=eps, amsbound=amsgrad)
    elif optimizer_name in ['adamax']:
        alpha   = cdata.train.adam_alpha
        beta1   = cdata.train.adam_beta1
        beta2   = cdata.train.adam_beta2
        eps     = cdata.train.adam_eps
        amsgrad = (optimizer_name == 'amsgrad')
        optimizer = optimizers.Adamax(param_groups, lr=alpha, betas=(beta1,beta2), eps=eps)
    elif optimizer_name in ['adabound', 'amsbound']:
        alpha   = cdata.train.adam_alpha
        beta1   = cdata.train.adam_beta1
        beta2   = cdata.train.adam_beta2
        eps     = cdata.train.adam_eps
        gamma   = cdata.train.adabound_gamma
        #final_lr = cdata.train.final_learning_rate
        final_lr = cdata.train.sgd_learning_rate
        final_lr_rate = final_lr / alpha
        amsgrad = (optimizer_name == 'amsbound')
        if amsgrad:
            logger.debug("using experimentally backported optimizer: AMSBound")
        else:
            logger.debug("using experimentally backported optimizer: AdaBound")
        dprint(alpha)
        dprint(beta1)
        dprint(beta2)
        dprint(gamma)
        dprint(final_lr)
        dprint(final_lr_rate)
        optimizer = optimizers.AdaBoundW(param_groups, lr=alpha, final_lr=final_lr, betas=(beta1,beta2), eps=eps, gamma=gamma, amsbound=amsgrad)
    elif optimizer_name in ['lamb']:
        alpha   = cdata.train.adam_alpha
        beta1   = cdata.train.adam_beta1
        beta2   = cdata.train.adam_beta2
        eps     = cdata.train.adam_eps
        optimizer = optimizers.Lamb(param_groups, lr=alpha, betas=(beta1,beta2), eps=eps)
    else:
        # otherwise using SGD
        cdata.train.optimizer = 'sgd'
        dprint(cdata.train.sgd_learning_rate)
        optimizer = optimizers.SGD(param_groups, lr=cdata.train.sgd_learning_rate)
    return optimizer

def get_record_name(model_path: str) -> "str | None":
    """Extract the record name out of a checkpoint path

    チェックポイントのパスから記録名を取り出す。
    例: 'workdir/record.best_dev_loss' -> 'best_dev_loss'
    """
    found = re.findall('record.([a-z_]*)', model_path)
    if found:
        record = found[0]
        return record
    return None

def build_batches_by_samples(df: pd.DataFrame, batch_size: int) -> "Iterator[pd.DataFrame]":
    for i in range(0, len(df), batch_size):
        yield df[i:i+batch_size]

def build_batches_by_tokens(df: pd.DataFrame, batch_size: int) -> "Iterator[pd.DataFrame]":
    batch_items = []
    num_tokens = 0
    for _i, row in df.iterrows():
        batch_items.append(row)
        num_tokens += row.len
        if num_tokens >= batch_size:
            yield pd.DataFrame(batch_items)
            num_tokens = 0
            batch_items = []
    if len(batch_items) > 0:
        yield pd.DataFrame(batch_items)

def build_batches(df: pd.DataFrame, batch_size: int,
                  batch_type: str = 'samples') -> "Iterator[pd.DataFrame]":
    """Split a dataframe into batches of samples or of tokens

    データフレームをサンプル単位またはトークン単位のバッチへ分割する。

    An unknown batch type used to fall off the end and return None, which
    the caller then tried to iterate.

    未知の種別では末尾に抜けて None を返しており、呼び出し側がそれを
    反復しようとしていた。
    """
    if batch_type == 'samples':
        return build_batches_by_samples(df, batch_size)
    elif batch_type == 'tokens':
        return build_batches_by_tokens(df, batch_size)
    raise ValueError(f"unknown batch type: {batch_type!r}")

def reduce_batch_size(batch: pd.DataFrame, batch_size: int,
                      batch_type: str) -> pd.DataFrame:
    """Shrink one batch to fit the given size

    1 つのバッチを指定した大きさに収まるまで縮める。
    メモリ不足からの再試行で使う。
    """
    if batch_type == 'samples':
        return batch[:batch_size]
    elif batch_type == 'tokens':
        while batch.len.sum() > batch_size:
            if len(batch) == 1:
                break
            #dprint(batch.len.sum())
            batch.drop(batch.len.idxmax(), inplace=True)
        return batch
    raise ValueError(f"unknown batch type: {batch_type!r}")

class Trainer:
    # 派生クラスが具体的な設定とモデルクラスで上書きする
    default: Any = default
    specific: Any = None
    Model: Any = None
    #def __init__(self):
    def __init__(self, args=None):
        self.args = args
        self.config = Config(self.default)
        # 構築時には未設定で、後から setup で入る
        self.idmaps: Any = None
        self.model: Any = None
        self.optimizer: Any = None
        self.sep = '\t'
        self.update_main_fields()

    def update_main_fields(self):
        self.main_fields = main_fields = self.specific.to_dict('format', flat=True, ordered=True, upstream=True)
        self.len_fields = ['len_'+column for column, ftype in main_fields.items() if ftype == 'seq']
        return self.main_fields

    def build_train_batches(self):
        cdata = self.config.data
        train_df = self.train_df
        method = cdata.train.curriculum
        train_size = len(self.train_data)
        #dprint(train_df.iloc[:5])
        if method in ['crit-len']:
            train_samples = train_df.sort_values(['priority', 'len'])
        elif method in ['crit']:
            #train_samples = train_df.sort_values(['priority'])
            train_samples = train_df.sort_values(['priority'], kind='mergesort')
        elif method in ['len']:
            train_samples = train_df.sort_values(['len'])
        elif method in ['none', None]:
            #train_samples = train_df
            train_samples = train_df.sample(len(train_df))
            method = 'none'
        else:
            raise ValueError(f"unknown curriculum method: {method}")
        #dprint(train_samples.iloc[:5])
        digest_data0 = train_samples[train_samples.priority <= 0]
        if True:
            pd.set_option('display.max_columns', 10)
            pd.set_option('display.max_rows', 30)
            pd.set_option('display.width', 200)
            digest_data0 = train_samples[train_samples.priority <= 0]
            if len(digest_data0) > 6:
                digest_data1 = pd.concat([digest_data0[:3], digest_data0[-3:]])
            else:
                digest_data1 = digest_data0
            digest_data2 = train_samples[train_samples.priority > 0]
            digest_data = pd.concat([digest_data1, digest_data2])
            digest_str = repr(digest_data[[*self.len_fields, 'cost', 'feed_count', 'last_epoch', 'last_step', 'criterion', 'priority']])
            logger.info("training data digest:\n" + digest_str)
            if cdata.log.epoch >= 2:
                evaluated = train_df[train_df.criterion > 0]
                if len(evaluated) > 0:
                    try:
                        desc = evaluated.describe().transpose()
                        logger.info(f"statistics for trained samples:\n{desc}")
                    except Exception as e:
                        dprint(repr(e))
                logger.info(f"training data size: {train_size:,}")
                logger.info(f"taken training data size: {len(train_samples):,}")
                taken_ratio = len(train_samples) / float(train_size)
                logger.info(f"taking: {taken_ratio * 100:.2f}%")
        if cdata.log.epoch >= 2:
            if cdata.train.review_rate > 0:
                # secure a chance to train unseen examples and review examples trained in previous epochs
                unseen_rate = max(0.01, 1 - cdata.train.review_rate)
                unseen_limit = int(cdata.train.batch_size * cdata.train.max_batches * unseen_rate + 1)
                #unseen_limit = int(cdata.train.batch_size * cdata.train.max_batches * 0.5 + 1)
                unseen = train_samples[train_samples.feed_count == 0]
                seen = train_samples[train_samples.feed_count > 0]
                logger.info(f"unseen data size: {len(unseen):,}")
                logger.info(f"seen data size: {len(seen):,}")
                #drop_indices = unseen.index[unseen_limit:]
                #curriculum_data.drop(drop_indices, inplace=True)
                train_samples = pd.concat([unseen[:unseen_limit], seen])
        self.challenge_batches = None
        if cdata.log.epoch >= 2:
            if cdata.train.challenge_rate > 0 and method != 'none':
                challenge_rate = cdata.train.challenge_rate
                challenge_data = train_df.sort_values(['criterion'])
                challenge_data = challenge_data[challenge_data.criterion > 0]
                #challenge_size = int( min(10**4, len(challenge_data)*0.01+1) )
                challenge_size = int( min(10**4, len(challenge_data)*challenge_rate+1) )
                #challenge_size = int( max(10**4, len(challenge_data)*0.01+1) )
                challenge_data = challenge_data[::-1][:challenge_size]
                challenge_data = challenge_data.sort_values(self.len_fields)
                #self.challenge_batches = list(chainer.iterators.SerialIterator(challenge_data, cdata.train.batch_size, repeat=False, shuffle=False))
                if len(challenge_data) > 0:
                    self.challenge_batches = list(build_batches(challenge_data, cdata.train.batch_size, cdata.train.batch_type))
                    train_samples = train_samples[~train_samples.index.isin(challenge_data.index)]
                    random.shuffle(self.challenge_batches)
                else:
                    self.challenge_batches = None
        #logger.info("curriculum data size: {:,}".format(len(train_samples)))
        logger.info(f"train data size: {len(train_samples):,}")
        #self.curriculum_batches = list(build_batches(train_samples, cdata.train.batch_size, cdata.train.batch_type))
        self.train_batches = list(build_batches(train_samples, cdata.train.batch_size, cdata.train.batch_type))
        #random.shuffle(self.train_batches)
        return True

    def evaluate(self, tag, df, args, feed_batches=None, report=None):
        if report is None:
            eval_report = pd.Series()
        else:
            eval_report = report
        if df is None:
            return eval_report
        cdata = self.config.data
        batch_size = max(cdata.train.min_batch_size, int(cdata.train.batch_size / 2))
        batches = list(build_batches(df, batch_size, cdata.train.batch_type))
        try:
            with torch.no_grad():
                self.model.zero_grad()
                if self.optimizer:
                    self.optimizer.zero_grad()
                if feed_batches:
                    logger.info(f"evaluating {tag} batches...")
                    #eval_report = self.feed_batches(batches, train=False, show_report=True)
                    eval_report = self.feed_batches(batches, train=False, show_report=True, df=df)
        except Exception as e:
            logger.exception(e)
        return eval_report

    def filter_noisy_samples(self):
        def drop_noise(noisy, based_on):
            noisy = noisy.sort_values('criterion')
            noisy_repr = repr(noisy[[*self.len_fields, 'cost', 'feed_count', 'last_epoch', 'last_step', 'criterion', 'priority']])
            logger.info(f"noisy training data (based on \"{based_on}\"):\n{noisy_repr}")
            logger.info(f"dropping {len(noisy)} noisy training samples...")
            self.train_df.drop(noisy.index, inplace=True)
        # dropping noisy training data
        cdata = self.config.data
        steps = cdata.log.train_step
        config = self.config
        #vocab_size = len(self.model.vocab)
        len(self.train_data)
        trained = self.train_df[self.train_df.last_epoch == cdata.log.epoch]
        if True:
            dprint(self.last_worst_criterion)
            if getattr(self, 'last_worst_criterion', None):
                if len(self.train_data) >= cdata.train.max_samples_per_epoch:
                    threshold = self.last_worst_criterion
                    based_on = 'criterion compared to last worst criterion'
                else:
                    threshold = self.last_worst_criterion + self.train_df.criterion.std()
                    based_on = 'criterion compared to last worst criterion + std'
                noisy = trained[(trained.feed_count >= 2) & (trained.criterion > threshold)]
                if len(noisy) > 0:
                    dprint(threshold)
                    drop_noise(noisy, based_on)
                    trained = self.train_df[self.train_df.last_epoch == cdata.log.epoch]
        if config.get('train.warmup_steps') is not None and cdata.train.warmup_steps > 0:
            step_threshold = cdata.train.warmup_steps
        else:
            step_threshold = cdata.train.max_batches * 2
        step_threshold = max(10000, step_threshold)
        if steps >= step_threshold:
            # may be outliers, extremely diffucult to learn
            if len(self.train_data) >= cdata.train.max_samples_per_epoch:
                # aggressive filtering
                #threshold = self.train_df.criterion.mean() + self.train_df.criterion.std() * 3
                threshold = self.train_df.criterion.mean() + self.train_df.criterion.std() * 3
                based_on = 'criterion compared to mean+3std'
            else:
                #threshold = self.train_df.criterion.mean() + self.train_df.criterion.std() * 10 + 10
                #based_on = 'criterion compared to mean+10std+10'
                threshold = self.train_df.criterion.mean() + self.train_df.criterion.std() * 5 + 1
                based_on = 'criterion compared to mean+5std+1'
            noisy = trained[(trained.feed_count >= 2) & (trained.criterion > threshold)]
            if len(noisy) > 0:
                dprint(threshold)
                #drop_noise(noisy, 'criterion compared to mean+3std')
                #drop_noise(noisy, 'criterion compared to mean+10std+10')
                drop_noise(noisy, based_on)
                trained = self.train_df[self.train_df.last_epoch == cdata.log.epoch]
        #if steps >= step_threshold:
        #    factor = 1
        #    if cdata.train.optimizer in ['adabound', 'amsbound'] and cdata.train.warmup_steps == 0:
        #        params_factor = 1 / max(1, cdata.model.num_params ** 0.5 / 1000)
        #        factor = params_factor
        #    if train_size > 10 ** 5:
        #        noisy = trained[ (trained.feed_count >= 2) & ((trained.criterion-1) * (trained.last_step ** 0.5) * factor > vocab_size) ]
        #        if len(noisy) > 0:
        #            drop_noise(noisy, 'criterion vs last training step')
        #            trained = self.train_df[self.train_df.last_epoch == cdata.log.epoch]
        #    noisy = trained[ (trained.feed_count >= 5) & ((trained.criterion-1) * (trained.feed_count ** 1.5 / 2) > vocab_size) ]
        #    if len(noisy) > 0:
        #        drop_noise(noisy, 'criterion vs last training step')
        #        trained = self.train_df[self.train_df.last_epoch == cdata.log.epoch]

    def get_config(self, key, val):
        if val is None:
            return default[key]
        else:
            return val

    @classmethod
    def get_extra_symbols(cls):
    #def get_extra_symbols(cls, field_name):
        if 'extra_symbols' in cls.specific:
            extra_symbols = cls.specific.to_dict(key='extra_symbols', ordered=True, upstream=True)
            #key = 'extra_symbols.{}'.format(field_name)
            #extra_symbols = cls.specific.to_dict(key=key, ordered=True, upstream=True)
            dprint(extra_symbols)
        else:
            extra_symbols = {}
        return extra_symbols

    def get_num_params(self):
        num_params = sum(param.numel() for param in self.model.parameters())
        self.config.data.model.num_params = num_params
        return num_params

    def init_random(self, args):
        cdata = self.config.data
        if cdata.train.random_seed < 0:
            #cdata.train.random_seed = random.randint(0, 2 ** 16)
            cdata.train.random_seed = torch.initial_seed()
        random.seed(cdata.train.random_seed)
        np.random.seed(cdata.train.random_seed)
        torch.manual_seed(cdata.train.random_seed)
        torch.backends.cudnn.deterministic = True

    def link_status(self, basedir, src_record, dist_record, log=True):
        link_filenames = [
            'config.json',
            'model.pt',
            'optimizer.pt',
            #'sp.model',
            'pred_dev.txt',
            'pred_test.txt',
            #'train.log',
            #'scores.json',
            'train_data.tsv',
        ]
        copy_filenames = [
            'train.log',
            'scores.json',
        ]
        link_patterns = [
            'plot*.pdf',
        ]
        src_dir  = os.path.join(basedir, 'record.'+src_record)
        dist_dir = os.path.join(basedir, 'record.'+dist_record)
        if not os.path.isdir(src_dir):
            return False
        safeMakeDirs(dist_dir)
        for pattern in link_patterns:
            for path in glob.glob(os.path.join(src_dir, pattern)):
                link_filenames.append( os.path.basename(path) )
        for filename in link_filenames + copy_filenames:
            #src_path = dist_path = os.path.join(basedir, filename)
            src_path  = os.path.join(src_dir, filename)
            dist_path = os.path.join(dist_dir, filename)
            if filename in copy_filenames:
                safe_copy(src_path, dist_path, log=log)
            else:
                safe_link(src_path, dist_path, log=log)

    def load_eval_data(self, path):
        #return load_eval_data(self.main_fields, path, self.vocab, self.sep)
        return load_eval_data(self.main_fields, path, self.sep)

    def load_labels(self, path=None):
        self.labels = []
        #self.label2id = {}
        self.label2id = defaultdict(lambda: 0)
        self.labels.append('<unk>')
        self.label2id['<unk>'] = 0
        if path:
            for line in open(path, encoding='utf-8'):
                #ids = tuple(self.vocab.encode(line.strip()))
                label = line.strip()
                #if ids not in self.label2id:
                if label not in self.label2id:
                    self.label2id[label] = len(self.labels)
                    self.labels.append(label)
        else:
            df = Dataset(self.main_train_data_path, self.sep).to_df(self.main_fields)
            for _i, row in df.iterrows():
                #ids = row.s2
                label = row.s2
                if label not in self.label2id:
                    self.label2id[label] = len(self.labels)
                    self.labels.append(label)

    def load_model(self, model_path, force=True):
        logger.info(f"loading model from '{model_path}' ...")
        loaded_state_dict = torch.load(model_path, 'cpu')
        config = Config(self.default)
        config.update(loaded_state_dict['config'])
        if self.args is not None:
            config = self.update_config(config, self.args)
        dprint(config.to_json(upstream=True, indent=2))
        params = config.to_dict(flat=True, upstream=True)
        if 'idmaps' not in loaded_state_dict:
            # 以前はここを素通りし、直後の参照が UnboundLocalError になっていた
            raise ValueError(
                f"the checkpoint has no vocabulary state ('idmaps'): {model_path}")
        idmaps = FieldMap().set_state(loaded_state_dict['idmaps'])
        #dprint(idmaps)
        model = self.Model(idmaps=idmaps, **params)
        loaded_model_state_dict = loaded_state_dict['model']
        try:
            model.load_state_dict(loaded_model_state_dict)
        except Exception as e:
            if force:
                #logger.exception(e)
                logger.error(repr(e))
                logger.info("failed to load model parameters")
                logger.info("falling back...")
                model_state_dict = model.state_dict()
                #for k, v in state_dict['model'].items():
                for k, v in loaded_model_state_dict.items():
                    if k in model_state_dict:
                        logger.debug(f"loading \"{k}\"")
                        model_state_dict[k] = v
                    else:
                        logger.debug(f"\"{k}\" is not found")
                model.load_state_dict(model_state_dict)
            else:
                raise e
        model.eval() # default mode as evaluation
        return config, model

    @classmethod
    def load_optimizer(cls, path, config, model, force=True):
        optimizer = setup_optimizer(config, model)
        optimizer_path = path
        if os.path.exists(optimizer_path):
            loaded_optimizer_state_dict = torch.load(optimizer_path, 'cpu')
            try:
                logger.info(f"loading optimizer from '{optimizer_path}' ...")
                optimizer.load_state_dict(loaded_optimizer_state_dict)
            except Exception as e:
                logger.error(repr(e))
                logger.info("failed to load optimizer parameters")
                logger.info("continuing without loading")
        else:
            raise RuntimeError(f"Not found file: {optimizer_path}")
        return optimizer

    def load_train_data(self, path=None):
        curriculum = self.config.data.train.curriculum
        if path is None:
            #path = self.worker_data_path
            path = self.train_data_path
        logger.info(f"loading train dataset: {path}")
        if curriculum in ['crit-len']:
            self.train_data = Dataset(path, sep='\t', priority_keys=['priority', 'len'])
        elif curriculum in ['crit']:
            self.train_data = Dataset(path, sep='\t', priority_keys=['priority'])
        elif curriculum in ['len']:
            self.train_data = Dataset(path, sep='\t', priority_keys=['len'])
        elif curriculum in ['none', None]:
            self.train_data = Dataset(path, sep='\t', priority_keys=None)
        else:
            raise ValueError(f"unknown curriculum: {curriculum}")
        train_size = len(self.train_data)
        max_samples = self.config.data.train.max_samples_per_epoch
        logger.info(f"converting {min(train_size, max_samples):,d} samples to pandas dataframe")
        self.train_df = self.train_data.to_df(self.main_fields, 0, max_samples)
        #dprint(self.train_df.iloc[:5])
        return self.train_df

    def load_status(self, path, record=None, load_optimizer=False, reuse_dataset=False):
        model_path = None
        path_candidates = []
        if os.path.isfile(path):
            path_candidates.append(path)
        else:
            path_candidates.append(os.path.join(path, 'model.pt'))
            if record is None:
                path_candidates.append(os.path.join(path, 'record.latest', 'model.pt'))
            else:
                if os.path.isfile(record):
                    path_candidates.append(record)
                path_candidates.append(os.path.join(record, 'model.pt'))
                path_candidates.append(os.path.join(path, f'record.{record}', 'model.pt'))
        for path in path_candidates:
            if os.path.isfile(path):
                model_path = path
        if model_path is None:
            raise RuntimeError("not found files: {}".format(str.join(', ', path_candidates)))
        config, model = self.load_model(model_path)
        # record_dir は reuse_dataset でも使うため、分岐の外で決める
        # (以前は load_optimizer の分岐内でしか代入されず、
        #  reuse_dataset=True かつ load_optimizer=False で
        #  UnboundLocalError になっていた)
        record_dir = os.path.dirname(model_path)
        if load_optimizer:
            optimizer_path = os.path.join(record_dir, 'optimizer.pt')
            optimizer = self.load_optimizer(optimizer_path, config, model)
        else:
            optimizer = None
        if reuse_dataset:
            dataset_path = os.path.join(record_dir, 'train_data.tsv')
            safe_link(dataset_path, os.path.join(record_dir, '..', 'record.tmp', 'train_data.tsv'), log=True)
        self.config = config
        self.model = model
        self.vocab = model.vocab
        self.optimizer = optimizer
        if self.config.data.train.schedule_num_steps:
            self.set_max_steps()
        return self

    def plot(self, infile_scores, field_x, fields_y, outfile_plot, labels=None):
        try:
            import matplotlib
        except ImportError:
            # matplotlib は任意依存 (plot エクストラ)。未導入の場合はグラフ出力
            # だけを諦める。広い try の中で import するとエポックごとに
            # トレースバックが出るため、ここで個別に処理する
            logger.debug("matplotlib is not installed, skipping the plot")
            return False
        try:
            import json
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            if not isinstance(fields_y, (tuple,list)):
                fields_y = fields_y
            lines = open(infile_scores, encoding='utf-8').readlines()
            data = [json.loads(line) for line in lines]
            df = pd.DataFrame(data)
            label_x = None
            label_y = None
            fields_y = [field for field in fields_y if field in df]
            if len(fields_y) == 0:
                return False
            if labels is None:
                labels = []
                for field in fields_y:
                    labels.append(field.replace('_', ' '))
            plt.figure()
            plt.grid(which='major', color='#000000', linestyle='dotted')
            plt.grid(which='minor', color='#000000', linestyle='dotted', alpha=0.1, linewidth=1)
            if field_x.find('elapsed') >= 0:
                max_seconds = df[field_x].max()
                if max_seconds >= 3600 * 2:
                    #df.elapsed /= 3600
                    df[field_x] /= 3600
                    label_x = 'Training Time [hours]'
                elif max_seconds >= 60 * 2:
                    #df.elapsed /= 60
                    df[field_x] /= 60
                    label_x = 'Training Time [minutes]'
                else:
                    label_x = 'Training Time [seconds]'
            if field_x.find('fed_samples') >= 0:
                max_samples = df[field_x].max()
                label_x = 'Fed Samples'
                if max_samples >= (1000**2) * 2:
                    df[field_x] /= 1000**2
                    label_x = 'Fed Samples [M samples]'
                elif max_samples >= 1000 * 2:
                    df[field_x] /= 1000
                    label_x = 'Fed Samples [K samples]'
            for _i, (field, label) in enumerate(zip(fields_y, labels, strict=False)):
                plt.plot(df[field_x], df[field], label=label, marker='.')
                if field.find('perplexity') >= 0:
                    max_y = plt.ylim()[1]
                    min_ppl = df[field].min()
                    #dprint(min_ppl)
                    plt.ylim(0, min(max_y, min_ppl*10))
                    label_y = 'Perplexity'
                if field.find('loss') >= 0:
                    label_y = 'Loss'
                if field.find('accuracy') >= 0:
                    #label_y = 'Accuracy'
                    label_y = 'Accuracy [%]'
            plt.xlim(0)
            plt.legend()
            if label_x is not None:
                plt.xlabel(label_x)
            if label_y is not None:
                plt.ylabel(label_y)
            logger.info(f"plotting into: {outfile_plot}")
            safe_remove(outfile_plot, log=False)
            plt.savefig(outfile_plot)
            plt.close()
        except Exception as e:
            logger.exception(e)
            return False
        return True

    def show_progress_report(self):
        cdata = self.config.data
        progress = self.progress
        delta = time.time() - progress['last_time']
        progress['elapsed'] += (time.time() - progress['last_time'])
        if progress['accum_batches'] > 0:
            accum_report = progress['accum_report']
            mean_report = accum_report / progress['accum_batches']
        else:
            accum_report = pd.Series()
            mean_report = pd.Series()
        msg = ''
        for field in self.specific.data.report:
            try:
                if msg:
                    msg = msg.strip(', ') + ', '
                if field in ['epoch']:
                    msg += f"{field}: {cdata.log.epoch}"
                elif field in ['proc', 'process', 'processing']:
                    num_batches = progress.num_batches
                    len_num_batches = len(str(num_batches))
                    str_i = str(progress.batch_i+1).rjust(len_num_batches)
                    msg += f"{field}: {str_i}/{num_batches}"
                elif field in ['lr']:
                    lr = self.optimizer.param_groups[0]['lr']
                    if cdata.train.optimizer in ['adabound', 'amdbound']:
                        final_lr = self.optimizer.param_groups[0]['final_lr']
                        gamma = self.optimizer.param_groups[0]['gamma']
                        step = cdata.log.train_step
                        lower = final_lr * (1.0 - 1.0 / (gamma * step + 1))
                        lr = max(lr, lower)
                    msg += f"{field}: {lr:.8f}"
                elif field in ['acc', 'accuracy']:
                    accuracy = float(mean_report.get('acc', 'nan'))
                    msg += f"{field}: {accuracy:.4f}"
                elif field in ['acc_seq', 'sequence_accuracy']:
                    accuracy = float(mean_report.get('acc_seq', 'nan'))
                    msg += f"{field}: {accuracy:.4f}"
                elif field in ['ntacc', 'non_trivial_accuracy']:
                    accuracy = float(mean_report.get('ntacc', 'nan'))
                    msg += f"{field}: {accuracy:.4f}"
                elif field in ['rest_acc', 'restore', 'restore_accuracy']:
                    accuracy = mean_report.get('rest_acc', 'nan')
                    msg += f"{field}: {accuracy:.4f}"
                elif field in ['cont_acc', 'continuity_accuracy']:
                    accuracy = mean_report.get('cont_acc', 'nan')
                    msg += f"{field}: {accuracy:.3f}"
                elif field in ['ppl', 'perplexity']:
                    ppl = float(mean_report.get('ppl', 'nan'))
                    msg += f"{field}: {ppl:.3f}"
                elif field in ['ntppl', 'non_trivial_perplexity']:
                    ppl = float(mean_report.get('ntppl', 'nan'))
                    msg += f"{field}: {ppl:.3f}"
                elif field in ['loss']:
                    loss = float(mean_report.get('loss', 'nan'))
                    msg += f"{field}: {loss:.4f}"
                elif field in ['gold', 'gold_score']:
                    gold_score = float(mean_report.get('gold', 'nan'))
                    #if gold_score > 0:
                    if math.isfinite(gold_score):
                        msg += f"{field}: {gold_score:.4f}"
                elif field in ['cost', 'pcost', 'ponder_cost']:
                    ponder_cost = float( mean_report.get('ponder_cost', 'nan') )
                    if ponder_cost > 0:
                        msg += f"{field}: {ponder_cost:.3f}"
                elif field in ['gnorm', 'gradient_norm']:
                    msg += "{}: {:.3f}".format(field, mean_report['gnorm'])
                elif field in ['clip', 'gradient_clip']:
                    msg += "{}: {}".format(field, int(accum_report['clip']))
                elif field in ['err', 'errors']:
                    msg += "{}: {}".format(field, progress['accum_errors'])
                elif field in ['samples', 'fed_samples']:
                    if self.model.training:
                        #msg += "{}: {}".format(field, config.fed_samples)
                        msg += f"{field}: {cdata.log.fed_samples:,d}"
                elif field in ['steps']:
                    if self.model.training:
                        #msg += "{}: {}".format(field, config.steps)
                        msg += f"{field}: {cdata.log.train_step}"
                elif field in ['elapsed']:
                    if self.model.training:
                        #str_elapsed = format_time(cdata.log.elapsed)
                        str_elapsed = format_time(progress['elapsed'])
                        msg += f"{field}: {str_elapsed}"
                elif field in ['tokens', 'fed_tokens']:
                    if self.model.training:
                        msg += f"{field}: {cdata.log.fed_tokens:,d}"
                elif field in ['tokens/s', 'tokens/sec']:
                    if self.model.training:
                        delta_tokens = cdata.log.fed_tokens - progress['last_tokens']
                        msg += f"{field}: {delta_tokens / delta:.1f}"
            except Exception:
                #logger.exception(e)
                pass
        logger.info(msg)
        progress['accum_batches'] = 0
        progress['accum_errors'] = 0
        progress['accum_report'] = 0
        progress['last_time'] = time.time()
        progress['last_tokens'] = cdata.log.fed_tokens
        return progress

    #def feed_batches(self, batches, report=False, timeout=None, fallback=False):
    def feed_batches(self, batches, train, show_report=False, timeout=None, fallback=False, df=None):
        if train:
            self.model.train()
        else:
            self.model.eval()
        if self.config.data.train.schedule_num_steps:
            self.set_max_steps()
        cdata = self.config.data
        epoch = cdata.log.epoch
        start_time = time.time()
        #self.prog = prog = {}
        self.progress = progress = ConfigData()
        progress['accum_batches'] = 0
        progress['accum_errors'] = 0
        progress['accum_report'] = 0
        progress['total_batches'] = 0
        progress['total_errors'] = 0
        progress['total_report'] = 0
        progress['elapsed'] = cdata.log.elapsed
        progress['last_time'] = start_time
        progress['last_tokens'] = cdata.log.fed_tokens
        progress.num_batches = len(batches)
        if not show_report:
            train_iterator = pview(batches, 'processing')
        else:
            train_iterator = batches
        if train:
            self.update_train_step(increment=False)
            if cdata.train.optimizer in ['adabound', 'amsbound']:
                opt = self.optimizer
                #dprint(opt.final_lr_rate)
                #dprint(opt.t)
                #if opt.t > 0:
                if cdata.log.train_step > 0:
                    #final_lr = opt.final_lr_rate * opt.alpha
                    final_lr = self.optimizer.param_groups[0]['final_lr']
                    gamma = self.optimizer.param_groups[0]['gamma']
                    step = cdata.log.train_step
                    lower = final_lr * (1.0 - 1.0 / (gamma * step + 1))
                    upper = final_lr * (1.0 + 1.0 / (gamma * step))
                    dprint(final_lr)
                    dprint(lower)
                    dprint(upper)
            if cdata.train.optimizer in ['adam', 'amsgrad', 'adabound', 'amsbound']:
                opt = self.optimizer
                #if opt.t > 0:
                if cdata.log.train_step > 0:
                    #if cdata.train.warmup_steps > 0:
                    #    #dprint(opt.alpha)
                    #    dprint(opt.lr)
                    if cdata.train.weight_decay_rate > 0:
                        weight_decay_warmup_steps = cdata.train.weight_decay_warmup_steps
                        if weight_decay_warmup_steps is not None and weight_decay_warmup_steps > 0:
                            # torch の最適化器に weight_decay_rate 属性は
                            # 無い (Chainer 時代の名前)。実際の値は
                            # param_groups に入っている
                            dprint(opt.param_groups[0]['weight_decay'])
                    #if default.train.warmup_steps > 0:
                    #    if opt.weight_decay_rate > 0:
                    #        dprint(opt.weight_decay_rate)
        for progress.batch_i, batch in enumerate(train_iterator):
            if train:
                batch = reduce_batch_size(batch, cdata.train.batch_size, cdata.train.batch_type)
                #dprint(len(batch))
            if len(batch) == 0:
                continue
            if train:
                self.update_train_step(increment=True)
            try:
                self.model.zero_grad()
                self.optimizer.zero_grad()
                if hasattr(self.model, 'reset_state'):
                    self.model.reset_state()
                #batch_report = self.feed_one_batch(batch)
                batch_report = self.feed_one_batch(batch, fallback=fallback, df=df)
            except Exception as e:
                # common error process
                self.progress['accum_errors'] += 1
                self.progress['total_errors'] += 1
                if train:
                    try:
                        cdata.log.train_step -= 1
                        self.train_df.loc[batch.index, 'criterion'] = -1
                    except Exception as e2:
                        logger.exception(e2)
                if isinstance(e, RuntimeError):
                    # torch.cuda.OutOfMemoryError (RuntimeError の派生) で
                    # 型として判定できる。文字列一致だけだと、メッセージの
                    # 文言が変わった時点で黙ってバッチ縮小が働かなくなり、
                    # 学習がそのまま落ちる
                    mem_error = isinstance(e, torch.cuda.OutOfMemoryError)
                    if str(e)[:18] == "CUDA out of memory":
                        mem_error = True
                    if str(e).find('CUDNN_STATUS_EXECUTION_FAILED') >= 0:
                        mem_error = True
                    #if str(e)[:18] == "CUDA out of memory":
                    if mem_error:
                        logger.debug(repr(e))
                        self.model.zero_grad()
                        self.optimizer.zero_grad()
                        gc.collect()
                        try:
                            torch.cuda.empty_cache()
                            # reset_max_memory_allocated と
                            # reset_max_memory_cached はいずれも
                            # reset_peak_memory_stats を呼ぶ別名で、後者は
                            # FutureWarning を出す。直接呼べば 1 回で済む
                            torch.cuda.reset_peak_memory_stats()
                        except Exception as e2:
                            logger.debug(repr(e2))
                        if cdata.train.batch_type == 'samples':
                            new_batch_size = cdata.train.batch_size - 1
                        else:
                            new_batch_size = cdata.train.batch_size - int(cdata.model.max_length / 2)
                        cdata.train.batch_size = max(cdata.train.min_batch_size, new_batch_size)
                        logger.debug(f"new batch size: {cdata.train.batch_size}")
                        continue
                #logger.exception(e)
                raise e
                #continue
            progress['total_report'] += batch_report
            progress['accum_report'] += batch_report
            progress['accum_batches'] += 1
            progress['total_batches'] += 1
            if train:
                try:
                    self.train_df.loc[batch.index, 'feed_count'] += 1
                    self.train_df.loc[batch.index, 'last_step'] = cdata.log.train_step
                    self.train_df.loc[batch.index, 'last_epoch'] = epoch
                except Exception as e:
                    #logging.debug(e)
                    logger.exception(e)
            if show_report:
                if cdata.log.interval > 0 and time.time() - progress['last_time'] >= cdata.log.interval:
                    self.show_progress_report()
            if timeout is not None and timeout > 0:
                if time.time() - start_time > timeout:
                    break
        if progress['accum_batches'] > 0 or progress['accum_errors'] > 0:
            self.show_progress_report()
        if timeout is not None and timeout > 0:
            if time.time() - start_time > timeout:
                logger.info("feeding batches is timed out, the process is truncated")
        if progress['total_batches'] >= 1:
            mean_report = progress['total_report'] / progress['total_batches']
        else:
            mean_report = pd.Series()
        if show_report:
            report_fields = []
            for key in ['loss', 'acc', 'ppl', 'ponder_cost']:
                if key in mean_report:
                    value = mean_report.get(key)
                    report_fields.append("{key}: {value}".format(**locals()))
            report_fields.append("errors: {progress[total_errors]}".format(**locals()))
            if train:
                msg = "train " + str.join(', ', report_fields)
            else:
                msg = "dev " + str.join(', ', report_fields)
            logger.info(msg)
        train_report = mean_report
        train_report['error_count'] = progress['total_errors']
        return mean_report

    def test_model(self):
        return False

    def train_epoch(self, args):
        cdata = self.config.data
        status = cdata.log
        #dprint(self.get_num_params())
        train_df = self.load_train_data(self.train_data_path)
        #dprint(args.filter_noisy_samples)
        if args.filter_noisy_samples:
            #self.last_worst_criterion = trained.criterion.max()
            #self.last_worst_criterion = self.train_df.criterion.max()
            self.last_worst_criterion = self.config.get('log.min_worst_train_ppl')
        len(self.train_data)
        if len(train_df) == 0:
            logger.info(f"train dataset: (following lines)\n{train_df!r}")
            logger.info("nothing to train, finishing the training")
            return False
        #dprint(self.config.to_json(indent=2))
        dprint(self.config.to_json(indent=2, purge=True))
        str_log = str(self.config.to_json('log', indent=2, purge=True))
        logger.info(f"training log (following lines):\n{str_log}")
        if 'max_steps' in self.config.data.model:
            if self.config.data.model.max_steps is not None:
                if self.config.data.train.schedule_num_steps:
                    self.set_max_steps()
                    dprint(self.model.max_steps)
        # update elapsed times
        elapsed = cdata.log.elapsed
        cdata.log.elapsed_hours = elapsed / 3600
        cdata.log.elapsed_days  = elapsed / 3600 / 24
        hostname = os.uname().nodename
        logger.debug("hostname: " + hostname)
        logger.debug("process id: " + str(os.getpid()))
        logger.debug(f"using devices: {args.gpu}")
        #trainer.optimizer.new_epoch()
        #self.optimizer.new_epoch()
        logger.info(f"Epoch: {status.epoch}")
        logger.info(f"Batch Size: {cdata.train.batch_size}")
        logger.info("building curriculum batches...")
        self.build_train_batches()
        train_batches = self.train_batches
        challenge_batches = self.challenge_batches
        if cdata.train.max_batches > 0 and len(train_batches) > cdata.train.max_batches:
            logger.info(f"having {len(train_batches)} batches, limiting up to {cdata.train.max_batches} batches")
            train_batches = train_batches[:cdata.train.max_batches]
        start = time.time()
        try:
            if challenge_batches is not None:
                logger.info(f"reviewing {len(challenge_batches)} difficult batches...")
                #self.feed_batches(challenge_batches, report=True)
                #self.feed_batches(challenge_batches, report=True, timeout=cdata.train.timeout)
                #self.feed_batches(challenge_batches, train=True, show_report=True, timeout=cdata.train.timeout)
                self.feed_batches(challenge_batches, train=True, show_report=True, timeout=cdata.train.timeout, df=self.train_df)
            #logger.info("training curriculum batches...")
            logger.info("training...")
            #train_report = self.train_report = self.feed_batches(curriculum_batches, report=True, timeout=cdata.train.timeout)
            #train_report = self.train_report = self.feed_batches(train_batches, train=True, show_report=True, timeout=cdata.train.timeout)
            train_report = self.train_report = self.feed_batches(train_batches, train=True, show_report=True, timeout=cdata.train.timeout, df=self.train_df)
        except Exception as e:
            logger.exception(e)
            return False
        if True:
            # batch size reduction is automatically done in training batches
            if train_report.error_count == 0:
                if args.auto_batch_size:
                    cdata.train.batch_size = int(min(cdata.train.batch_size * 1.01 + 1, len(self.train_df)))
        elapsed = cdata.log.elapsed + (time.time() - start)
        cdata.log.elapsed = elapsed
        #logging.debug(model.optimizer.lr)
        if (self.last_loss is not None) and (train_report.get('loss') > self.last_loss):
            #logging.log("Changing optimizer to SGD")
            #model.set_optimizer('SGD')
            #if curriculum_data.curriculum_criterion.min() > 0:
            #    train_factor = model.train_factor
            #    logging.log("Changing learning rate from {} to {}".format(train_factor, train_factor / 2.0))
            #    config['train']['train_factor'] = str(train_factor / 2.0)
            #    model.train_factor = train_factor / 2.0
            pass
        self.last_loss = train_report.get('loss')
        #if dev_src_sents:
        #if args.dev_files:

        if train_report.get('error_count') > 0:
            # fall backing
            try:
                #BATCH_SIZE = 32
                #log("falling back with batch size {}...".format(BATCH_SIZE))
                if cdata.train.batch_type == 'samples':
                    min_batch_size = cdata.train.min_batch_size
                else:
                    #min_batch_size = cdata.train.min_batch_size * cdata.model.max_length
                    #min_batch_size = cdata.train.min_batch_size * cdata.model.max_length / 2
                    min_batch_size = cdata.train.min_batch_size * int(max(1, cdata.model.max_length / 2))
                logger.info(f"falling back with batch size {min_batch_size}...")
                curriculum_data = train_df.loc[pd.concat(train_batches).index]
                fallback_data = curriculum_data[curriculum_data.criterion <= 0]
                #fallback_data = fallback_data.sort_values(['len_x', 'len_t'])
                fallback_data = fallback_data.sort_values(self.len_fields)
                #fallback_batches = list( chainer.iterators.SerialIterator(fallback_data, BATCH_SIZE, repeat=False, shuffle=False) )
                #fallback_batches = list( chainer.iterators.SerialIterator(fallback_data, cdata.train.min_batch_size, repeat=False, shuffle=False) )
                #fallback_batches = list(build_batches(fallback_data, cdata.train.min_batch_size, cdata.train.batch_type))
                fallback_batches = list(build_batches(fallback_data, min_batch_size, cdata.train.batch_type))
                if cdata.train.max_batches > 0:
                    fallback_batches = fallback_batches[:cdata.train.max_batches]
                start = time.time()
                #fallback_loss, fallback_accuracy, fallback_ppl, fallback_cost, error_count = trainer.train(fallback_batches, report=True)
                #fallback_report = trainer.train(fallback_batches, report=True)
                #fallback_report = self.feed_batches(fallback_batches, report=True, timeout=cdata.train.timeout)
                #fallback_report = self.feed_batches(fallback_batches, report=True, timeout=cdata.train.timeout, fallback=True)
                #fallback_report = self.feed_batches(fallback_batches, train=True, show_report=True, timeout=cdata.train.timeout, fallback=True)
                fallback_report = self.feed_batches(fallback_batches, train=True, show_report=True, timeout=cdata.train.timeout, fallback=True, df=self.train_df)
                if fallback_report.get('error_count') > 0:
                    curriculum_data = train_df.loc[pd.concat(train_batches).index]
                    long_data = curriculum_data[curriculum_data.criterion <= 0]
                    #long_data = long_data.sort_values(['len_x', 'len_t'])
                    #long_data = long_data.sort_values(self.len_fields)
                    long_data = long_data.sort_values(['len'])
                    long_data = long_data[::-1][:len(fallback_batches)]
                    #long_repr = repr(long_data[['len_x', 'len_t', 'cost', 'feed_count', 'last_epoch', 'last_step', 'criterion', 'priority']])
                    long_repr = repr(long_data[[*self.len_fields, 'cost', 'feed_count', 'last_epoch', 'last_step', 'criterion', 'priority']])
                    logger.debug("long training samples:\n" + long_repr)
                    logger.info(f"dropping {len(long_data)} training examples with too long sentences")
                    train_df.drop(long_data.index, inplace=True)
                elapsed += (time.time() - start)
                cdata.log.elapsed = elapsed
            except Exception as e:
                #logging.warn(traceback.format_exc())
                logger.exception(e)
        if args.save_models:
            cdata.log.timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        if args.eval_train:
            #eval_train_report = self.evaluate('train', self.train_df, args)
            #eval_train_report = self.evaluate('train', self.train_df, args, report=train_report)
            train_report = self.evaluate('train', self.train_df, args, feed_batches=False, report=train_report)
            #train_report = eval_train_report.combine_first(train_report)
        if args.test_file:
            test_report = self.evaluate('test', self.test_df, args, feed_batches=None)
        else:
            test_report = None
        if args.dev_file:
            dev_report = self.evaluate('dev', self.dev_df, args, feed_batches=True)
        else:
            dev_report = None
        if True:
            self.save_scores(args.workdir, train_report, dev_report, test_report)
            #outdir = os.path.join(args.workdir, 'record.latest')
            outdir = os.path.join(args.workdir, 'record.tmp')
            score_path = os.path.join(outdir, 'scores.json')
            self.plot(score_path, 'elapsed', ['train_perplexity', 'dev_perplexity'], os.path.join(outdir, 'plot.time-ppl.pdf'))
            self.plot(score_path, 'elapsed', ['train_accuracy', 'dev_accuracy'], os.path.join(outdir, 'plot.time-acc.pdf'))
            self.plot(score_path, 'elapsed', ['train_loss', 'dev_loss'], os.path.join(outdir, 'plot.time-loss.pdf'))
            self.plot(score_path, 'elapsed', ['train_mrr', 'dev_mrr', 'test_mrr'], os.path.join(outdir, 'plot.time-mrr.pdf'))
            self.plot(score_path, 'elapsed', ['dev_bleu', 'test_bleu'], os.path.join(outdir, 'plot.time-bleu.pdf'))
            self.plot(score_path, 'elapsed', ['dev_f1', 'test_f1'], os.path.join(outdir, 'plot.time-f1.pdf'))
            self.plot(score_path, 'elapsed', ['dev_boundary_f1', 'test_boundary_f1'], os.path.join(outdir, 'plot.time-bf1.pdf'))
            self.plot(score_path, 'fed_samples', ['train_perplexity', 'dev_perplexity'], os.path.join(outdir, 'plot.samp-ppl.pdf'))
            self.plot(score_path, 'fed_samples', ['train_accuracy', 'dev_accuracy'], os.path.join(outdir, 'plot.samp-acc.pdf'))
            self.plot(score_path, 'fed_samples', ['train_loss', 'dev_loss'], os.path.join(outdir, 'plot.samp-loss.pdf'))
            self.plot(score_path, 'fed_samples', ['train_mrr', 'dev_mrr', 'test_mrr'], os.path.join(outdir, 'plot.samp-mrr.pdf'))
        if args.filter_noisy_samples:
            self.filter_noisy_samples()
        if args.save_models:
            self.save_latest_status(link_only=False)
            self.update_best_scores(args.workdir, train_report, dev_report, test_report)
            self.save_latest_status(link_only=True)
        #self.save_dataset()
        try:
            self.test_model()
        except Exception as e:
            logger.exception(e)
        return True

    def try_loading(self, workdir, list_resume):
        """Load the first of the given records that can be loaded

        与えられた記録のうち、最初に読み込めたものを読み込む。

        `--resume` takes a list precisely so that a later entry can stand in
        for one that does not load. The loop used to re-raise on the first
        failure, and the line recording it was unreachable behind that
        raise, so only the first entry was ever tried. A failure of every
        entry still raises, since carrying on without a loaded model would
        silently start the run from scratch.

        `--resume` が一覧を取るのは、読み込めなかった記録を後続で代替する
        ためである。ループは最初の失敗でそのまま送出しており、それを記録
        する行は raise の後ろで到達不能だった。つまり最初の 1 件しか
        試されていなかった。全ての候補が失敗した場合は送出する。
        読み込めないまま先へ進むと、黙って最初から学習し直すことになる
        ためである。
        """
        list_failed = []
        last_error = None
        for resume_entry in list_resume:
            try:
                dprint(resume_entry)
                #self.load_status(workdir, record=resume_entry, load_optimizer=True)
                self.load_status(workdir, record=resume_entry, load_optimizer=True, reuse_dataset=True)
                logger.info(f"successfully loaded: {resume_entry}")
                return self
            except Exception as e:
                #logger.debug(repr(e))
                logger.exception(e)
                logger.info(f"failed to load: {resume_entry}")
                list_failed.append(resume_entry)
                last_error = e
        if last_error is not None:
            logger.error(f"none of the given records could be loaded: {list_failed}")
            raise last_error
        return self

    def save_config(self, workdir, record=None, log=True):
        cdata = self.config.data
        recdir = workdir
        if record:
            recdir = os.path.join(workdir, 'record.'+record)
        safeMakeDirs(recdir)
        elapsed = cdata.log.elapsed
        cdata.log.elapsed_hours = elapsed / 3600
        cdata.log.elapsed_days  = elapsed / 3600 / 24
        config_name = 'config.json'
        config_path = os.path.join(recdir, config_name)
        safe_remove(config_path, log=False)
        with open(config_path, 'w', encoding='utf-8') as fobj:
            if log:
                logger.info(f"saving configuration into '{config_path}'")
            #fobj.write(self.config.to_json(indent=2))
            fobj.write(self.config.to_json(indent=2, purge=True),)
        return

    def save_labels(self, path):
        with open(path, 'w', encoding='utf-8') as fobj:
            for label in self.labels:
                fobj.write(self.vocab.decode(label))
                fobj.write("\n")

    def save_scores(self, workdir, train_report, dev_report=None, test_report=None):
        def assign_score(report, scores, report_field, score_field):
            if report_field in report:
                scores[score_field] = report[report_field]
        cdata = self.config.data
        # score logging
        scores = Config()
        # X-axis
        scores.data.epoch = cdata.log.epoch
        scores.data.train_step = cdata.log.train_step
        scores.data.elapsed = cdata.log.elapsed
        scores.data.fed_samples = cdata.log.fed_samples
        # Y-axis
        assign_score(train_report, scores, 'loss', 'train_loss')
        assign_score(train_report, scores, 'acc', 'train_accuracy')
        assign_score(train_report, scores, 'ppl', 'train_perplexity')
        assign_score(train_report, scores, 'bleu', 'train_bleu')
        assign_score(train_report, scores, 'map', 'train_map')
        assign_score(train_report, scores, 'mr', 'train_mr')
        assign_score(train_report, scores, 'mrr', 'train_mrr')
        assign_score(train_report, scores, 'map', 'train_map')
        if dev_report is not None:
            assign_score(dev_report, scores, 'acc', 'dev_accuracy')
            assign_score(dev_report, scores, 'loss', 'dev_loss')
            assign_score(dev_report, scores, 'ppl', 'dev_perplexity')
            assign_score(dev_report, scores, 'bleu', 'dev_bleu')
            assign_score(dev_report, scores, 'f1', 'dev_f1')
            assign_score(dev_report, scores, 'boundary_f1', 'dev_boundary_f1')
            assign_score(dev_report, scores, 'mr', 'dev_mr')
            assign_score(dev_report, scores, 'mrr', 'dev_mrr')
            assign_score(dev_report, scores, 'map', 'dev_map')
        if test_report is not None:
            assign_score(test_report, scores, 'bleu', 'test_bleu')
            assign_score(test_report, scores, 'f1', 'test_f1')
            assign_score(test_report, scores, 'boundary_f1', 'test_boundary_f1')
            assign_score(test_report, scores, 'mr', 'test_mr')
            assign_score(test_report, scores, 'mrr', 'test_mrr')
            assign_score(test_report, scores, 'map', 'test_map')
        #with open(os.path.join(workdir, 'scores.json'), 'a') as fobj:
        #scores_path = os.path.join(workdir, 'record.latest', 'scores.json')
        scores_path = os.path.join(workdir, 'record.tmp', 'scores.json')
        #safe_remove(scores_path)
        #with open(os.path.join(workdir, 'record.latest', 'scores.json'), 'a') as fobj:
        with open(scores_path, 'a', encoding='utf-8') as fobj:
            fobj.write(scores.to_json())
            fobj.write("\n")

    def save_status(self, workdir, record=None):
        recdir = workdir
        if record:
            #recdir = os.path.join(workdir, record)
            recdir = os.path.join(workdir, 'record.'+record)
        safeMakeDirs(recdir)
        model_name = 'model.pt'
        opt_name = 'optimizer.pt'
        model_path = os.path.join(recdir, model_name)
        opt_path   = os.path.join(recdir, opt_name)
        safe_remove(model_path, log=False)
        logger.info(f"saving model into '{model_path}' ...")
        #torch.save(self.model, 'test.pt')
        state_dict = OrderedDict()
        #dprint(self.config.to_dict(ordered=True))
        state_dict['config'] = self.config.to_dict(ordered=True)
        #state_dict['vocab'] = self.model.vocab.get_state()
        state_dict['idmaps'] = self.model.idmaps.get_state()
        self.model.zero_grad()
        if hasattr(self.model, 'reset_state'):
            self.model.reset_state()
        self.model.eval()
        #state_dict['model'] = self.model.to('cpu')
        state_dict['model'] = self.model.state_dict()
        #state_dict['model'] = self.model.to('cpu').state_dict()
        torch.save(state_dict, model_path)
        #dprint(state_dict['model'])
        safe_remove(opt_path, log=False)
        logger.info(f"saving optimizer params into '{opt_path}' ...")
        torch.save(self.optimizer.state_dict(), opt_path)
        self.save_config(workdir, record=record)
        return

    def save_latest_status(self, link_only=False):
        if not link_only:
            self.save_status(self.args.workdir, 'tmp')
            self.save_dataset()
        self.link_status(self.args.workdir, 'latest', 'prev', log=False)
        self.link_status(self.args.workdir, 'tmp', 'latest', log=True)

    def save_best_status(self, path, report, report_name, config_name, ascend=True, best='best', save_model=True):
        score = report.get(report_name)
        #last_best_score = self.config.get(config_name)
        record_name = best + '_' + config_name.replace(' ', '_')
        config_field = 'log.' + record_name
        #dprint(config_field)
        last_best_score = self.config.get(config_field)
        try:
            score = float(score)
        except Exception:
            return False
        if math.isfinite(score):
            update = False
            if last_best_score is None:
                update = True
            elif ascend and score > last_best_score:
                update = True
            elif not ascend and score < last_best_score:
                update = True
            if update:
                if last_best_score is None:
                    #log("new {} ({})".format(config_name, score))
                    logger.info(f"new {config_name}: {score}")
                else:
                    if ascend:
                        logger.info(f"new {config_name} ({score}) > last best ({last_best_score})")
                    else:
                        logger.info(f"new {config_name} ({score}) < last best ({last_best_score})")
                self.config[config_field] = score
                self.save_config(path, record='tmp', log=False)
                if save_model:
                    self.link_status(path, 'tmp', record_name, log=False)
                return True
        return False

    def save_last_score(self, path, report, report_name, config_name, last='last'):
        score = report.get(report_name)
        record_name = last + '_' + config_name.replace(' ', '_')
        config_field = 'log.' + record_name
        try:
            score = float(score)
        except Exception:
            return False
        if math.isfinite(score):
            self.config[config_field] = score
        return True

    @classmethod
    def set_config(cls, config, key, value=None):
        if value is None:
            if key in config:
                return config[key]
            else:
                raise KeyError(key)
        else:
            config[key] = value
            return value

    def save_dataset(self):
        cdata = self.config.data
        df = self.train_df
        df.priority = df.criterion * df.cost * df.feed_count * df.last_step / (cdata.log.train_step + 1)
        safe_remove(self.temp_train_data_path)
        logger.info(f"saving dataset into: {self.temp_train_data_path}")
        self.train_data.save(self.temp_train_data_path, cdata.train.max_samples_per_epoch, None, 1)
        with open(self.temp_train_data_path, 'a', encoding='utf-8') as fobj_data:
            df.to_csv(fobj_data, sep='\t', header=False)
        #safe_rename(self.train_data_path, self.prev_train_data_path)
        safe_rename(self.temp_train_data_path, self.train_data_path)
        return True

    def _fix_max_steps(self, min_steps=1):
        cdata = self.config.data
        max_steps = getattr(self.model, 'max_steps', None)
        if max_steps is not None:
            max_steps = max(max_steps, min_steps)
            if self.config.get('model.universal'):
                max_steps = min(max_steps, cdata.model.max_steps)
            elif self.config.get('model.num_layers'):
                max_steps = min(max_steps, cdata.model.num_layers)
            self.model.max_steps = max_steps
        return max_steps
    def set_max_steps(self, max_steps=None, min_steps=1):
        if 'max_steps' in self.config.data.model:
            if self.config.data.model.max_steps is not None:
                self.model.max_steps = max_steps
                self._fix_max_steps(min_steps)
                return self.model.max_steps
        return None

    def setup_model(self, args):
        #self.args = args
        cdata = self.config.data
        #dprint(self.config.to_json(indent=2))

        ### random seed initialization
        self.init_random(args)

        ### buffering if necessary
        if args.train_file:
            try:
                # パスでない (PastedFile 等) 場合は TypeError、開けない場合は
                # OSError になる。いずれもシーク不可として扱う。
                # 元実装はハンドルを閉じていなかった
                with open(args.train_file) as probe:
                    probe.seek(0)
                seekable = True
            except (OSError, TypeError, ValueError):
                seekable = False
            dprint(seekable)
            if not seekable:
                temp = load_to_temp(args.train_file)
                dprint(temp.name)
                args.train_file = temp.name
            #elif 'labels' in args:
            #    temp = load_to_temp(args.train_file)
            #    dprint(temp.name)
            #    args.train_file = temp.name

        ### vocabulary settings
        sp_model_path = os.path.join(args.workdir, 'sp.model')
        #self.vocab = None
        #if 'extra_symbols' in self.specific:
        #    extra_symbols = self.specific.to_dict(key='extra_symbols', ordered=True, upstream=True)
        #    dprint(extra_symbols)
        #else:
        #    extra_symbols = {}
        extra_symbols = self.get_extra_symbols()
        if self.model:
            args.sentencepiece = sp_model_path
            self.vocab = self.model.vocab
        if args.sentencepiece:
            # using existing sp model
            if args.sentencepiece != sp_model_path:
                if not os.path.exists(args.sentencepiece):
                    logger.warn(f"[ERROR] {args.sentencepiece} does not exist!")
                    sys.exit(1)
                dprint(sp_model_path)
                #safe_link(args.sentencepiece, sp_model_path)
                safe_copy(args.sentencepiece, sp_model_path)
        if self.model is None:
            # training sentencepiece model
            FieldMap.train(args.workdir, self.main_fields, args.train_file, cdata.model.vocab_size, extra_symbols)
        if self.idmaps is None:
            self.idmaps = FieldMap().load(args.workdir, self.main_fields)
            self.idmaps.set_symbols(self.get_extra_symbols())
        cdata.model.vocab_size = len(self.idmaps['seq'])
        dprint(self.config.to_json(indent=2, purge=True),)

        ### dataset settings
        #self.train_data_path = os.path.join(args.workdir, 'train_data.tsv')
        #self.train_data_path = os.path.join(args.workdir, 'record.latest', 'train_data.tsv')
        self.train_data_path = os.path.join(args.workdir, 'record.tmp', 'train_data.tsv')
        #self.train_data_path = os.path.join(args.workdir, 'record.latest', 'train_data.tsv')
        #self.temp_train_data_path = os.path.join(args.workdir, 'temp_train_data.tsv')
        #self.temp_train_data_path = os.path.join(args.workdir, 'record.latest', 'temp_train_data.tsv')
        #self.temp_train_data_path = os.path.join(args.workdir, 'record.tmp', 'temp_train_data.tsv')
        self.temp_train_data_path = os.path.join(args.workdir, 'record.tmp', 'temp_train_data.tsv')
        dprint(self.main_fields)
        dprint(self.len_fields)
        if 'labels' in args:
            self.labels_path = os.path.join(args.workdir, 'labels.txt')
        #if not os.path.exists(self.main_train_data_path):
        #if not os.path.exists(self.train_data_path):
        if not args.resume or not os.path.exists(self.train_data_path):
            #if comm_main:
            if True:
                #if 'labels' in args:
                #    if args.labels is not None:
                #        with open(args.labels, 'r') as fobj_in:
                #            with open(args.train_file, 'a') as fobj_out:
                #                for line in fobj_in:
                #                    line = line.strip()
                #                    fobj_out.write("{}\t{}\n".format(line,line))
                #safe_remove(self.main_train_data_path)
                safe_remove(self.train_data_path)
                #logger.info("formatting train data into: {}".format(self.main_train_data_path))
                #logger.info("formatting train data into: {}".format(self.worker_temp_path))
                logger.info(f"formatting train data into: {self.temp_train_data_path}")
                #build_train_data(self.main_train_data_path, args.train_file, self.vocab)
                #build_train_data(self.main_fields, self.worker_temp_path, args.train_file, self.vocab)
                #build_train_data(self.main_fields, self.worker_temp_path, args.train_file, self.vocab, max_length=cdata.model.max_length)
                #build_train_data(self.main_fields, self.worker_temp_path, args.train_file, max_length=cdata.model.max_length)
                build_train_data(self.main_fields, self.temp_train_data_path, args.train_file, max_length=cdata.model.max_length)
                #safe_link(self.worker_temp_path, self.main_train_data_path)
                #safe_rename(self.worker_temp_path, self.main_train_data_path)
                safe_rename(self.temp_train_data_path, self.train_data_path)
                if 'labels' in args:
                    self.load_labels(args.labels)
                    self.save_labels(self.labels_path)
        if args.dev_file:
            self.dev_df = self.load_eval_data(args.dev_file)
        else:
            self.dev_df = None
        if args.test_file:
            self.test_df = self.load_eval_data(args.test_file)
        else:
            self.test_df = None
        if 'labels' in args:
            self.load_labels(args.labels)
            cdata.model.num_classes = len(self.labels)

        ### model setting
        if self.model is None:
            #self.model = self.Model(self.config)
            params = self.config.to_dict(flat=True)
            dprint(params)
            #self.model = self.Model(self.vocab, **params)
            self.model = self.Model(self.idmaps, **params)
            dprint(self.model)
            #self.model.apply(init_weights)
            #for name, module in reversed(list(self.model.named_modules())):
            apply_init_weights(self.model)
            self.update_model_config()
            #try:
            #    dprint(self.Model.get_config(**params))
            #except Exception as e:
            #    #dprint(e)
            #    logger.exception(e)
            #dprint(cdata.model.num_params)
        else:
            dprint(self.model)

        if self.config.data.train.schedule_num_steps:
            self.set_max_steps()
        self.last_loss = None

    def setup_dataset(self):
        pass

    def setup_optimizer(self, renew=False):
        if renew or self.optimizer is None:
            self.optimizer = setup_optimizer(self.config, self.model)

    def setup_device(self, args):
        if args.float16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        if args.gpu[0] >= 0:
            self.device = torch.device(args.gpu[0])
        else:
            self.device = torch.device('cpu')
        if self.model is not None:
            logger.debug(f"moving model to device: {self.device}, dtype: {self.dtype}")
            self.model.to(self.device, self.dtype)
            dprint(self.optimizer)
            dprint(self.model.device)
            dprint(self.model.dtype)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

    @classmethod
    def get_default(cls, field, value):
        if value is not None:
            return value
        else:
            return cls.default[field]

    @classmethod
    def update_config(cls, config, args):
        global PRESET_CHOICES
        #dprint(config.base)
        #dprint(Config(config.base).to_json(indent=2, upstream=True),)
        #dprint(config)
        #dprint(config.to_json(indent=2, upstream=True))
        if config is None:
            config = Config(cls.default)
        default = config.base
        cdata = config.data
        ignore = []
        ignore.append('//')
        ignore.append('help')
        ignore.append('workdir')
        ignore.append('train_file')
        ignore.append('train_files')
        ignore.append('dev_file')
        ignore.append('dev_files')
        ignore.append('test_file')
        ignore.append('test_files')
        ignore.append('gpu')
        ignore.append('debug')
        ignore.append('device')
        ignore.append('filter_noisy_samples')
        ignore.append('eval_only')
        ignore.append('import_embed')
        ignore.append('save_models')
        #ignore.append('max_epochs')
        ignore.append('num_epochs')
        ignore.append('move_optimizer')
        params = vars(args)
        if args.float16:
            params['dtype'] = 'float16'
        else:
            params['dtype'] = 'float32'
        if args.batch_size is not None:
            if args.batch_size < 0:
                args.batch_size = max(cdata.train.batch_size+args.batch_size, cdata.train.min_batch_size)
        #params = config.to_dict(flat=True, upstream=True)
        params = {key: val for key, val in params.items() if key not in ignore and val is not None}
        #for key, val in config.to_dict(flat=True, upstream=True).items():
        for key, val in config.to_dict(flat=True).items():
            if key in ignore:
                continue
            if val is not None:
                if params.get(key) is None:
                    params[key] = val
        dprint(params)
        #params.setdefault('preset_choices', set(['ref', 'reference']))
        params = cls.Model.get_config(**params)
        dprint(params)
        if 'preset_choices' in params:
            PRESET_CHOICES = ['ref', 'reference', *sorted(params['preset_choices'])]
            #dprint(PRESET_CHOICES)
        #dprint(config.to_json(indent=2, upstream=True))
        config.get('model.//')
        config.get('train.//')
        config.get('log.//')
        optimizer = cls.set_config(config, 'train.optimizer', args.optimizer)
        dprint(optimizer)
        weight_decay_rate = cls.get_default('train.weight_decay_rate', args.weight_decay_rate)
        if weight_decay_rate <= 0:
            default.train.weight_decay_warmup_steps = 0
        if cls.get_default('model.universal', args.universal):
            default.model.num_layers = None
            default.train.schedule_num_steps = True
        for section in ['model', 'train', 'log']:
            #for key in config[section]:
            for key in default[section]:
                #dprint( (section, key, config[section][key]), )
                #dprint( (section, key, default[section][key]), )
                if key in params and params[key] is not None:
                    config[section][key] = params[key]
                elif default[section][key] is not None:
                    config[section][key] # acceess to appear in config file
                    #config[section][key] = default[section][key]
        #dprint(config.to_json(indent=2, upstream=True),)
        return config

    def update_model_config(self):
        if hasattr(self.model, 'config'):
            hparams = self.model.config
            config = self.config
            for section in ['model', 'train', 'log']:
                for key in self.default[section]:
                    if key in hparams and hparams[key] is not None:
                        config[section][key] = hparams[key]
        self.get_num_params()
        return self.config

    def update_to_reduce_ponder_cost(self, ponder_cost, penalty):
        if isinstance(ponder_cost, torch.Tensor):
            try:
                self.model.zero_grad()
                gc.collect()
                cost = ponder_cost * penalty
                cost.backward()
                self.optimizer.step()
            except Exception as e:
                logger.exception(e)

    def update_parameters(self, loss, report=None):
        if not torch.isfinite(loss):
            return
        cdata = self.config.data
        #with torch.autograd.detect_anomaly():
        #    loss.backward()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cdata.train.gradient_clipping)
        # 署名は report=None を許すが、ここだけ確認していなかった
        if report is not None:
            report['gnorm'] = gnorm
        if not math.isfinite(gnorm):
            #dprint(gnorm)
            logger.debug(f"detected NaN on backward, gnorm: {gnorm}")
            #report['clip'] = 1
            return
        if cdata.train.gradient_clipping:
            #torch.nn.utils.clip_grad_value_(self.model.parameters(), cdata.train.gradient_clipping)
            if cdata.train.gradient_clipping > 0:
                if report is not None:
                    report['clip'] = int(gnorm > cdata.train.gradient_clipping)
                #dprint(report)
        if self.model.device.type == 'cuda':
            if self.args.move_optimizer:
                #logger.debug("moving optimizer to video memory")
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
        self.optimizer.step()
        if self.model.device.type == 'cuda':
            if self.args.move_optimizer:
                #logger.debug("moving optimizer to system memory")
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cpu()

    def update_best_scores(self, workdir, train_report=None, dev_report=None, test_report=None):
        if train_report is not None:
            if 'ppl' in train_report:
                if len(self.train_df[self.train_df.criterion > 0]) > 0:
                    worst_train_ppl = self.train_df.criterion.max()
                    train_report['worst_ppl'] = worst_train_ppl
            self.save_best_status(workdir, train_report, 'loss', 'train loss', ascend=False)
            self.save_best_status(workdir, train_report, 'ppl',  'train ppl',  ascend=False)
            self.save_best_status(workdir, train_report, 'worst_ppl', 'worst train ppl', ascend=False, best='min')
            self.save_best_status(workdir, train_report, 'ntppl',  'train non trivial ppl',  ascend=False)
            self.save_best_status(workdir, train_report, 'acc',  'train acc',  ascend=True)
            self.save_best_status(workdir, train_report, 'acc_seq',  'train acc seq',  ascend=True, save_model=False)
            self.save_best_status(workdir, train_report, 'ntacc',  'train non trivial acc',  ascend=True)
            self.save_best_status(workdir, train_report, 'map',  'train map',  ascend=True, save_model=False)
            self.save_best_status(workdir, train_report, 'mr',  'train mr',  ascend=False, save_model=False)
            self.save_best_status(workdir, train_report, 'mrr',  'train mrr',  ascend=True, save_model=False)
        if test_report is not None:
            self.save_last_score(workdir, test_report, 'acc',  'test acc')
            self.save_last_score(workdir, test_report, 'acc_seq',  'test acc seq')
            self.save_last_score(workdir, test_report, 'bleu',  'test bleu')
            self.save_last_score(workdir, test_report, 'mr',    'test mr')
            self.save_last_score(workdir, test_report, 'mrr',   'test mrr')
            self.save_last_score(workdir, test_report, 'map',   'test map')
            for k in LIST_K_FOR_RECALL:
                field = f'r_at_{k}'
                display = f'test {field}'
                self.save_last_score(workdir, test_report, field, display)
            self.save_last_score(workdir, test_report, 'precision', 'test precision')
            self.save_last_score(workdir, test_report, 'recall', 'test recall')
            self.save_last_score(workdir, test_report, 'f1', 'test f1')
            self.save_last_score(workdir, test_report, 'boundary_precision', 'test boundary precision')
            self.save_last_score(workdir, test_report, 'boundary_recall', 'test boundary recall')
            self.save_last_score(workdir, test_report, 'boundary_f1', 'test boundary f1')
            #self.save_config(workdir, record='tmp', log=False)
        if dev_report is not None:
            if 'ppl' in dev_report:
                if len(self.dev_df[self.dev_df.criterion > 0]) > 0:
                    worst_dev_ppl = self.dev_df.criterion.max()
                    dev_report['worst_ppl'] = worst_dev_ppl
            self.save_best_status(workdir, dev_report, 'loss',  'dev loss', ascend=False)
            self.save_best_status(workdir, dev_report, 'ppl',   'dev ppl',  ascend=False)
            self.save_best_status(workdir, dev_report, 'worst_ppl', 'worst dev ppl', ascend=False, best='min')
            self.save_best_status(workdir, dev_report, 'ntppl', 'dev non trivial ppl', ascend=False)
            self.save_best_status(workdir, dev_report, 'acc',   'dev acc',  ascend=True)
            self.save_best_status(workdir, dev_report, 'acc_seq',   'dev acc seq', ascend=True)
            self.save_best_status(workdir, dev_report, 'ntacc',   'dev non trivial acc', ascend=True)
            self.save_best_status(workdir, dev_report, 'bleu',  'dev bleu', ascend=True)
            self.save_best_status(workdir, dev_report, 'mr',    'dev mr', ascend=False, save_model=False)
            self.save_best_status(workdir, dev_report, 'mrr',   'dev mrr', ascend=True)
            self.save_best_status(workdir, dev_report, 'map',   'dev map', ascend=True)
            for k in LIST_K_FOR_RECALL:
                field = f'r_at_{k}'
                display = f'dev {field}'
                self.save_best_status(workdir, dev_report, field, display, ascend=True, save_model=False)
            self.save_best_status(workdir, dev_report, 'precision', 'dev precision', ascend=True)
            self.save_best_status(workdir, dev_report, 'recall', 'dev recall', ascend=True)
            self.save_best_status(workdir, dev_report, 'f1', 'dev f1', ascend=True)
            self.save_best_status(workdir, dev_report, 'boundary_precision', 'dev boundary precision', ascend=True)
            self.save_best_status(workdir, dev_report, 'boundary_recall', 'dev bounary recall', ascend=True)
            self.save_best_status(workdir, dev_report, 'boundary_f1', 'dev boundary f1', ascend=True)
        #self.link_status(workdir, 'tmp', 'latest')

    def update_train_step(self, increment=True):
        if self.model.training:
            cdata = self.config.data
            status = cdata.log
            if increment:
                train_step = status.train_step = status.train_step + 1
            else:
                train_step = status.train_step
            warmup_steps = cdata.train.warmup_steps
            optimizer = cdata.train.optimizer
            if warmup_steps is None or warmup_steps <= 0:
                for param_group in self.optimizer.param_groups:
                    if optimizer in ['adam', 'amsgrad', 'adabound', 'amsbound', 'lamb']:
                        param_group['lr'] = cdata.train.adam_alpha
            else:
                embed_size = cdata.model.embed_size
                warmup_factor = cdata.train.warmup_factor
                if train_step > 0:
                    if optimizer in ['adam', 'amsgrad', 'adabound', 'amsbound']:
                        alpha = warmup_factor * (embed_size ** -0.5) * min(train_step ** -0.5, train_step * warmup_steps ** -1.5)
                        alpha = min(0.001, alpha)
                        self.optimizer.lr = alpha
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = alpha
                    elif optimizer in ['lamb']:
                        #alpha = train_factor * (embed_size ** -0.5) * min(train_step ** -0.5, train_step * warmup_steps ** -1.5) * 100
                        lr = warmup_factor * (embed_size ** -0.5) * min(train_step ** -0.5, train_step * warmup_steps ** -1.5)
                        lr = min(0.1, lr)
                        self.optimizer.lr = lr
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
                    elif cdata.train.optimizer == 'sgd':
                        #alpha = train_factor * (embed_size ** -0.5) * min(train_step ** -0.5, train_step * warmup_steps ** -1.5) * 100
                        lr = warmup_factor * (embed_size ** -0.5) * min(train_step ** -0.5, train_step * warmup_steps ** -1.5) * 100
                        lr = min(0.2, lr)
                        self.optimizer.lr = lr
                        #self.optimizer.param_groups[0]['lr'] = lr
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
            if cdata.train.weight_decay_rate > 0:
                weight_decay_warmup_steps = cdata.train.weight_decay_warmup_steps
                if weight_decay_warmup_steps is not None and weight_decay_warmup_steps > 0:
                    if cdata.train.optimizer in ['adam', 'amsgrad', 'adabound', 'amsbound']:
                        weight_decay_rate = cdata.train.weight_decay_rate * min(1.0, train_step / weight_decay_warmup_steps)
                        self.optimizer.weight_decay_rate = weight_decay_rate
                        #self.optimizer.param_groups[0]['weight_decay'] = weight_decay_rate
                        for param_group in self.optimizer.param_groups:
                            param_group['weight_decay'] = weight_decay_rate

    @classmethod
    def add_argument(cls, parser, default, *args, **kwargs):
        if default is None:
            kwargs['help'] = argparse.SUPPRESS
        else:
            kwargs['default'] = None
            kwargs['help'] = '{} (current: {})'.format(kwargs['help'], default)
        parser.add_argument(*args, **kwargs)

    @classmethod
    def add_argument_group(cls, parser, check, name, help):
        if check:
            return parser.add_argument_group(name, help)
        else:
            return parser.add_argument_group(name, argparse.SUPPRESS)

    @classmethod
    def create_parser(cls, model_name, default=None):
        if default is None:
            default = cls.default
        parser = {}
        parser['main'] = main= argparse.ArgumentParser(f"{model_name} Trainer", add_help=False)
        main.add_argument('--help', '-h', action='store_true', help=_('show this help message and exit'))
        main.add_argument('workdir', help='directory path to write the training dataset, trained models, evaluation status, etc', nargs='?')
        main.add_argument('train_files', metavar='train_file', help='path to the training data (tab separated values, or column-split files)', nargs='*')
        if PRESET_CHOICES:
            main.add_argument('--preset', '--pre', '-P', type=str, default=None, choices=PRESET_CHOICES, help='preset name of hyper-parameters (choices: %(choices)s)')
        else:
            main.add_argument('--preset', '--pre', '-P', type=str, default=None, help='preset name of hyper-parameters')

        parser['files'] = group = main.add_argument_group('files', 'arguments for extra files')
        group.add_argument('--dev-files', '--dev', metavar='dev_file', type=str, help='path to the validation data (tab separated values, or column-split files)', nargs='+')
        group.add_argument('--test-files', '--test', metavar='test_file', type=str, help='path to the evaluation data (tab separated values, or column-split files)', nargs='+')
        group.add_argument('--sentencepiece', '--tokenizer', '--spmodel', '--sp', type=str, default=None, help='path to sentencepiece tokenizer model (if not given, new model is automatically trained with {train_files})')

        parser['model'] = group = main.add_argument_group('model', 'hyper-parameters for the model')
        cls.add_argument(group, False, '--float16', '--fp16', '--half', type=strtobool, nargs='?', const=True, help='16bit floating point mode (experimental)')
        group.add_argument('--vocab-size', '--vocab', '-V', type=int, default=None, help=f'Vocabulary size (number of unique tokens) (default: {default.model.vocab_size})')
        cls.add_argument(group, default.model.activation, '--activation', '--act', '-A', type=str, choices=['gelu', 'mish', 'relu', 'swish'], help='activation function')
        group.add_argument('--embed-size', '--embed', '--es', '-E', type=int, default=None, help=f'Number of embedding nodes (default: {default.model.embed_size})')
        cls.add_argument(group, default.model.hidden_size, '--hidden-size', '--hidden', '--hs', '-H', type=int, help='Number of hidden layer nodes')
        cls.add_argument(group, default.model.inner_size, '--inner-size', '--inner', '--is', type=int, help='Number of inner nodes in feed-forward layer')
        cls.add_argument(group, default.model.recurrence, '--recurrence', '--rec', type=str, choices=['basic', 'act', 'act-prob', 'act-accum'], help='Auto-regression type for universal model')
        cls.add_argument(group, default.model.share_embedding, '--share-embedding', '-S', type=strtobool, nargs='?', const=True, help='Using single shared embedding weight for source/target input and target output')
        cls.add_argument(group, default.model.max_length, '--max-length', '--length', type=int, help='Maximum length (number of tokens) for training')
        cls.add_argument(group, default.model.embed_positions, '--embed-positions', '--embed-pos', '--emb-pos', '--ep', type=strtobool, nargs='?', const=True, help='Using learnable position embeddings')
        cls.add_argument(group, default.model.relative_attention, '--relative-attention', '--relative', '--rel', type=strtobool, nargs='?', const=True, help='Using relative position representations for self-attention')
        cls.add_argument(group, default.model.clip_distance, '--clip-distance', '--distance', '--dist', '-c', type=int, help='Threshold of relative distance for clipping in relative-positional embedding')
        #cls.add_argument(group, default.model.num_layers, '--num-layers', '--layers', '-L', type=int, help='Number of RNN/Transformer layers')
        #cls.add_argument(group, default.model.num_blocks, '--num-blocks', '--blocks', '--nb', type=int, help='Number of main component blocks')
        cls.add_argument(group, default.model.num_heads, '--num-heads', '--heads', '--head', type=int, help='Number of ensembles for multi-head attention mechanism')
        cls.add_argument(group, default.model.universal, '--universal', '-U', type=strtobool, nargs='?', const=True, help='Using universal transformer model')

        parser['logging'] = group = main.add_argument_group('logging', 'arguments for logging options')
        group.add_argument('--debug', '-D', action='store_true', help='Debug mode')
        group.add_argument('--logging', '--log', type=str, default=None, help='Path of file to log (default: %(default)s')

        parser['training'] = group = main.add_argument_group('training', 'arguments for training options')
        cls.add_argument(group, default.train.batch_size, '--batch-size', '--batch', '--bs', '-B', type=int, help='Size of mini-batch')
        group.add_argument('--auto-batch-size', '--auto-batch', '--ab', type=strtobool, default=True, help='Using auto-justify mode of mini-batch size (default: %(default)s)')
        group.add_argument('--batch-type', '--bt', type=str, default=None, choices=['samples', 'tokens'], help=f'Batch type (default: {default.train.batch_type})')
        group.add_argument('--curriculum', '-C', type=str, default=None, choices=['none', 'len', 'crit', 'crit-len'], help=f'First priority criterion for training (default: {default.train.curriculum})')
        group.add_argument('--dropout-ratio', '--dropout', type=float, default=None, help=f'Dropout Rate (default: {default.train.dropout_ratio})')
        group.add_argument('--num-epochs', '--epochs', '-ne', type=int, default=100, help='Number of epochs (default: %(default)s)')
        group.add_argument('--gpu', '-G', type=int, default=[-1], help='GPU IDs (negative value indicates CPU) (default: %(default)s)', nargs='+')
        group.add_argument('--import-embed', '--pre-trained-vectors', type=str, default=None, help='Path to pre-trained vectors to import for embedding initialization')
        cls.add_argument(group, default.train.fix_imported_vectors, '--fix-imported-vectors', '--freeze-imported-vectors', '--fix-vectors', '--fix', '--freeze', type=strtobool, nargs='?', help='Whether to fix (not train) imported token vectors')
        group.add_argument('--max-batches', '--batches', type=int, default=None, help=f'Maximum batches to train in one epoch (default: {default.train.max_batches})')
        group.add_argument('--max-samples-per-epoch', '--samples', type=int, default=None, help=f'Maximum samples to train in one epoch (default: {default.train.max_samples_per_epoch})')
        group.add_argument('--min-batch-size', '--min-batch', '--mb', type=int, default=None, help=f'Minimum batch size for fallbacking (default: {default.train.min_batch_size})')
        #group.add_argument('--process_size', '--proc', '-P', type=int, default=-1, help='Maximum training samples taken for this process (save extra data to storage')
        group.add_argument('--warmup-steps', '--warmup', '--ws', '-W', type=int, default=None, help=f'Number of warming up steps (default: {default.train.warmup_steps})')
        #group.add_argument('--start_steps', '--start', type=int, default=None, help='Step count starting from (default: %(default)s)')
        #group.add_argument('--train-factor', '--factor', '--tf', '-F', type=float, default=None, help='Training factor for learning rate (default: {})'.format(default.train.factor))
        cls.add_argument(group, default.train.warmup_factor, '--warmup-factor', '--train-factor', '--factor', '--wf', '-F', type=float, help='Training factor for learning rate')
        group.add_argument('--resume', '-R', type=str, help='list of path to the resuming models (ends with ".npz") or suffix name (e.g. "latest", "best_dev_loss")', nargs='*')
        group.add_argument('--interval', '-I', type=float, default=None, help=f'Interval of training report (in seconds, default: {default.log.interval})')
        group.add_argument('--filter-noisy-samples', '--filter-noise', '--filter', type=strtobool, default=None, nargs='?', const=True, help='Filtering noisy training examples gradually with training steps')
        group.add_argument('--max-steps', type=int, default=None, help=f'Maximum number of universal transformer steps (default: {default.model.max_steps})')
        group.add_argument('--save-models', '--save-model', '--save', type=strtobool, default=True, help='Enable to save trained models (default: %(default)s')
        group.add_argument('--initializer', '--initialize', '--init', type=str, default=None, choices=['he-normal', 'orthogonal', 'pytorch'], help='Parameter initializing method')
        group.add_argument('--random-seed', '--seed', '--rs', type=int, default=None, help=f'Random seed (default: {default.train.random_seed})')
        group.add_argument('--time-penalty', '--tp', type=float, default=None, help=f'Penalty for pondering time (default: {default.train.time_penalty})')
        group.add_argument('--timeout', '--train-timeout', '-T', type=float, default=None, help=f'Timeout duration for truncation in feeding training batches (default: {default.train.timeout})')
        group.add_argument('--schedule-num-steps', '--schedule-steps', type=strtobool, default=None, nargs='?', const=True, help=f'Scheduling number of steps (default: {default.train.schedule_num_steps})')
        group.add_argument('--review-rate', '--review', '--rr', type=float, default=None, help=f'Ratio to review (feed already trained samples) in each epoch (default: {default.train.review_rate})')
        group.add_argument('--eval-only', '--no-train', '--eval', action='store_true', help='Evaluation only (skip training)')
        group.add_argument('--eval-train', '--et', action='store_true', help='Evaluate also training data with task specific measurement (not suggested for large data)')

        parser['optimizers'] = group = main.add_argument_group('optimizers', 'general optimizers options')
        group.add_argument('--optimizer', '-O', type=str, default=None, choices=['sgd', 'adam', 'amsgrad', 'adabound', 'amsbound', 'lamb'], help=f'Optimizer (default: {default.train.optimizer})')
        group.add_argument('--move-optimizer', '-M', action='store_true', help='move optimizer states into CPU memory (more efficifient for video memory usage, less for computation')
        group.add_argument('--gradient-clipping', '--grad-clip', '--gc', type=float, default=None, help='Gradient clipping (default: %(default)s)')
        cls.add_argument(group, default.train.sgd_learning_rate, '--learning-rate', '--lr', type=float, help='Optimizer learning rate')
        group.add_argument('--weight-decay-rate', '--weight-decay', '--l2-decay', '--decay', '--wd', type=float, default=None, help=f'Gradient clipping (default: {default.train.weight_decay_rate})')
        group.add_argument('--weight-decay-warmup-steps', '--weight-decay-warmup', '--wdws', type=int, default=None, help=f'Number of warming up steps for weight decay (default: {default.train.weight_decay_warmup_steps})')

        #parser['optimizers-adam'] = group = main.add_argument_group('optimizers-adam', 'specific optimizers options for Adam/AMSGrad/AdaBound/AMSBound')
        use_adam_family = (default.train.optimizer in ['adam', 'adamax', 'amsgrad', 'adabound', 'amsbound', 'lamb'])
        parser['optimizers-adam'] = group = cls.add_argument_group(main, use_adam_family, 'optimizers-adam', 'specific optimizers options for Adam/AMSGrad/AdaBound/AMSBound/LAMB')
        cls.add_argument(group, default.train.adam_alpha, '--adam-alpha', '--alpha', type=float, help='Alpha value for Adam/AMSGrad/AdaBound/AMSBound/LAMB')
        cls.add_argument(group, default.train.adam_beta1, '--adam-beta1', '--beta1', type=float, help='Beta1 value for Adam/AMSGrad/AdaBound/AMSBound/LAMB')
        cls.add_argument(group, default.train.adam_beta2, '--adam-beta2', '--beta2', type=float, help='Beta2 value for Adam/AMSGrad/AdaBound/AMSBound/LAMB')
        cls.add_argument(group, default.train.adam_eps, '--adam-eps', '--eps', '--epsilon', type=float, help='Epsilon value for Adam/AMSGrad/AdaBound/AMSBound/LAMB')
        cls.add_argument(group, default.train.adabound_gamma, '--adabound-gamma', '--gamma', type=float, help='Gamma value for AdaBound/AMSBound')

        use_sgd_family = (default.train.optimizer in ['sgd', 'adabound', 'amsbound'])
        parser['optimizers-sgd'] = group = cls.add_argument_group(main, use_sgd_family, 'optimizers-sgd', 'specific optimizers options for SGD/AdaBound/AMSBound')
        cls.add_argument(group, default.train.sgd_learning_rate, '--sgd-learning-rate', '--sgd-lr', type=float, help='Learning rate for SGD/AdaBound/AMSBound')
        return parser

def main(Trainer, modelname):
    parser = Trainer.create_parser(modelname)
    main_parser = parser['main']
    args = main_parser.parse_args()
    if args.help:
        if args.debug:
            logging.using_config(target_loggers, debug=True)
        # reconstruct parser with model-specific default values
        dprint(args)
        config = Trainer.update_config(None, args)
        dprint(config.to_json(indent=2, purge=True),)
        parser = Trainer.create_parser(modelname, config.data)
        main_parser = parser['main']
        print(main_parser.format_help())
        # Printing the help is what was asked for, so it is a success.
        # Exiting with -1 (255) made `cmd --help` look like a failure to
        # shell scripts and to CI.
        # (ヘルプの表示は要求どおりの動作なので正常終了とする。-1 (255) で
        #  終了すると、シェルスクリプトや CI からは失敗に見えてしまう)
        sys.exit(0)

    # setting debug mode
    if args.debug:
        logging.using_config(target_loggers, debug=True)
        for l in target_loggers:
            l = logging.getLogger(l)
        dprint(args)

    # setting logging files
    if not args.workdir:
        logger.error("please specify workdir")
        raise ValueError("working directory is not given")
    # record.tmp holds the process id file, and the training log unless
    # --logging overrides its path, so it must exist either way. The original
    # created it only on the else branch, which made --logging crash here.
    # (record.tmp にはプロセス ID ファイルと、--logging で上書きされない限り
    #  訓練ログが置かれるため、どちらの経路でも作成が必要。元実装は else 節
    #  でのみ作成しており、--logging 指定時にここでクラッシュしていた)
    tmp_record_dir = os.path.join(args.workdir, 'record.tmp')
    safeMakeDirs(tmp_record_dir)
    if args.logging is not None:
        logpath = args.logging
    else:
        logpath = os.path.join(tmp_record_dir, 'train.log')

    # checking conflicting process
    pidpath = os.path.join(tmp_record_dir, 'training.pid')
    if os.path.isfile(pidpath):
        with open(pidpath, encoding='utf-8') as fobj_pid:
            pid = int(fobj_pid.readline())
        dprint(pid)
        try:
            os.kill(pid, 0)
        except OSError:
            # process is not running
            if not args.resume:
                ask_continue_if_exist(pidpath)
        else:
            logger.error(f"other training process ({pid}) is running")
            return False
    # saving new process id
    with open(pidpath, 'w') as fobj:
        fobj.write(str(os.getpid()))
        dprint(os.getpid())

    # checking gpu availability
    if len(args.gpu) > 0:
        for gpu_id in args.gpu:
            if gpu_id >= 0:
                device = torch.device(gpu_id)
            else:
                device = torch.device('cpu')
            logger.debug(f"testing device: {device}")
            try:
                torch.empty(0).to(device)
                logger.debug("-> OK")
            except Exception as e:
                logger.debug("-> NG")
                raise e

    if args.resume is None:
        safe_remove(logpath)
        safe_remove(os.path.join(args.workdir, 'record.tmp', 'scores.json'))
    set_logfile_handler(logpath)

    # argment rewriting
    if args.resume == []:
        args.resume = ['latest', 'prev']
    if args.eval_only:
        if not args.resume:
            args.resume = ['latest', 'prev']

    # file loading
    args.train_file = None
    args.test_file  = None
    args.dev_file   = None
    if isinstance(args.train_files, list):
        if len(args.train_files) == 0:
            if not args.resume:
                logger.error("please specify train_files")
                raise ValueError("train file is not given")
            args.train_file = None
        elif len(args.train_files) == 1:
            args.train_file = args.train_files[0]
        else: # len(args.train_files) >= 2:
            args.train_file = PastedFile(args.train_files)
    if isinstance(args.test_files, list):
        if len(args.test_files) >= 2:
            args.test_file = PastedFile(args.test_files, 'rt')
        else:
            args.test_file = args.test_files[0]
    if isinstance(args.dev_files, list):
        if len(args.dev_files) >= 2:
            args.dev_file = PastedFile(args.dev_files, 'rt')
        else:
            args.dev_file = args.dev_files[0]

    # trainer setting
    trainer = Trainer(args)
    if args.resume:
        trainer.try_loading(args.workdir, args.resume)
    set_logfile_handler(logpath)

    trainer.config = trainer.update_config(trainer.config, args)
    if args.import_embed and not args.resume:
        vocab.import_vectors(args.import_embed)
    trainer.setup_model(args)
    trainer.setup_optimizer()
    trainer.setup_device(args)

    cdata = trainer.config.data
    status = cdata.log
    if args.eval_only:
        if args.eval_train:
            trainer.load_train_data()
            if trainer.train_df is not None:
                train_report  = trainer.evaluate('train', trainer.train_df, args)
                logger.info("evaluation result:\n" + str(train_report))
        if trainer.dev_df is not None:
            dev_report  = trainer.evaluate('dev', trainer.dev_df, args)
            logger.info("evaluation result:\n" + str(dev_report))
        if trainer.test_df is not None:
            test_report = trainer.evaluate('test', trainer.test_df, args)
            logger.info("evaluation result:\n" + str(test_report))
    else:
        try:
            #for status.epoch in range(status.epoch+1, args.max_epochs+1):
            # ruff B020 は誤検出。range() はループ開始前に一度だけ評価される。
            # status への代入は、中断時に再開位置を保存するための意図的な設計
            for status.epoch in range(status.epoch+1, args.num_epochs+1):  # noqa: B020
                if trainer.train_epoch(args):
                    pass #ok
                else:
                    break
        except KeyboardInterrupt:
            logger.warning("received keyboard interruption")
            try:
                if trainer.device.type == 'cuda':
                    logger.info("moving model parameters from video memory into main memory")
                    trainer.model.zero_grad()
                    trainer.model.to('cpu')
                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                        # reset_max_memory_cached は FutureWarning を出す別名
                        torch.cuda.reset_peak_memory_stats()
                    except Exception as exc:
                        # 後片付けの失敗で終了処理を止めない。ただし
                        # 握り潰さず記録は残す
                        logger.debug(f"failed to release CUDA memory: {exc!r}")
                if args.save_models:
                    status.epoch -= 1
                    trainer.save_latest_status()
                logger.info("exiting training")
            except KeyboardInterrupt:
                logger.warning("received keyboard interruption again")
            sys.exit(1)

if __name__ == '__main__':
    main()
