#!/usr/bin/env python3

import csv
import glob
import io
from functools import reduce

import pandas as pd

from lpu.common import logging
from lpu.common import progress

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

def min_tuple(t1, t2):
    return tuple(min(e1,e2) for e1, e2 in zip(t1,t2))

def max_tuple(t1, t2):
    return tuple(max(e1,e2) for e1, e2 in zip(t1,t2))

def inter_tuple(t1, t2):
    return tuple((e1+e2) / 2.0 for e1,e2 in zip(t1,t2))

def get_indices(fields, keys):
    return tuple(fields.index(key) for key in keys)

def to_float(s, default=-1):
    try:
        return float(s)
    except Exception:
        return default

def get_values(fields, indices):
    #return tuple(float(fields[index]) for index in indices)
    try:
        return tuple(to_float(fields[index]) for index in indices)
    except Exception as e:
        dprint(e)
        dprint(fields)
        dprint(indices)

#def gen_converters(main_fields):
#    converters = dict()
#    for column, ftype in main_fields.items():
#        if ftype == 'seq':
#            converters[column] = literal_eval
#    return converters

class Dataset:
    def __init__(self, path, sep='\t', priority_keys=None):
        self.load(path, sep, priority_keys)

    def iter(self, start=None, stop=None, step=None, headers=True, binmode=False):
        if headers:
            if binmode:
                yield self.str_headers
            else:
                yield self.str_headers.decode('utf-8', 'backslashreplace')
        if step is None:
            step = 1
        if isinstance(start, int) and start < 0:
            start = len(self) - start
        if isinstance(stop, int) and stop < 0:
            stop = len(self) - stop
        if step >= 1:
            if start is None:
                start = 0
            if stop is None:
                stop = len(self)
            current = start
            while current < stop:
                if current < 0 or current >= len(self):
                    break
                #yield self[current]
                yield self.getline(current, binmode)
                current += step
        elif step <= -1:
            if start is None:
                start = len(self) - 1
            if stop is None:
                stop = -1
            current = start
            while current > stop:
                if current < 0 or current >= len(self):
                    break
                #yield self[current]
                yield self.getline(current, binmode)
                current += step

    def getbuffer(self, start=None, stop=None, step=None, headers=True):
        buf = io.StringIO()
        for line in self.iter(start, stop, step, headers, binmode=False):
            buf.write(line)
            buf.write("\n")
        buf.seek(0)
        return buf

    def getline(self, index, binmode=False):
        pos = self.positions[index]
        self.fobj.seek(pos)
        line = self.fobj.readline().strip()
        if binmode:
            return line
        else:
            return line.decode('utf-8', 'backslashreplace')

    def load(self, path, sep='\t', priority_keys=None):
        self.sep = sep
        self.path = path
        positions1 = []
        positions2 = []
        f = progress.FileReader(path)
        header_line = f.read_byte_line().strip()
        self.str_headers = header_line
        self.headers = header_line.decode('utf-8', 'backslashreplace').split(sep)
        if isinstance(priority_keys, str):
            priority_keys = [priority_keys]
        if priority_keys:
            priority_indices = get_indices(self.headers, priority_keys)
            #dprint(priority_keys)
            #dprint(self.headers)
            #dprint(priority_indices)
        repeated_errors = 0
        while True:
            pos = f.tell()
            line = f.read_byte_line()
            if not line:
                break
            try:
                line = line.strip()
                if line:
                    if priority_keys:
                        line = line.decode('utf-8', 'backslashreplace')
                        priority = get_values(line.split(sep), priority_indices)
                        # fuzzy sorting
                        if len(positions1) == 0:
                            min_priority = priority
                            max_priority = priority
                        else:
                            min_priority = min_tuple(priority, min_priority)
                            max_priority = max_tuple(priority, max_priority)
                        inter_priority = inter_tuple(min_priority, max_priority)
                        if priority <= inter_priority:
                            positions1.append(pos)
                        else:
                            positions2.append(pos)
                    else:
                        positions1.append(pos)
                repeated_errors = 0
            except Exception as e:
                dprint(pos)
                dprint(line)
                dprint(repeated_errors)
                logger.exception(e)
                repeated_errors += 1
                if repeated_errors > 5:
                    logger.debug("repeated errors were detected, breaking the load loop")
                    break
        f.close()
        self.positions = positions1 + positions2
        self.fobj = open(path, 'rb')

    def save(self, path_or_buffer, start=None, stop=None, step=None):
        if isinstance(path_or_buffer, str):
            path_or_buffer = open(path_or_buffer, 'wb')
        for line in self.iter(start, stop, step, headers=True, binmode=True):
            path_or_buffer.write(line)
            path_or_buffer.write(b"\n")

    def to_df(self, main_fields, start=None, stop=None, step=None):
        buf = self.getbuffer(start, stop, step)
        #converters = gen_converters(main_fields)
        #return pd.read_csv(buf, self.sep, converters=converters, index_col='index')
        # pandas 3.0 で sep の位置引数渡しは廃止された (キーワード必須)
        return pd.read_csv(buf, sep=self.sep, index_col='index')

    def __getitem__(self, i):
        #print(i)
        if isinstance(i, slice):
            return list(self.iter(i.start, i.stop, i.step, False))
        else:
            return self.getline(i, False)

    def __iter__(self):
        return self.iter()

    def __len__(self):
        return len(self.positions)

#def find_sub(full, sub, index=0):
#    length = len(sub)
#    if len(full) < index + length:
#        #return -1, None
#        return -1
#    if full[index:index+length] == sub:
#        #return index, full[index:index+length]
#        return index
#    else:
#        return find_sub(full, sub, index+1)

#if __name__ == '__main__':
#    full = list('abracadabra')
#    dprint(full)
#    dprint(find_sub(full, list('abra')),)
#    dprint(find_sub(full, list('bra')),)
#    dprint(find_sub(full, list('abca')),)

#def build_train_data(main_fields, save_path, train_file, vocab, sep= '\t'):
#def build_train_data(main_fields, save_path, train_file, vocab, sep= '\t', max_length=None):
def build_train_data(main_fields, save_path, train_file, sep= '\t', max_length=None):
    writer = csv.writer(open(save_path, 'w'), delimiter=sep)
    #header = ['index', 'x', 't', 'len_x', 'len_t', 'cost', 'last_epoch', 'last_step', 'feed_count', 'criterion', 'last_pred']
    #header = ['index', 'x', 't', 'criterion']
    #header = ['index', 'x', 't', 'len_x', 'len_t', 'cost', 'last_epoch', 'last_step', 'feed_count', 'criterion', 'priority', 'last_pred']
    #header = ['index', 'x', 't', 'len_x', 'len_t', 'cost', 'last_epoch', 'last_step', 'feed_count', 'criterion', 'priority']
    main_columns = list(main_fields.keys())
    len_columns = ['len_'+column for column, type in main_fields.items() if type == 'seq']
    header = ['index'] + main_columns + len_columns + ['len', 'cost', 'last_epoch', 'last_step', 'feed_count', 'criterion', 'priority']
    writer.writerow(header)
    format_types = main_fields.values()
    too_long_count = 0
    #for i, line in enumerate( progress.view(train_file) ):
    for i, line in enumerate( progress.view(train_file, header=f"scanning file '{train_file}'") ):
        try:
            fields = line.strip().split(sep)
            #fields = line.strip('\n').split(sep)
            #dprint(i)
            #dprint(fields)
            len_values = []
            row = []
            row.append(i)
            too_long = False
            #x = fields[0]
            #x = tuple(vocab.encode_ids(x))
            #row.append(x)
            #t = fields[1]
            #t = tuple(vocab.encode_ids(t))
            #row.append(t)
            #len_x = len(x)
            #row.append(len_x)
            #len_t = len(t)
            #row.append(len_t)
            #for j, field in enumerate(fields):
            for j, ftype in enumerate(format_types):
                value = fields[j]
                if ftype == 'seq':
                    #value = tuple(vocab.encode(value))
                    len_value = len(value)
                    if max_length is not None and len_value > max_length:
                        too_long = True
                        break
                    len_values.append(len_value)
                row.append(value)
            if too_long:
                too_long_count += 1
                continue
            for len_value in len_values:
                row.append(len_value)
            len_total = sum(len_values)
            row.append(len_total)
            cost = 1
            row.append(cost)
            last_epoch = 0
            row.append(last_epoch)
            last_step = 0
            row.append(last_step)
            feed_count = 0
            row.append(feed_count)
            criterion = 0.0
            row.append(criterion)
            priority = 0.0
            row.append(priority)
            #criterion = (len_x + len_t) - 4096
            #last_pred = ()
            #row.append(last_pred)
            writer.writerow(row)
        except Exception as e:
            dprint(i)
            dprint(fields)
            logger.exception(e)
            #logger.debug(repr(e))
    if too_long_count > 0:
        logger.info(f"skipped {too_long_count:,d} too long samples")

#def load_eval_data(main_fields, file_or_buffer, vocab, sep='\t'):
def load_eval_data(main_fields, file_or_buffer, sep='\t'):
    #with open(eval_file, 'r') as fobj:
    fobj = file_or_buffer
    if isinstance(file_or_buffer, str):
        fobj = open(file_or_buffer, encoding='utf-8', errors='backslashreplace')
    rows = []
    #converters = gen_converters(main_fields)
    format_types = main_fields.values()
    for i, line in enumerate(fobj):
        try:
            fields = line.strip().split(sep)
            #fields = [vocab.encode_ids(field) for field in fields]
            #for j, ftype in enumerate(format_types):
            #    if ftype == 'seq':
            #        #fields[j] = vocab.encode_ids(fields[j])
            #        fields[j] = tuple(vocab.encode(fields[j]))
            rows.append( fields )
        except Exception as e:
            logger.exception(e)
    #df = pd.DataFrame(rows, columns=['x', 't'])
    main_columns = list(main_fields.keys())
    df = pd.DataFrame(rows, columns=main_columns)
    len_total = 0
    for column, type in main_fields.items():
        if type == 'seq':
            df.loc[:, 'len_'+column] = df[column].apply(lambda seq: len(seq))
            len_total += df.loc[:, 'len_'+column]
    df.loc[:, 'len'] = len_total
    #df.loc[:, 'len_x'] = df.x.apply(lambda x: len(x))
    #df.loc[:, 'len_t'] = df.t.apply(lambda t: len(t))
    # The criterion column later receives per-sample perplexities, which are
    # floats. Initializing it with the int -1 made it an int64 column, and
    # pandas 3.0 refuses the float assignment instead of upcasting, so the
    # evaluation scores were silently dropped. A row-wise apply was also a
    # needlessly expensive way to fill a constant.
    # (criterion 列には後からサンプルごとの perplexity (float) が入る。
    #  int の -1 で初期化すると int64 列になり、pandas 3.0 は型を広げずに
    #  float の代入を拒むため、評価スコアが静かに捨てられていた。
    #  定数を埋めるのに行単位の apply を使う必要も無い)
    df.loc[:, 'criterion'] = -1.0
    return df

def merge_tsv_files(tsv_paths, target_path, sep='\t'):
    if isinstance(tsv_paths, str):
        tsv_paths = [tsv_paths]
    tsv_paths = reduce(list.__add__, [glob.glob(p) for p in tsv_paths])
    for i, tsv_path in enumerate(tsv_paths):
        msg = f"merging '{tsv_path}' into '{target_path}'"
        logger.info(msg)
        fobj_in = progress.view(tsv_path, header = msg)
        if i == 0:
            header = fobj_in.read_byte_line()
            fobj_out = open(target_path, 'wb')
            fobj_out.write(header)
        else:
            header = fobj_in.read_byte_line()
            # skip header
        for chunk in fobj_in.read_byte_chunks():
            fobj_out.write(chunk)
    logger.info(f"merged tsv files '{tsv_path}' into: '{target_path}'")

