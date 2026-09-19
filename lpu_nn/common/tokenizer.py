#!/usr/bin/env python3

# system
import argparse
import functools
import tempfile
import unicodedata

# 3rd 
import sentencepiece as spm

# local
from lpu.common import logging
from lpu.common.progress import view as pview

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

DEFAULT_VOCAB_SIZE = 16000
DEFAULT_MIN_COUNT = 2
DEFAULT_UNK_SURFACE = '<unk>'
DEFAULT_MODEL_TYPE = 'unigram'
MODEL_TYPE_CHOICES = ['unigram', 'bpe', 'word', 'char']

from collections import defaultdict

#def split_digits(segment):
def split_digit_chars(segment):
    tokens = []
    buf = b''
    for code in segment:
        c = bytes([code])
        if c.isdigit():
            if buf:
                tokens.append(buf)
                buf = b''
            tokens.append(c)
        else:
            buf += c
    if buf:
        tokens.append(buf)
    return tokens

def load_coverage(path):
    coverage = set()
    #for line in pview(path).read_byte_lines():
    for line in pview(path):
        fields = line.strip().split(' ')
        for field in fields:
            low = field.lower()
            norm = unicodedata.normalize('NFKC', field)
            low_norm = unicodedata.normalize('NFKC', low)
            coverage.add(field.encode('utf-8'))
            coverage.add(low.encode('utf-8'))
            coverage.add(norm.encode('utf-8'))
            coverage.add(low_norm.encode('utf-8'))
    return coverage

def preprocess(train_files,
               sep=b'\t', min_count=None, max_count=None, max_bytes=None, max_segments=None,
               force_coverage=False, split_digits=True, cover_segments=None):
    #MAX_SEGMENT_LENGTH = 30
    MAX_SEGMENT_LENGTH = 50
    count_segment = defaultdict(int)
    count_char = defaultdict(int)
    max_char_count = 0
    space_keywords = [
        'ARABIC', # "پ", "خ", ...
        'CYRILLIC', # "Б", "Д", ..., "б", "д", ...
        'GREEK', # "Γ", "Δ", "Θ", ..., "α", "β", "γ", ...
        'LATIN', # "Ī", "Ñ", "ú", ... (latin letters with diacritical mark) "œ", ... (latin ligatures)
        'SIGN', # 'µ', '£', ...
    ]
    if not cover_segments:
        cover_segments = set()
    if isinstance(train_files, str):
        train_files = [train_files]
    # A hardcoded 'TMP.txt' was left here in place of the temporary file: it
    # polluted the working directory, was never removed, and two concurrent
    # runs would overwrite each other's data.
    # (一時ファイルの代わりに 'TMP.txt' が固定名で書かれていた。作業
    #  ディレクトリを汚し、削除もされず、並行実行では互いのデータを
    #  上書きしてしまう)
    temp = tempfile.NamedTemporaryFile('w+b')
    dprint(temp.name)
    logger.info("starting pre-proprocess")
    if max_segments:
        logger.info(f"max total segment count: {max_segments:,d}")
    if max_bytes:
        logger.info(f"max bytes: {max_bytes:,d}")
    count = 0
    byte_count = 0
    for path in train_files:
        reader = pview(path)
        for row in reader.read_byte_lines():
            row = row.rstrip(b'\n')
            if not row.strip():
                continue
            if max_segments and count >= max_segments:
                break
            fields = row.split(sep)
            # forcing character-wise split for numerical expressions
            if split_digits:
                #fields = functools.reduce(list.__add__, map(split_digits, fields))
                fields = functools.reduce(list.__add__, map(split_digit_chars, fields))
            for field in fields:
                segments = field.split(b' ')
                if force_coverage:
                    # inserting space
                    segments.append(b' ')
                for segment in segments:
                    low_segment = segment.lower()
                    if not segment:
                        continue
                    if (len(segment) <= MAX_SEGMENT_LENGTH
                            or segment in cover_segments
                            or low_segment in cover_segments):
                        count_segment[segment] += 1
                        if max_count:
                            if segment == b' ':
                                if count_segment[segment] > max_char_count:
                                    continue
                            else:
                                if count_segment[segment] > max_count:
                                    continue
                        if min_count:
                            #dprint(segment)
                            #dprint(count_segment[segment])
                            if count_segment[segment] < min_count:
                                #dprint("NG")
                                #dprint(segment)
                                #dprint(count_segment[segment])
                                #dprint(segment in cover_segments)
                                #dprint(low_segment in cover_segments)
                                if (segment not in cover_segments
                                    and low_segment not in cover_segments):
                                    continue
                            #dprint("OK")
                    else:
                        continue
                    if force_coverage:
                        if segment != b' ':
                            for code in segment:
                                count_char[code] += 1
                                max_char_count = max(max_char_count, count_char[code])
                    else: # not force_coverage
                        unicode_head = segment[:3].decode('utf-8', 'ignore')
                        if len(unicode_head) == 0:
                            continue
                        char = unicode_head[0]
                        if len(char.encode('utf-8')) == 1:
                            dummy_prefix = b' '
                        else:
                            try:
                                name = unicodedata.name(char)
                            except ValueError:
                                name = ''
                            if any(name.find(keyword) >= 0 for keyword in space_keywords):
                                dummy_prefix = b' '
                            else:
                                dummy_prefix = b''
                        segment = dummy_prefix + segment
                    byte_count += (len(segment)+1)
                    #dprint(byte_count)
                    if max_bytes and byte_count >= max_bytes:
                        reader.close()
                        logger.info("finished pre-process")
                        return temp
                    temp.write(segment)
                    temp.write(b'\n')
                    count += 1
        del reader
    logger.info("finished pre-process")
    return temp

def train_tokenizer(model_prefix, train_files, vocab_size=DEFAULT_VOCAB_SIZE,
    model_type=DEFAULT_MODEL_TYPE, unk_surface=DEFAULT_UNK_SURFACE, force_coverage=False,
    min_count=DEFAULT_MIN_COUNT, cover_segments=None, normalize=False,
    hard_vocab_limit=False):
    spm_force_coverage = force_coverage
    if cover_segments:
        cover_segments = load_coverage(cover_segments)
    split_digits = True
    if model_type == 'char':
        split_digits = True
        if force_coverage:
            #temp = preprocess(train_files, min_count=min_count, max_count=100, force_coverage=force_coverage)
            max_count = 100
        else:
            #temp = preprocess(train_files, min_count=min_count, max_count=1000, force_coverage=force_coverage)
            max_count = 1000
        max_bytes = None
    else:
        if model_type == 'word':
            split_digits = False
            spm_force_coverage = force_coverage
            force_coverage = False
        if force_coverage:
            #temp = preprocess(train_files, min_count=min_count, max_count=500, max_bytes=2**31-1, force_coverage=force_coverage, split_digits=split_digits)
            max_count=500
            max_bytes = 2**31 - 1
        else:
            #temp = preprocess(train_files, min_count=min_count, max_count=5000, max_bytes=2**31-1, force_coverage=force_coverage, split_digits=split_digits)
            max_count=5000
            max_bytes = 2**31 - 1
    temp = preprocess(
        train_files,
        min_count = min_count,
        max_count = max_count,
        force_coverage = force_coverage,
        max_bytes = max_bytes,
        split_digits = split_digits,
        cover_segments = cover_segments,
    )
    args = []
    args.append(f'--model_prefix={model_prefix}')
    args.append(f'--input={temp.name}')
    args.append(f'--vocab_size={vocab_size}')
    args.append(f'--model_type={model_type}')
    args.append(f'--unk_surface={unk_surface}')
    #if model_type == 'word':
    #    args.append('--hard_vocab_limit=false')
    if not hard_vocab_limit:
        args.append('--hard_vocab_limit=false')
    if normalize:
        args.append('--normalization_rule_name={}'.format("nmt_nfkc"))
    else:
        args.append('--normalization_rule_name={}'.format("identity"))
    #args.append('--input_sentence_size=10000000')
    args.append('--add_dummy_prefix=false') # testing
    args.append('--remove_extra_whitespaces=false') # testing
    #if force_coverage:
    if spm_force_coverage:
        args.append('--character_coverage=1')
        args.append('--use_all_vocab=true')
    model_path = model_prefix + '.model'
    str_args = str.join(' ', args)
    logger.info(f"started training SentencePiece with arguments: {str_args}")
    spm.SentencePieceTrainer.train(str_args)
    logger.info("finished training SentencePiece")
    return model_path

def main():
    parser = argparse.ArgumentParser("Tokenizer Trainer")
    parser.add_argument("model_prefix", type=str, help="output model prefix")
    parser.add_argument("train_files", type=str, nargs="+", help="input text files to train tokenizer")
    parser.add_argument("--vocab-size", "--vsize", "--size", "-V", type=int, default=DEFAULT_VOCAB_SIZE, help="vocabulary size (default: %(default)s)")
    parser.add_argument("--model-type", "--token-type", "--type", "-T", type=str, default=DEFAULT_MODEL_TYPE, choices=MODEL_TYPE_CHOICES,
                        help="model algorithm (default: %(default)s)")
    parser.add_argument("--unk-surface", "--unk", "-U", type=str, default=DEFAULT_UNK_SURFACE, help="dummy surface string for unknown token (default: %(default)s)")
    parser.add_argument("--debug", "-D", action="store_true", help="enable debug mode")
    parser.add_argument("--force-coverage", "--cover", "-C", action="store_true", help="force to cover the all characters (if false, minimum character coverage is 0.9995)")
    parser.add_argument("--normalize", "--unicode-normalization", "-N", action="store_true", help="enable unicode normalization")
    parser.add_argument("--min-count", "--min", type=int, default=DEFAULT_MIN_COUNT, help="lower bound of each segment count")
    parser.add_argument("--cover-segments", "--cover-list", "-L", type=str, default=None, help="path to the list of covered segments (e.g. list of pre-trained words)")
    parser.add_argument("--hard-vocab-limit", "--hard-limit", "-H", action="store_true", help="consider --vocab-size as a hard limit")
    args = parser.parse_args()
    with logging.using_config(logger, debug=args.debug):
        dprint(args)
        return train_tokenizer(args.model_prefix, args.train_files,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            unk_surface=args.unk_surface,
            force_coverage=args.force_coverage,
            min_count=args.min_count,
            cover_segments=args.cover_segments,
            normalize = args.normalize,
            hard_vocab_limit = args.hard_vocab_limit,
        )

if __name__ == '__main__':
    main()
