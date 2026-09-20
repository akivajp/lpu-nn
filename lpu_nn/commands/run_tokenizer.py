#!/usr/bin/env python3

# system
import argparse
import sys

# 3-rd party
import sentencepiece as spm

# local
from lpu.common import logging

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

FORMAT_CHOICES = ['id', 'token', 'piece']
DEFAULT_FORMAT = 'token'

def run_tokenizer(model, output_format=DEFAULT_FORMAT):
    sp = spm.SentencePieceProcessor()
    sp.load(model)
    #dprint(sp)
    dprint(len(sp))
    for line in sys.stdin:
        line = line.rstrip('\n')
        #dprint(line)
        ids = sp.encode_as_ids(line)
        #dprint(ids)
        if output_format in ['token', 'piece']:
            tokens = [sp.id_to_piece(id) for id in ids]
            #dprint(tokens)
            print(str.join(' ', tokens))
        elif output_format in ['id']:
            #dprint(ids)
            print(str.join(' ', map(str, ids)))
        else:
            raise ValueError(f'Unknown format: {output_format}')

def main():
    parser = argparse.ArgumentParser("Tokenizer")
    parser.add_argument("model", type=str, help="model path")
    parser.add_argument("--format", "-F", default=DEFAULT_FORMAT, choices=FORMAT_CHOICES, help='output format')
    parser.add_argument("--debug", "-D", action="store_true", help="enable debug mode")
    args = parser.parse_args()
    # 他のコマンドと同じく、対象はロガーオブジェクトではなくパッケージ名
    loggers = ['__main__', 'lpu_nn', 'lpu']
    with logging.using_config(loggers, debug=args.debug):
        dprint(args)
        return run_tokenizer(args.model, args.format)

if __name__ == '__main__':
    main()
