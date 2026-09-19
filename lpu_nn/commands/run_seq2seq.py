#!/usr/bin/env python3

# system
import argparse
import sys
import time

# 3rd

# local
from lpu.common import logging
from lpu_nn.commands import train_seq2seq

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

def main():
    parser = argparse.ArgumentParser(description = 'Sequence-to-Sequence Decoder')
    parser.add_argument('model', help='path to read the trained model (directory or config path)')
    parser.add_argument('--gpu', '-G', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--debug', '-D', action='store_true', help='Debug mode')
    parser.add_argument('--logging', '--log', type=str, default=None, help='Path of file to log (default: %(default)s')
    parser.add_argument('--beam_width', '--beam', '--n-best', '--nbest', '-n', type=int, default=10, help='Beam width')
    parser.add_argument('--incomplete-cost', '--incomp', '-c', type=int, default=100, help='Cost of end-of-sentence symbold of incomplete sentence')
    parser.add_argument('--repetition-cost', '--rep', '-r', type=float, default=0, help='Cost of symbol repetition in short term of sentence')
    parser.add_argument('--normalize', '--norm', '-N', action='store_true', help='Scale n-best scores with sentence length')
    parser.add_argument('--max-steps', type=int, default=None, help='Number of adaptive computation time steps (for universal transformers)')
    parser.add_argument('--timeout', '--time-limit', '-T', type=float, default=None, help='Time limit for each generation (in seconds)')

    args = parser.parse_args()
    if args.debug:
        logging.using_config(['__main__', 'common', 'models'], debug=True)
        dprint(args)

    trainer = train_seq2seq.Seq2SeqTrainer().load_status(args.model)
    model = trainer.model
    if args.gpu >= 0:
        model.to(args.gpu)
    logger.info("loaded")
    #trainer.config.max_steps = args.max_steps
    trainer.set_max_steps(args.max_steps)

    for sent_number, line in enumerate(sys.stdin):
        line = line.strip()
        logger.debug("-----")
        dprint(sent_number)
        dprint(line)
        try:
            #trainer.set_max_steps()
            time_start = time.time()
            max_length = max(100, len(line) * 2 + 20)
            #pred = model.generate(line, max_length=max_length, timeout=args.timeout,)
            #pred_str = trainer.model.restore_batch(pred, str)
            #dprint(pred_str)
            #if True:
            #    break
            pred = model.beam_search(
                line,
                beam_width=args.beam_width,
                max_length=max_length,
                incomplete_cost=args.incomplete_cost,
                repetition_cost=args.repetition_cost,
                normalize=args.normalize,
                timeout=args.timeout,
            )
            if args.debug:
                if args.normalize:
                    #dprint(pred)
                    for seq, score in pred:
                        dprint( (score, seq) )
            if len(pred) > 0:
                sys.stdout.write(f"{trainer.vocab.decode(pred[0][0])}\n")
            else:
                sys.stdout.write("\n")
            dprint(time.time() - time_start)
        except Exception as e:
            logger.exception(e)
            #logger.debug(repr(e))
            sys.stdout.write("<err>\n")
        sys.stdout.flush()

if __name__ == '__main__':
    main()

