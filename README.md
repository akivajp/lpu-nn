# LPU-NN

Neural language processing models on PyTorch, built on
[LPU](https://github.com/akivajp/lpu).

日本語版のドキュメントは [README.ja.md](README.ja.md) にあります。

## Status

This package revives a private research codebase written in 2019-2020 for
reproducing and prototyping neural language processing models. It is being
ported and modernized incrementally, so the API is not yet stable.

What currently runs end to end on PyTorch 2.x / Python 3.13:

- sequence-to-sequence training (`lpu-nn-train-seq2seq`)
- decoding with beam search (`lpu-nn-run-seq2seq`)

The BERT, sequence tagging, sequence matching and language modeling parts of
the original codebase are not ported yet.

## Requirements

- Python 3.10 or later
- PyTorch 2.4 or later (a CUDA build is recommended for training)

## Installation

```shell
$ pip install 'lpu-nn @ git+https://github.com/akivajp/lpu-nn.git'
```

For development:

```shell
$ uv sync
```

Training curves are written only when the optional `plot` extra is installed:

```shell
$ pip install 'lpu-nn[plot] @ git+https://github.com/akivajp/lpu-nn.git'
```

## Usage

The trainer takes a working directory and a training corpus. The corpus is
either a TSV file (source and target in two columns) or one file per column.

```shell
$ lpu-nn-train-seq2seq workdir train.tsv --dev-files dev.tsv --test-files test.tsv --gpu 0
```

It trains a SentencePiece tokenizer, builds the dataset, and writes a
checkpoint directory for every metric it improves on
(`record.best_dev_loss`, `record.best_dev_bleu`, ...), each holding the
model, the optimizer state, the configuration and the scores.

Decoding reads from the standard input and writes to the standard output:

```shell
$ lpu-nn-run-seq2seq workdir/record.best_dev_loss --gpu 0 < test.txt > hyp.txt
```

Run either command with `--help` for the full list of options.

## Layout

| Module | Contents |
| --- | --- |
| `lpu_nn.common` | the trainer, the dataset, the vocabulary, the criteria |
| `lpu_nn.modeling` | transformer, universal transformer, LSTM, attention, embeddings |
| `lpu_nn.optimizers` | AdaBound, LAMB, and the torch optimizers used by the trainer |
| `lpu_nn.commands` | the command line entry points |

The configuration, logging, progress display and file utilities come from
`lpu`, so they are not duplicated here.

## License

MIT, except for the bundled third-party optimizers; see [LICENSE](LICENSE)
and [licenses/NOTICE.md](licenses/NOTICE.md).
