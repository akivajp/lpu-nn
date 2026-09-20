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
- sequence matching and ranking (`lpu-nn-train-match-ranker`,
  `lpu-nn-run-match-ranker`), with the RE2 and Compare-Aggregate poolers
- BERT pre-training (`lpu-nn-train-bert`) and fine-tuning for classification
  (`lpu-nn-train-bert-classifier`) and pair ranking
  (`lpu-nn-train-bert-ranker`)

The sequence tagging and language modeling parts of the original codebase
are not ported yet.

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

### Sequence matching and ranking

The match ranker scores a pair of sequences. Its corpus is a TSV file of
three columns: the two sequences and the target score.

```shell
$ lpu-nn-train-match-ranker workdir match-train.tsv --dev-files match-dev.tsv --gpu 0
```

`--match-pooler-type` selects the architecture: `re2`
([Yang et al., 2019](https://aclanthology.org/P19-1465/)) or
`compare-aggregate` ([Wang and Jiang, 2017](https://arxiv.org/abs/1611.01747)).
`--loss-method` selects how the target is used: `point` for regression on the
score, `pair` for a pairwise ranking loss, `classify` for a label
distribution. The checkpoints are written per ranking metric
(`record.best_dev_mrr`, `record.best_dev_map`, ...).

Scoring reads pairs from the standard input, one per line:

```shell
$ lpu-nn-run-match-ranker workdir/record.best_dev_mrr --gpu 0 < pairs.tsv
```

`--evaluate` reports MRR, MAP and recall at k on a labelled corpus instead,
and `--replies` ranks a whole candidate file against each query.

### BERT

Pre-training takes a TSV file of sentence pairs and learns a masked language
model together with next-sentence prediction.

```shell
$ lpu-nn-train-bert workdir train.tsv --dev-files dev.tsv --gpu 0
```

`--universal` uses a Universal Transformer (with an adaptive number of steps
and a ponder cost) instead of a fixed stack, and `--num-token-types 2` adds
the segment embedding that distinguishes the two sides of a pair.

Fine-tuning starts from a pre-trained checkpoint. The classifier takes a TSV
file of a sentence and its label; the ranker takes a TSV file of pairs.

```shell
$ lpu-nn-train-bert-classifier workdir class-train.tsv --dev-files class-dev.tsv \
    --pre-trained-model bert-workdir/record.best_dev_loss \
    --sentencepiece bert-workdir/sp.model --gpu 0
$ lpu-nn-run-bert-classifier workdir/record.best_dev_acc < sentences.txt
```

`--sentencepiece` is required alongside `--pre-trained-model`: each work
directory trains its own tokenizer, and fine-tuning reuses the pre-trained
embedding, so the two vocabularies have to be the same one. The command
refuses to start when they differ rather than writing a checkpoint that
cannot be loaded back.

The scorer writes one predicted label per line. `--ranking` reads
`sentence<TAB>label` instead and reports MRR and precision at k over the
known labels. The pair ranker's scorer, `lpu-nn-run-bert-ranker`, reads
`sentence1|||sentence2` and writes one score per line, or ranks a candidate
file against each query with `--replies`.

Run any command with `--help` for the full list of options.

## Layout

| Module | Contents |
| --- | --- |
| `lpu_nn.common` | the trainer, the dataset, the vocabulary, the criteria |
| `lpu_nn.modeling` | transformer, universal transformer, LSTM, attention, embeddings, RE2, Compare-Aggregate, BERT |
| `lpu_nn.optimizers` | AdaBound, LAMB, and the torch optimizers used by the trainer |
| `lpu_nn.commands` | the command line entry points |

The configuration, logging, progress display and file utilities come from
`lpu`, so they are not duplicated here.

## License

MIT, except for the bundled third-party optimizers; see [LICENSE](LICENSE)
and [licenses/NOTICE.md](licenses/NOTICE.md).
