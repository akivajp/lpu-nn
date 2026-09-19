# Changelog

日本語版は [CHANGELOG.ja.md](CHANGELOG.ja.md) にあります。

## Unreleased

### Added

- The sequence-to-sequence path of a private research codebase written in
  2019-2020, ported onto PyTorch 2.x and Python 3.13: the trainer, the
  dataset, the SentencePiece vocabulary, the transformer / universal
  transformer / LSTM modules, the attention layers, the AdaBound and LAMB
  optimizers, and the `lpu-nn-train-seq2seq` / `lpu-nn-run-seq2seq`
  commands.
- The configuration, logging, progress display, colors, dialog and file
  utilities are taken from `lpu` instead of being carried over, which drops
  about 1,900 lines of duplicated code.

### Fixed

`lpu_nn.modeling.__all__` listed `match_ranker` and `re2`, which are not
ported, so `from lpu_nn.modeling import *` raised. Both `__all__` lists now
match what the packages actually export.

`FieldMap.load` carried a block copied from `FieldMap.train` that logged
"feeding corpus" and computed a path it never used. It has been removed.

Four methods took a mutable default argument, and two of them passed it
straight to `dict.update`. The bare `except` clauses now name the exception
they mean to catch: a signature mismatch when probing `init_weights`, a
failure to seek a non-seekable input (which also leaked its file handle),
and a label that does not parse as a number.

`Dataset.iter` resolved a negative `start` or `stop` as
`len(self) - index`, which cancels the sign and lands past the end, so a
negative start yielded nothing at all. It now counts from the end, as
elsewhere in Python.

`Dataset.load` called `.strip()` on the first line without checking it:
`read_byte_line` returns `None` at the end of the input, so an empty file
raised an `AttributeError`, and a file that began with a blank line built
a dataset with no column names at all. It now says which file has no
header.

`dataset.get_values` swallowed a missing column and returned an implicit
`None`, which then made the caller's comparison fail with a `TypeError`
far from the cause. The caller already handles a bad row, so the
`IndexError` is left to propagate.

`utils.flip` built a list of index tensors and applied it as `t[slices]`,
working around `torch.flip` not supporting bool CPU tensors in torch 1.3.
That restriction is long gone, and indexing with a non-tuple sequence is
deprecated: with more than one dimension PyTorch already reads it as
advanced indexing and raises an `IndexError`. It now calls `torch.flip`.

`criteria.cross_entropy` and `criteria.perplexity` accepted extra
positional arguments and forwarded them positionally to
`nn.functional.cross_entropy`, where they land on `weight` /
`size_average` / `ignore_index` and collide with the keywords set
alongside them. No caller used them, and they are no longer accepted.
Passing a list of ignored indices with any reduction other than `'hmean'`
now says so, rather than surfacing a message from inside torch.

`criteria.accuracy` masked nothing when `ignore_index` was given as a
list: it compared the boolean mask against the index (`t_valid != ignore`)
rather than the targets, and that is true for every ordinary index. Only
the integer form worked. `cross_entropy` next to it has this right.

`--help` printed the help and then exited with -1 (255), so `cmd --help`
looked like a failure to shell scripts and to CI. The trainer builds its
help manually, in order to show the model-specific defaults, which is how
this went unnoticed.

Deriving the number of attention heads from a hidden size below the
default key size of 64 produced zero heads, which surfaced much later as a
`ZeroDivisionError` inside a module constructor. The derivation is now
shared by the two places that had a copy of it, and reports what to pass
instead.

`ModuleConnection` reads an `init_gamma` parameter that `get_parameters`
computes a default for, but nothing applies it: the scaled LayerNorm gain
initialization was lost in a refactor, so passing the parameter has no
effect. Restoring it would change how every model initializes, so it is
documented in place rather than changed silently.

The original code had no tests and had not been run since 2020. The
following defects were found and fixed while making it run again.

Broken code paths (each raising `NameError` when reached):

- `Trainer.train_epoch` called `pview` without importing it, so every epoch
  that did not print a report crashed.
- `TransformerBase.add_positional_encoding` and `.forward` called
  `feed_seq`, a helper left over from the Chainer implementation that no
  longer exists. The same class applies its other `LayerNorm` directly, so
  these two call sites now do the same.
- `AttentionBase.get_local_attention` used `h_dec` without taking it as an
  argument, so local (predictive) attention could never run. It is now
  passed from `forward`.
- `EncoderDecoder.prepare_features` assigned `y` to a feature after the
  variable had been renamed to `seq`.
- `CharacterMap.convert` built a list comprehension over `codes` before
  `codes` was assigned.
- `Trainer.split_train_data` referenced an undefined `comm` (MPI) and
  formatted its message with a `.format` that sat inside the string
  literal. It was unreachable, and has been removed.
- `run_seq2seq --batch_size` selected a branch that called `Manager`,
  `Process`, `chainer` and `args.ideep`, none of which were defined or
  imported. The branch and the option have been removed.

Latent defects:

- The trainer created `record.tmp` only when `--logging` was *not* given,
  but always wrote its pid file there, so passing `--logging` crashed on
  startup.
- `load_eval_data` initialized the `criterion` column with the integer -1,
  making it an `int64` column that later rejected the float perplexities.
  Under pandas 3.0 this silently discarded every development-set score, so
  only the loss-based checkpoints were ever written and the per-sample
  report found nothing to show. The column is now a float column.
- The SentencePiece preprocessor wrote its scratch data to a hardcoded
  `TMP.txt` in the working directory (the `tempfile.NamedTemporaryFile`
  call next to it was commented out), so it left the file behind and two
  concurrent runs overwrote each other.
- `Trainer.test_model` called `idxmin` on a frame that is empty until the
  first criteria are computed.
- `train_seq2seq` compared a configuration string with `is` instead of
  `==`.
- `Trainer.format_state` used a regular expression written as a plain
  string with `\.`, which Python 3.12 warns about and a later version will
  reject.

Compatibility with current dependencies:

- `distutils.util.strtobool`, removed in Python 3.12, is replaced by
  `lpu_nn.common.args.strtobool`, which returns a real `bool`.
- `Module.to` unpacked the result of the private `torch._C._nn._parse_to`
  into three values; it has returned four since PyTorch 1.5.
- `pandas.read_csv` no longer accepts `sep` positionally.
- The optimizers used the deprecated `Tensor.add_(alpha, other)` and
  `Tensor.addcmul_(value, t1, t2)` overloads.
- Reported scalars are now detached before `float()`, which PyTorch warns
  about otherwise.
- `matplotlib` is an optional extra (`plot`); the trainer skips the plot
  with a debug message instead of logging a traceback every epoch.
