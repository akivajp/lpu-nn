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

The SentencePiece pre-processor never flushed the scratch file it hands
to the trainer. SentencePiece is given `temp.name` and reads it from
disk, so whatever remained in the buffer was simply not there: for a
small corpus the file was 0 bytes and training failed with an internal
error from inside SentencePiece, and for a large one it was truncated at
the last 8 KiB boundary. Every tokenizer this codebase has trained was
therefore trained on less text than it was given. The file is flushed
now, and an empty result says which knob to turn instead of surfacing a
message from inside the library.

This changes the vocabulary a given corpus produces, and with it the loss
a training run reports, because the tokenizer now sees the whole corpus.

`AdaBoundW` lost its optimizer state whenever that state and the
parameters sat on different devices, which is what resuming from a
checkpoint saved on the CPU produces. It moved the moments with
`.to(p.device)`, and `.to()` returns a copy when it actually moves, so
every in-place update landed on a copy that was dropped at the end of the
step: the moments stayed frozen at whatever they were, and the optimizer
effectively restarted on every step while reporting nothing.

Both AdaBound variants raised `ZeroDivisionError` when the learning rate
reached zero: the dynamic bound divides the current rate by the rate the
optimizer was constructed with, to let a scheduler decay `final_lr` too.
A warmup schedule starting from zero hits this on its first step. A zero
base rate now means a zero effective rate, so the parameters stay put.

`AdaBound` also still used the deprecated `Tensor.add(alpha, other)`
overload in its weight decay path, which the earlier sweep missed.

`universal_transformer` could not be imported on its own: `transformer`
imports it at module level and it imports `transformer` back, so whichever
came first raised `ImportError`. It worked only because every other path
happened to reach `transformer` first. The back edge is a local import at
its two use sites now.

`UniversalTransformer` never initialized `last_state`, so a forward pass
on a freshly constructed module raised `AttributeError` unless
`reset_state()` had been called by hand. Its `recurrence='basic'` branch,
one of the four the command line offers, called `self.transform`, which
does not exist, and lacked the per-step state handling the ACT branch has;
without it the transformer accumulates its inputs across steps and the
attention mask stops matching on the second one. `max_ponder` stored the
`(values, indices)` pair that `torch.max` returns rather than the step
counts, and `init_weights` replaced the halting bias with a fresh CPU
float32 tensor instead of filling the existing one.

`ContextualStringEmbedding` could not be constructed: it assigned its
submodules without calling `nn.Module.__init__` first, which PyTorch
refuses. Past that, `forward` concatenated along `dim=3` tensors that have
three dimensions, and `prepare_batch` reached for `self.weight` and
`self.str2tensor`, neither of which the class has. That last body belongs
to `CharacterEmbedding`, whose own `prepare_batch` was a bare `pass`
returning `None`; it has been moved there, and the case where no branch
assigned the batch now raises instead of falling through.

`DotAttention` merged a bidirectional memory with
`memory.split(2, dim=2)`, taking adjacent pairs of features. The encoder
concatenates the two directions in blocks, `cat([forward, backward])`, so
that averaged each direction with itself and never paired the two
together. It now splits the memory in half and averages dimension by
dimension. The default attention is `mlp`, so this affected
`--attention dot` with an LSTM encoder.

The attention layers read `hidden_size` with `params.get` and passed the
result straight to `nn.Linear`, so a missing value arrived as
`None * 3`. The MLP attention did the same with its activation. Both say
what is missing now.

`CharacterEmbedding.tensor2bytes` asserted on `tensor.ndim()`; `ndim` is a
property, so the call raised `TypeError`.

`activation.Swish1` called `super(Swish, self).__init__()`, naming a class
it does not derive from, so `get_activator('swish1')` raised a `TypeError`
every time. Nothing could have used it.

`SequenceConvolution1d` built its padding on `self.device`, an attribute
that only exists once `to()` has been called, so a freshly constructed
module raised `AttributeError` for any n-gram order above 1; and the
padding stayed float32, so it could not be concatenated with a float16
sequence. Both now follow the input tensor. Its `init_weights` also
assumed a bias, which `nn.Conv1d` does not guarantee.

`to()` and `__repr__` were shared by borrowing the unbound methods of
`Module` from classes that do not derive from it, which works only
because Python does not check the receiver. They are plain functions now,
`modeling.apply_to` and `modeling.format_module`, and `format_module`
skips the `None` entries torch can leave in `_modules`.

`lpu_nn.common.vocab` carried its own `IDMap` and `LabelMap`, copied from
an older lpu and since diverged, plus a `CharacterMap` that nothing used
and that could not have run: its `decode`, `sample` and `__iter__`
referred to attributes its constructor never set. The tested
implementations now come from `lpu.common.vocab`, and `CharacterMap` is
gone; 380 lines of duplicated code went with them. `FieldMap` keeps the
state handling those maps used to provide.

`Vocabulary` decided whether it had been initialized by asking
`hasattr(self, 'symbols')`, so its symbol ids existed only after
`set_symbols`. They are declared up front now, with -1 for undefined, as
SentencePiece itself reports it.

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
