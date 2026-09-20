# Changelog

日本語版は [CHANGELOG.ja.md](CHANGELOG.ja.md) にあります。

## Unreleased

### Changed

- `lpu` is required as `lpu>=0.6` rather than from its repository, now
  that the release carrying `IDMap`, the `safe_*` file helpers and
  `lpu.metrics.ranking` is on PyPI. A direct URL dependency is also what
  PyPI refuses, so this is what makes this package publishable.

### Added

- The BERT path of the same codebase: pre-training with a masked language
  model and next-sentence prediction, and fine-tuning for classification and
  for pair ranking, as `lpu-nn-train-bert`,
  `lpu-nn-train-bert-classifier` / `lpu-nn-run-bert-classifier` and
  `lpu-nn-train-bert-ranker` / `lpu-nn-run-bert-ranker`.
- `train_tokenizer` takes `user_defined_symbols`, and `FieldMap.train`
  passes the trainer's extra symbols to it. Without this the symbols a task
  declares (BERT's `<cls>` / `<sep>` / `<mask>`) never enter the
  SentencePiece vocabulary, and building the vocabulary fails.
- A CI smoke test that pre-trains BERT for one epoch, fine-tunes a
  classifier from that checkpoint and classifies with the result.
- The sequence matching and ranking path of the same codebase: the RE2
  ([Yang et al., 2019](https://aclanthology.org/P19-1465/)) and
  Compare-Aggregate ([Wang and Jiang, 2017](https://arxiv.org/abs/1611.01747))
  poolers, the sequence pooling functions they share, the `SequenceMatcher`
  head, and the `lpu-nn-train-match-ranker` / `lpu-nn-run-match-ranker`
  commands. It trains with a point-wise, a pair-wise or a classification
  loss, and reports MRR, MAP, mean rank and recall at k.
- A CI smoke test that trains the match ranker for one epoch and scores the
  development set with the checkpoint it wrote.
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

`--pre-trained-model` replaced the whole `mod_bert` without checking that
the two vocabularies match. Each work directory trains its own tokenizer, so
they normally do not, and the fine-tuned checkpoint was written with an
embedding that disagreed with its own recorded vocabulary size: loading it
back raised a size mismatch. Fine-tuning now refuses to start unless
`--sentencepiece` points at the pre-trained tokenizer.

`create_parser` in both fine-tuning commands was a `staticmethod` taking
only the model name, while the base and `train_bert` are classmethods that
also take the defaults. `main` builds the parser twice, and the second call,
which is what prints `--help`, passes the defaults, so neither command could
print its help.

`modeling/bert_classifier.py` and `modeling/bert_ranker.py` imported
`modules`, the name `modeling` was renamed from. Neither could be imported
since that rename, so the classifier and the ranker were unreachable
together with the two trainers and the two scorers that build on them.

`Bert` built its token and segment embeddings as plain `nn.Embedding` with
`padding_idx` set to the vocabulary's pad id. SentencePiece has no pad token
and reports `-1`, so the padded positions of every batch indexed the
embedding with `-1` and training stopped at "index out of range in self".
Both now use `modeling.embeddings.Embedding`, which is written for exactly
this and substitutes a valid index before the lookup.

`Bert` looked for `UniversalTransformer` in `modeling.transformer`, but it
lives in `modeling.universal_transformer` since the split, so `--universal`
raised `AttributeError` at construction.

`BertRanker.forward` passed the segment information as `segment_id_seq`,
while `Bert.forward` reads `segment_info`. The value was swallowed by
`**features` and silently discarded, so the ranker never told the model
which of the two sequences each token belonged to.

`BertClassifier.__init__` took its first argument as `vocab` and read
`vocab.pad` from it, but what it receives is the field map. Constructing it
raised `AttributeError`.

`feed_one_batch` and `evaluate` in all three BERT trainers had the
signatures the base trainer used before it gained `df`, `feed_batches` and
`report`. The resulting `TypeError` was caught by the training loop's
`except`, so a run reported success while learning nothing.

`BertClassifierTrainer` and `BertRankerTrainer` overrode `__init__` without
the `args` parameter, which `main` passes, so neither could be constructed.

Both fine-tuning commands registered `--pre` and `-P` as aliases of
`--pre-trained-model`, although the base parser already uses them for
`--preset`. `argparse` rejects the conflict while building the parser, so
neither command could even print `--help`.

`Trainer.save_labels` called `self.vocab.decode(label)`. `load_labels` had
moved to keeping labels as plain strings, and `self.vocab` no longer exists
at all, so every task with labels failed at setup.

The classifier and the ranker still read `self.vocab` in four places, a
name that was replaced by `self.idmaps['seq']`.

`train_bert` put a tensor into the report for `rest_acc` while every other
field is converted with `float()`, so pandas refused to average the report
and the autograd graph was retained for the whole epoch.

`train_bert_classifier` assigned a list of predictions with `df.at`, which
takes a single label, not an index array.

`training.infomain` and `training.comm_main` are leftovers of the
multi-process support that was dropped on the port, and neither exists.
They were reached by the fine-tuning commands and by `--schedule-num-steps`.
The same branch then read `self.model.max_steps`, which `set_max_steps`
leaves unset unless the configuration carries `model.max_steps`.

The `--replies` branch of `run_bert_ranker` called `Trainer.load_status`
unbound on the base class, passed it a `model_path` it no longer takes, and
encoded the candidates with `sent2idvec`, a method that no longer exists.
The scoring loop referenced `xp` and `F`, chainer names that were never
imported, and the classifier's loop wrote its predictions with `dprint`, so
nothing reached the standard output even had it run.

`run_bert_classifier --ranking` encoded the correct label into token ids
before comparing it against the label strings, so no rank was ever found and
the metrics divided by zero.

Both scorers assigned `logging.using_config(...)` to a variable instead of
entering it, so `--debug` did nothing, and targeted the logger `logger`
rather than the package names.

The four ranking metrics these commands duplicated now come from
`lpu.metrics.ranking`.

`Fusion.forward` called `self.mod_direct` for all three of its views, so
`mod_sub` and `mod_mult` were built, counted among the parameters and never
reached by a gradient. RE2's augmented fusion had collapsed onto a single
projection: the sub and mult features went through the same weights as the
direct one.

`SequenceAttentionPooling` wrapped `torch.Tensor(vector_size)`, which
returns uninitialized memory, and its `init_weights` was commented out. The
query vector therefore started as whatever the allocator handed back, in
practice NaN. A one-dimensional parameter does not reach the generic
`apply_init_weights` branch either, which only initializes `weight.dim() >
1`, so nothing rescued it later. Attention is the default sequence pooling
for every preset but the reference one, so the default configuration
produced a NaN loss on the first step.

`NGramPooler.forward` fed its `Conv2d` an input shorter than the kernel
whenever a sentence was shorter than the n-gram order. With the default
`ngram_orders` of `[1, 2, 3, 4, 5]`, any corpus containing a sentence of
fewer than five tokens could not be trained at all. The input is now padded
on the right, and the output is realigned to the input length rather than
by a fixed `n - 1`.

`SequenceMatcher.get_config` set `num_classes` only when `idmaps` carried a
label map, so constructing the model for regression raised `KeyError:
'num_classes'`. It also fell through silently for an unknown
`match_pooler_type`, leaving every default unset, so the error surfaced as
`KeyError: 'dropout_ratio'` rather than naming the pooler.

`RE2Pooler.get_config` resolved the `pooling` alias after the preset
defaults had already filled `sequence_pooling`, so the alias never took
effect. It is resolved first now.

`score` in the scorer compared `timeout > elapsed` and broke out of the
loop when it was true, which is the case on the first batch. Passing
`--eval-timeout` therefore scored nothing at all rather than stopping at
the limit.

The `--replies` branch of `eval_ranker` called `rank` with the signature of
an older version, unpacking three values from what is a dict, and passed it
id vectors, which cannot be used as dict keys. Every line raised and was
swallowed by the enclosing `except`, leaving the rank list empty for a
`ZeroDivisionError` in the metrics below. It ranks the candidate file
against each query now, and the two metric helpers it duplicated are taken
from `lpu.metrics.ranking`.

The scorer's `main` targeted the loggers `common` and `models`, names from
before the port, so `--debug` and `--logging` produced no output from
`lpu_nn` or `lpu`.

Ranked replies were ordered out of a `set`, so replies with equal scores
came out in an order that varied between runs and the reported metrics
varied with them. Ties break on the reply text now.

Scoring built an autograd graph it never used, for every batch of every
candidate.

`mean` divided by the length of a list that is empty when no query has a
correct answer.

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

`Trainer.load_labels` read `self.main_train_data_path`, a name that was
renamed to `train_data_path` and left behind here, so calling it without
an explicit path raised `AttributeError`. mypy found this one.

The debug branch of `main` ran a loop whose body fetched a logger and
discarded it, doing nothing; the line above it already configures the
same loggers. The module ended with `if __name__ == '__main__': main()`,
although `main` takes a trainer class and a model name, so running the
module directly could only raise `TypeError`. The entry points live in
`lpu_nn.commands`.

`Trainer.train_epoch` compared `train_report.get('loss')` against the
previous loss while checking only the previous one for `None`, so a
report without a loss compared `None` with a number.

The progress report spelled `amsbound` as `amdbound`, so the dynamic
lower bound on the learning rate was left out of the line for
`--optimizer amsbound`. The rate it printed was the raw one, which on a
warmup step is far below what the optimizer actually applies: 0.00000389
where the effective rate is 0.00006296.

Two of its fields read a missing metric as the string `'nan'` and handed
it to a numeric format, so the field vanished from the line rather than
reading `nan`. Its `except` clause was a bare `pass`, which is what let
that go unnoticed; it logs at debug level now. The token rate divided by
an interval that can be zero.

`--eval-train` was folded into the saved configuration while every other
`store_true` flag is left out of it. The default of such a flag is
`False`, not `None`, so an unspecified flag reads as an explicit `False`
and overwrites whatever the saved configuration held. It is a choice for
one run, like `--eval-only` next to it, and is ignored the same way now.

`Trainer.train_epoch` read the host name with `os.uname`, which does not
exist on Windows. `platform.node` returns the same thing everywhere.

Two statements in the trainer evaluated a value and discarded it, left
behind when the earlier unused-variable pass removed their assignments.

The out-of-memory recovery in `Trainer.feed_batches` recognized the
error only by the first 18 characters of its message. PyTorch raises
`torch.cuda.OutOfMemoryError`, a `RuntimeError` subclass, so the type is
checked first now; a change of wording would otherwise have stopped the
batch from shrinking, and the run would simply have died where it used
to recover.

That recovery path, and the keyboard-interrupt cleanup, both called
`torch.cuda.reset_max_memory_cached`, which is an alias for
`reset_peak_memory_stats` that warns, right next to
`reset_max_memory_allocated`, which is the same alias again. One call
replaces the pair.

`Trainer.update_parameters` assigned the gradient norm into the report
without checking it, although its signature accepts `report=None` and
guards the other assignment.

It also read `opt.weight_decay_rate`, a name from the Chainer
implementation that no torch optimizer carries; the value lives in
`param_groups`. I could not construct a run that reaches that line, so
this is a latent defect rather than one with a known trigger.

`Trainer.try_loading` re-raised on the first record that failed to load,
and the line recording the failure sat unreachable behind that raise, so
only the first entry of `--resume` was ever tried. `--resume` takes a
list precisely so a later record can stand in for one that does not load.
It now tries each in turn and raises only when none of them loads.

`Trainer.load_status` assigned `record_dir` inside the `load_optimizer`
branch and then used it for `reuse_dataset`, so asking to reuse the
dataset without loading the optimizer raised `UnboundLocalError`.
`Trainer.load_model` referenced `idmaps` whether or not the checkpoint
carried it, with the same result; it says what the checkpoint is missing
now.

`set_logfile_handler` removed the previous handler without closing it,
and closing a `StreamHandler` does not close the stream it was given. It
builds a `FileHandler` now, which owns its file.

Several writers used the platform default encoding, which corrupts
non-ASCII content on Windows: the configuration, the labels, the scores
and the dataset append.

`build_batches` and `reduce_batch_size` fell off the end and returned
`None` for a batch type they did not recognize, which the caller then
tried to iterate or index. They name the type they were given now.

`set_logfile_handler` removed the previous handler without closing it or
the file behind it.

The logger names the trainer configures, `target_loggers`, were the
top-level module names from before the port: `common`, `modeling`,
`optimizers`. Logger names are dotted, so a name only reaches the loggers
beneath it, and `common` is not an ancestor of `lpu_nn.common.training`.
Neither the `--logging` file handler nor `--debug` reached any module of
this package: the log file held the `__main__` lines and nothing else,
92 of the 300 lines it should have had. `run_seq2seq` had the same list
with `models`, a name that predates even `modeling`. Both name `lpu_nn`
and `lpu` now.

`LSTMDecoder` never initialized `last_state`, so a forward pass on a
freshly built encoder-decoder raised `AttributeError` unless
`reset_state()` had been called by hand. It also could not be constructed
on its own: `memory_size` and `share_embedding` were set only by
`EncoderDecoder.get_config`, so `LSTMDecoder.get_config` left its
`__init__` to raise `KeyError`. Its `reset_state` returned `None` where
the sibling classes return `self`, and `set_state` neither accepted
`None` nor tolerated a state without an `rnn_state` entry.

`LSTMDecoder.decode_one` called `self.prepare_features(seq_enc=seq, ...)`
where the parameter is named `seq`, so the keyword landed in `**features`
and the branch that records the token ids never ran. The feature it would
have set, `id_seq`, is written in four places and read in none, so
nothing depended on it; the call says what it means now.

Its `__init__` also read `vocab.pad` into `self.padding` and overwrote it
with `params['padding']` on the next line.

`MultiStepTransformer` built a `LayerNorm` for its input or its output
when the sublayer pre- or post-processing carried no `'n'`, then guarded
the call with `hasattr(self, 'normalize_input')` and
`hasattr(self, 'normalize_output')` — names that are never assigned
anywhere, so the module was created and never applied.
`UniversalTransformer` asks for the module itself, which is what
`__init__` decides on. Neither setting is reachable from the command line
today, so no model trained here was affected; it matters the moment
anyone switches to the alternative arrangement the comments in
`ModuleConnection.get_parameters` recommend.

`EmbedPosition.get_config` wrote `params.setdefault('embed_size', ...)`
twice where the second line should have derived `hidden_size`, as every
sibling `get_config` does. Calling it without an explicit `hidden_size`
raised `KeyError`.

`MultiStepTransformer.set_state` returned `None` while `reset_state`
returned `self`.

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
