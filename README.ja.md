# LPU-NN

[LPU](https://github.com/akivajp/lpu) の上に構築された、PyTorch による
ニューラル言語処理モデル。

English version is available in [README.md](README.md).

## 現状

本パッケージは、2019-2020 年に既存論文の再現や手法の検証のために書かれた
非公開の研究コードを、公開可能な形に整備し直しているものです。移植と
近代化は段階的に進めており、API はまだ安定していません。

PyTorch 2.x / Python 3.13 上で現在一通り動作するのは以下です。

- 系列変換モデルの訓練 (`lpu-nn-train-seq2seq`)
- ビームサーチによる復号 (`lpu-nn-run-seq2seq`)
- 系列マッチングとランキング (`lpu-nn-train-match-ranker`,
  `lpu-nn-run-match-ranker`)。RE2 と Compare-Aggregate の 2 方式に対応
- BERT の事前学習 (`lpu-nn-train-bert`) と、分類
  (`lpu-nn-train-bert-classifier`)・ペアランキング
  (`lpu-nn-train-bert-ranker`) へのファインチューニング
- 系列タギング (`lpu-nn-train-tagger`)。BiLSTM / Transformer / BERT の
  符号化器と、線形 / CRF の復号器に対応
- 文字言語モデル (`lpu-nn-train-embedding`)、トークナイザコマンド
  (`lpu-nn-run-tokenizer`)、学習済み系列変換モデルの HTTP サーバ
  (`lpu-nn-serve-seq2seq`)

元コードは全て移植済みです。

## 動作要件

- Python 3.10 以降
- PyTorch 2.4 以降 (訓練には CUDA ビルドを推奨)

## インストール

```shell
$ pip install 'lpu-nn @ git+https://github.com/akivajp/lpu-nn.git'
```

開発用:

```shell
$ uv sync
```

訓練曲線の出力は、任意依存の `plot` エクストラを入れた場合のみ行われます。

```shell
$ pip install 'lpu-nn[plot] @ git+https://github.com/akivajp/lpu-nn.git'
```

## 使い方

訓練コマンドは作業ディレクトリと訓練コーパスを受け取ります。コーパスは
TSV ファイル (原言語と目的言語の 2 列)、または列ごとに分けたファイル群の
いずれでも構いません。

```shell
$ lpu-nn-train-seq2seq workdir train.tsv --dev-files dev.tsv --test-files test.tsv --gpu 0
```

SentencePiece のトークナイザを学習し、データセットを構築した上で、
各指標を更新するたびにチェックポイントのディレクトリ
(`record.best_dev_loss`, `record.best_dev_bleu` など) を書き出します。
各ディレクトリにはモデル、最適化器の状態、設定、スコアが収められます。

復号は標準入力から読み、標準出力へ書き出します。

```shell
$ lpu-nn-run-seq2seq workdir/record.best_dev_loss --gpu 0 < test.txt > hyp.txt
```

### 系列マッチングとランキング

マッチングランカーは 2 本の系列の組にスコアを与えます。コーパスは
2 本の系列と目標スコアの 3 列からなる TSV ファイルです。

```shell
$ lpu-nn-train-match-ranker workdir match-train.tsv --dev-files match-dev.tsv --gpu 0
```

`--match-pooler-type` でモデル構造を選びます。`re2`
([Yang+ 2019](https://aclanthology.org/P19-1465/)) または
`compare-aggregate` ([Wang & Jiang 2017](https://arxiv.org/abs/1611.01747))
です。`--loss-method` は目標値の使い方を選びます。`point` はスコアへの回帰、
`pair` はペアワイズのランキング損失、`classify` はラベル分布です。
チェックポイントはランキング指標ごとに
(`record.best_dev_mrr`, `record.best_dev_map` など) 書き出されます。

採点は標準入力から 1 行 1 組で読み込みます。

```shell
$ lpu-nn-run-match-ranker workdir/record.best_dev_mrr --gpu 0 < pairs.tsv
```

`--evaluate` を付けると、代わりに正解付きコーパスに対する MRR・MAP・
再現率@k を報告します。`--replies` は候補ファイル全体を各問い合わせに対して
順位付けします。

### BERT

事前学習は文ペアの TSV ファイルを受け取り、マスク言語モデルと
次文予測を同時に学習します。

```shell
$ lpu-nn-train-bert workdir train.tsv --dev-files dev.tsv --gpu 0
```

`--universal` は固定段数の代わりに Universal Transformer
(適応的な段数と ponder cost を持つ) を用います。`--num-token-types 2`
を指定すると、ペアの左右を区別するセグメント埋め込みが有効になります。

ファインチューニングは事前学習済みチェックポイントから始めます。
分類器は「文 + ラベル」、ランカーは「文ペア」の TSV を受け取ります。

```shell
$ lpu-nn-train-bert-classifier workdir class-train.tsv --dev-files class-dev.tsv \
    --pre-trained-model bert-workdir/record.best_dev_loss \
    --sentencepiece bert-workdir/sp.model --gpu 0
$ lpu-nn-run-bert-classifier workdir/record.best_dev_acc < sentences.txt
```

`--pre-trained-model` には `--sentencepiece` を併せて指定します。
作業ディレクトリごとにトークナイザを学習する一方、ファインチューニングは
事前学習済みの埋め込みをそのまま引き継ぐため、語彙は同一である必要が
あります。食い違う場合は、読み直せないチェックポイントを書き出す前に
起動を拒否します。

採点コマンドは 1 行 1 件で予測ラベルを書き出します。`--ranking` を付けると
`文<TAB>ラベル` を読み込み、既知ラベル全体に対する MRR と
精度@k を報告します。ペアランカー用の `lpu-nn-run-bert-ranker` は
`文1|||文2` を読んでスコアを 1 行 1 件で出力し、`--replies` を付けると
候補ファイル全体を各問い合わせに対して順位付けします。

### 系列タギング

タガーは「文 + トークン毎のタグ」の TSV ファイルを受け取ります。
タグは BIO 方式 (`O`, `B-ラベル`, `I-ラベル`) です。

```shell
$ lpu-nn-train-tagger workdir tag-train.tsv --dev-files tag-dev.tsv --gpu 0
```

`--encoder-type` で `lstm` (既定で双方向)・`transformer`・`bert` を、
`--decoder-type` で `linear`・`crf` を選びます。`bert` を使う場合は、
他のファインチューニングコマンドと同様に `--pre-trained-model` と
`--sentencepiece` を指定します。評価のたびにタグ付けした開発セットを
`record.latest/pred_dev.txt` へ書き出し、固有表現の適合率・再現率・F1 を
(ラベル一致あり / 境界のみ の 2 通りで) 報告します。

### 学習の再開

`--resume latest` で、作業ディレクトリのチェックポイントから学習を再開
します。モデルは保存時の設定から組み直したうえで重みを読み込むため、
構造を決める設定 (`--embed-size`, `--hidden-size`, `--num-layers` など) は
チェックポイントの値を保ちます。異なる値を渡した場合は、重みの読み込みに
失敗する代わりに、無視した指定を報告します。

それ以外はコマンドラインに従います。これは継続学習に必要な挙動です。
コーパス、`--num-epochs`、`--batch-size`、`--optimizer`、
`--learning-rate`、`--dropout-ratio` などの学習側の設定は、再開時に
差し替えられます。

```shell
$ lpu-nn-train-seq2seq workdir more-data.tsv --resume latest \
    --num-epochs 20 --batch-size 64 --optimizer adam
```

`--override-model-params` を付けると、この制限を外せます。`--max-length`
を広げるなど、安全に変更できる場合に使います。重みの形が変わる変更は
やはり読み込めず、その旨が表示されます。

なお、到達済みのエポック数を超えて `--num-epochs` を増やさずに再開しても
何も起きません。実行すべきエポックが残っておらず、チェックポイントも
書き出されないためです。

### 言語モデルとトークナイザ

言語モデルは 1 行 1 文のプレーンテキストで学習し、両方向の次トークンを
予測します。

```shell
$ lpu-nn-train-embedding workdir corpus.txt --dev-files dev.txt --gpu 0
```

`lpu-nn-run-tokenizer` は、これらのコマンドが学習した SentencePiece
モデルを標準入力に適用します。

```shell
$ lpu-nn-run-tokenizer workdir/sp.model < text.txt
$ lpu-nn-run-tokenizer workdir/sp.model --format id < text.txt
```

### 系列変換モデルのサーバ

```shell
$ pip install 'lpu-nn[serve]'
$ lpu-nn-serve-seq2seq ja-en=workdir/record.best_dev_bleu --port 8000
```

`GET /` は動作確認用のページ、`/api/models` は登録したモデル名、
`/api/decode` は復号結果を JSON で返します。`名前=パス` の組を複数
指定すれば、同時に複数のモデルを提供できます。

待ち受けは `--host` を指定しない限り `127.0.0.1` のみです。また bottle の
デバッグモードは使いません (例外の内容を要求元へ返してしまうため)。

いずれのコマンドも `--help` で全オプションを確認できます。

## 構成

| モジュール | 内容 |
| --- | --- |
| `lpu_nn.common` | 訓練ループ、データセット、語彙、評価基準 |
| `lpu_nn.modeling` | Transformer, Universal Transformer, LSTM, 注意機構, 埋め込み, RE2, Compare-Aggregate, BERT, CRF |
| `lpu_nn.optimizers` | AdaBound, LAMB と、訓練で用いる torch の最適化器 |
| `lpu_nn.commands` | コマンドラインのエントリポイント |

設定・ロギング・進捗表示・ファイル操作は `lpu` が提供するものを用いており、
本パッケージでは重複して実装していません。

## ライセンス

MIT。ただし同梱しているサードパーティの最適化器は除きます。
[LICENSE](LICENSE) と [licenses/NOTICE.md](licenses/NOTICE.md) を参照。
