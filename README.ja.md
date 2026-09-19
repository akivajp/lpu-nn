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

元コードに含まれる BERT、系列タギング、系列マッチング、言語モデリングの
各部分は未移植です。

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

いずれのコマンドも `--help` で全オプションを確認できます。

## 構成

| モジュール | 内容 |
| --- | --- |
| `lpu_nn.common` | 訓練ループ、データセット、語彙、評価基準 |
| `lpu_nn.modeling` | Transformer, Universal Transformer, LSTM, 注意機構, 埋め込み |
| `lpu_nn.optimizers` | AdaBound, LAMB と、訓練で用いる torch の最適化器 |
| `lpu_nn.commands` | コマンドラインのエントリポイント |

設定・ロギング・進捗表示・ファイル操作は `lpu` が提供するものを用いており、
本パッケージでは重複して実装していません。

## ライセンス

MIT。ただし同梱しているサードパーティの最適化器は除きます。
[LICENSE](LICENSE) と [licenses/NOTICE.md](licenses/NOTICE.md) を参照。
