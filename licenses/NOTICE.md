# Third-party code bundled in this distribution

本配布物に同梱しているサードパーティのコードと、そのライセンス。

`lpu_nn` 自体は MIT ライセンス (ルートの [LICENSE](../LICENSE)) ですが、
以下のファイルは上流の実装に由来し、それぞれのライセンス条件に従います。

## `lpu_nn/optimizers/adabound.py`

- 由来: [Luolc/AdaBound](https://github.com/Luolc/AdaBound) `adabound/adabound.py`
- 取得コミット: `d5bb5ffcea39f5733d642567e7fc55cb35ef6c82`
- ライセンス: Apache License 2.0 — 全文は
  [AdaBound-LICENSE.txt](AdaBound-LICENSE.txt)
- 著作権: the AdaBound authors (Luolc)
- 論文: Luo et al., *Adaptive Gradient Methods with Dynamic Bound of Learning
  Rate*, ICLR 2019
- 変更点:
  - 最適化器の内部状態 (`exp_avg` 等) を float32 に保ち、勾配とパラメータが
    float16 の場合でも動作するようにした
  - 内部状態を毎ステップ、パラメータと同じデバイスへ移すようにした
  - `final_lr=None` を許可し、その場合は AdaBound の動的な学習率の
    上下限によるクランプを行わない (この分岐では実質 Adam として動作する)
  - 非推奨となった `Tensor.add_` / `Tensor.addcmul_` の位置引数形式を
    現行形式へ書き換えた
  - ロギングを `lpu.common.logging` 経由に差し替え、`lpu_nn` の
    パッケージ構成に合わせて import を変更

## `lpu_nn/optimizers/lamb.py`

- 由来: [cybertronai/pytorch-lamb](https://github.com/cybertronai/pytorch-lamb)
  `pytorch_lamb/lamb.py`
- 取得コミット: `ff2245eaa458278b096e682a66c29a2d73f690d7`
- ライセンス: MIT License — 全文は
  [pytorch-lamb-LICENSE.txt](pytorch-lamb-LICENSE.txt)
- 著作権: Copyright (c) 2019 cybertronai
- 論文: You et al., *Large Batch Optimization for Deep Learning: Training BERT
  in 76 minutes*, ICLR 2020
- 変更点:
  - 上流が TensorBoard への出力に用いていた `tensorboardX` への依存と、
    それを使う `log_lamb_rs()` を除去した
  - 最適化器の内部状態を float32 に保ち、パラメータが float16 の場合でも
    動作するようにした
  - 非推奨となった `Tensor.add_(alpha, other)` / `Tensor.addcmul_(value, t1, t2)`
    の位置引数形式を、現行の `add_(other, alpha=...)` /
    `addcmul_(t1, t2, value=...)` へ書き換えた
  - ロギングを `lpu.common.logging` 経由に差し替え、`lpu_nn` の
    パッケージ構成に合わせて import を変更
