# 変更履歴

English version is available in [CHANGELOG.md](CHANGELOG.md).

## 未リリース

### 変更

- `lpu` への依存を、リポジトリの直接指定から `lpu>=0.6` に変更しました。
  `IDMap`、`safe_*` 系のファイル補助、`lpu.metrics.ranking` を含む
  リリースが PyPI に出たためです。PyPI は直接 URL 依存を受け付けないため、
  この変更が本パッケージを公開可能にする条件でもあります。

### 追加

- 同じコードベースの BERT 部分。マスク言語モデルと次文予測による
  事前学習と、分類・ペアランキングへのファインチューニングを、
  `lpu-nn-train-bert`、`lpu-nn-train-bert-classifier` /
  `lpu-nn-run-bert-classifier`、`lpu-nn-train-bert-ranker` /
  `lpu-nn-run-bert-ranker` として提供する。
- `train_tokenizer` に `user_defined_symbols` を追加し、
  `FieldMap.train` がトレーナーの特殊記号を渡すようにした。これが無いと
  タスクが宣言した記号 (BERT の `<cls>` / `<sep>` / `<mask>`) が
  SentencePiece の語彙に入らず、語彙の構築に失敗する。
- BERT を 1 エポック事前学習し、そのチェックポイントから分類器を
  ファインチューニングして分類まで通す CI スモークテスト。
- 同じコードベースの系列マッチング・ランキング部分。RE2
  ([Yang+ 2019](https://aclanthology.org/P19-1465/)) と Compare-Aggregate
  ([Wang & Jiang 2017](https://arxiv.org/abs/1611.01747)) の 2 方式、
  両者が共有する系列プーリング、`SequenceMatcher` の出力層、そして
  `lpu-nn-train-match-ranker` / `lpu-nn-run-match-ranker` コマンド。
  ポイントワイズ・ペアワイズ・分類の 3 つの損失で訓練でき、
  MRR・MAP・平均順位・再現率@k を報告する。
- マッチングランカーを 1 エポック訓練し、書き出したチェックポイントで
  開発セットを採点する CI スモークテスト。
- 2019-2020 年に書かれた非公開の研究コードのうち、系列変換 (seq2seq) 経路を
  PyTorch 2.x / Python 3.13 へ移植しました。訓練ループ、データセット、
  SentencePiece 語彙、Transformer / Universal Transformer / LSTM の各モジュール、
  注意機構、AdaBound と LAMB の最適化器、および
  `lpu-nn-train-seq2seq` / `lpu-nn-run-seq2seq` コマンドが対象です。
- 設定・ロギング・進捗表示・色付け・対話・ファイル操作は移植せず `lpu` の
  ものを用います。これにより約 1,900 行の重複コードが不要になりました。

### 修正

`--pre-trained-model` が語彙の一致を確認せずに `mod_bert` をまるごと
差し替えていた。作業ディレクトリごとにトークナイザを学習するため通常は
一致せず、ファインチューニング後のチェックポイントは、自身が記録した
語彙サイズと食い違う埋め込みを持って書き出されていた。読み直すと
サイズ不一致で失敗する。`--sentencepiece` で事前学習側のトークナイザを
指していない限り、起動を拒否するようにした。

ファインチューニング 2 コマンドの `create_parser` が、モデル名のみを
取る `staticmethod` だった。基底と `train_bert` は既定値も受け取る
classmethod である。`main` はパーサを 2 回構築し、`--help` を表示する
2 回目は既定値を渡すため、どちらのコマンドもヘルプを表示できなかった。

`modeling/bert_classifier.py` と `modeling/bert_ranker.py` が、
`modeling` の旧称である `modules` を import していた。改名以降
どちらも import できず、分類器とランカーは、それらの上に立つ
2 つのトレーナーと 2 つの採点コマンドごと到達不能だった。

`Bert` はトークン埋め込みとセグメント埋め込みを生の `nn.Embedding` で
構築し、`padding_idx` に語彙のパディング ID を渡していた。
SentencePiece はパディング記号を持たず `-1` を返すため、
全バッチのパディング位置が埋め込みを `-1` で参照し、
"index out of range in self" で学習が止まっていた。どちらも
まさにこの状況のために書かれた `modeling.embeddings.Embedding`
(参照前に有効な添字へ差し替える) を使うようにした。

`Bert` が `UniversalTransformer` を `modeling.transformer` から
引いていたが、分割以降 `modeling.universal_transformer` にあるため、
`--universal` は構築時に `AttributeError` になっていた。

`BertRanker.forward` はセグメント情報を `segment_id_seq` という名前で
渡していたが、`Bert.forward` が読むのは `segment_info` である。
値は `**features` に吸い込まれて黙って捨てられており、ランカーは
各トークンが 2 系列のどちらに属するかをモデルに伝えていなかった。

`BertClassifier.__init__` は第 1 引数を `vocab` として受け取り
`vocab.pad` を引いていたが、実際に渡るのはフィールド対応表であり、
構築時に `AttributeError` になっていた。

BERT の 3 つのトレーナーの `feed_one_batch` と `evaluate` が、
基底トレーナーに `df` / `feed_batches` / `report` が加わる前の署名の
ままだった。生じる `TypeError` は訓練ループの `except` に捕まるため、
実行は成功と報告されながら 1 件も学習していなかった。

`BertClassifierTrainer` と `BertRankerTrainer` が、`main` が渡す
`args` を受け取らない `__init__` を上書きしており、構築できなかった。

ファインチューニングの 2 コマンドが `--pre` と `-P` を
`--pre-trained-model` の別名に登録していたが、基底パーサが既に
`--preset` で使っている。`argparse` はパーサ構築時に衝突を拒否するため、
どちらのコマンドも `--help` すら出せなかった。

`Trainer.save_labels` が `self.vocab.decode(label)` を呼んでいた。
`load_labels` はラベルを文字列として保持するよう変更されており、
`self.vocab` 自体も存在しないため、ラベルを扱うタスクは
すべて初期化時に失敗していた。

分類器とランカーが、`self.idmaps['seq']` に置き換えられた
`self.vocab` を 4 箇所で参照したままだった。

`train_bert` が `rest_acc` だけテンソルのまま report に入れていた
(他の項目はすべて `float()` を通している) ため、pandas が report の
平均を取れず、さらに自動微分グラフがエポック中保持されていた。

`train_bert_classifier` が予測のリストを `df.at` で代入していた。
`.at` は単一ラベル専用であり、索引の配列は受け取れない。

`training.infomain` と `training.comm_main` は移植時に落とされた
複数プロセス対応の残骸で、どちらも存在しない。前者は
ファインチューニングの 2 コマンドから、後者は `--schedule-num-steps`
から到達していた。同じ分岐はさらに `self.model.max_steps` を読むが、
設定に `model.max_steps` が無ければ `set_max_steps` は代入しない。

`run_bert_ranker` の `--replies` 分岐は、基底クラスの
`Trainer.load_status` を未束縛で呼び、既に受け取らなくなった
`model_path` を渡し、候補を存在しない `sent2idvec` で符号化していた。
採点ループは import されていない chainer 由来の `xp` と `F` を参照し、
分類器側のループは予測を `dprint` で書いていたため、仮に動いたとしても
標準出力には何も出なかった。

`run_bert_classifier --ranking` は、正解ラベルをラベル文字列と
突き合わせる前にトークン ID へ符号化していた。順位が 1 件も得られず、
指標計算がゼロ除算になっていた。

採点コマンド 2 つが `logging.using_config(...)` を変数に代入するだけで
文脈に入れておらず `--debug` が効かなかった。対象もパッケージ名ではなく
`logger` オブジェクトを渡していた。

これらのコマンドが重複実装していた 4 つのランキング指標は
`lpu.metrics.ranking` から取るようにした。

`Fusion.forward` が 3 つの特徴すべてに `self.mod_direct` を使っており、
`mod_sub` と `mod_mult` は構築されパラメータとして数えられながら、
勾配が一度も届いていなかった。RE2 の augmented fusion が単一射影に
退化しており、差と積の特徴が直和の特徴と同じ重みを通っていた。

`SequenceAttentionPooling` が未初期化メモリを返す
`torch.Tensor(vector_size)` をそのままパラメータにしており、
`init_weights` はコメントアウトされていた。問い合わせベクトルは
アロケータが返した値のまま、実際には NaN で始まっていた。1 次元の
パラメータは `apply_init_weights` の汎用処理 (`weight.dim() > 1`) にも
掛からないため、後から救われることもなかった。参照プリセット以外では
注意プーリングが既定なので、既定設定は 1 ステップ目で損失が NaN に
なっていた。

`NGramPooler.forward` は、文が n-gram 長より短いときに畳み込みへ
カーネルより小さい入力を渡していた。既定の `ngram_orders` は
`[1, 2, 3, 4, 5]` なので、5 語未満の文を含むコーパスは訓練を開始
できなかった。入力を右側で埋め、出力長は固定の `n - 1` ではなく
入力長に合わせて揃えるようにした。

`SequenceMatcher.get_config` は `idmaps` にラベル表がある場合のみ
`num_classes` を設定しており、回帰用に構築すると
`KeyError: 'num_classes'` になっていた。未知の `match_pooler_type` でも
黙って素通りするため既定値が一切入らず、種別ではなく
`KeyError: 'dropout_ratio'` として表面化していた。

`RE2Pooler.get_config` は、プリセットの既定値が `sequence_pooling` を
埋めた後に `pooling` 別名を解決していたため、別名が効かなかった。
先に解決するようにした。

採点側の `score` は `timeout > 経過時間` で打ち切っており、これは
1 バッチ目で成立する。`--eval-timeout` を指定すると制限まで採点される
どころか 1 件も採点されなかった。

`eval_ranker` の `--replies` 分岐は、旧版の署名で `rank` を呼んで
dict から 3 要素を取り出そうとし、さらに dict のキーにできない id 列を
渡していた。毎行が例外となって外側の `except` に握り潰され、
順位リストが空のまま下の指標計算が `ZeroDivisionError` になっていた。
候補ファイル全体を各問い合わせに対して順位付けするようにし、
重複実装していた 2 つの指標は `lpu.metrics.ranking` から取るようにした。

採点コマンドの `main` が移植前のロガー名 `common` と `models` を
対象にしていたため、`--debug` や `--logging` を付けても `lpu_nn` や
`lpu` の出力が得られなかった。

順位付けした応答を `set` から取り出していたため、同点の応答の並びが
実行毎に変わり、報告される指標もそれに応じて揺れていた。同点は
応答文字列で決定的に並べるようにした。

採点が、候補全バッチについて使われない自動微分グラフを構築していた。

`mean` が、正解を含む問い合わせが 1 件も無いときに空リストの長さで
割っていた。

`lpu_nn.modeling.__all__` に未移植の `match_ranker` と `re2` が残っており、
`from lpu_nn.modeling import *` が失敗していました。両パッケージの
`__all__` を実際の公開内容に合わせました。

`FieldMap.load` に `FieldMap.train` から複製された残骸があり、"feeding
corpus" というログを出した上で使われないパスを算出していました。削除
しました。

4 つのメソッドが可変オブジェクトを既定引数に取っており、うち 2 つは
それをそのまま `dict.update` へ渡していました。また、bare `except` は
意図した例外を明示するようにしました (`init_weights` の署名不一致、
シーク不可能な入力の判定 (ファイルハンドルの閉じ漏れも併せて修正)、
数値として解釈できないラベル)。

`Trainer.load_labels` が `self.main_train_data_path` を読んでいました。
この属性は `train_data_path` へ改名されており、ここだけ旧名が残っていた
ため、パスを指定しない呼び出しは `AttributeError` になりました。mypy が
検出したものです。

`main` のデバッグ分岐に、ロガーを取得して捨てるだけの、何もしないループが
ありました。直前の行が同じロガーを設定済みです。またモジュール末尾に
`if __name__ == '__main__': main()` がありましたが、`main` は Trainer
クラスとモデル名を要求するため、直接実行しても `TypeError` にしかなり
ません。実行の入口は `lpu_nn.commands` にあります。

`Trainer.train_epoch` は `train_report.get('loss')` を直前の損失と比較する
際、直前側しか `None` を確認しておらず、損失を欠く報告では `None` と数値の
比較になっていました。

進捗表示が `amsbound` を `amdbound` と綴っており、`--optimizer amsbound`
では学習率の動的な下限が表示に反映されていませんでした。表示されていたのは
素の学習率で、ウォームアップ中は最適化器が実際に適用する値より遥かに小さく
なります (0.00000389 に対し実効値は 0.00006296)。

2 つの項目が、欠損した指標を文字列 `'nan'` のまま数値書式へ渡しており、
`nan` と表示される代わりに項目ごと行から消えていました。`except` 節が
`pass` だけだったことがこれを覆い隠していたので、デバッグログを出すように
しました。トークン速度は 0 になりうる間隔で割っていました。

`--eval-train` だけが保存される設定へ畳み込まれていました (他の
`store_true` はいずれも対象外です)。この種のフラグの既定値は `None` では
なく `False` のため、無指定でも「明示的な False」として、保存済みの設定を
上書きしてしまいます。隣の `--eval-only` と同じく 1 回の実行限りの指定
なので、同様に対象外としました。

`Trainer.train_epoch` はホスト名を `os.uname` で取得していましたが、これは
Windows に存在しません。`platform.node` はどの環境でも同じものを返します。

訓練側に、値を評価して捨てるだけの式が 2 つ残っていました。先の未使用変数の
整理で代入だけが外れた跡です。

`Trainer.feed_batches` のメモリ不足からの復帰は、例外をメッセージの先頭
18 文字だけで判定していました。PyTorch は `RuntimeError` の派生である
`torch.cuda.OutOfMemoryError` を送出するため、まず型で判定するように
しました。文言が変われば、バッチ縮小が働かなくなり、復帰できていた場面で
そのまま落ちることになります。

この復帰経路とキーボード中断時の後片付けは、いずれも
`torch.cuda.reset_max_memory_cached` を呼んでいました。これは警告を出す
`reset_peak_memory_stats` の別名で、隣の `reset_max_memory_allocated` も
同じ別名です。1 回の呼び出しにまとめました。

`Trainer.update_parameters` は、署名が `report=None` を許し他方の代入では
確認しているにもかかわらず、勾配ノルムの代入だけ確認していませんでした。

また `opt.weight_decay_rate` を読んでいました。これは Chainer 実装時代の
名前で、torch のどの最適化器も持ちません (実際の値は `param_groups` に
あります)。この行に到達する実行条件は構成できなかったため、引き金が
判明している欠陥ではなく潜在的なものとして扱います。

`Trainer.try_loading` は最初に読み込めなかった記録でそのまま送出しており、
失敗を記録する行はその raise の後ろで到達不能でした。つまり `--resume` の
1 件目しか試されていませんでした。`--resume` が一覧を取るのは、読み込め
なかった記録を後続で代替するためです。順に試し、どれも読み込めなかった
場合にのみ送出するようにしました。

`Trainer.load_status` は `record_dir` を `load_optimizer` の分岐内で代入
しながら `reuse_dataset` でも使っていたため、最適化器を読み込まずに
データセットだけ再利用しようとすると `UnboundLocalError` になりました。
`Trainer.load_model` もチェックポイントが持つかどうかに関わらず `idmaps`
を参照しており、同じ結果でした。何が欠けているかを述べるようにしました。

`set_logfile_handler` は差し替え時に古いハンドラを閉じておらず、また
`StreamHandler` を閉じても渡したストリームは閉じられません。自分で
ファイルを持つ `FileHandler` を使うようにしました。

設定・ラベル・スコア・データセット追記の書き出しがプラットフォーム既定の
エンコーディングのままで、Windows では非 ASCII が壊れます。明示しました。

`build_batches` と `reduce_batch_size` は、認識できないバッチ種別に対して
末尾に抜けて `None` を返しており、呼び出し側がそれを反復したり添字を
取ったりしていました。受け取った種別を述べて弾くようにしました。

`set_logfile_handler` は、差し替え時に古いハンドラもその先のファイルも
閉じていませんでした。

訓練側が設定するロガー名 `target_loggers` が、移植前のトップレベル名
(`common` / `modeling` / `optimizers`) のままでした。ロガー名はドット区切り
で、その配下にしか効きません。`common` は `lpu_nn.common.training` の祖先
ではないため、`--logging` のファイルハンドラも `--debug` も、本パッケージの
どのモジュールにも届いていませんでした。ログファイルには `__main__` の行
しか残らず、本来 300 行あるべきところが 92 行でした。`run_seq2seq` も同じ
一覧を持っており、そちらは `modeling` より更に古い `models` という名前
でした。いずれも `lpu_nn` と `lpu` を指すようにしました。

`LSTMDecoder` は `last_state` を初期化しておらず、構築直後の
encoder-decoder に対する forward は、手動で `reset_state()` を呼ばない限り
`AttributeError` になりました。単独での構築もできませんでした。
`memory_size` と `share_embedding` が `EncoderDecoder.get_config` 側でしか
設定されず、`LSTMDecoder.get_config` は自身の `__init__` を `KeyError` の
まま放置していました。`reset_state` は同種のクラスが `self` を返す場面で
`None` を返し、`set_state` は `None` も `rnn_state` を欠いた状態も
受け付けませんでした。

`LSTMDecoder.decode_one` は `self.prepare_features(seq_enc=seq, ...)` と
呼んでいましたが、引数名は `seq` です。キーワードは `**features` へ流れ込み、
トークン ID を記録する分岐は一度も実行されませんでした。設定されるはずだった
`id_seq` は 4 箇所で書かれ 0 箇所で読まれるため依存するものは無く、呼び出しが
意図どおりの形になっただけです。

`__init__` には、`vocab.pad` を `self.padding` へ読み込んだ直後に
`params['padding']` で上書きする死んだ代入もありました。

`MultiStepTransformer` は、サブレイヤの前処理・後処理に `'n'` が無い場合に
入力側・出力側の `LayerNorm` を作りますが、その適用を
`hasattr(self, 'normalize_input')` / `hasattr(self, 'normalize_output')` で
判定していました。これらの名前はどこにも代入されないため、モジュールは
生成されるだけで一度も適用されていませんでした。`UniversalTransformer` は
`__init__` の条件と同じ、モジュール自体の有無を見ています。現状これらの
設定はコマンドラインから変更できないため、ここで学習されたモデルへの影響は
ありませんが、`ModuleConnection.get_parameters` のコメントが推奨する別の
配置に切り替えた瞬間に問題になります。

`EmbedPosition.get_config` は `params.setdefault('embed_size', ...)` を
2 度書いており、2 行目は他の全ての `get_config` と同じく `hidden_size` を
導出すべきものでした。`hidden_size` を明示せずに呼ぶと `KeyError` に
なっていました。

`MultiStepTransformer.set_state` は `None` を返していました
(`reset_state` は `self` を返します)。

SentencePiece の前処理が、学習へ渡す作業用ファイルを flush していません
でした。SentencePiece には `temp.name` を渡してディスクから読ませるため、
バッファに残った分はそこに存在しません。小さなコーパスではファイルが
0 バイトになり SentencePiece 内部のエラーで失敗し、大きなコーパスでも
最後の 8 KiB 境界で切り詰められていました。つまり、このコードで学習された
トークナイザは全て、与えられたより少ないテキストで学習されていたことに
なります。flush するようにし、結果が空の場合はライブラリ内部のメッセージ
ではなく、どの設定を変えればよいかを述べるようにしました。

この修正により、同じコーパスから得られる語彙が変わり、訓練が報告する損失も
変わります。トークナイザがコーパス全体を見るようになったためです。

`AdaBoundW` は、内部状態とパラメータが別のデバイスにある場合に最適化器の
状態を失っていました。CPU で保存したチェックポイントから再開するとこの
状態になります。モーメントを `.to(p.device)` で移していましたが、`.to()` は
実際に移動する際にコピーを返すため、その場更新はすべてステップ終了時に
捨てられるコピーに対して行われていました。モーメントは凍結されたままで、
最適化器は毎ステップ実質的に再出発していながら、何も報告しません。

AdaBound は両クラスとも、学習率が 0 になると `ZeroDivisionError` になり
ました。動的な上下限は、スケジューラによる減衰を `final_lr` にも反映
させるため、現在の学習率を構築時の学習率で割ります。0 から始まる
ウォームアップは初回ステップでこれを踏みます。基準の学習率が 0 なら実効
学習率も 0 として扱い、パラメータを動かさないようにしました。

`AdaBound` には、重み減衰の経路に非推奨の `Tensor.add(alpha, other)` 形式が
残っていました。先の一掃で取りこぼしていたものです。

`universal_transformer` は単独では import できませんでした。`transformer`
がモジュールレベルでこれを import し、こちらも `transformer` を import し
返すため、先に読み込まれた方が `ImportError` になります。他の経路がたまたま
先に `transformer` へ到達していたために表面化していませんでした。逆向きの
辺を、利用箇所での局所 import に変更しました。

`UniversalTransformer` は `last_state` を初期化しておらず、構築直後の
forward は手動で `reset_state()` を呼ばない限り `AttributeError` に
なりました。コマンドラインが提供する 4 つのうちの 1 つである
`recurrence='basic'` の分岐は、存在しない `self.transform` を呼んでおり、
さらに ACT 分岐にあるステップごとの状態管理を欠いていました。これが無いと
Transformer がステップを跨いで入力を溜め込み、2 歩目で注意マスクの形が
合わなくなります。`max_ponder` は歩数ではなく `torch.max` が返す
`(values, indices)` の組を保存しており、`init_weights` は停止バイアスを
その場で埋めるのではなく、新しい CPU / float32 のテンソルで置き換えて
いました。

`ContextualStringEmbedding` は構築すらできませんでした。`nn.Module.__init__`
を呼ばないままサブモジュールを代入しており、PyTorch がこれを拒みます。
その先でも、`forward` が 3 次元のテンソルを `dim=3` で連結しようとし、
`prepare_batch` はこのクラスに存在しない `self.weight` と `self.str2tensor`
を参照していました。この `prepare_batch` の中身は本来 `CharacterEmbedding`
のもので、そちらの `prepare_batch` は `pass` だけで `None` を返していました。
実装を移すとともに、どの分岐にも当たらない入力がそのまま先へ進んでいた
点を例外にしました。

`DotAttention` は双方向の記憶を `memory.split(2, dim=2)` で、隣り合う
2 要素ずつ平均していました。エンコーダは `cat([前向き, 後向き])` と
ブロックで連結するため、これは同じ向き同士を平均しているだけで、2 つの
向きが対応することはありませんでした。記憶を前半と後半に割り、次元ごとに
平均するようにしました。既定の注意機構は `mlp` のため、影響を受けるのは
LSTM エンコーダで `--attention dot` を指定した場合です。

注意機構は `hidden_size` を `params.get` で読み、そのまま `nn.Linear` へ
渡していたため、値が無い場合は `None * 3` として届いていました。MLP 注意は
活性化関数の指定についても同様でした。いずれも何が足りないかを述べる
ようにしました。

`CharacterEmbedding.tensor2bytes` は `tensor.ndim()` を表明していました。
`ndim` はプロパティのため、この呼び出しは `TypeError` になります。

`activation.Swish1` は `super(Swish, self).__init__()` と、継承していない
クラスを渡していたため、`get_activator('swish1')` は毎回 `TypeError` で
落ちていました。利用できた経路はありません。

`SequenceConvolution1d` はパディングを `self.device` から作っていました。
この属性は `to()` を呼んだ後にしか存在しないため、構築直後のモジュールは
n-gram 次数が 2 以上で `AttributeError` になり、さらにパディングが float32
のままなので float16 の系列と連結できませんでした。いずれも入力テンソルに
従うようにしました。`init_weights` がバイアスの存在を前提にしていた点も
修正しました (`nn.Conv1d` はバイアスを保証しません)。

`to()` と `__repr__` は、`Module` を継承していないクラスからその未束縛
メソッドを借用する形で共有されていました。これは Python が受け手の型を
検査しないから動いていただけです。通常の関数 `modeling.apply_to` /
`modeling.format_module` に切り出し、`format_module` は torch が
`_modules` に残しうる `None` を読み飛ばすようにしました。

`lpu_nn.common.vocab` は独自の `IDMap` / `LabelMap` (古い lpu からの複製で
その後分岐したもの) と、どこからも使われずかつ動作し得ない `CharacterMap`
を抱えていました。`CharacterMap` は `decode` / `sample` / `__iter__` が
コンストラクタで設定されない属性を参照していました。検証済みの実装は
`lpu.common.vocab` から取るようにし、`CharacterMap` は削除しました
(あわせて重複コード 380 行が不要になりました)。これらのマップが担っていた
状態の保存・復元は `FieldMap` 側で引き受けます。

`Vocabulary` は初期化済みかどうかを `hasattr(self, 'symbols')` で判定して
いたため、記号の ID は `set_symbols` の後にしか存在しませんでした。
SentencePiece 自身の表現に倣い、未定義を -1 として最初から宣言するように
しました。

`Dataset.iter` は負の `start` / `stop` を `len(self) - 添字` として解決
していました。負号が打ち消されて範囲外になるため、負の start を渡すと
何も返りませんでした。Python の慣習どおり末尾からの位置として扱うように
しました。

`Dataset.load` は先頭行を確認せずに `.strip()` を呼んでいました。
`read_byte_line` は入力の終端で `None` を返すため空ファイルでは
`AttributeError` になり、空行で始まるファイルでは列名の無いデータセットが
黙って出来ていました。どのファイルにヘッダが無いかを述べるようにしました。

`dataset.get_values` は列の不足を握り潰して暗黙の `None` を返しており、
呼び出し側の比較が原因から離れた場所で `TypeError` になっていました。
呼び出し側には不正な行の処理があるため、`IndexError` はそのまま送出します。

`utils.flip` は添字テンソルのリストを組み立てて `t[slices]` として適用して
いました。torch 1.3 の `torch.flip` が CPU 上の bool テンソルを扱えなかった
ことへの回避策ですが、その制限は既に無く、非タプル列による添字指定は
非推奨です。2 次元以上を指定すると PyTorch は既に高度な添字指定として
解釈し `IndexError` になります。`torch.flip` を呼ぶようにしました。

`criteria.cross_entropy` と `criteria.perplexity` は追加の位置引数を受け
取り、`nn.functional.cross_entropy` へ位置のまま転送していました。転送先
では `weight` / `size_average` / `ignore_index` に割り当てられ、同時に
指定しているキーワードと衝突します。利用箇所は無かったため受け付けない
ようにしました。また、`'hmean'` 以外の reduction で無視添字のリストを
渡した場合は、torch 内部のメッセージではなく理由を述べるようにしました。

`criteria.accuracy` は `ignore_index` をリストで渡すと何も無視しません
でした。正解ではなく真偽マスクと添字を比較しており (`t_valid != ignore`)、
通常の添字では常に真になるためです。整数で渡した場合のみ正しく動作して
いました。隣の `cross_entropy` は正しく書かれています。

`--help` はヘルプを表示した後に -1 (255) で終了しており、シェルスクリプト
や CI からは失敗に見えていました。モデル固有の既定値を表示するために
ヘルプを独自に組み立てている都合で見落とされていたものです。

既定の key_size (64) より小さい hidden_size からアテンションのヘッド数を
導出すると 0 になり、ずっと後のモジュール構築時に `ZeroDivisionError`
として現れていました。同じ導出処理が 2 箇所に複製されていたため共通化し、
0 になる場合は何を指定すればよいかを示すようにしました。

`ModuleConnection` は `get_parameters` が既定値を算出する `init_gamma`
を読み出しますが、適用する箇所がありません。LayerNorm のゲイン初期化の
スケーリングがリファクタで失われており、現状この引数を渡しても効果は
ありません。復活させると全モデルの初期化が変わるため、黙って変更せず
その場に注記を残すに留めました。

元コードにはテストが無く、2020 年以降実行されていませんでした。再び動作
させる過程で以下の不具合を発見し、修正しました。

到達すると `NameError` になる経路:

- `Trainer.train_epoch` が未 import の `pview` を呼んでおり、レポートを
  表示しないエポックで必ずクラッシュしていました。
- `TransformerBase.add_positional_encoding` と `.forward` が、Chainer 実装
  時代の補助関数 `feed_seq` を呼んでいましたが、その定義は既に存在しません。
  同クラスの他の `LayerNorm` は直接適用しているため、この 2 箇所も同様に
  しました。
- `AttentionBase.get_local_attention` が引数に取っていない `h_dec` を参照
  しており、局所注意 (予測的アライメント) は一度も動作しませんでした。
  `forward` から渡すようにしました。
- `EncoderDecoder.prepare_features` が、変数を `seq` へ改名した後も `y` を
  特徴量に代入していました。
- `CharacterMap.convert` が、未代入の `codes` に対する内包表記を評価して
  いました。
- `Trainer.split_train_data` が未定義の `comm` (MPI) を参照し、かつ書式化の
  `.format` が文字列リテラルの内側に入り込んでいました。到達不能なため
  削除しました。
- `run_seq2seq --batch_size` の分岐が `Manager` / `Process` / `chainer` /
  `args.ideep` を参照していましたが、いずれも未定義・未 import でした。
  分岐ごとオプションを削除しました。

潜在的な不具合:

- 訓練側は `--logging` が**指定されていない**場合にのみ `record.tmp` を作成
  する一方、プロセス ID ファイルは常にそこへ書いていたため、`--logging` を
  指定すると起動直後にクラッシュしていました。
- `load_eval_data` が `criterion` 列を整数の -1 で初期化していたため
  `int64` 列になり、後から代入される float の perplexity を受け付けません
  でした。pandas 3.0 ではこれが静かに捨てられるため、dev の評価値が一切
  記録されず、損失ベース以外のチェックポイントが作られず、サンプル単位の
  レポートも空になっていました。float 列に修正しました。
- SentencePiece の前処理が、作業用データを作業ディレクトリ直下の
  `TMP.txt` という固定名で書き出していました (隣にある
  `tempfile.NamedTemporaryFile` の呼び出しはコメントアウトされていました)。
  ファイルが残り続ける上、並行実行では互いのデータを上書きします。
- `Trainer.test_model` が、評価値が書き込まれるまで空になるフレームに対して
  `idxmin` を呼んでいました。
- `train_seq2seq` が設定文字列の比較に `==` ではなく `is` を使っていました。
- `Trainer.format_state` の正規表現が `\.` を含む通常の文字列リテラルで
  書かれており、Python 3.12 が警告し将来のバージョンでは構文エラーに
  なります。

現行の依存関係への追従:

- Python 3.12 で削除された `distutils.util.strtobool` を、本来の `bool` を
  返す `lpu_nn.common.args.strtobool` に置き換えました。
- `Module.to` が私的関数 `torch._C._nn._parse_to` の戻り値を 3 要素で展開
  していましたが、PyTorch 1.5 以降は 4 要素を返します。
- `pandas.read_csv` は `sep` の位置引数渡しを受け付けなくなりました。
- 最適化器が非推奨の `Tensor.add_(alpha, other)` /
  `Tensor.addcmul_(value, t1, t2)` の形式を使っていました。
- 報告用のスカラーを `float()` へ渡す前に detach するようにしました
  (しない場合 PyTorch が警告します)。
- `matplotlib` を任意依存 (`plot` エクストラ) とし、未導入時はエポック毎に
  トレースバックを出すのではなくデバッグログ 1 行で描画を省略します。
