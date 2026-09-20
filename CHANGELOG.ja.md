# 変更履歴

English version is available in [CHANGELOG.md](CHANGELOG.md).

## 未リリース

### 追加

- 2019-2020 年に書かれた非公開の研究コードのうち、系列変換 (seq2seq) 経路を
  PyTorch 2.x / Python 3.13 へ移植しました。訓練ループ、データセット、
  SentencePiece 語彙、Transformer / Universal Transformer / LSTM の各モジュール、
  注意機構、AdaBound と LAMB の最適化器、および
  `lpu-nn-train-seq2seq` / `lpu-nn-run-seq2seq` コマンドが対象です。
- 設定・ロギング・進捗表示・色付け・対話・ファイル操作は移植せず `lpu` の
  ものを用います。これにより約 1,900 行の重複コードが不要になりました。

### 修正

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
