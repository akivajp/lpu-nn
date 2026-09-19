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
