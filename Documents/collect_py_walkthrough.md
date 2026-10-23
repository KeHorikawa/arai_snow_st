# collect.py コード解説（初心者向け）

`collect.py` が「何を」「どの順番で」「どうやって」いるかを、上から順に読めるようにまとめたドキュメントです。
Python を書き始めたばかりの人でも読めるよう、使われている文法やライブラリのメソッドにも短い説明を付けています。

> 行番号は 2026-10-23 時点の [collect.py](../collect.py) に対応しています。コードが変わるとずれることがあります。

---

## 目次

1. [このスクリプトは何をするのか](#1-このスクリプトは何をするのか)
2. [処理の全体像](#2-処理の全体像)
3. [インポートしているモジュール](#3-インポートしているモジュール)
4. [冒頭の準備（ロギング・定数・独自エラー）](#4-冒頭の準備ロギング定数独自エラー)
5. [メイン処理を順番に追う](#5-メイン処理を順番に追う)
6. [よく出てくる Python の書き方まとめ](#6-よく出てくる-python-の書き方まとめ)

---

## 1. このスクリプトは何をするのか

妙高市の Web サイトに掲載されている「降積雪観測データ」（新井消防署・頸南消防署・妙高支所の3地点の降雪量と積雪量）を取ってきて、ファイルに保存するスクリプトです。

GitHub Actions（[.github/workflows/collect.yml](../.github/workflows/collect.yml)）から毎朝1回自動で実行されます。

### 実行方法

```bash
python collect.py              # 一覧ページと、当シーズンの最新月ページの2枚だけ取得（毎朝の通常運転）
python collect.py --all        # 全期間をまとめて取得（初回に1回だけ）
python collect.py --dry-run    # 取得して画面に表示するだけ。ファイルは書かない（動作確認用）
```

### 出力されるファイル（3つ）

| ファイル | 役割 |
|---|---|
| `snow_data_history.csv` | 全シーズンの全データ。**これが唯一の正しいデータ**（他のファイルはここから作る） |
| `docs/data/snow_data.json` | 静的サイト（GitHub Pages）が読む表示用データ。最新シーズンの分だけ |
| `data/status.json` | 実行記録。成功・失敗にかかわらず毎回更新する |

もう1つ、市サイトから見つけた月別ページの URL 一覧 `data_urls.csv` も毎回作り直しています。

---

## 2. 処理の全体像

```
main()                          コマンドライン引数を読む・エラー時の後始末
 └─ run()                       収集の本体
     ├─ http_get()              ① 市サイトの一覧ページ（インデックス）を取得
     ├─ discover_month_urls()   ② 一覧ページから「月別ページのURL」を全部見つける
     ├─ save_url_csv()          ③ 見つけたURLを data_urls.csv に保存
     ├─ load_history()          ④ これまでの履歴CSVを読み込む
     ├─ （取得対象を決める）      ⑤ 通常は「当シーズンの最新月」1ページだけ
     ├─ fetch_month()           ⑥ 月別ページを取得して表を読み取る
     │   └─ parse_month_page()      HTMLの表 → DataFrame に変換
     ├─ check_consistency()     ⑦ 取得結果がおかしくないか確認
     ├─ upsert_history()        ⑧ 履歴に新しいデータを合体（追加 or 上書き）
     ├─ save_history()          ⑨ 履歴CSVを保存
     ├─ build_snow_data()       ⑩ 表示用JSONの中身を組み立てる
     ├─ build_status()          ⑪ 実行記録の中身を組み立てる
     └─ write_json()            ⑫ JSONファイル2つを書き出す
```

ファイルの最後にある次の2行が、スクリプトの入り口です（[collect.py:715-716](../collect.py#L715-L716)）。

```python
if __name__ == "__main__":
    sys.exit(main())
```

- `if __name__ == "__main__":` … 「このファイルが `python collect.py` として直接実行されたときだけ」中身を動かす、という Python の定番の書き方。テストなどから `import collect` されたときは動かない。
- `sys.exit(main())` … `main()` の戻り値（0 = 成功、1 = 失敗）を**終了コード**としてプログラムを終える。GitHub Actions はこの終了コードを見て成功・失敗を判断する。

---

## 3. インポートしているモジュール

[collect.py:19-35](../collect.py#L19-L35)

### 標準ライブラリ（Python に最初から入っているもの）

| モジュール | 何に使っているか |
|---|---|
| `from __future__ import annotations` | 型ヒント（`-> int` など）の評価を後回しにする指定。古めの Python でも新しい型ヒントの書き方を使えるようにするおまじない |
| `argparse` | `--all` や `--dry-run` などのコマンドライン引数を読み取る |
| `calendar` | 「その月が何日まであるか」を調べる（`calendar.monthrange`） |
| `json` | Python の辞書 ⇔ JSON 文字列の変換 |
| `logging` | 時刻付きのログ（実行記録のメッセージ）を出す。`print` より情報が多く、重要度（INFO / WARNING / ERROR）を付けられる |
| `re` | 正規表現。文字列から「(R8.3)」のようなパターンを探す |
| `sys` | `sys.exit()` で終了コードを返す |
| `time` | `time.sleep()` で待機する（市サーバーに連続アクセスしないため） |
| `datetime`（`date, datetime, timedelta, timezone`） | 日付・時刻を扱う。日本時間（JST）の今日の日付を出すのに使う |
| `pathlib.Path` | ファイルパスをオブジェクトとして扱う。`BASE_DIR / "data"` のように `/` でパスをつなげられる |
| `typing`（`Dict, List, Optional, Tuple`） | 型ヒント用。`Optional[float]` は「float か None」の意味 |
| `urllib.parse.urljoin` | 相対URL（`/life-info/...`）を完全なURL（`https://www.city...`）に変換する |

### 外部ライブラリ（`pip install` が必要なもの）

[requirements-collect.txt](../requirements-collect.txt) に書かれています。

| ライブラリ | 何に使っているか |
|---|---|
| `pandas`（`pd`） | 表形式データ（DataFrame）を扱う。CSV の読み書き・並べ替え・重複削除など |
| `requests` | Web ページを HTTP で取得する |
| `bs4.BeautifulSoup` | 取得した HTML を解析し、`<a>` や `<table>` などのタグを探し出す |
| `lxml` | BeautifulSoup が内部で使う HTML 解析エンジン（コード中では `"lxml"` という文字列で指定するだけ） |

---

## 4. 冒頭の準備（ロギング・定数・独自エラー）

### ロギングの設定（[collect.py:40-45](../collect.py#L40-L45)）

```python
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", ...)
logger = logging.getLogger("collect")
```

- `basicConfig` … ログの出し方を一括設定。`level=logging.INFO` で INFO 以上（INFO / WARNING / ERROR）を表示する。
- `format` … 1行の見た目。`2026-10-23 10:17:00 [INFO] 取得: ...` のようになる。
- 以降は `logger.info(...)`、`logger.warning(...)`、`logger.error(...)` でメッセージを出す。

### 定数（[collect.py:50-87](../collect.py#L50-L87)）

大文字の変数名は「途中で書き換えない値（定数）」という Python の慣習です。

| 定数 | 中身 |
|---|---|
| `BASE_URL` / `INDEX_URL` | 市サイトのURLと、月別ページへのリンクが並ぶ一覧ページのURL |
| `LOCATIONS` | 観測地点3つ。表の列の順番と一致している |
| `LOCATION_NORMALIZE` | 「頚」→「頸」の表記ゆれ補正（2009年12月のページだけ異体字のため） |
| `BASE_DIR` | このスクリプトが置かれているフォルダ。`Path(__file__).resolve().parent` で求める |
| `HISTORY_CSV` など | 各出力ファイルのパス |
| `HISTORY_COLS` / `HISTORY_KEY` | 履歴CSVの列名と、「同じ行」とみなすキー（年・月・日・地点） |
| `JST` | 日本時間。`timezone(timedelta(hours=9))` = UTC+9時間 |
| `REQUEST_INTERVAL_SEC` / `REQUEST_TIMEOUT_SEC` | アクセス間隔 1.5秒 / 応答待ちの上限 20秒 |
| `USER_AGENT` | 「誰がアクセスしているか」を市サーバーに伝える名乗り |
| `CONSISTENCY_RATIO` | 0.8。データが急に8割未満に減ったら異常とみなす基準 |
| `MONTH_LINK_RE` | リンク文字列から「(R8.3)」を見つける正規表現（下で解説） |

#### `MONTH_LINK_RE` の正規表現を分解すると

```python
re.compile(r"[（(]\s*([RH])\s*(\d{1,2})\s*[.．]\s*(\d{1,2})\s*[）)]")
```

| 部分 | 意味 |
|---|---|
| `r"..."` | raw 文字列。`\` をそのまま書ける（正規表現でよく使う） |
| `[（(]` | 全角か半角の開きカッコ |
| `\s*` | 空白が0個以上（あってもなくてもよい） |
| `([RH])` | R（令和）か H（平成）。`()` で囲むと後で `group(1)` として取り出せる |
| `(\d{1,2})` | 数字1〜2桁（年）。`group(2)` |
| `[.．]` | 半角か全角のピリオド |
| `(\d{1,2})` | 数字1〜2桁（月）。`group(3)` |
| `[）)]` | 閉じカッコ |

`re.compile` は正規表現を事前に準備しておく関数で、何度も使うときに効率が良くなります。

### 独自エラー `CollectError`（[collect.py:90-91](../collect.py#L90-L91)）

```python
class CollectError(Exception):
    """ジョブを失敗させる（＝メール通知してほしい）エラー"""
```

- Python の標準の `Exception` を**継承**して、自分専用のエラーの種類を作っている。
- `raise CollectError("メッセージ")` で発生させ、`except CollectError` で受け止める。
- 「想定内の失敗（市サイトの構造が変わった等）」と「想定外のバグ」を区別するために使う。

---

## 5. メイン処理を順番に追う

### 5-0. `main()` — 入り口（[collect.py:686-712](../collect.py#L686-L712)）

**役割**: コマンドライン引数を読み、`run()` を呼び、エラーが起きたら記録して終了コードを返す。

**処理の流れ**

1. `argparse` で `--all` と `--dry-run` を受け付ける
2. 日本時間の今日の日付を求める
3. `run()` を呼ぶ
4. `CollectError` が起きたら → ログに出し、`data/status.json` に `"result": "error"` を書いて `1` を返す
5. それ以外の想定外エラーも同様に記録して `1` を返す
6. 成功なら `0` を返す

**使われている書き方**

| コード | 説明 |
|---|---|
| `argparse.ArgumentParser(...)` | 引数を解析する道具を作る |
| `parser.add_argument("--all", action="store_true", dest="fetch_all")` | `--all` が付いていれば `True`。`dest` で受け取る変数名を `fetch_all` に変えている |
| `args = parser.parse_args(argv)` | 実際に解析。`args.fetch_all`、`args.dry_run` で値を取り出せる（`--dry-run` の `-` は `_` に変わる） |
| `datetime.now(JST).date()` | 日本時間の現在時刻 → 日付部分だけ |
| `try: ... except CollectError as e: ... except Exception as e:` | エラーを捕まえる。上から順に判定され、`Exception` は「その他すべて」 |
| `logger.exception(...)` | エラーの詳細（どの行で起きたか＝トレースバック）付きでログを出す |
| `run(fetch_all=..., dry_run=..., today=...)` | **キーワード引数**。引数名を明示して渡すので、順番を間違えにくい |

> **ポイント**: 失敗しても `status.json` を書いておくことで、「いつ・なぜ失敗したか」が Git の履歴に残ります。

---

### 5-1. `run()` — 収集の本体（[collect.py:589-683](../collect.py#L589-L683)）

**役割**: ①〜⑫の処理を順番に実行する司令塔。以下、`run()` の中で呼ばれる順に各関数を見ていきます。

---

### ① `http_get()` — Webページを取得（[collect.py:134-138](../collect.py#L134-L138)）

```python
def http_get(url: str) -> str:
    resp = requests.get(url, timeout=REQUEST_TIMEOUT_SEC, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    return resp.text
```

| コード | 説明 |
|---|---|
| `requests.get(url, timeout=..., headers=...)` | URL にアクセスしてページを取ってくる。20秒応答がなければ諦める |
| `resp.raise_for_status()` | 404（ページなし）や 500（サーバーエラー）なら例外を発生させる |
| `resp.apparent_encoding` | ページの中身から文字コードを推測する（文字化け対策） |
| `resp.text` | ページの HTML を文字列として取り出す |

`run()` 側では `try / except requests.RequestException` で囲み、通信エラーを `CollectError` に変換しています。

```python
raise CollectError(...) from e
```

`from e` を付けると「元のエラー e が原因で起きた」という情報が残り、デバッグしやすくなります。

---

### ② `discover_month_urls()` — 月別ページのURLを見つける（[collect.py:144-192](../collect.py#L144-L192)）

**役割**: 一覧ページの全リンクを調べ、「令和7年度 降積雪観測(R8.3)」のようなリンクから **年・月・URL** を取り出す。

**処理の流れ**

1. HTML を BeautifulSoup で解析
2. すべての `<a href="...">` タグを順に見る
3. リンク文字列が `MONTH_LINK_RE` に一致するものだけ残す
4. 和暦を西暦に変換（`wareki_to_year`）
5. 月が 1〜12 か確認
6. 相対URL を完全なURL に変換し、`(年, 月)` をキーに辞書へ登録
7. 1件も見つからなければ `CollectError`（市サイトの形式が変わった可能性）
8. 年月の昇順に並べ、年度ラベルを付けたリストを返す

**使われている書き方**

| コード | 説明 |
|---|---|
| `BeautifulSoup(html, "lxml")` | HTML 文字列を解析して、タグを探せるオブジェクトにする |
| `soup.find_all("a", href=True)` | `href` 属性を持つ `<a>` タグを全部探す |
| `a.get_text(" ", strip=True)` | タグ内の文字を取り出す。前後の空白を除き、子要素の間は `" "` でつなぐ |
| `MONTH_LINK_RE.search(text)` | 文字列の中からパターンを探す。見つからなければ `None` |
| `m.group(1)` | 正規表現の1つ目の `()` に一致した部分 |
| `continue` | ループの残りを飛ばして次の要素へ |
| `a["href"]` | タグの属性値を取り出す |
| `urljoin(BASE_URL, a["href"])` | 相対URL を完全なURL に |
| `found.setdefault(key, url)` | キーがまだなければ登録。あれば何もしない |
| `{"年度": ..., ...} for (y, m), url in sorted(found.items())` | **リスト内包表記**。ループで作ったものをまとめてリストにする（[6章](#6-よく出てくる-python-の書き方まとめ)参照） |
| `f"...{変数}..."` | **f文字列**。`{}` の中に変数や式を埋め込める |
| `{text!r}` | f文字列で `!r` を付けると `'...'` のように引用符付きで表示（空白などが見やすい） |

#### ここで使われる補助関数（和暦・年度）

[collect.py:97-128](../collect.py#L97-L128)

| 関数 | 何をするか | 例 |
|---|---|---|
| `wareki_to_year(era, n)` | 和暦 → 西暦。令和は +2018、平成は +1988 | `("R", 8)` → `2026` |
| `fiscal_year(year, month)` | その月がどのシーズンに属するか。**6月以降は今年、5月以前は前年** | `(2026, 3)` → `2025` |
| `fiscal_year_label(fy)` | シーズンを和暦の年度表記に。令和1年は「令和元年度」 | `2025` → `"令和7年度"` |
| `season_label(fy)` | シーズンを JSON 用の表記に | `2025` → `"2025-26"` |
| `current_fiscal_year(today)` | 今日がどのシーズンか | `2026-10-23` → `2026` |

- `str(fy + 1)[2:]` … 数値を文字列にして3文字目以降を取り出す（`"2026"` → `"26"`）。`[2:]` は**スライス**という書き方。
- `"令和元年度" if n == 1 else f"令和{n}年度"` … **条件式（三項演算子）**。条件が真なら左、偽なら右。

---

### ③ `save_url_csv()` — URL一覧を保存（[collect.py:195-202](../collect.py#L195-L202)）

見つけた URL 一覧を `data_urls.csv` に書き出します。`--dry-run` のときは呼ばれません。

| コード | 説明 |
|---|---|
| `pd.DataFrame([...], columns=URLS_COLS)` | 辞書のリストから表（DataFrame）を作る。列の順番を `columns` で指定 |
| `df.to_csv(URLS_CSV, index=False)` | CSV ファイルに保存。`index=False` で行番号の列を出さない |

---

### ④ `load_history()` — 履歴CSVを読み込む（[collect.py:359-386](../collect.py#L359-L386)）

**役割**: `snow_data_history.csv` を読み込み、型をそろえ、おかしな行を取り除く。

**処理の流れ**

1. ファイルがなければ空の DataFrame を返す（初回実行）
2. CSV を読み込み、必要な列がそろっているか確認
3. 年・月・日を数値に変換。変換できない行は捨てる
4. 実在しない日付（4月31日など）の行を捨てる
5. 列の順番をそろえて返す

| コード | 説明 |
|---|---|
| `HISTORY_CSV.exists()` | ファイルが存在するか（`Path` のメソッド） |
| `pd.read_csv(HISTORY_CSV)` | CSV を DataFrame として読み込む |
| `[c for c in HISTORY_COLS if c not in df.columns]` | 足りない列名を集めるリスト内包表記 |
| `pd.to_numeric(df[col], errors="coerce")` | 数値に変換。変換できない値は `NaN`（欠損値）にする |
| `df.dropna(subset=[...])` | 指定列に `NaN` がある行を削除 |
| `.copy()` | 元のデータと切り離したコピーを作る（pandas の警告対策） |
| `df[col].astype(int)` | 列の型を整数に変換 |
| `zip(df["year"], df["month"], df["day"])` | 3つの列を1行ずつ組にして取り出す |
| `sum(valid)` | `True` は1、`False` は0として数えられるので、`True` の個数になる |
| `df[valid]` | `True` の行だけ残す（**ブールインデックス**） |

#### 補助関数 `_is_real_date()`（[collect.py:260-262](../collect.py#L260-L262)）

```python
return 1 <= month <= 12 and 1 <= day <= calendar.monthrange(year, month)[1]
```

- `calendar.monthrange(年, 月)` は `(その月1日の曜日, その月の日数)` を返す。`[1]` で日数を取り出す。
- `1 <= month <= 12` のように比較を連ねて書けるのは Python の特徴。

> 関数名の先頭の `_`（アンダースコア）は「このファイルの中だけで使う補助関数」という慣習です。

---

### ⑤ 取得対象を決める（[collect.py:608-628](../collect.py#L608-L628)）

ここは関数ではなく `run()` の中に直接書かれています。3つのケースに分かれます。

| 条件 | 取得対象 |
|---|---|
| `--all` 指定 | 見つけた全ページ |
| 当シーズンのページがある | 当シーズンの**最新月1ページだけ** |
| 当シーズンのページがない | 何も取得しない（オフシーズンの正常な状態。失敗ではない） |

`--dry-run` かつオフシーズンの場合は、動作確認のため公開済みの最新ページを1枚だけ取ります。

| コード | 説明 |
|---|---|
| `[e for e in entries if fiscal_year(...) == cur_fy]` | 当シーズンのものだけ絞り込む |
| `max(season_entries, key=lambda e: (e["year"], e["month"]))` | `(年, 月)` が最大のもの＝最新月を選ぶ |
| `lambda e: ...` | **無名関数**。その場限りの小さな関数。ここでは「比較に使う値」を指定している |
| `entries[-1]` | リストの最後の要素（マイナスの番号は後ろから数える） |

---

### ⑥ `fetch_month()` → `parse_month_page()` — 月別ページを読み取る

#### `fetch_month()`（[collect.py:341-353](../collect.py#L341-L353)）

月別ページを `http_get()` で取得し、`parse_month_page()` に渡すだけの関数です。

`run()` 側ではこれをループで呼び、2回目以降は `time.sleep(1.5)` で1.5秒待ちます（`--all` のときに市サーバーへ負荷をかけないため）。`--all` のときは1ページ失敗しても最後まで続け、最後にまとめて報告します。

| コード | 説明 |
|---|---|
| `for i, entry in enumerate(targets):` | `enumerate` で「何番目か（i）」と「要素」を同時に受け取る |
| `frames.append(...)` | リストの末尾に追加 |
| `raise`（引数なし） | 今捕まえたエラーをそのまま投げ直す |
| `df['day'].nunique()` | 重複を除いた値の種類数（＝何日分あるか） |

#### `parse_month_page()`（[collect.py:265-338](../collect.py#L265-L338)）

**役割**: 月別ページの HTML の表を読み、1行＝「ある日・ある地点の降雪量と積雪量」という形の DataFrame にする。

市サイトの表はこういう形をしています（横に3地点が並ぶ）。

| 日 | 新井 降雪 | 新井 積雪 | 頸南 降雪 | 頸南 積雪 | 妙高 降雪 | 妙高 積雪 |
|---|---|---|---|---|---|---|
| 1日 | 5 | 30 | 8 | 45 | 10 | 60 |

これを次のような**縦長の形（tidy データ）**に変換します。

| year | month | day | location | snowfall_cm | snowdepth_cm |
|---|---|---|---|---|---|
| 2026 | 1 | 1 | 新井消防署 | 5.0 | 30.0 |
| 2026 | 1 | 1 | 頸南消防署 | 8.0 | 45.0 |
| 2026 | 1 | 1 | 妙高市役所 妙高支所 | 10.0 | 60.0 |

**処理の流れ**

1. `_pick_data_table()` で観測地点名を含む表を選ぶ
2. 表の行（`<tr>`）を全部取り出す。3行未満なら異常
3. 先頭行（見出し）を除いて1行ずつ処理
   - セルが7個未満の行は飛ばす
   - 1列目から「3日」→ `3` のように日を取り出す。数字でなければ飛ばす（見出し行など）
   - 実在しない日付なら飛ばす
   - 3地点それぞれについて、列番号を計算して降雪量・積雪量を取り出す
   - マイナスの値は警告を出しつつ、そのまま保存する（後で原因を調べられるように）
4. 1行も取れなければ `CollectError`
5. DataFrame にして、数値列を float 型にそろえて返す

**列番号の計算**（[collect.py:298-300](../collect.py#L298-L300)）

```python
for i, location in enumerate(LOCATIONS):   # i = 0, 1, 2
    snowfall_idx = i * 2 + 1               # 1, 3, 5
    snowdepth_idx = i * 2 + 2              # 2, 4, 6
```

| コード | 説明 |
|---|---|
| `table.find_all("tr")` | 表の行を全部取り出す |
| `rows[1:]` | 先頭（0番目）を除いた残り |
| `row.find_all(["td", "th"])` | `<td>` と `<th>` のどちらのセルも取り出す |
| `.replace("日", "").strip()` | 「日」を消し、前後の空白を除く |
| `.isdigit()` | 文字列が数字だけか |
| `A if 条件 else B` | 列が足りないときは `"-"` を代わりに使う |
| `logger.debug(...)` | INFO より低い重要度のログ。通常の設定では表示されない |
| `df[[列1, 列2]].astype(float)` | 複数の列をまとめて float 型に |

#### `parse_month_page()` が使う補助関数

**`_pick_data_table()`**（[collect.py:214-227](../collect.py#L214-L227)）
ページ内のすべての `<table>` を調べ、3地点の名前がすべて含まれる表を選びます。見つからなければ最初の表を使います。

- `all(loc in text for loc in normalized_locations)` … 3地点すべてが `text` に含まれていれば `True`。`all()` は「全部 True か」を判定する関数。

**`_normalize()`**（[collect.py:208-211](../collect.py#L208-L211)）
「頚」→「頸」に置き換え、空白をすべて取り除きます。地点名の表記ゆれや空白の違いで判定が失敗しないようにするためです。

- `re.sub(r"\s+", "", text)` … 空白（`\s`）が1個以上続く部分を空文字に置き換える。

**`_cell_text()`**（[collect.py:230-241](../collect.py#L230-L241)）
セルの文字を取り出します。ただし、**画面に表示されない隠し要素を先に削除**します。

市のページには、見た目は「18」なのに、画面外に隠された「-」が入っていて、そのまま文字を取ると「-18」になってしまうセルがあります。そこで `style` 属性に `position:absolute` と `left:-`（画面外への移動）がある要素を消してから文字を取ります。

| コード | 説明 |
|---|---|
| `cell.find_all(style=True)` | `style` 属性を持つ要素を全部探す |
| `el["style"].replace(" ", "").lower()` | 空白を除いて小文字に（表記の違いを吸収） |
| `el.decompose()` | その要素を HTML から完全に削除する |

**`_to_float_or_none()`**（[collect.py:244-257](../collect.py#L244-L257)）
「12」→ `12.0`、「-5」→ `-5.0`、「-」や空欄 → `None` に変換します。

- `re.search(r"-?\d+(?:\.\d+)?", x)` … 「マイナス記号（あってもなくても）＋数字＋（小数点以下があれば）」を探す。`(?:...)` は「グループ化するが `group()` で取り出さない」という書き方。
- `(s or "")` … `s` が `None` なら `""` を使う、という短い書き方。

---

### ⑦ `check_consistency()` — 取得結果の健全性チェック（[collect.py:389-405](../collect.py#L389-L405)）

**役割**: 「静かなデータ破壊」を防ぐ。たとえば市サイトの表の形が変わって一部の日しか読めなくなったとき、そのまま上書き保存すると正しいデータが失われます。

そこで、同じ年月について**既存の履歴と比べて日数が8割未満に減っていたら** `CollectError` で止めます。

| コード | 説明 |
|---|---|
| `history.empty` | DataFrame が空か |
| `new_df.groupby(["year", "month"])` | 年月ごとにグループ分け。`for (year, month), group in ...` で1グループずつ取り出せる |
| `history[(history["year"] == year) & (history["month"] == month)]` | 条件に合う行だけ取り出す。複数条件は `&`（かつ）でつなぎ、それぞれを `()` で囲む |

---

### ⑧ `upsert_history()` — 履歴に合体（[collect.py:416-441](../collect.py#L416-L441)）

**役割**: 新しく取得したデータを履歴に合体する。**upsert** = update（更新）+ insert（追加）。同じキー（年・月・日・地点）の行があれば新しい値で上書き、なければ追加します。

**処理の流れ**

1. 既存の履歴を `{(年,月,日,地点): (降雪, 積雪)}` の辞書にする
2. 新データを1行ずつ見て、「新規」または「値が変わった」行を数える
3. 履歴と新データを縦につなげ、同じキーの重複は**後ろ（新データ）を残して**削除
4. 並べ替えて返す

| コード | 説明 |
|---|---|
| `history.itertuples(index=False)` | DataFrame を1行ずつ取り出す。`row.year` のように列名でアクセスできる |
| `old_map.get(key)` | 辞書から値を取り出す。キーがなければ `None` |
| `pd.concat([history, new_df], ignore_index=True)` | 2つの表を縦につなげる。行番号は振り直す |
| `drop_duplicates(subset=HISTORY_KEY, keep="last")` | キーが同じ行のうち最後のものだけ残す |
| `return combined, changed` | 2つの値を**タプル**として同時に返す。受け取り側は `history, rows_written = ...` |

#### 補助関数

**`_values_equal()`**（[collect.py:408-413](../collect.py#L408-L413)）
2つの値が同じかを比べます。`NaN`（欠損値）同士は普通に `==` で比べると `False` になってしまうため、`pd.isna()` で欠損かどうかを先に判定しています。

**`sort_history()`**（[collect.py:444-449](../collect.py#L444-L449)）
年→月→日→地点の順に並べ替えます。地点は文字コード順ではなく `LOCATIONS` の順（新井→頸南→妙高）にしたいので、工夫しています。

| コード | 説明 |
|---|---|
| `pd.Categorical(..., categories=LOCATIONS, ordered=True)` | 「順序付きのカテゴリ」型にする。並べ替え時に `categories` の順番が使われる |
| `df.assign(_loc_order=order)` | 一時的な列 `_loc_order` を追加した新しい DataFrame を作る |
| `sort_values([...], kind="mergesort")` | 指定列の順で並べ替え。`mergesort` は同じ値の行の元の順番を崩さない（安定ソート） |
| `drop(columns="_loc_order")` | 一時列を削除 |
| `reset_index(drop=True)` | 行番号を 0 から振り直す |

> 取得がなかった日でも、`run()` は `sort_history()` を通してから保存します。これにより並び順の正規化と不正な日付の除去が毎日効きます。内容が変わらなければ Git の差分は出ないので、空のコミットにはなりません。

---

### ⑨ `save_history()` — 履歴CSVを保存（[collect.py:452-454](../collect.py#L452-L454)）

```python
df.reindex(columns=HISTORY_COLS).to_csv(HISTORY_CSV, index=False)
```

- `reindex(columns=...)` … 列を指定した順番に並べ替える（余計な列は落ちる）。
- `--dry-run` のときは保存しません。

---

### ⑩ `build_snow_data()` — 表示用JSONを組み立てる（[collect.py:468-567](../collect.py#L468-L567)）

**役割**: 静的サイトが読む `snow_data.json` の中身（Python の辞書）を作る。このスクリプトで一番長い関数です。

**作られる JSON の形（概略）**

```json
{
  "generated_at": "2026-10-23T10:20:00+09:00",
  "source_url": "https://...（最新観測日の月ページ）",
  "season": "2025-26",
  "season_status": "finished",
  "is_current_season": false,
  "locations": ["新井消防署", "頸南消防署", "妙高市役所 妙高支所"],
  "latest": {
    "observed_date": "2026-03-31",
    "by_location": {
      "新井消防署": { "snowdepth_cm": 0, "snowdepth_diff_cm": -2, "snowfall_cm": 0 }
    }
  },
  "series": {
    "dates": ["2025-12-01", "2025-12-02", "..."],
    "新井消防署": { "snowfall_cm": [0, 5, "..."], "snowdepth_cm": [0, 5, "..."] }
  }
}
```

**処理の流れ**

1. **ひな形（`base`）を作る** … データがなくても形が崩れないよう、空の値で全項目を用意
2. **対象シーズンを決める** … 「今シーズン」ではなく「**値が1つでもある最も新しいシーズン**」。オフシーズンでも前シーズンのデータでサイトを確認できるようにするため
3. **シーズンの状態を設定** … 今シーズンなら `"in-season"`、過去なら `"finished"`。データが1件も無いときはひな形のまま `"no-data"`
4. **日付列を作る** … 年・月・日の3列から日付型の列を作る
5. **末尾の空白日を落とす** … 最後に値がある日より後の日（全部欠測）は除く
6. **`series` を作る** … 日付の一覧と、地点ごとに日付順に並んだ降雪量・積雪量の配列
7. **`latest` を作る** … 最も新しい観測日の各地点の値と、**前日比**（「暦の前日」ではなく「値がある直前の観測日」との差）
8. **`source_url`** … 最新観測日が載っている月ページのURL

| コード | 説明 |
|---|---|
| `datetime.now(JST).isoformat(timespec="seconds")` | `2026-10-23T10:20:00+09:00` 形式の文字列に |
| `df["snowfall_cm"].notna() \| df["snowdepth_cm"].notna()` | どちらかに値がある行は `True`。`\|` は「または」 |
| `df.loc[has_value, "fy"]` | `loc[行の条件, 列名]` で行と列を同時に指定して取り出す |
| `seasons_with_data.max()` | 最大値（＝最新シーズン） |
| `pd.to_datetime(dict(year=..., month=..., day=...))` | 年・月・日の列から日付型を作る |
| `sorted(season_df["date"].unique())` | 重複を除いた日付を昇順に |
| `pd.Timestamp(d).strftime("%Y-%m-%d")` | 日付を `"2026-01-05"` 形式の文字列に |
| `set_index("date")` | `date` 列を行のラベル（インデックス）にする。`loc_df.loc[日付]` で直接引けるようになる |
| `d in loc_df.index` | その日付の行があるか |
| `prev.iloc[-1]` | `iloc` は位置で取り出す。`-1` で最後の行（＝直前の観測日） |
| `.iloc[0]` | 最初の行 |

#### 補助関数 `_num()`（[collect.py:460-465](../collect.py#L460-L465)）

JSON に書く数値を整えます。欠損なら `None`（JSON では `null`）、`5.0` のような整数値は `5` にして見やすくします。

- `f.is_integer()` … float が整数値か（`5.0` → `True`、`5.5` → `False`）。

---

### ⑪ `build_status()` — 実行記録を組み立てる（[collect.py:576-583](../collect.py#L576-L583)）

`data/status.json` に書く辞書を作るだけの関数です。

| キー | 中身 |
|---|---|
| `last_run_at` | 実行日時（日本時間） |
| `result` | `"ok"` または `"error"` |
| `note` | 何をしたか／エラー内容 |
| `rows_written` | 追加・更新した行数 |
| `latest_observed_date` | 最新の観測日 |

`run()` では、`--all` で失敗したページがあればこの前に `CollectError` を発生させます。

---

### ⑫ `write_json()` — JSONファイルを書き出す（[collect.py:570-573](../collect.py#L570-L573)）

```python
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
```

| コード | 説明 |
|---|---|
| `path.parent` | ファイルが入るフォルダ |
| `mkdir(parents=True, exist_ok=True)` | フォルダを作る。途中のフォルダもまとめて作り、既にあってもエラーにしない |
| `json.dumps(payload, ...)` | 辞書を JSON 文字列に変換 |
| `ensure_ascii=False` | 日本語を `\u65b0` のような記号の並びにせず、そのまま書く |
| `indent=2` | 2スペースでインデントして見やすく |
| `path.write_text(..., encoding="utf-8")` | 文字列をファイルに書き込む |
| `path.relative_to(BASE_DIR)` | ログ表示用に、プロジェクトからの相対パスにする |

`--dry-run` のときはファイルに書かず、`print()` で画面に表示します（JSON は長いので `[:2000]` で先頭2000文字だけ）。

---

## 6. よく出てくる Python の書き方まとめ

| 書き方 | 例 | 意味 |
|---|---|---|
| 型ヒント | `def http_get(url: str) -> str:` | 引数 `url` は文字列、戻り値は文字列、という注釈。動作には影響しない |
| `Optional[X]` | `Optional[float]` | X か `None` |
| f文字列 | `f"{year}年{month}月"` | 文字列に変数を埋め込む |
| リスト内包表記 | `[c for c in cols if c not in df.columns]` | 「cols の各 c について、条件を満たすものを集めたリスト」 |
| 辞書内包表記 | `{(e["year"], e["month"]): e["url"] for e in entries}` | 同様に辞書を作る |
| 条件式 | `A if 条件 else B` | 条件が真なら A、偽なら B |
| アンパック | `history, rows_written = upsert_history(...)` | 複数の戻り値を別々の変数に受け取る |
| スライス | `rows[1:]`、`text[:2000]` | リストや文字列の一部を取り出す |
| 負のインデックス | `entries[-1]` | 後ろから数える（-1 が最後） |
| `lambda` | `key=lambda e: (e["year"], e["month"])` | その場で使う小さな関数 |
| 例外処理 | `try: ... except X as e: ...` | エラーを捕まえて対処する |
| `raise ... from e` | | 原因のエラーを引き継いで新しいエラーを投げる |
| docstring | `"""..."""`（関数の直下） | 関数の説明文。`help(関数名)` で表示される |
| 先頭 `_` の関数名 | `_normalize` | 「内部用」という慣習 |
| 大文字の変数名 | `BASE_URL` | 「定数」という慣習 |

### pandas でよく出てくるもの

| 書き方 | 意味 |
|---|---|
| `df["列名"]` | 1列を取り出す（Series） |
| `df[["列1", "列2"]]` | 複数列を取り出す（DataFrame） |
| `df[条件]` | 条件が `True` の行だけ取り出す |
| `df.loc[行, 列]` | ラベル（名前・条件）で取り出す |
| `df.iloc[行番号]` | 位置（何番目か）で取り出す |
| `NaN` / `pd.isna()` | 欠損値と、その判定 |
| `& / \| / ~` | pandas の条件での「かつ / または / 否定」（`and / or / not` は使えない） |
