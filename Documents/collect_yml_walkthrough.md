# collect.yml（定期収集ワークフロー）解説（初心者向け）

[.github/workflows/collect.yml](../.github/workflows/collect.yml) が「いつ」「どこで」「何を」「どの順番で」しているかを、上から順に読めるようにまとめたドキュメントです。
GitHub Actions や YAML を初めて触る人でも読めるよう、出てくるキーワードやシェルのコマンドにも短い説明を付けています。

`collect.py` 自体の中身は [collect_py_walkthrough.md](collect_py_walkthrough.md) を参照してください。

> 行番号は 2026-10-29 時点の [collect.yml](../.github/workflows/collect.yml) に対応しています。ファイルが変わるとずれることがあります。

---

## 目次

1. [このファイルは何をするのか](#1-このファイルは何をするのか)
2. [前提知識：GitHub Actions と YAML](#2-前提知識github-actions-と-yaml)
3. [処理の全体像](#3-処理の全体像)
4. [ワークフロー全体の設定を順番に読む](#4-ワークフロー全体の設定を順番に読む)
5. [ジョブとステップを順番に追う](#5-ジョブとステップを順番に追う)
6. [なぜ「失敗しても先にコミットする」のか](#6-なぜ失敗しても先にコミットするのか)
7. [手動で動かす・結果を確かめる](#7-手動で動かす結果を確かめる)
8. [よく出てくる書き方まとめ](#8-よく出てくる書き方まとめ)

---

## 1. このファイルは何をするのか

毎朝1回、GitHub のサーバー上で `collect.py` を自動実行し、更新されたデータファイルをリポジトリにコミット・push するための設定ファイルです。

自分のパソコンを起動しておく必要はありません。GitHub が用意した使い捨てのサーバー（**ランナー**）が、毎回まっさらな状態から次のことを行います。

1. リポジトリのコードを取ってくる
2. Python を用意して、必要なライブラリを入れる
3. `collect.py` を実行する
4. ファイルに変化があればコミットして push する
5. `collect.py` が失敗していたら、ジョブを「失敗」にする（GitHub からメールが届く）

Git の履歴に毎日並ぶ `data: update snow data (2026-10-28)` のようなコミットは、このワークフローが作ったものです。

---

## 2. 前提知識：GitHub Actions と YAML

### GitHub Actions の用語

| 用語 | 意味 | このファイルでは |
|---|---|---|
| **ワークフロー** | 自動化したい作業のまとまり。`.github/workflows/` に置いた YAML ファイル1つが1ワークフロー | `collect snow data` |
| **トリガー（イベント）** | ワークフローを動かすきっかけ | 毎日の定時（`schedule`）と手動（`workflow_dispatch`） |
| **ジョブ** | 1台のランナー上で実行される作業の単位。ワークフローに複数書ける | `collect` の1つだけ |
| **ステップ** | ジョブの中で上から順に実行される1つ1つの作業 | 6つ |
| **ランナー** | ジョブを実行するサーバー | `ubuntu-latest`（GitHub が用意する Linux） |
| **アクション** | 他の人が作って公開している、再利用できるステップ部品 | `actions/checkout`、`actions/setup-python` |

### YAML の読み方（最低限）

YAML は設定ファイルによく使われる書式です。

| 書き方 | 意味 |
|---|---|
| `キー: 値` | 名前と値の組 |
| 字下げ（スペース2つ） | 階層。字下げされた行は、1つ上の行の「中身」になる。**タブは使えない** |
| `- ` で始まる行 | リスト（配列）の1要素 |
| `#` 以降 | コメント（実行には影響しない） |
| `run: \|` | `\|` の次の行から、字下げされている部分を**改行を保ったまま**1つの文字列として扱う（複数行のコマンドを書くのに使う） |
| `"..."` | 文字列。`*` や `:` など YAML で特別な意味を持つ記号を含むときは引用符で囲む |

---

## 3. 処理の全体像

```
トリガー：毎日 10:17 JST（実際は数時間遅れることが多い）／手動実行
 │
 └─ ジョブ collect（ubuntu-latest、最長10分）
     ├─ ① actions/checkout          リポジトリのコードを取得
     ├─ ② actions/setup-python      Python 3.12 を用意（pip のキャッシュ付き）
     ├─ ③ 依存パッケージのインストール   pip install -r requirements-collect.txt
     ├─ ④ データ収集                python collect.py（失敗してもここでは止めない）
     ├─ ⑤ 変更があればコミット        差分があれば commit & push
     └─ ⑥ 失敗していればジョブを失敗   ④の終了コードが 0 以外なら exit 1
```

---

## 4. ワークフロー全体の設定を順番に読む

### `name` — ワークフローの名前（[collect.yml:1](../.github/workflows/collect.yml#L1)）

```yaml
name: collect snow data
```

GitHub の「Actions」タブに表示される名前です。

### `on` — いつ動かすか（[collect.yml:3-13](../.github/workflows/collect.yml#L3-L13)）

```yaml
on:
  schedule:
    - cron: "17 1 * * *"
  workflow_dispatch:
```

#### `schedule` と cron 式

`cron` は「いつ実行するか」を5つの欄で表す書き方です。左から順に次の意味です。

| 欄 | 分 | 時 | 日 | 月 | 曜日 |
|---|---|---|---|---|---|
| 値 | `17` | `1` | `*` | `*` | `*` |
| 意味 | 17分 | 1時 | 毎日 | 毎月 | 全曜日 |

`*` は「すべて」の意味です。つまり「毎日 1時17分」ですが、**GitHub Actions の cron は UTC（協定世界時）** で解釈されます。日本時間（JST）は UTC + 9時間なので、**毎日 10:17 JST** になります。

コメント（[collect.yml:5-11](../.github/workflows/collect.yml#L5-L11)）には、この時刻にした理由が書かれています。

- 観測は朝9時だが、市サイトに掲載される時刻が分からないので余裕を取って10時台にした
- 分を `:30` から `:17` にずらした。GitHub は「毎時0分など混む時刻を避けて分を散らす」ことを推奨しているため
- 実際には指定時刻から平均5時間半ほど遅れて実行されていた（2026-09-27〜10-04 の観測）

> **注意**: GitHub Actions の `schedule` は「その時刻ちょうどに必ず動く」ものではありません。GitHub 側が混んでいると遅れたり、まれに飛ばされたりします。たとえば 2026-10-28 の実行記録（`data/status.json` の `last_run_at`）は 15:54 JST でした。

#### `workflow_dispatch`

値を何も書いていませんが、書いておくだけで **GitHub の画面から手動で実行できる** ようになります（[7章](#7-手動で動かす結果を確かめる)参照）。

### `permissions` — 何を許可するか（[collect.yml:15-16](../.github/workflows/collect.yml#L15-L16)）

```yaml
permissions:
  contents: write
```

ワークフローには、GitHub が自動で発行する一時的な鍵（`GITHUB_TOKEN`）が渡されます。この鍵でできることを指定するのが `permissions` です。

- `contents: write` … リポジトリの中身への**書き込み**（コミットの push）を許可する。
- これがないと、⑤のステップで `git push` が権限エラーになります。
- 必要な権限だけを書いておくと、それ以外（Issue の操作など）はできなくなるので安全です。

### `concurrency` — 同時に動かさない（[collect.yml:18-20](../.github/workflows/collect.yml#L18-L20)）

```yaml
concurrency:
  group: collect-snow-data
  cancel-in-progress: false
```

- `group` … 同じグループ名のワークフローは**同時に1つしか動かない**。定時実行と手動実行が重なっても、片方は前の実行が終わるまで待つ。
- `cancel-in-progress: false` … 後から来た実行のために、**動いている途中の実行を取り消さない**。

同時に2つ動くと、両方が同じファイルを書き換えて push しようとし、片方の push が失敗する（衝突する）おそれがあります。それを防ぐための設定です。

---

## 5. ジョブとステップを順番に追う

### ジョブの設定（[collect.yml:22-25](../.github/workflows/collect.yml#L22-L25)）

```yaml
jobs:
  collect:
    runs-on: ubuntu-latest
    timeout-minutes: 10
```

| キー | 説明 |
|---|---|
| `jobs:` | この下にジョブを並べる |
| `collect:` | ジョブの名前（ID）。自分で好きに付けられる |
| `runs-on: ubuntu-latest` | GitHub が用意する最新版の Ubuntu（Linux）で動かす |
| `timeout-minutes: 10` | 10分を超えたら強制終了する。市サイトが応答しないなどで止まり続けるのを防ぐ（通常は1分程度で終わる） |

`steps:` の下に並んだステップが、上から順に実行されます。

---

### ① リポジトリを取得する（[collect.yml:28](../.github/workflows/collect.yml#L28)）

```yaml
- uses: actions/checkout@v4
```

- `uses:` … 公開されているアクションを使う。`作者/リポジトリ名@バージョン` の形で指定する。
- `actions/checkout` … GitHub 公式のアクション。ランナー上にリポジトリを `git clone` 相当の方法で取ってくる。
- `@v4` … バージョン4系を使う。バージョンを固定しておくと、アクション側が大きく変わっても突然壊れにくい。

ランナーは毎回まっさらなので、まずこれをしないと `collect.py` もデータファイルもありません。
また checkout は、後で `git push` できるように `GITHUB_TOKEN` を Git の設定に入れておいてくれます。

---

### ② Python を用意する（[collect.yml:30-34](../.github/workflows/collect.yml#L30-L34)）

```yaml
- uses: actions/setup-python@v5
  with:
    python-version: "3.12"
    cache: pip
    cache-dependency-path: requirements-collect.txt
```

| キー | 説明 |
|---|---|
| `actions/setup-python@v5` | 指定したバージョンの Python を使えるようにする公式アクション |
| `with:` | アクションに渡す設定（引数） |
| `python-version: "3.12"` | Python 3.12 を使う。引用符がないと `3.1` と `3.10` のように数値として誤解されることがあるので、文字列にしておくのが定番 |
| `cache: pip` | pip がダウンロードしたライブラリを保存（キャッシュ）しておき、次回の実行で使い回す。インストールが速くなる |
| `cache-dependency-path` | このファイルの中身が変わったらキャッシュを作り直す、という目印のファイル |

---

### ③ 依存パッケージのインストール（[collect.yml:36-37](../.github/workflows/collect.yml#L36-L37)）

```yaml
- name: 依存パッケージのインストール
  run: pip install -r requirements-collect.txt
```

- `name:` … ステップの表示名。Actions の画面でこの名前が出る（①②のように省略するとアクション名が表示される）。
- `run:` … ランナー上で実行するシェルのコマンド。
- `pip install -r ファイル名` … ファイルに書かれたライブラリをまとめてインストールする。

[requirements-collect.txt](../requirements-collect.txt) には `collect.py` に必要な最小限（pandas・requests・beautifulsoup4・lxml）だけが書かれています。Streamlit 版用の `requirements.txt` を使わないのは、使わない大きなライブラリ（streamlit・plotly）を毎朝インストールする時間を省くためです。

---

### ④ データ収集（[collect.yml:39-48](../.github/workflows/collect.yml#L39-L48)）

```yaml
- name: データ収集
  id: collect
  run: |
    set +e
    python collect.py
    code=$?
    echo "exit_code=$code" >> "$GITHUB_OUTPUT"
    echo "collect.py の終了コード: $code"
```

このワークフローで一番の工夫がある部分です。**`collect.py` が失敗しても、このステップ自体は成功扱いで終わらせ、結果（終了コード）だけを後のステップに渡します**（理由は[6章](#6-なぜ失敗しても先にコミットするのか)）。

| コード | 説明 |
|---|---|
| `id: collect` | このステップに名前を付ける。⑥で `steps.collect.outputs...` として結果を参照するのに使う |
| `set +e` | 「コマンドが失敗してもスクリプトを止めない」設定。GitHub Actions の `run` は、途中のコマンドが失敗するとその場で止まる設定（`set -e`）で動くため、それを解除している |
| `python collect.py` | 収集スクリプトを実行。成功なら終了コード `0`、失敗なら `1` で終わる（`collect.py` の `main()` が返す値） |
| `code=$?` | `$?` は「直前のコマンドの終了コード」。それを変数 `code` に入れる。**`=` の前後に空白を入れない**のがシェルのルール |
| `echo "exit_code=$code" >> "$GITHUB_OUTPUT"` | ステップの**出力**として `exit_code` を記録する。`$GITHUB_OUTPUT` は GitHub が用意する特別なファイルで、ここに `名前=値` を書くと後のステップから読める。`>>` はファイルの末尾に追記する記号 |
| `echo "collect.py の終了コード: $code"` | ログに表示するだけ |

---

### ⑤ 変更があればコミット（[collect.yml:50-60](../.github/workflows/collect.yml#L50-L60)）

```yaml
- name: 変更があればコミット
  run: |
    git config user.name "github-actions[bot]"
    git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
    if [ -z "$(git status --porcelain)" ]; then
      echo "差分なし。コミットしません。"
      exit 0
    fi
    git add -A
    git commit -m "data: update snow data ($(TZ=Asia/Tokyo date +%F))"
    git push
```

**処理の流れ**

1. コミットする人の名前とメールアドレスを設定する
2. ファイルに変化がなければ、何もせずにこのステップを成功で終える
3. 変化があれば、すべての変更をコミットして push する

| コード | 説明 |
|---|---|
| `git config user.name / user.email` | コミットの作者情報。ランナーには設定がないので毎回指定する。この名前とアドレスは GitHub の公式ボット `github-actions[bot]` のもので、`41898282` はそのボットのユーザーID。こうすると GitHub の画面でボットのアイコン付きで表示される |
| `$(コマンド)` | **コマンド置換**。中のコマンドを実行し、その出力に置き換える |
| `git status --porcelain` | 変更されたファイルを、機械が読みやすい簡潔な形式で一覧表示する。変更がなければ何も出力しない |
| `[ -z "文字列" ]` | 文字列が**空なら真**。つまり「変更が何もなければ」 |
| `if [ 条件 ]; then ... fi` | シェルの if 文。`fi` は `if` を逆から書いたもので、if の終わりを表す |
| `exit 0` | このステップを成功（0）で終える。以降のコマンドは実行しない |
| `git add -A` | 追加・変更・削除をすべてコミット対象にする |
| `TZ=Asia/Tokyo date +%F` | 日本時間の今日の日付を `2026-10-28` の形で出す。`TZ=...` はこのコマンドだけタイムゾーンを変える指定、`%F` は `年-月-日` の書式 |
| `git commit -m "..."` | コミットする。メッセージは `data: update snow data (2026-10-28)` のようになる |
| `git push` | GitHub に送る。`permissions: contents: write` と、①で設定された `GITHUB_TOKEN` のおかげで成功する |

**差分チェックを入れている理由**: 変更がないのに `git commit` すると「コミットするものがない」というエラーになり、ステップが失敗してしまいます。オフシーズンでも `data/status.json` の `last_run_at`（実行日時）は毎回変わるので、実際には毎日コミットされることがほとんどです。

---

### ⑥ 収集が失敗していればジョブを失敗させる（[collect.yml:62-69](../.github/workflows/collect.yml#L62-L69)）

```yaml
- name: 収集が失敗していればジョブを失敗させる
  run: |
    code="${{ steps.collect.outputs.exit_code }}"
    if [ "$code" != "0" ]; then
      echo "collect.py が失敗しました（終了コード $code）。原因は data/status.json に記録されています。"
      exit 1
    fi
    echo "正常終了しました。"
```

| コード | 説明 |
|---|---|
| `${{ ... }}` | GitHub Actions の**式**。シェルが動く前に GitHub が値を埋め込む |
| `steps.collect.outputs.exit_code` | `id: collect` のステップ（④）が `$GITHUB_OUTPUT` に書いた `exit_code` の値 |
| `[ "$code" != "0" ]` | `code` が `"0"` でなければ真 |
| `exit 1` | 0 以外で終わると、このステップ＝ジョブ全体が**失敗**になる。Actions の画面に赤い ✕ が付き、GitHub から通知メールが届く |

> `${{ }}` はシェルの `$変数` とは別物です。`${{ }}` は GitHub が先に文字を置き換え、`$code` はその後シェルが実行時に置き換えます。

---

## 6. なぜ「失敗しても先にコミットする」のか

素直に書くなら、`python collect.py` が失敗した時点でジョブを止めればよさそうです。しかしそうすると、⑤のコミットが実行されません。

`collect.py` は失敗したときも `data/status.json` に `"result": "error"` と原因を書き込みます。**この失敗の記録もコミットして Git の履歴に残したい**ので、次の順番にしています。

```
④ collect.py を実行 ── 失敗しても止めない。終了コードだけメモ
        │
⑤ コミット & push ──── 成功時はデータ、失敗時は status.json（失敗の記録）が残る
        │
⑥ 終了コードを確認 ─── 0 以外なら、ここで初めてジョブを失敗にする → 通知メール
```

これで「失敗したことを通知で知る」ことと「失敗の内容を履歴に残す」ことを両立しています。`collect.py` 側のコメントにある「Git履歴がそのまま成績表になる」という考え方です。

### `collect.py` の終了コードとの対応

| `collect.py` の結果 | 終了コード | `status.json` の `result` | ジョブ |
|---|---|---|---|
| 成功（データ更新あり） | 0 | `ok` | 成功 |
| 成功（オフシーズンで取得なし） | 0 | `ok`（note: 当シーズンのページ未公開） | 成功 |
| `CollectError`（市サイトの構造変更など） | 1 | `error` | **失敗** |
| 想定外のエラー（バグなど） | 1 | `error` | **失敗** |

---

## 7. 手動で動かす・結果を確かめる

### 手動で実行する（`workflow_dispatch`）

- **画面から**: GitHub のリポジトリ → 「Actions」タブ → 左の一覧から「collect snow data」→ 右の「Run workflow」ボタン
- **ターミナルから**（`gh` コマンドを使う場合）:
  ```bash
  gh workflow run collect.yml
  ```

### 結果を確かめる

```bash
gh run list --workflow=collect.yml --limit 5   # 最近の実行一覧（成功・失敗・所要時間）
gh run view --log-failed                        # 失敗した実行のログを表示
cat data/status.json                            # 最後の実行の記録（git pull してから）
```

### 失敗の通知はどこに届くか

スケジュール実行が失敗したときの通知は、GitHub の仕様上「**そのワークフローの cron 設定を最後に変更した人**」に届きます。通知を受けるかどうかは、GitHub の Settings → Notifications → Actions で設定できます。

---

## 8. よく出てくる書き方まとめ

### GitHub Actions

| 書き方 | 意味 |
|---|---|
| `on:` | トリガー（いつ動かすか） |
| `schedule: - cron: "..."` | 定時実行。**時刻は UTC** |
| `workflow_dispatch:` | 手動実行を可能にする |
| `permissions:` | `GITHUB_TOKEN` に与える権限 |
| `concurrency:` | 同時実行の制御 |
| `runs-on:` | どの OS のランナーで動かすか |
| `uses: 作者/名前@版` | 公開アクションを使う |
| `with:` | アクションへの設定 |
| `run:` | シェルコマンドを実行 |
| `id:` | ステップに名前を付け、後から結果を参照できるようにする |
| `$GITHUB_OUTPUT` | ステップの出力を書き込む特別なファイル |
| `${{ steps.<id>.outputs.<名前> }}` | 前のステップの出力を読む式 |

### シェル（bash）

| 書き方 | 意味 |
|---|---|
| `set +e` / `set -e` | 失敗しても続ける / 失敗したら止める |
| `$?` | 直前のコマンドの終了コード（0 = 成功） |
| `変数=値` | 変数に代入（`=` の前後に空白を入れない） |
| `$変数` / `"$変数"` | 変数の値を使う。`"..."` で囲むと空白を含んでいても安全 |
| `$(コマンド)` | コマンドの出力に置き換える |
| `>>` | ファイルの末尾に追記 |
| `[ -z "文字列" ]` | 文字列が空なら真 |
| `[ "A" != "B" ]` | A と B が違えば真 |
| `if ...; then ... fi` | 条件分岐 |
| `exit 0` / `exit 1` | 成功 / 失敗で終了 |
| `TZ=Asia/Tokyo コマンド` | そのコマンドだけ日本時間で動かす |
