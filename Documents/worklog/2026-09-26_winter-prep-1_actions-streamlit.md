# 実装ログ：冬支度① データ収集の分離（後半／タスク4〜5）

| 項目 | 内容 |
|---|---|
| 作業日 | 2026-09-26 |
| ブランチ | `winterprep260922`（PR #4）／`winterprep2-streamlit260926`（PR #5）。どちらもマージ済み・削除済み |
| 範囲 | タスク4（GitHub Actions）、タスク5（Streamlit改修） |
| 前半 | [2026-09-22_winter-prep-1_collect.md](2026-09-22_winter-prep-1_collect.md)（タスク0〜3） |
| 残 | タスク6-6（翌朝 cron が自動で動いたことの確認） |

---

## 1. タスク4：GitHub Actions ワークフロー

### 1-1. 作成したファイル

`.github/workflows/collect.yml`（64行）。

```yaml
on:
  schedule:
    - cron: "30 1 * * *"   # 10:30 JST
  workflow_dispatch:
permissions:
  contents: write
concurrency:
  group: collect-snow-data
  cancel-in-progress: false
```

ジョブは `ubuntu-latest` / `timeout-minutes: 10`、Python 3.12。
依存は `requirements-collect.txt` のみ（`streamlit` は入れない）。`actions/setup-python@v5` の pip キャッシュを使用。

ステップは6つ。

1. `actions/checkout@v4`
2. `actions/setup-python@v5`
3. 依存パッケージのインストール
4. データ収集
5. 変更があればコミット
6. 収集が失敗していればジョブを失敗させる

### 1-2. 設計上の判断：終了コードを持ち越す

**collect.py が失敗してもステップ自体は失敗させず、終了コードだけ次へ渡して最後に判定する。**

```bash
# ステップ4
set +e
python collect.py
code=$?
echo "exit_code=$code" >> "$GITHUB_OUTPUT"
```

```bash
# ステップ6
code="${{ steps.collect.outputs.exit_code }}"
if [ "$code" != "0" ]; then exit 1; fi
```

理由：`collect.py` は失敗時にも `data/status.json` へ原因を書く。
ステップ4で即座に落とすとコミットされないため、**失敗の記録がGitに残らない**。
コミットを挟んでから落とすことで、「いつ・何が壊れたか」が後から履歴で追える。

### 1-3. 差分が無ければコミットしない

```bash
if [ -z "$(git status --porcelain)" ]; then
  echo "差分なし。コミットしません。"
  exit 0
fi
```

`git diff --quiet` ではなく `git status --porcelain` を使った。前者は未追跡ファイルを見ないため、
`data/` を新規作成した回を取りこぼす。

### 1-4. リポジトリ設定の変更

事前確認で、Actions のワークフロー既定権限が `read` になっていた。

```
$ gh api repos/KeHorikawa/arai_snow_st/actions/permissions
{"enabled":true,"allowed_actions":"all","sha_pinning_required":false}

$ gh api repos/KeHorikawa/arai_snow_st/actions/permissions/workflow
{"default_workflow_permissions":"read","can_approve_pull_request_reviews":false}
```

ワークフロー側で `permissions: contents: write` を宣言しているが、
リポジトリ既定が `read` のときに宣言で上書きできるか確証がなかったため、**先に `write` へ変更した**。

```bash
gh api -X PUT repos/KeHorikawa/arai_snow_st/actions/permissions/workflow \
  -f default_workflow_permissions=write -F can_approve_pull_request_reviews=false
```

`can_approve_pull_request_reviews` は `false` のまま据え置き。

### 1-5. `workflow_dispatch` の制約

**ワークフローがデフォルトブランチに存在しないと `workflow_dispatch` は実行できない。**
そのため、タスク4の検証はPRのマージ後に行った。

一方、**デフォルトブランチに存在してさえいれば、別のブランチを指定して実行できる**
（`gh workflow run collect.yml --ref <branch>`）。失敗テストはこれを利用した。

---

## 2. タスク4の検証

### 2-1. 成功パターン（手動実行）

[Run 36212279312](https://github.com/KeHorikawa/arai_snow_st/actions/runs/36212279312) — 結果：**success**

```
[INFO] 実行日（JST）: 2026-09-26 / 当シーズン: 2026-27
[INFO] 月別リンクを 97 件検出しました（2009年12月 〜 2026年3月）
[INFO] 既存の履歴: 8796行
[INFO] 当シーズン（2026-27）のページ未公開。取得はスキップします（オフシーズンの正常な状態です）
[INFO] data/snow_data.json を書き出しました
[INFO] data/status.json を書き出しました
[INFO] 完了: 当シーズン（2026-27）のページ未公開（追加・更新 0行）
collect.py の終了コード: 0
[main c5ac042] data: update snow data (2026-09-26)
   2ce00ce..c5ac042  main -> main
```

確認できたこと：

- **オフシーズンが成功として扱われる**（市サイトへのリクエストはインデックス1回のみ）
- bot コミット `c5ac042` の作者は `github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>`
- コミット内容は2ファイル・4行のみ。**`snow_data_history.csv` は差分が無いため含まれない**

```diff
 data/status.json
-  "last_run_at": "2026-09-22T11:39:49+09:00",
+  "last_run_at": "2026-09-26T11:37:59+09:00",
-  "note": "全期間バックフィル（97ページ）",
-  "rows_written": 3,
+  "note": "当シーズン（2026-27）のページ未公開",
+  "rows_written": 0,
```

- JSTの時刻が正しい（11:37 JST = 02:37 UTC）

### 2-2. 失敗パターン（意図的に壊す）

`main` から `test/actions-failure` ブランチを作り、`INDEX_URL` を存在しないパスに書き換えて
`gh workflow run collect.yml --ref test/actions-failure` で実行。

[Run 36212341082](https://github.com/KeHorikawa/arai_snow_st/actions/runs/36212341082) — 結果：**failure**

```
[ERROR] 失敗: インデックスページの取得に失敗: ... - 404 Client Error: Not Found for url: ...
[INFO] data/status.json を書き出しました
collect.py の終了コード: 1
[test/actions-failure 73f5b68] data: update snow data (2026-09-26)
```

ブランチに記録された `data/status.json`：

```json
{
  "last_run_at": "2026-09-26T11:39:07+09:00",
  "result": "error",
  "note": "インデックスページの取得に失敗: ... 404 Client Error: Not Found for url: ...",
  "rows_written": 0,
  "latest_observed_date": null
}
```

**メール通知の受信を確認**（件名：`[KeHorikawa/arai_snow_st] Run failed: collect snow data - test/actions-failure (da82712`）。
これで完了条件のひとつ「ワークフロー失敗時にメール通知が届くこと」を満たした。

検証用ブランチは削除済み。**`main` には一切入れていない。**
Actions の履歴には赤い実行が1件残るが、通知確認の記録として意図的に残している。

なお、GitHub の通知APIは `gh` のトークンに `notifications` スコープが無いため参照できなかった。
メール受信の確認は人が行った。

---

## 3. タスク5：Streamlit版からスクレイピングを外す

### 3-1. 削除したもの

| 対象 | 理由 |
|---|---|
| `fetch_snow_data()` / `_pick_data_table()` / `_to_float_or_none()` | `collect.py` へ移動済み |
| `save_history_data()` | Streamlit Cloud では書いても再起動で消える。今後は Actions が更新する |
| `get_month_df()` の分岐 | 「最新月は毎回Web取得」「履歴に無ければWeb取得して追記」を削除し、すべて `snow_data_history.csv` から読む |
| `is_latest_month()` | 不参照になった |
| `requests` / `BeautifulSoup` / `re` の import | 使わなくなった |
| `st.spinner` | 取得待ちが無くなり、CSV読み込みは一瞬で終わるため |

**456行 → 295行**（161行減）。

### 3-2. データの供給元

**`data/snow_data.json` ではなく `snow_data_history.csv` を読む。**
JSONは当シーズン分しかなく、年・月を18シーズン分から選べる既存機能を賄えないため。

`data_urls.csv` は引き続き**年・月の選択肢としてのみ**使う（URLは参照しない）。
自動生成により97件へ増えたので、年の選択肢は6件から**18件（2009〜2026）**になった。
増えた分もすべてバックフィル済みなので、選択肢に対してデータが存在する。

### 3-3. 追加した表示

依頼にあった「データが無い場合の表示」を実装する際、
**行はあるが観測値が1つも無い組み合わせが42件**あることが分かった。
そのままだと空のグラフが描画されるため、文言を分けた。

```python
if df.empty:
    st.info(f"ℹ️ {year}年{month}月 / {location} のデータはまだ収集されていません")
    continue

if not has_any_value(df):
    st.info(f"ℹ️ {year}年{month}月 / {location} は全日が欠測のため、表示できるデータがありません")
    continue
```

42件の内訳（不具合ではなく、実際に雪が無かった月）：

| 区分 | 内訳 |
|---|---|
| 月別 | 4月 23件、11月 17件、5月 2件 |
| 地点別 | 新井消防署 24件、妙高市役所 妙高支所 14件、頸南消防署 4件 |

シーズンの端の月に集中している。頸南消防署が少ないのは、標高が高く端の月でも積雪が残るためと思われる。

### 3-4. 「🔄 データを再読み込み」ボタンの判断：残した

位置もラベルも変えていない。ただし**意味が変わる**。

- 変更前：キャッシュをクリアして市のサイトから取り直す
- 変更後：キャッシュをクリアして保存ファイルを読み直す

あわせて `load_history_data()` に `@st.cache_data(ttl=3600)` を追加した。
これによりボタンに実際の効果が生まれる（`snow_data_history.csv` が更新されたのに画面が古いままのときに使う）。

**残した理由**：このボタンは連載第9回で意図して追加されたもの。
消すのではなく役割を移す形にした。READMEにも「市のサイトへ取りに行くものではない」と明記した。

### 3-5. UIの変更有無

| 要素 | 変更 |
|---|---|
| `st.set_page_config` / タイトル / 区切り線 | なし |
| サイドバーの構成・ラベル・既定値 | なし |
| `create_snow_graph()` | **1行も変更なし** |
| フッターのリンクボタン・キャプション | なし |
| 重複チェックの警告 | なし |
| データが無いときの `st.info` 2種 | **新規追加**（依頼による） |
| `st.spinner` | 削除（表示時間がゼロになったため） |

---

## 4. READMEの更新

| 節 | 変更内容 |
|---|---|
| 概要・主な機能 | 「閲覧のたびに取得」から「Actions が毎朝取得」へ書き換え |
| 対象期間 | 令和1年度〜 → 平成21年度（2009年）〜。97件すべて収録と明記 |
| ファイル構成 | `collect.py` / `.github/` / `data/` / `plans/` / `requirements-collect.txt` を追加 |
| **データ収集の自動化（新設）** | 構成図、`collect.py` の使い方、手動実行、失敗条件の表、**60日ルールの注意2点**、既知の制約 |
| トラブルシューティング | 「インターネット接続を確認」を削除。`status.json` と Actions タブを見る手順へ |
| HTML構造の変更について | `fetch_snow_data()` → `parse_month_page()`。失敗時にメールが届くことを追記 |
| キャッシュの削除 | 再読み込みボタンの意味の変化を明記 |
| 今後の拡張案 | マイナス値の項目にチェック。隠し文字の修正についても追記 |

**60日ルールの注意2点**（依頼により記載）：

1. GitHubからの「停止予告メール」を見逃さないこと
2. 11月の降雪開始前に必ず手動実行して生きていることを確認すること

---

## 5. 検証結果（タスク5）

| 確認項目 | 結果 |
|---|---|
| `streamlit run main.py` の起動 | `/healthz` 200、エラーログなし |
| **2026年3月の表示** | 31行を読み込み描画OK（これまで毎回Web取得していた月） |
| 通常の月（2026年2月） | 28行、描画OK |
| 全日欠測の月（2014年11月） | 30行あるが全欠測 → 案内メッセージ |
| 最古の月（2009年12月） | 31行、描画OK |
| ファイル書き込み | `snow_data_history.csv` の更新時刻が変わらないことを確認 |
| 年の選択肢 | 2009〜2026 の18件 |
| 描画できる組み合わせ | **249 / 291**（残り42件は全日欠測） |

マージ後の `main` でも同じ確認を再実施し、結果が変わらないことを確認した。

---

## 5-2. 追加作業：画面テストの整備と非推奨APIの修正

タスク5の検証方法について質問を受けたことをきっかけに、検証手段を見直した。

### きっかけ：当時の検証は弱かった

タスク5の時点でやったのは次の3つ。

| 手段 | 分かること | 限界 |
|---|---|---|
| ヘッドレス起動 + `/healthz` | サーバーが起動した | **スクリプト本体が実行されていない**。Streamlit はブラウザ接続で初めて実行するため、`main()` は走っていない |
| `main.py` を import して関数を直接呼ぶ | 各関数の戻り値は正しい | **`main()` の中身（サイドバー構築・ループ・`st.info` の分岐）が未実行** |
| ファイルの mtime 比較 | 書き込みが起きない | 表示とは無関係 |

画面は一度も見ていない。レイアウトも、メッセージが実際に出るところも未確認だった。

### `streamlit.testing.v1.AppTest` を使う

Streamlit 1.28 以降に公式のテスト機能がある（インストール済みは 1.54）。
ブラウザなしでスクリプトを実行し、ウィジェットを操作して、描画された要素を検査できる。

`tests/test_app.py` として9件のテストを追加した。pytest は未導入のため、
**`python tests/test_app.py` で単体実行できる**形にしつつ、pytest でも動くようにしてある。実行は約1.1秒。

| テスト | 内容 |
|---|---|
| `test_起動して例外が出ない` | 例外なし、タイトルが一致 |
| `test_既定の表示はグラフ3件` | グラフ3件、error/info が0件 |
| `test_サイドバーの構成` | selectbox 9件、ラベル、ボタン、年の選択肢が18件・最小2009 |
| `test_既定の選択は最新公開月` | 2026年3月が既定 |
| `test_最新月がCSVから表示できる` | 2026年3月でグラフ3件・エラーなし |
| `test_全日欠測の月は案内を出す` | 2014年11月でメッセージが出て、グラフが2件に減る |
| `test_条件の重複は警告して1件にまとめる` | 重複警告が出て、グラフが2件に |
| `test_再読み込みボタンが動く` | クリックして例外なし、再描画される |
| `test_非推奨のAPIを使っていない` | 非推奨警告のログが出ていない |

これで初めて「画面に何が出るか」を機械的に確認できるようになった。

### 見つかった問題：非推奨APIを使っていた

AppTest で実行したところ、警告が3回出ていた。

```
Please replace `use_container_width` with `width`.
`use_container_width` will be removed after 2025-12-31.
```

`main.py` の2箇所。**削除予定日（2025-12-31）はすでに過ぎているが、まだ動いている状態だった。**
今回の変更で入ったものではなく、以前からあるコード。

```diff
- st.plotly_chart(fig, use_container_width=True)
+ st.plotly_chart(fig, width="stretch")

- st.link_button(..., use_container_width=True)
+ st.link_button(..., width="stretch")
```

`plotly_chart` の `width` は既定値が `stretch` なので省略もできるが、
`link_button` は既定が `content` のため明示が必要。統一して両方とも明示した。見た目は変わらない。

### テストが素通りしていた問題（2回直した）

非推奨を検出するテストは、**最初の2回とも機能していなかった**。
修正前の `main.py` に戻して確かめたところ、落ちるはずが通ってしまった。

| 版 | 方法 | 結果 |
|---|---|---|
| 1回目 | 画面要素（`at.warning`）を調べる | **素通り**。警告は画面に出ず、ログに流れるだけだった |
| 2回目 | `streamlit` ロガーにハンドラを付ける | **素通り**。理由は下記 |
| 3回目 | ルートロガーに付ける | **素通り**。同じ理由 |
| 4回目 | `streamlit` 配下のロガーすべてに付ける | **検出できた** |

原因は、**`streamlit.deprecation_util` ロガーが `propagate = False`** だったこと。
親（`streamlit`）にもルートにもレコードが伝わらないため、そのロガー自身にハンドラを付ける必要があった。

```python
targets = [logging.getLogger()] + [
    logging.getLogger(name)
    for name in list(logging.root.manager.loggerDict)
    if name == "streamlit" or name.startswith("streamlit.")
]
```

最終的に、**修正後は PASS・非推奨に戻すと FAIL** することを両方向で確認した。

> テストを書いたら、**わざと壊して落ちることを確かめる**。
> 今回は3回続けて「通るけれど何も見ていない」テストを書いていた。
> 通ったことは、テストが機能している証拠にはならない。

---

## 6. コミットとPR

| PR | 内容 | マージ |
|---|---|---|
| [#4](https://github.com/KeHorikawa/arai_snow_st/pull/4) | 冬支度①：データ収集を GitHub Actions に分離する（タスク0〜4） | `2ce00ce` / 2026-09-26 02:36 UTC |
| [#5](https://github.com/KeHorikawa/arai_snow_st/pull/5) | 冬支度①（後半）：Streamlit版からスクレイピングを外す（タスク5） | `e7acf6f` / 2026-09-26 02:50 UTC |

```
e7acf6f Merge pull request #5 （タスク5）
3bde30a refactor: Streamlit版からスクレイピングを外す
c5ac042 data: update snow data (2026-09-26)   ← botによる初の自動コミット
2ce00ce Merge pull request #4 （タスク0〜4）
7693e37 feat: 毎朝データを収集する GitHub Actions ワークフローを追加
```

作業ブランチ・検証用ブランチはすべて削除済み。リモートは `main` のみ。

---

## 7. 達成状況

| | タスク | 状態 |
|---|---|---|
| 0 | 計画書をこのリポジトリに保存 | ✅ 完了 |
| 1 | `collect.py` を切り出す | ✅ 完了 |
| 2 | 月別URLの自動発見 | ✅ 完了 |
| 3 | 出力3ファイル | ✅ 完了 |
| 4 | GitHub Actions ワークフロー | ✅ 完了 |
| 5 | Streamlit版からスクレイピングを外す | ✅ 完了 |
| 6 | 動作確認 | 🔶 **6件中5件完了** |

### タスク6の内訳

| # | 確認項目 | 状態 |
|---|---|---|
| 1 | `--dry-run` で取得とパースが通る | ✅ |
| 2 | `--all` で2026年3月と4月分が入る | ✅ |
| 3 | `streamlit run main.py` で2026年3月が表示できる | ✅ |
| 4 | `workflow_dispatch` で手動実行して緑になる | ✅ |
| 5 | わざと失敗させてメール通知が届く | ✅ |
| 6 | **翌朝 cron が自動で動いた** | ⬜ **2026-09-27 に確認** |

### 完了条件に対して

> 手を触れずに毎朝データが更新され、Streamlit版がそれを表示している。
> 加えて、ワークフロー失敗時にメール通知が届くことを確認済みであること。

- **メール通知の確認**：✅ 済み
- **Streamlit版が表示している**：✅ 済み（`main` にマージ済み）
- **手を触れずに毎朝更新される**：🔶 実装と手動実行までは確認済み。**cron による無人実行は未確認**

---

## 8. 明朝（2026-09-27）の確認ポイント

最初の自動実行は **10:30 JST ごろ**。cron は混雑で数分〜数十分遅れるため、**11時すぎに確認**するのが確実。

| # | 見る場所 | 期待される状態 |
|---|---|---|
| 1 | [Actions タブ](https://github.com/KeHorikawa/arai_snow_st/actions) | 緑の実行が1件増えている。契機（Event）が `schedule` になっている |
| 2 | [コミット履歴](https://github.com/KeHorikawa/arai_snow_st/commits/main) | `data: update snow data (2026-09-27)` が増えている |
| 3 | [data/status.json](https://github.com/KeHorikawa/arai_snow_st/blob/main/data/status.json) | `last_run_at` が 2026-09-27、`result` が `ok`、`note` が「当シーズン（2026-27）のページ未公開」 |

**あわせて確認したいこと**：実際に何時に実行されたか。
`cron` に指定した 01:30 UTC からどれだけ遅れたかを記録しておくと、
10月の観察と合わせて実行時刻を前倒しする判断材料になる。

### 動かなかった場合に見るところ

| 症状 | 確認 |
|---|---|
| 実行自体が無い | Actions タブでワークフローが `active` か。リポジトリ設定で Actions が無効化されていないか |
| 実行はあるが赤い | ログの `[ERROR]` 行と `data/status.json` の `note` |
| 緑だがコミットが無い | 「差分なし」と出ていないか（`status.json` は毎回変わるので、通常は起こらない） |
| push で 403 | ワークフロー権限が `write` のままか（本日 `read` から変更済み） |

### 実機で見てほしいこと（Streamlit Cloud）

PR #5 のマージで [公開アプリ](https://araisnowst-7py5ykyrj27appzniuh9aaa.streamlit.app/) が再デプロイされているはず。

- **2026年3月が表示されること**（これまで毎回Web取得していた月）
- **年のプルダウンが2009年まで並んでいること**
- 表示速度（スクレイピングが無くなったぶん速くなっているか）

---

## 9. 申し送り・未解決

前半（9/22）からの繰り越しを含む。

- **60日ルール**：GitHub は公開リポジトリで60日間活動がないと cron を自動停止する。
  bot コミットが「活動」とみなされるかは未確認。停止予告メールの監視と、11月前の手動実行で担保する。
  本日の人間の活動を起点にすると、**2026年11月25日ごろ**が目安になる（降雪開始と重なる）
- **月替わり直後の前月訂正を拾えない**：毎朝取得するのは当シーズンの最新月1枚のみ。
  必要なときは `python collect.py --all` を手動実行する
- **`latest` の地点欠測**：シーズン末や欠測日は、全地点共通の観測日を採るため一部地点が `null` になる。
  静的サイト側で「—」表示の想定が必要
- **隠し文字の正体**：訂正前の値と推測されるが未確認。数字を含む隠し文字（`1035-`）の例があるため、
  他のパターンが潜んでいる可能性は残る
- **実行時刻**：10:30 JST は暫定。市サイトへの掲載時刻が未確認のため余裕をとっている。
  10月の観察（`status.json` の `latest_observed_date`）で前倒しを判断する
- **Python バージョン**：ワークフローは 3.12、ローカルの venv は 3.10。
  いまのところ差は出ていないが、再現時は留意する

---

## 10. 次回（冬支度②）

Phase 2：`data/snow_data.json` を読んで描く静的サイト＋PWA。今回は着手しない。

10月は無人稼働の観察期間。**Actions の成功率が検証数字になる**ので、Git履歴がそのまま成績表になる。

---

## 11. 記事にする際の要点

- **失敗の記録もコミットしてから落とす**という設計。ステップ4で即座に落とすと `status.json` が残らない
- **`workflow_dispatch` はデフォルトブランチに無いと実行できない**。一方、置いてさえあれば別ブランチを指定して実行できる。
  失敗テストをこの仕組みで安全に行った（`main` を汚さない）
- **通知が届くことを実際に確かめた**。「届くはず」で終わらせない
- **ワークフロー権限の既定が `read`** だった。宣言で上書きできるか確証が持てず、先に設定を変えた
- **ボタンを消さずに役割を移す**判断。第9回で意図して追加したものを、意味だけ差し替えた
- **42件の「全日欠測」**は不具合ではなく、実際に雪が無かった月。4月と11月に集中し、頸南消防署だけ少ない
- **60日ルール**と、その期限が降雪開始と重なるという巡り合わせ
- **通ったテストは、機能しているテストとは限らない**。非推奨を検出するテストを3回続けて素通りさせた。
  わざと壊して落ちることを確かめて初めて、テストとして成立した
- **Streamlit にはブラウザ不要の公式テスト機能（`AppTest`）がある**。
  「画面は目で見るしかない」と思い込んでいた部分が、機械で確認できるようになった
