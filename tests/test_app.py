"""Streamlit版の画面テスト（streamlit.testing.v1.AppTest を使用）

ブラウザを開かずにアプリを実行し、描画された要素やウィジェット操作の結果を検査する。

    python tests/test_app.py     そのまま実行（pytest不要）
    pytest tests/test_app.py     pytest があればこちらでも動く

前提：`snow_data_history.csv` と `data_urls.csv` がリポジトリ直下にあること。
テストは読むだけで、ファイルを書き換えない。
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

from streamlit.testing.v1 import AppTest

BASE_DIR = Path(__file__).resolve().parent.parent
APP_FILE = str(BASE_DIR / "main.py")

# 全日が欠測（行はあるが観測値が1つも無い）と分かっている月。
# 市のページにその月の観測値が入っていないケースで、不具合ではない。
ALL_MISSING = (2014, 11, "新井消防署")

# これまで「最新月」として毎回Web取得していた月。履歴CSVから読めるようになった。
LATEST_MONTH = (2026, 3)


def _run() -> AppTest:
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    at.run()
    return at


def _charts(at: AppTest) -> int:
    return Counter(e.type for e in at.main)["plotly_chart"]


def _select(at: AppTest, index: int, year: int, month: int, location: str) -> AppTest:
    """条件 index（0始まり）の 年・月・観測地点 を設定する。

    年を変えると月の選択肢が変わるため、1つずつ run() を挟む。
    """
    at.sidebar.selectbox[index * 3].set_value(year).run()
    at.sidebar.selectbox[index * 3 + 1].set_value(month).run()
    at.sidebar.selectbox[index * 3 + 2].set_value(location).run()
    return at


def test_起動して例外が出ない():
    at = _run()
    assert not at.exception, f"例外が発生しました: {at.exception}"
    assert [t.value for t in at.title] == ["❄️ 妙高市 降雪・積雪データ可視化"]


def test_既定の表示はグラフ3件():
    at = _run()
    assert _charts(at) == 3
    assert len(at.error) == 0
    assert len(at.info) == 0


def test_サイドバーの構成():
    at = _run()
    # 条件3件 × （年・月・観測地点）
    assert len(at.sidebar.selectbox) == 9
    assert [sb.label for sb in at.sidebar.selectbox[:3]] == ["年", "月", "観測地点"]
    assert [b.label for b in at.sidebar.button] == ["🔄 データを再読み込み"]

    # 年の選択肢は data_urls.csv 由来（自動生成で2009年まで遡る）。
    # AppTest は選択肢を文字列で返すため、数値に直して比べる。
    years = [int(y) for y in at.sidebar.selectbox[0].options]
    assert len(years) >= 18, years
    assert min(years) == 2009, years


def test_既定の選択は最新公開月():
    at = _run()
    year, month = LATEST_MONTH
    assert at.sidebar.selectbox[0].value == year
    assert at.sidebar.selectbox[1].value == month


def test_最新月がCSVから表示できる():
    """以前は毎回Web取得していた月。スクレイピングなしで描けることを確かめる。"""
    at = _run()
    year, month = LATEST_MONTH
    _select(at, 0, year, month, "新井消防署")
    assert _charts(at) == 3
    assert len(at.error) == 0
    assert len(at.info) == 0


def test_全日欠測の月は案内を出す():
    at = _run()
    year, month, location = ALL_MISSING
    _select(at, 0, year, month, location)

    messages = [i.value for i in at.info]
    assert any("全日が欠測" in m for m in messages), messages
    assert any(f"{year}年{month}月" in m for m in messages), messages
    # その条件だけグラフが出ず、残り2件は描画される
    assert _charts(at) == 2


def test_条件の重複は警告して1件にまとめる():
    at = _run()
    year, month = LATEST_MONTH
    _select(at, 0, year, month, "新井消防署")
    _select(at, 1, year, month, "新井消防署")

    warnings = [w.value for w in at.sidebar.warning]
    assert any("重複しています" in w for w in warnings), warnings
    assert _charts(at) == 2


def test_再読み込みボタンが動く():
    at = _run()
    at.sidebar.button[0].click().run()
    assert not at.exception, f"例外が発生しました: {at.exception}"
    assert _charts(at) == 3


class _LogCatcher(logging.Handler):
    """Streamlit が出すログを拾うためのハンドラ"""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_非推奨のAPIを使っていない():
    """use_container_width など、Streamlit の非推奨APIを使っていないことを確かめる。

    非推奨の警告は画面には出ず、Streamlit のロガーへ流れる。そのため
    画面要素ではなくログを捕まえて判定する。
    """
    # 非推奨の警告は streamlit.deprecation_util が出す。このロガーは
    # propagate=False のため、ルートに付けたハンドラには届かない。
    # 特定のロガー名に頼りすぎないよう、streamlit 配下すべてに付ける。
    catcher = _LogCatcher()
    targets = [logging.getLogger()] + [
        logging.getLogger(name)
        for name in list(logging.root.manager.loggerDict)
        if name == "streamlit" or name.startswith("streamlit.")
    ]
    for lg in targets:
        lg.addHandler(catcher)
    try:
        _run()
    finally:
        for lg in targets:
            lg.removeHandler(catcher)

    deprecated = [
        m for m in catcher.messages
        if "deprecat" in m.lower() or "will be removed" in m.lower()
    ]
    assert not deprecated, f"非推奨APIの警告が出ています: {deprecated}"


def main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = 0
    for name, func in tests:
        try:
            func()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}\n        {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}\n        {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
