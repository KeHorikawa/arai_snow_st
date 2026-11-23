"""Streamlit版の画面テスト（streamlit.testing.v1.AppTest を使用）

ブラウザを開かずにアプリを実行し、描画された要素やウィジェット操作の結果を検査する。

    python tests/test_app.py     そのまま実行（pytest不要）
    pytest tests/test_app.py     pytest があればこちらでも動く

前提：`snow_data_history.csv` がリポジトリ直下にあること。
テストは読むだけで、ファイルを書き換えない。

履歴CSVは毎朝増えていく。テストに「今の最新月」を直書きすると、
季節が変わっただけで落ちる（2026年11月に実際に起きた）。
最新月はCSVから求め、件数を確かめるテストには値のそろった過去の月を使う。
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

BASE_DIR = Path(__file__).resolve().parent.parent
APP_FILE = str(BASE_DIR / "main.py")

# 全日が欠測（行はあるが観測値が1つも無い）と分かっている月。
# 市のページにその月の観測値が入っていないケースで、不具合ではない。
ALL_MISSING = (2014, 11, "新井消防署")

# 3地点とも値がそろっていると分かっている過去の月。グラフの件数を確かめるのに使う。
# 3地点目は CSV では「妙高市役所 妙高支所」だが、画面では「関山」として読めること。
FULL_MONTH = (2025, 1)

LOCATIONS = ["新井消防署", "頸南消防署", "関山"]


def _latest_month_with_data() -> tuple[int, int]:
    """履歴CSVで、観測値が1つでもある最新の（年, 月）"""
    df = pd.read_csv(BASE_DIR / "snow_data_history.csv")
    df = df[df["snowfall_cm"].notna() | df["snowdepth_cm"].notna()]
    last = df.sort_values(["year", "month"]).iloc[-1]
    return int(last["year"]), int(last["month"])


def _run() -> AppTest:
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    at.run()
    return at


def _charts(at: AppTest) -> int:
    return Counter(e.type for e in at.main)["plotly_chart"]


def _shown(at: AppTest) -> int:
    """表示された条件の件数（グラフ＋「欠測」「未収集」の案内）。

    シーズン初めは地点によって全日欠測のことがあり、グラフの数だけでは数えられない。
    """
    return _charts(at) + len(at.info)


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


def test_既定の表示は3件で地点は3つ():
    at = _run()
    assert len(at.error) == 0, [e.value for e in at.error]
    # 季節の初めは全日欠測の地点があるので、グラフと案内を合わせて3件
    assert _shown(at) == 3
    assert [sb.value for sb in at.sidebar.selectbox[2::3]] == LOCATIONS


def test_サイドバーの構成():
    at = _run()
    # 条件3件 × （年・月・観測地点）
    assert len(at.sidebar.selectbox) == 9
    assert [sb.label for sb in at.sidebar.selectbox[:3]] == ["年", "月", "観測地点"]
    assert [b.label for b in at.sidebar.button] == ["🔄 データを再読み込み"]
    assert list(at.sidebar.selectbox[2].options) == LOCATIONS

    # 年の選択肢は履歴CSV由来。市の一覧から2009年12月のリンクが外れても2009年から選べること。
    # AppTest は選択肢を文字列で返すため、数値に直して比べる。
    years = [int(y) for y in at.sidebar.selectbox[0].options]
    assert len(years) >= 18, years
    assert min(years) == 2009, years


def test_既定の選択は値のある最新月():
    at = _run()
    year, month = _latest_month_with_data()
    assert at.sidebar.selectbox[0].value == year
    assert at.sidebar.selectbox[1].value == month


def test_値のそろった月は3件ともグラフになる():
    at = _run()
    year, month = FULL_MONTH
    for i, loc in enumerate(LOCATIONS):
        _select(at, i, year, month, loc)
    assert _charts(at) == 3
    assert len(at.error) == 0
    assert len(at.info) == 0


def test_全日欠測の月は案内を出す():
    at = _run()
    year, month, location = ALL_MISSING
    _select(at, 0, year, month, location)

    messages = [i.value for i in at.info]
    assert any("全日が欠測" in m for m in messages), messages
    assert any(f"{year}年{month}月 / {location}" in m for m in messages), messages
    assert _shown(at) == 3


def test_条件の重複は警告して1件にまとめる():
    at = _run()
    year, month = FULL_MONTH
    _select(at, 0, year, month, "新井消防署")
    _select(at, 1, year, month, "新井消防署")

    warnings = [w.value for w in at.sidebar.warning]
    assert any("重複しています" in w for w in warnings), warnings
    assert _shown(at) == 2


def test_再読み込みボタンが動く():
    at = _run()
    at.sidebar.button[0].click().run()
    assert not at.exception, f"例外が発生しました: {at.exception}"
    assert _shown(at) == 3


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
