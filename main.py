import logging
from datetime import datetime
from typing import Tuple, List

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========
# ロギング
# ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========
# 定数
# ==========
LOCATIONS: List[str] = ["新井消防署", "頸南消防署", "妙高市役所 妙高支所"]

CSV_FILE = "data_urls.csv"
HISTORY_CSV_FILE = "snow_data_history.csv"

HISTORY_REQUIRED_COLS = ["year", "month", "day", "location", "snowfall_cm", "snowdepth_cm"]


# ==========
# データ読み込み
# ==========
# データの取得は collect.py（GitHub Actions が毎朝実行）が行う。
# このアプリは保存されたファイルを読むだけで、市サイトへは取りに行かない。
@st.cache_data(ttl=3600)
def load_url_data() -> pd.DataFrame:
    """URL一覧CSVファイルを読み込む（列: 年, 月, URL を想定）

    このアプリでは年・月の選択肢としてのみ使う（URLは参照しない）。
    ファイル自体は collect.py が毎朝インデックスページから再生成する。
    """
    try:
        df = pd.read_csv(CSV_FILE)
        # 最低限の列チェック
        required = {"年", "月", "URL"}
        if not required.issubset(df.columns):
            st.error(f"{CSV_FILE} に必要な列（年, 月, URL）がありません。")
            return pd.DataFrame()
        return df
    except Exception as e:
        logger.error(f"CSVファイルの読み込みに失敗: {e}")
        st.error("データURLの読み込みに失敗しました")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_history_data() -> pd.DataFrame:
    """観測データCSVファイルを読み込む（無ければ空DataFrame）"""
    try:
        df = pd.read_csv(HISTORY_CSV_FILE)
        if df.empty:
            return pd.DataFrame(columns=HISTORY_REQUIRED_COLS)

        # 必須列チェック
        missing_cols = [c for c in HISTORY_REQUIRED_COLS if c not in df.columns]
        if missing_cols:
            logger.warning(f"履歴CSVに必要列が不足: {missing_cols}")
            return pd.DataFrame(columns=HISTORY_REQUIRED_COLS)

        # 型を安全に整える（壊れた値があっても落ちにくく）
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
        df["day"] = pd.to_numeric(df["day"], errors="coerce").astype("Int64")

        # year/month/day が欠損の行は捨てる
        df = df.dropna(subset=["year", "month", "day"]).copy()
        df["year"] = df["year"].astype(int)
        df["month"] = df["month"].astype(int)
        df["day"] = df["day"].astype(int)

        # location は文字列で統一
        df["location"] = df["location"].astype(str)

        return df

    except FileNotFoundError:
        logger.warning("観測データファイルが存在しません。")
        return pd.DataFrame(columns=HISTORY_REQUIRED_COLS)
    except Exception as e:
        logger.error(f"観測データファイルの読み込みに失敗: {e}")
        return pd.DataFrame(columns=HISTORY_REQUIRED_COLS)


# ==========
# 最新公開月（URL一覧ベース）
# ==========
def get_latest_available_month(url_df: pd.DataFrame) -> Tuple[int, int]:
    """URL一覧から、最新の（年,月）を返す"""
    if url_df.empty:
        now = datetime.now()
        return now.year, now.month

    sorted_df = url_df.sort_values(["年", "月"], ascending=False)
    latest = sorted_df.iloc[0]
    return int(latest["年"]), int(latest["月"])


# ==========
# データ取得
# ==========
def get_month_df(
    *,
    year: int,
    month: int,
    location: str,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """指定年月・指定地点のデータを観測データCSVから取り出す"""
    if history_df.empty:
        return history_df

    return history_df[
        (history_df["year"] == year)
        & (history_df["month"] == month)
        & (history_df["location"] == location)
    ].copy()


def has_any_value(df: pd.DataFrame) -> bool:
    """降雪量・積雪量のどちらかに値がある行が1つでもあるか（全日欠測の月を判別する）"""
    if df.empty:
        return False
    return bool(df["snowfall_cm"].notna().any() or df["snowdepth_cm"].notna().any())


# ==========
# グラフ
# ==========
def create_snow_graph(df: pd.DataFrame, year: int, month: int, location: str) -> go.Figure:
    """降雪・積雪のグラフを作成"""
    filtered_df = df[
        (df["year"] == year) & (df["month"] == month) & (df["location"] == location)
    ].sort_values("day")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 降雪量（棒）: 左
    fig.add_trace(
        go.Bar(
            x=filtered_df["day"],
            y=filtered_df["snowfall_cm"],
            name="降雪量",
            opacity=0.7,
        ),
        secondary_y=False,
    )

    # 積雪量（線）: 右
    fig.add_trace(
        go.Scatter(
            x=filtered_df["day"],
            y=filtered_df["snowdepth_cm"],
            name="積雪量",
            mode="lines+markers",
            line=dict(color="red"),
            marker=dict(color="red"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=f"{year}年{month}月 / {location}",
        xaxis_title="日",
        hovermode="x unified",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(range=[0.5, 31.5], dtick=5)

    fig.update_yaxes(
        title_text="降雪量 (cm)",
        range=[0, 100],
        dtick=20,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="積雪量 (cm)",
        range=[0, 300],
        dtick=60,
        secondary_y=True,
    )

    return fig


# ==========
# メイン
# ==========
def main() -> None:
    st.set_page_config(page_title="妙高市 降雪・積雪データ可視化", page_icon="❄️", layout="wide")

    st.title("❄️ 妙高市 降雪・積雪データ可視化")
    st.markdown("---")

    url_df = load_url_data()
    if url_df.empty:
        st.error("データURLが読み込めません。data_urls.csv を確認してください。")
        return

    latest_year, latest_month = get_latest_available_month(url_df)

    # 利用可能な年
    available_years = sorted(url_df["年"].unique())

    # サイドバー
    st.sidebar.header("📊 表示条件設定")
    st.sidebar.markdown("最大3件まで選択できます")

    selections = []
    for i in range(3):
        st.sidebar.markdown(f"### 条件 {i + 1}")
        col1, col2 = st.sidebar.columns(2)

        with col1:
            default_year_idx = available_years.index(latest_year) if latest_year in available_years else 0
            year = st.selectbox("年", options=available_years, index=default_year_idx, key=f"year_{i}")

        with col2:
            available_months = sorted(url_df[url_df["年"] == year]["月"].unique())
            default_month_idx = available_months.index(latest_month) if latest_month in available_months else 0
            month = st.selectbox("月", options=available_months, index=default_month_idx, key=f"month_{i}")

        default_location_idx = i if i < len(LOCATIONS) else 0
        location = st.sidebar.selectbox(
            "観測地点", options=LOCATIONS, index=default_location_idx, key=f"location_{i}"
        )

        selections.append({"year": year, "month": month, "location": location})
        st.sidebar.markdown("---")

    # 重複チェック
    unique_selections = []
    seen = set()
    for sel in selections:
        key = (sel["year"], sel["month"], sel["location"])
        if key in seen:
            st.sidebar.warning(f"⚠️ {sel['year']}年{sel['month']}月 / {sel['location']} が重複しています")
            continue
        seen.add(key)
        unique_selections.append(sel)

    if st.sidebar.button("🔄 データを再読み込み"):
        st.cache_data.clear()
        st.rerun()

    # 観測データ読み込み
    history_df = load_history_data()

    st.markdown("## 📈 グラフ表示")

    for sel in unique_selections:
        year = int(sel["year"])
        month = int(sel["month"])
        location = sel["location"]

        df = get_month_df(
            year=year,
            month=month,
            location=location,
            history_df=history_df,
        )

        if df.empty:
            st.info(f"ℹ️ {year}年{month}月 / {location} のデータはまだ収集されていません")
            continue

        if not has_any_value(df):
            st.info(f"ℹ️ {year}年{month}月 / {location} は全日が欠測のため、表示できるデータがありません")
            continue

        fig = create_snow_graph(df, year, month, location)
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.link_button(
            "🌨️ 妙高市 雪情報ホームページ",
            "https://www.city.myoko.niigata.jp/life-info/snow-info/snow/",
            width="stretch",
        )
    st.caption("観測時刻: 9時 | 降雪量: 前日分 | 積雪量: 当日分")


if __name__ == "__main__":
    main()
