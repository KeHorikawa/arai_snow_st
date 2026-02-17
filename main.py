import logging
from datetime import datetime
from typing import Optional, Tuple, List

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
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
@st.cache_data(ttl=3600)
def load_url_data() -> pd.DataFrame:
    """URL一覧CSVファイルを読み込む（列: 年, 月, URL を想定）"""
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


def load_history_data() -> pd.DataFrame:
    """過去データCSVファイルを読み込む（無ければ空DataFrame）"""
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
        logger.info("過去データファイルが存在しません。新規作成します。")
        return pd.DataFrame(columns=HISTORY_REQUIRED_COLS)
    except Exception as e:
        logger.error(f"過去データファイルの読み込みに失敗: {e}")
        return pd.DataFrame(columns=HISTORY_REQUIRED_COLS)


def save_history_data(df: pd.DataFrame) -> None:
    """過去データをCSVファイルに保存する"""
    try:
        # 列順を揃える（読者が見ても分かりやすい）
        df = df.reindex(columns=HISTORY_REQUIRED_COLS)
        df.to_csv(HISTORY_CSV_FILE, index=False)
        logger.info(f"過去データを保存しました: {len(df)}件")
    except Exception as e:
        logger.error(f"過去データの保存に失敗: {e}")


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


def is_latest_month(year: int, month: int, latest_year: int, latest_month: int) -> bool:
    """指定年月が、URL一覧における最新公開月かどうか"""
    return year == latest_year and month == latest_month


# ==========
# スクレイピング
# ==========
def _pick_data_table(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """
    妙高市ページ内の tables から、観測所名が含まれるテーブルを優先して選ぶ。
    見つからなければ最初のテーブルを返す。
    """
    tables = soup.find_all("table")
    if not tables:
        return None

    # 観測所名が含まれるテーブルを優先
    for tbl in tables:
        text = tbl.get_text(" ", strip=True)
        if all(loc in text for loc in LOCATIONS):
            return tbl

    return tables[0]


def _to_float_or_none(s: str) -> Optional[float]:
    """
    数値っぽい文字列をfloatに。 "-", "--", "" は None。
    "30-" のような末尾 '-' は除去。
    """
    x = (s or "").strip()
    if x in {"-", "--", ""}:
        return None
    x = x.rstrip("-").strip()
    if x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


@st.cache_data(ttl=3600)
def fetch_snow_data(url: str, year: int, month: int) -> Optional[pd.DataFrame]:
    """指定URLから降雪・積雪データを取得して tidy DataFrame を返す（失敗時None）"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding

        soup = BeautifulSoup(response.text, "lxml")

        table = _pick_data_table(soup)
        if table is None:
            logger.warning(f"テーブルが見つかりません: {url}")
            return None

        rows = table.find_all("tr")
        if len(rows) < 3:
            logger.warning(f"テーブル行数が少なすぎます: {url}")
            return None

        data_rows = []

        # ヘッダーは基本2行想定。ただし壊れにくいよう、日付列が取れた行だけ採用する
        for row in rows[1:]:
            cols = row.find_all(["td", "th"])
            if len(cols) < 7:
                continue

            cols_text = [c.get_text(strip=True) for c in cols]

            # 1列目から日付を取る（例: "3日"）
            day_text = cols_text[0].replace("日", "").strip()
            if not day_text.isdigit():
                continue
            day = int(day_text)

            # 実際の列順: [日, 降雪1, 積雪1, 降雪2, 積雪2, 降雪3, 積雪3]
            for i, location in enumerate(LOCATIONS):
                snowfall_idx = i * 2 + 1
                snowdepth_idx = i * 2 + 2

                snowfall_raw = cols_text[snowfall_idx] if snowfall_idx < len(cols_text) else "-"
                snowdepth_raw = cols_text[snowdepth_idx] if snowdepth_idx < len(cols_text) else "-"

                snowfall_cm = _to_float_or_none(snowfall_raw)
                snowdepth_cm = _to_float_or_none(snowdepth_raw)

                # 積雪量が負になるのは仕様的におかしいので None 扱い
                if snowdepth_cm is not None and snowdepth_cm < 0:
                    snowdepth_cm = None

                data_rows.append(
                    {
                        "year": year,
                        "month": month,
                        "day": day,
                        "location": location,
                        "snowfall_cm": snowfall_cm,
                        "snowdepth_cm": snowdepth_cm,
                    }
                )

        if not data_rows:
            logger.warning(f"データが抽出できませんでした: {url}")
            return None

        return pd.DataFrame(data_rows)

    except requests.RequestException as e:
        logger.error(f"HTTPリクエストエラー: {url} - {e}")
        return None
    except Exception as e:
        logger.error(f"データ取得エラー: {url} - {e}")
        return None


# ==========
# データ取得戦略（分岐ロジックを集約）
# ==========
def history_has_month(history_df: pd.DataFrame, year: int, month: int) -> bool:
    if history_df.empty:
        return False
    return not history_df[(history_df["year"] == year) & (history_df["month"] == month)].empty


def get_month_df(
    *,
    year: int,
    month: int,
    url: str,
    location: str,
    history_df: pd.DataFrame,
    latest_year: int,
    latest_month: int,
) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, bool]:
    """
    指定年月のデータを返す。
    - 最新公開月: 毎回Web取得（更新される可能性があるため）
    - それ以外: 履歴CSVから（無ければWeb取得して履歴に追記）
    戻り値:
      (df, updated_history_df, history_updated_flag)
    """
    history_updated = False

    if is_latest_month(year, month, latest_year, latest_month):
        # 最新公開月は毎回取得
        df = fetch_snow_data(url, year, month)
        return df, history_df, False

    # 過去月：履歴にあればそれを使う
    if history_has_month(history_df, year, month):
        df = history_df[(history_df["year"] == year) & (history_df["month"] == month)].copy()
        return df, history_df, False

    # 履歴にない → Web取得して履歴に追記
    df = fetch_snow_data(url, year, month)
    if df is not None and not df.empty:
        if history_df.empty:
            history_df = df.copy()
        else:
            history_df = pd.concat([history_df, df], ignore_index=True)
        history_updated = True

    return df, history_df, history_updated


# ==========
# グラフ
# ==========
def create_snow_graph(df: pd.DataFrame, year: int, month: int, location: str) -> go.Figure:
    """降雪・積雪のグラフを作成"""
    filtered_df = df[
        (df["year"] == year) & (df["month"] == month) & (df["location"] == location)
    ].sort_values("day")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 積雪量（棒）: 左
    fig.add_trace(
        go.Bar(
            x=filtered_df["day"],
            y=filtered_df["snowdepth_cm"],
            name="積雪量",
            opacity=0.7,
        ),
        secondary_y=False,
    )

    # 降雪量（線）: 右
    fig.add_trace(
        go.Scatter(
            x=filtered_df["day"],
            y=filtered_df["snowfall_cm"],
            name="降雪量",
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
        title_text="積雪量 (cm)",
        range=[0, 300],
        dtick=60,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="降雪量 (cm)",
        range=[0, 100],
        dtick=20,
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

    # 履歴読み込み
    history_df = load_history_data()

    st.markdown("## 📈 グラフ表示")

    # 表示ごとに取得（必要なら履歴更新）
    history_updated_any = False

    for sel in unique_selections:
        year = int(sel["year"])
        month = int(sel["month"])
        location = sel["location"]

        url_row = url_df[(url_df["年"] == year) & (url_df["月"] == month)]
        if url_row.empty:
            st.warning(f"⚠️ {year}年{month}月のデータはありません")
            continue

        url = url_row.iloc[0]["URL"]

        with st.spinner(f"{year}年{month}月のデータを準備中..."):
            df, history_df, history_updated = get_month_df(
                year=year,
                month=month,
                url=url,
                location=location,
                history_df=history_df,
                latest_year=latest_year,
                latest_month=latest_month,
            )
            if history_updated:
                history_updated_any = True

        if df is None or df.empty:
            st.error(f"❌ {year}年{month}月 / {location} のデータ取得に失敗しました")
            continue

        fig = create_snow_graph(df, year, month, location)
        st.plotly_chart(fig, use_container_width=True)

    # 履歴が更新されたら保存（最後にまとめて1回）
    if history_updated_any:
        save_history_data(history_df)

    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
        データ出典: <a href='https://www.city.myoko.niigata.jp/life-info/snow-info/snow/' target='_blank'>妙高市 雪情報ホームページ</a><br>
        観測時刻: 9時 | 降雪量: 前日分 | 積雪量: 当日分
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
