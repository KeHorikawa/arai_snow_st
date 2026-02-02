import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple
import logging
from datetime import datetime

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定数
LOCATIONS = ["新井消防署", "頸南消防署", "妙高市役所 妙高支所"]
CSV_FILE = "data_urls.csv"
HISTORY_CSV_FILE = "snow_data_history.csv"


@st.cache_data(ttl=3600)
def load_url_data() -> pd.DataFrame:
    """URL一覧CSVファイルを読み込む"""
    try:
        df = pd.read_csv(CSV_FILE)
        return df
    except Exception as e:
        logger.error(f"CSVファイルの読み込みに失敗: {e}")
        st.error("データURLの読み込みに失敗しました")
        return pd.DataFrame()


def load_history_data() -> pd.DataFrame:
    """過去データCSVファイルを読み込む"""
    try:
        df = pd.read_csv(HISTORY_CSV_FILE)
        # データ型を適切に変換
        df["year"] = df["year"].astype(int)
        df["month"] = df["month"].astype(int)
        df["day"] = df["day"].astype(int)
        return df
    except FileNotFoundError:
        logger.info("過去データファイルが存在しません。新規作成します。")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"過去データファイルの読み込みに失敗: {e}")
        return pd.DataFrame()


def save_history_data(df: pd.DataFrame) -> None:
    """過去データをCSVファイルに保存する"""
    try:
        df.to_csv(HISTORY_CSV_FILE, index=False)
        logger.info(f"過去データを保存しました: {len(df)}件")
    except Exception as e:
        logger.error(f"過去データの保存に失敗: {e}")


def get_current_month() -> Tuple[int, int]:
    """現在の年月を取得"""
    current_date = datetime.now()
    return current_date.year, current_date.month


def is_current_month(year: int, month: int) -> bool:
    """指定された年月が当月かどうかを判定"""
    current_year, current_month = get_current_month()
    return year == current_year and month == current_month


@st.cache_data(ttl=3600)
def fetch_snow_data(url: str, year: int, month: int) -> Optional[pd.DataFrame]:
    """
    指定URLから降雪・積雪データを取得してDataFrame化する

    Args:
        url: データ取得元のURL
        year: 年
        month: 月

    Returns:
        tidyデータ形式のDataFrame、取得失敗時はNone
    """
    try:
        # HTMLを取得
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding

        # BeautifulSoupでパース
        soup = BeautifulSoup(response.text, "lxml")

        # テーブルを探す
        tables = soup.find_all("table")
        if not tables:
            logger.warning(f"テーブルが見つかりません: {url}")
            return None

        # 最初のテーブルを使用（通常、データテーブルは最初にある）
        table = tables[0]

        # データを格納するリスト
        data_rows = []

        # テーブルの行を解析
        rows = table.find_all("tr")

        # ヘッダー行（2行）をスキップして、データ行を処理
        for row in rows[2:]:  # ヘッダーは2行あるためスキップ
            cols = row.find_all(["td", "th"])
            if len(cols) < 7:  # 日、降雪×3、積雪×3 の最低7列必要
                continue

            # テキストを抽出
            cols_text = [col.get_text(strip=True) for col in cols]

            # 日付を取得（「日」という文字を除去）
            try:
                day = int(cols_text[0].replace("日", ""))
            except (ValueError, IndexError):
                continue

            # 各観測地点のデータを処理
            # 実際の列順: [日, 降雪1, 積雪1, 降雪2, 積雪2, 降雪3, 積雪3]
            if len(cols_text) >= 7:
                for i, location in enumerate(LOCATIONS):
                    # 降雪量: インデックス 1, 3, 5 (= i*2 + 1)
                    snowfall_idx = i * 2 + 1
                    # 積雪量: インデックス 2, 4, 6 (= i*2 + 2)
                    snowdepth_idx = i * 2 + 2

                    snowfall = (
                        cols_text[snowfall_idx]
                        if snowfall_idx < len(cols_text)
                        else "-"
                    )
                    snowdepth = (
                        cols_text[snowdepth_idx]
                        if snowdepth_idx < len(cols_text)
                        else "-"
                    )

                    # 降雪量の処理: "-", "--", 空文字、"30-"のような表記を処理
                    snowfall_clean = snowfall.strip()
                    if snowfall_clean in ["-", "--", ""]:
                        snowfall_cm = None
                    else:
                        # "30-"のような表記から"-"を除去
                        snowfall_clean = snowfall_clean.rstrip("-")
                        try:
                            snowfall_cm = (
                                float(snowfall_clean) if snowfall_clean else None
                            )
                        except ValueError:
                            snowfall_cm = None

                    # 積雪量の処理: "-", "--", 空文字を処理、負の値は無視
                    snowdepth_clean = snowdepth.strip()
                    if snowdepth_clean in ["-", "--", ""]:
                        snowdepth_cm = None
                    else:
                        try:
                            snowdepth_value = float(snowdepth_clean)
                            # 積雪量が負の値の場合は、降雪量の"-"と混同している可能性があるためNoneにする
                            if snowdepth_value < 0:
                                snowdepth_cm = None
                            else:
                                snowdepth_cm = snowdepth_value
                        except ValueError:
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

        df = pd.DataFrame(data_rows)
        return df

    except requests.RequestException as e:
        logger.error(f"HTTPリクエストエラー: {url} - {e}")
        return None
    except Exception as e:
        logger.error(f"データ取得エラー: {url} - {e}")
        return None


def create_snow_graph(
    df: pd.DataFrame, year: int, month: int, location: str
) -> go.Figure:
    """
    降雪・積雪のグラフを作成

    Args:
        df: 降雪・積雪データ
        year: 年
        month: 月
        location: 観測地点

    Returns:
        Plotlyのグラフオブジェクト
    """
    # 指定された条件でフィルタ
    filtered_df = df[
        (df["year"] == year) & (df["month"] == month) & (df["location"] == location)
    ].sort_values("day")

    # グラフ作成（サブプロットで2つのy軸を使用）
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 積雪量（棒グラフ）- 左軸（secondary_y=False）
    fig.add_trace(
        go.Bar(
            x=filtered_df["day"],
            y=filtered_df["snowdepth_cm"],
            name="積雪量",
            marker_color="lightblue",
            opacity=0.7,
        ),
        secondary_y=False,
    )

    # 降雪量（折れ線グラフ）- 右軸（secondary_y=True）
    fig.add_trace(
        go.Scatter(
            x=filtered_df["day"],
            y=filtered_df["snowfall_cm"],
            name="降雪量",
            mode="lines+markers",
            line=dict(color="red", width=2),
            marker=dict(size=6),
        ),
        secondary_y=True,
    )

    # レイアウト設定
    fig.update_layout(
        title=f"{year}年{month}月 / {location}",
        xaxis_title="日",
        hovermode="x unified",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # X軸の範囲を1-31に設定
    fig.update_xaxes(range=[0.5, 31.5], dtick=5)

    # 左軸（積雪量）の設定: 0~300cm（60cm間隔で目盛り）
    fig.update_yaxes(
        title_text="積雪量 (cm)",
        range=[0, 300],
        dtick=60,
        secondary_y=False,
    )

    # 右軸（降雪量）の設定: 0~100cm（20cm間隔で目盛り）
    fig.update_yaxes(
        title_text="降雪量 (cm)",
        range=[0, 100],
        dtick=20,
        secondary_y=True,
    )

    return fig


def get_latest_available_month(url_df: pd.DataFrame) -> Tuple[int, int]:
    """
    データが存在する最新の年月を取得

    Args:
        url_df: URL一覧データ

    Returns:
        (年, 月) のタプル
    """
    if url_df.empty:
        current_date = datetime.now()
        return current_date.year, current_date.month

    # 年月でソート
    sorted_df = url_df.sort_values(["年", "月"], ascending=False)
    latest = sorted_df.iloc[0]
    return int(latest["年"]), int(latest["月"])


def main():
    """メインアプリケーション"""

    # ページ設定
    st.set_page_config(
        page_title="妙高市 降雪・積雪データ可視化", page_icon="❄️", layout="wide"
    )

    # タイトル
    st.title("❄️ 妙高市 降雪・積雪データ可視化")
    st.markdown("---")

    # URL一覧の読み込み
    url_df = load_url_data()

    if url_df.empty:
        st.error("データURLが読み込めません。data_urls.csv を確認してください。")
        return

    # 利用可能な年月のリストを作成
    available_years = sorted(url_df["年"].unique())

    # 最新の年月を取得
    latest_year, latest_month = get_latest_available_month(url_df)

    # サイドバー
    st.sidebar.header("📊 表示条件設定")
    st.sidebar.markdown("最大3件まで選択できます")

    # 条件選択（最大3件）
    selections = []

    for i in range(3):
        st.sidebar.markdown(f"### 条件 {i + 1}")

        col1, col2 = st.sidebar.columns(2)

        with col1:
            # デフォルト値を設定（最新月）
            default_year_idx = (
                available_years.index(latest_year)
                if latest_year in available_years
                else 0
            )
            year = st.selectbox(
                "年", options=available_years, index=default_year_idx, key=f"year_{i}"
            )

        with col2:
            # 選択された年で利用可能な月を取得
            available_months = sorted(url_df[url_df["年"] == year]["月"].unique())
            default_month_idx = (
                available_months.index(latest_month)
                if latest_month in available_months
                else 0
            )
            month = st.selectbox(
                "月",
                options=available_months,
                index=default_month_idx,
                key=f"month_{i}",
            )

        # 観測地点（デフォルトで各地点を割り当て）
        default_location_idx = i if i < len(LOCATIONS) else 0
        location = st.sidebar.selectbox(
            "観測地点",
            options=LOCATIONS,
            index=default_location_idx,
            key=f"location_{i}",
        )

        selections.append({"year": year, "month": month, "location": location})

        st.sidebar.markdown("---")

    # 重複チェック
    unique_selections = []
    seen = set()
    for sel in selections:
        key = (sel["year"], sel["month"], sel["location"])
        if key not in seen:
            unique_selections.append(sel)
            seen.add(key)
        else:
            st.sidebar.warning(
                f"⚠️ {sel['year']}年{sel['month']}月 / {sel['location']} が重複しています"
            )

    # 過去データの読み込み
    history_df = load_history_data()
    current_year, current_month = get_current_month()

    # 不足年月のデータを自動取得・保存
    if not url_df.empty:
        missing_data_fetched = False
        for _, row in url_df.iterrows():
            year = int(row["年"])
            month = int(row["月"])
            url = row["URL"]

            # 当月のデータは毎回取得するためスキップ
            if is_current_month(year, month):
                continue

            # 過去データに該当年月のデータが存在するかチェック
            if not history_df.empty:
                existing_data = history_df[
                    (history_df["year"] == year) & (history_df["month"] == month)
                ]
                if not existing_data.empty:
                    continue

            # 不足データを取得
            with st.spinner(f"不足データを取得中: {year}年{month}月..."):
                new_df = fetch_snow_data(url, year, month)
                if new_df is not None and not new_df.empty:
                    # 過去データに追加
                    if history_df.empty:
                        history_df = new_df
                    else:
                        history_df = pd.concat([history_df, new_df], ignore_index=True)
                    missing_data_fetched = True

        # 不足データを取得した場合は保存
        if missing_data_fetched:
            save_history_data(history_df)
            # キャッシュをクリアして再読み込み
            st.cache_data.clear()

    # メイン表示エリア
    st.markdown("## 📈 グラフ表示")

    # 各条件についてグラフを表示
    for idx, sel in enumerate(unique_selections):
        year = sel["year"]
        month = sel["month"]
        location = sel["location"]

        # URLを取得
        url_row = url_df[(url_df["年"] == year) & (url_df["月"] == month)]

        if url_row.empty:
            st.warning(f"⚠️ {year}年{month}月のデータはありません")
            continue

        url = url_row.iloc[0]["URL"]

        # データ取得: 当月は毎回取得、それ以外は過去データから読み込み
        df = None
        if is_current_month(year, month):
            # 当月は毎回取得
            with st.spinner(f"{year}年{month}月のデータを取得中..."):
                df = fetch_snow_data(url, year, month)
        else:
            # 過去データから読み込み
            if not history_df.empty:
                df = history_df[
                    (history_df["year"] == year) & (history_df["month"] == month)
                ].copy()
                if df.empty:
                    # 過去データにない場合は取得
                    with st.spinner(f"{year}年{month}月のデータを取得中..."):
                        df = fetch_snow_data(url, year, month)
                        # 取得したデータを保存
                        if df is not None and not df.empty:
                            if history_df.empty:
                                history_df = df
                            else:
                                history_df = pd.concat(
                                    [history_df, df], ignore_index=True
                                )
                            save_history_data(history_df)
            else:
                # 過去データファイルが存在しない場合は取得
                with st.spinner(f"{year}年{month}月のデータを取得中..."):
                    df = fetch_snow_data(url, year, month)
                    # 取得したデータを保存
                    if df is not None and not df.empty:
                        history_df = df
                        save_history_data(history_df)

        if df is None or df.empty:
            st.error(f"❌ {year}年{month}月 / {location} のデータ取得に失敗しました")
            continue

        # グラフ作成
        fig = create_snow_graph(df, year, month, location)
        st.plotly_chart(fig, use_container_width=True)

    # フッター
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
