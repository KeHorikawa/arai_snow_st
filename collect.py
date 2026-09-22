#!/usr/bin/env python3
"""妙高市 降積雪観測データの収集スクリプト（Streamlit非依存）

GitHub Actions から毎朝1回実行することを想定している。

    python collect.py              当シーズンの最新月ページ1枚だけ取得
    python collect.py --all        全期間をバックフィル（初回に1回だけ）
    python collect.py --dry-run    取得して結果を表示するがファイルは書かない

出力は3つ。
    snow_data_history.csv   唯一の正・全シーズン（upsert）
    data/snow_data.json     派生・最新データのあるシーズンの表示用
    data/status.json        実行記録（毎日必ず更新する）

ジョブを失敗させる条件は CollectError の送出箇所を参照。
「当シーズンのページがまだ無い」はオフシーズンの正常な状態であり、失敗ではない。
"""

from __future__ import annotations

import argparse
import calendar
import json
import logging
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ==========
# ロギング
# ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("collect")

# ==========
# 定数
# ==========
BASE_URL = "https://www.city.myoko.niigata.jp"
INDEX_URL = "https://www.city.myoko.niigata.jp/life-info/snow-info/snow/"

LOCATIONS: List[str] = ["新井消防署", "頸南消防署", "妙高市役所 妙高支所"]

# 2009年12月のページだけ「頚南消防署」（異体字）になっている。
# 行の出力は列位置で決まるので実害はないが、テーブル判定のために表記を寄せる。
LOCATION_NORMALIZE = {"頚": "頸"}

BASE_DIR = Path(__file__).resolve().parent
HISTORY_CSV = BASE_DIR / "snow_data_history.csv"
URLS_CSV = BASE_DIR / "data_urls.csv"
DATA_DIR = BASE_DIR / "data"
SNOW_JSON = DATA_DIR / "snow_data.json"
STATUS_JSON = DATA_DIR / "status.json"

HISTORY_COLS = ["year", "month", "day", "location", "snowfall_cm", "snowdepth_cm"]
HISTORY_KEY = ["year", "month", "day", "location"]
URLS_COLS = ["年度", "年", "月", "URL"]

JST = timezone(timedelta(hours=9))

# 市サーバーへの礼儀。--all のときだけ効く（毎朝の実行は2リクエストしか出さない）
REQUEST_INTERVAL_SEC = 1.5
REQUEST_TIMEOUT_SEC = 20
USER_AGENT = "arai_snow_st-collector/1.0 (+https://github.com/KeHorikawa/arai_snow_st)"

# 既存の履歴と比べて、その月の日数がこの割合を下回ったら「静かなデータ破壊」とみなす
CONSISTENCY_RATIO = 0.8

# リンクテキストの「(R8.3)」「（H31.4）」「(H26.04)」部分だけを信じる。
# 先頭の「令和7年度」は市サイト側で不整合があるため使わない。
MONTH_LINK_RE = re.compile(r"[（(]\s*([RH])\s*(\d{1,2})\s*[.．]\s*(\d{1,2})\s*[）)]")


class CollectError(Exception):
    """ジョブを失敗させる（＝メール通知してほしい）エラー"""


# ==========
# 和暦・年度
# ==========
def wareki_to_year(era: str, n: int) -> int:
    """令和n年 = 2018 + n、平成n年 = 1988 + n"""
    if era == "R":
        return 2018 + n
    if era == "H":
        return 1988 + n
    raise ValueError(f"未知の元号: {era}")


def fiscal_year(year: int, month: int) -> int:
    """年度（シーズン）の開始年を返す。11・12月はその年、1〜5月は前年。

    暦の月で「冬かどうか」を判定しないこと。6月以降を次シーズンの始まりとして扱う。
    """
    return year if month >= 6 else year - 1


def fiscal_year_label(fy: int) -> str:
    """年度を「令和3年度」「平成30年度」の形にする（data_urls.csv の列用）"""
    if fy >= 2019:
        n = fy - 2018
        return "令和元年度" if n == 1 else f"令和{n}年度"
    return f"平成{fy - 1988}年度"


def season_label(fy: int) -> str:
    """年度を「2025-26」の形にする（JSON用）"""
    return f"{fy}-{str(fy + 1)[2:]}"


def current_fiscal_year(today: date) -> int:
    return fiscal_year(today.year, today.month)


# ==========
# HTTP
# ==========
def http_get(url: str) -> str:
    resp = requests.get(url, timeout=REQUEST_TIMEOUT_SEC, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    return resp.text


# ==========
# 月別URLの自動発見
# ==========
def discover_month_urls(html: str) -> List[Dict]:
    """インデックスページから月別ページのURLを全件見つける。

    戻り値は [{"year": 2026, "month": 3, "url": "...", "年度": "令和7年度"}, ...]（年月昇順）
    """
    soup = BeautifulSoup(html, "lxml")

    found: Dict[Tuple[int, int], str] = {}
    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        m = MONTH_LINK_RE.search(text)
        if not m:
            continue

        era, era_year, month_str = m.group(1), int(m.group(2)), int(m.group(3))
        try:
            year = wareki_to_year(era, era_year)
        except ValueError as e:
            logger.warning(f"元号の解釈に失敗したので無視します: {text!r} ({e})")
            continue

        if not 1 <= month_str <= 12:
            logger.warning(f"月が範囲外なので無視します: {text!r}")
            continue

        url = urljoin(BASE_URL, a["href"])
        key = (year, month_str)
        if key in found and found[key] != url:
            logger.warning(f"{year}年{month_str}月のリンクが重複しています。先に見つけた方を使います: {url}")
            continue
        found.setdefault(key, url)

    if not found:
        raise CollectError(
            f"インデックスページから月別リンクが1件も見つかりませんでした: {INDEX_URL}\n"
            "リンクテキストの形式（例: 令和7年度 降積雪観測(R8.3)）が変わった可能性があります。"
        )

    entries = [
        {
            "年度": fiscal_year_label(fiscal_year(y, m)),
            "year": y,
            "month": m,
            "url": url,
        }
        for (y, m), url in sorted(found.items())
    ]
    logger.info(f"月別リンクを {len(entries)} 件検出しました（{entries[0]['year']}年{entries[0]['month']}月 〜 {entries[-1]['year']}年{entries[-1]['month']}月）")
    return entries


def save_url_csv(entries: List[Dict]) -> None:
    """data_urls.csv を再生成する（手で追記する台帳ではなく、毎朝作り直す派生ファイル）"""
    df = pd.DataFrame(
        [{"年度": e["年度"], "年": e["year"], "月": e["month"], "URL": e["url"]} for e in entries],
        columns=URLS_COLS,
    )
    df.to_csv(URLS_CSV, index=False)
    logger.info(f"{URLS_CSV.name} を再生成しました: {len(df)}件")


# ==========
# 月ページのパース（main.py からの移植）
# ==========
def _normalize(text: str) -> str:
    for src, dst in LOCATION_NORMALIZE.items():
        text = text.replace(src, dst)
    return re.sub(r"\s+", "", text)


def _pick_data_table(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """観測所名が含まれるテーブルを優先して選ぶ。無ければ最初のテーブル。"""
    tables = soup.find_all("table")
    if not tables:
        return None

    normalized_locations = [_normalize(loc) for loc in LOCATIONS]
    for tbl in tables:
        text = _normalize(tbl.get_text(" ", strip=True))
        if all(loc in text for loc in normalized_locations):
            return tbl

    logger.warning("観測所名がそろったテーブルが見つからないため、最初のテーブルを使います")
    return tables[0]


def _to_float_or_none(s: str) -> Optional[float]:
    """数値っぽい文字列をfloatに。数字が含まれなければ None（"-", "--", "" 等）。

    符号を拾う。旧実装の `\\d+(?:\\.\\d+)?` は "-5" を 5.0 にしていた。
    """
    x = (s or "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", x)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def _is_real_date(year: int, month: int, day: int) -> bool:
    """4月のページに空の「31日」行があるなど、実在しない日付が並ぶことがある"""
    return 1 <= month <= 12 and 1 <= day <= calendar.monthrange(year, month)[1]


def parse_month_page(html: str, year: int, month: int, url: str) -> pd.DataFrame:
    """月ページのHTMLから tidy DataFrame を作る。取れなければ CollectError。"""
    soup = BeautifulSoup(html, "lxml")

    table = _pick_data_table(soup)
    if table is None:
        raise CollectError(f"テーブルが見つかりません: {url}")

    rows = table.find_all("tr")
    if len(rows) < 3:
        raise CollectError(f"テーブル行数が少なすぎます（{len(rows)}行）: {url}")

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

        if not _is_real_date(year, month, day):
            logger.debug(f"実在しない日付なので無視します: {year}年{month}月{day}日")
            continue

        # 実際の列順: [日, 降雪1, 積雪1, 降雪2, 積雪2, 降雪3, 積雪3]
        for i, location in enumerate(LOCATIONS):
            snowfall_idx = i * 2 + 1
            snowdepth_idx = i * 2 + 2

            snowfall_raw = cols_text[snowfall_idx] if snowfall_idx < len(cols_text) else "-"
            snowdepth_raw = cols_text[snowdepth_idx] if snowdepth_idx < len(cols_text) else "-"

            snowfall_cm = _to_float_or_none(snowfall_raw)
            snowdepth_cm = _to_float_or_none(snowdepth_raw)

            # 負値は None に潰さずそのまま保存する。
            # 市のデータ側の異常か自分のパースの誤りかを、後から切り分けられるようにするため。
            if snowfall_cm is not None and snowfall_cm < 0:
                logger.warning(
                    f"降雪量が負値です（そのまま保存します）: {year}年{month}月{day}日 "
                    f"{location} raw={snowfall_raw!r} -> {snowfall_cm}"
                )
            if snowdepth_cm is not None and snowdepth_cm < 0:
                logger.warning(
                    f"積雪量が負値です（そのまま保存します）: {year}年{month}月{day}日 "
                    f"{location} raw={snowdepth_raw!r} -> {snowdepth_cm}"
                )

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
        raise CollectError(f"日付行が0件でした（HTML構造の変更の可能性）: {url}")

    df = pd.DataFrame(data_rows, columns=HISTORY_COLS)
    # 全欠測の月があっても dtype がぶれないよう明示しておく（concat時の警告対策）
    df[["snowfall_cm", "snowdepth_cm"]] = df[["snowfall_cm", "snowdepth_cm"]].astype(float)
    return df


def fetch_month(entry: Dict) -> pd.DataFrame:
    """月ページ1枚を取得してパースする"""
    url = entry["url"]
    year, month = entry["year"], entry["month"]
    logger.info(f"取得: {year}年{month}月 {url}")
    try:
        html = http_get(url)
    except requests.RequestException as e:
        raise CollectError(f"月ページの取得に失敗: {url} - {e}") from e

    df = parse_month_page(html, year, month, url)
    logger.info(f"  → {len(df)}行 / {df['day'].nunique()}日分")
    return df


# ==========
# 履歴CSV（唯一の正）
# ==========
def load_history() -> pd.DataFrame:
    """履歴CSVを読み込む（無ければ空DataFrame）"""
    if not HISTORY_CSV.exists():
        logger.info("履歴CSVがまだありません。新規作成します。")
        return pd.DataFrame(columns=HISTORY_COLS)

    df = pd.read_csv(HISTORY_CSV)
    missing = [c for c in HISTORY_COLS if c not in df.columns]
    if missing:
        raise CollectError(f"履歴CSVに必要列がありません: {missing}")

    for col in ["year", "month", "day"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["year", "month", "day"]).copy()
    for col in ["year", "month", "day"]:
        df[col] = df[col].astype(int)
    df["location"] = df["location"].astype(str)
    for col in ["snowfall_cm", "snowdepth_cm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # upsert では古い行を消せないので、実在しない日付はここで落とす（自己修復）
    valid = [_is_real_date(y, m, d) for y, m, d in zip(df["year"], df["month"], df["day"])]
    dropped = len(df) - sum(valid)
    if dropped:
        logger.warning(f"実在しない日付の行を履歴から除きます: {dropped}行")
        df = df[valid]

    return df[HISTORY_COLS]


def check_consistency(history: pd.DataFrame, new_df: pd.DataFrame) -> None:
    """既存の履歴と矛盾していないか（静かなデータ破壊の防止）"""
    if history.empty:
        return

    for (year, month), group in new_df.groupby(["year", "month"]):
        existing = history[(history["year"] == year) & (history["month"] == month)]
        if existing.empty:
            continue
        existing_days = existing["day"].nunique()
        new_days = group["day"].nunique()
        if new_days < existing_days * CONSISTENCY_RATIO:
            raise CollectError(
                f"{year}年{month}月の日数が既存の履歴より大幅に減っています "
                f"（既存 {existing_days}日 → 今回 {new_days}日）。"
                "HTML構造の変更かページの差し替えが疑われます。"
            )


def _values_equal(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return float(a) == float(b)


def upsert_history(history: pd.DataFrame, new_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """(year, month, day, location) をキーに、取得した値で置き換える。

    戻り値: (更新後の履歴, 追加または変更された行数)
    """
    if new_df.empty:
        return history, 0

    old_map = {}
    if not history.empty:
        for row in history.itertuples(index=False):
            old_map[(row.year, row.month, row.day, row.location)] = (row.snowfall_cm, row.snowdepth_cm)

    changed = 0
    for row in new_df.itertuples(index=False):
        key = (row.year, row.month, row.day, row.location)
        before = old_map.get(key)
        if before is None:
            changed += 1
        elif not (_values_equal(before[0], row.snowfall_cm) and _values_equal(before[1], row.snowdepth_cm)):
            changed += 1

    combined = pd.concat([history, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=HISTORY_KEY, keep="last")
    combined = sort_history(combined)
    return combined, changed


def sort_history(df: pd.DataFrame) -> pd.DataFrame:
    """year, month, day, 観測地点の並び順でそろえる"""
    order = pd.Categorical(df["location"], categories=LOCATIONS, ordered=True)
    df = df.assign(_loc_order=order)
    df = df.sort_values(["year", "month", "day", "_loc_order"], kind="mergesort")
    return df.drop(columns="_loc_order").reset_index(drop=True)


def save_history(df: pd.DataFrame) -> None:
    df.reindex(columns=HISTORY_COLS).to_csv(HISTORY_CSV, index=False)
    logger.info(f"{HISTORY_CSV.name} を保存しました: {len(df)}行")


# ==========
# data/snow_data.json（派生・表示用）
# ==========
def _num(value) -> Optional[float]:
    """JSONに書く数値。整数なら int にして見やすくする。"""
    if value is None or pd.isna(value):
        return None
    f = float(value)
    return int(f) if f.is_integer() else f


def build_snow_data(history: pd.DataFrame, url_map: Dict[Tuple[int, int], str], today: date) -> Dict:
    """最新データのあるシーズンから、静的サイト用のJSONを組む。

    「今シーズン」ではなく「最も新しくデータがあるシーズン」を入れる。
    オフシーズンに空のJSONを出すと、次回の静的サイトを実データで試せなくなるため。
    """
    generated_at = datetime.now(JST).isoformat(timespec="seconds")
    cur_fy = current_fiscal_year(today)

    base = {
        "generated_at": generated_at,
        "source_url": None,
        "season": None,
        "season_status": "no-data",
        "is_current_season": False,
        "locations": LOCATIONS,
        "latest": None,
        "series": {"dates": []},
    }
    for loc in LOCATIONS:
        base["series"][loc] = {"snowfall_cm": [], "snowdepth_cm": []}

    if history.empty:
        return base

    df = history.copy()
    df["fy"] = [fiscal_year(y, m) for y, m in zip(df["year"], df["month"])]

    # データ（非欠測値）が1つでもあるシーズンのうち、最も新しいもの
    has_value = df["snowfall_cm"].notna() | df["snowdepth_cm"].notna()
    seasons_with_data = df.loc[has_value, "fy"]
    if seasons_with_data.empty:
        return base
    fy = int(seasons_with_data.max())

    season_df = df[df["fy"] == fy].copy()
    season_df["date"] = pd.to_datetime(
        dict(year=season_df["year"], month=season_df["month"], day=season_df["day"]),
        errors="coerce",
    )
    season_df = season_df.dropna(subset=["date"]).sort_values("date")

    base["season"] = season_label(fy)
    base["is_current_season"] = fy == cur_fy
    base["season_status"] = "in-season" if fy == cur_fy else "finished"

    # 値のある日付だけを対象に、末尾の全欠測日を落とす
    dated_values = season_df[season_df["snowfall_cm"].notna() | season_df["snowdepth_cm"].notna()]
    if dated_values.empty:
        return base
    last_date = dated_values["date"].max()
    season_df = season_df[season_df["date"] <= last_date]

    dates = sorted(season_df["date"].unique())
    base["series"]["dates"] = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]

    for loc in LOCATIONS:
        loc_df = season_df[season_df["location"] == loc].set_index("date")
        snowfall, snowdepth = [], []
        for d in dates:
            if d in loc_df.index:
                row = loc_df.loc[d]
                snowfall.append(_num(row["snowfall_cm"]))
                snowdepth.append(_num(row["snowdepth_cm"]))
            else:
                snowfall.append(None)
                snowdepth.append(None)
        base["series"][loc] = {"snowfall_cm": snowfall, "snowdepth_cm": snowdepth}

    # latest: 「今日」ではなく「値がある最も新しい観測日」
    observed = pd.Timestamp(last_date)
    by_location = {}
    for loc in LOCATIONS:
        loc_df = season_df[season_df["location"] == loc].set_index("date").sort_index()
        row = loc_df.loc[observed] if observed in loc_df.index else None
        snowdepth = _num(row["snowdepth_cm"]) if row is not None else None
        snowfall = _num(row["snowfall_cm"]) if row is not None else None

        # 前日比の相手は「暦の前日」ではなく「値がある直近の観測日」
        diff = None
        if snowdepth is not None:
            prev = loc_df[(loc_df.index < observed) & loc_df["snowdepth_cm"].notna()]
            if not prev.empty:
                diff = _num(snowdepth - float(prev.iloc[-1]["snowdepth_cm"]))

        by_location[loc] = {
            "snowdepth_cm": snowdepth,
            "snowdepth_diff_cm": diff,
            "snowfall_cm": snowfall,
        }

    base["latest"] = {
        "observed_date": observed.strftime("%Y-%m-%d"),
        "by_location": by_location,
    }

    latest_row = season_df[season_df["date"] == observed].iloc[0]
    base["source_url"] = url_map.get((int(latest_row["year"]), int(latest_row["month"])))

    return base


def write_json(path: Path, payload: Dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(f"{path.relative_to(BASE_DIR)} を書き出しました")


def build_status(result: str, note: str, rows_written: int, latest_observed_date: Optional[str]) -> Dict:
    return {
        "last_run_at": datetime.now(JST).isoformat(timespec="seconds"),
        "result": result,
        "note": note,
        "rows_written": rows_written,
        "latest_observed_date": latest_observed_date,
    }


# ==========
# メイン
# ==========
def run(fetch_all: bool, dry_run: bool, today: date) -> Dict:
    """収集を1回実行する。戻り値は status の内容。"""
    logger.info(f"インデックス取得: {INDEX_URL}")
    try:
        index_html = http_get(INDEX_URL)
    except requests.RequestException as e:
        raise CollectError(f"インデックスページの取得に失敗: {INDEX_URL} - {e}") from e

    entries = discover_month_urls(index_html)
    url_map = {(e["year"], e["month"]): e["url"] for e in entries}

    if dry_run:
        logger.info(f"[dry-run] {URLS_CSV.name} は書き換えません")
    else:
        save_url_csv(entries)

    history = load_history()
    logger.info(f"既存の履歴: {len(history)}行")

    cur_fy = current_fiscal_year(today)
    season_entries = [e for e in entries if fiscal_year(e["year"], e["month"]) == cur_fy]

    targets: List[Dict] = []
    note = ""

    if fetch_all:
        targets = entries
        note = f"全期間バックフィル（{len(targets)}ページ）"
    elif season_entries:
        # 毎朝取りに行くのは当シーズンの該当月ページ1枚だけ
        targets = [max(season_entries, key=lambda e: (e["year"], e["month"]))]
        note = f"{targets[0]['year']}年{targets[0]['month']}月ページを取得"
    else:
        # オフシーズンの正常な状態。失敗ではない。
        note = f"当シーズン（{season_label(cur_fy)}）のページ未公開"
        logger.info(f"{note}。取得はスキップします（オフシーズンの正常な状態です）")
        if dry_run and entries:
            latest = entries[-1]
            logger.info("[dry-run] 取得とパースの確認のため、公開済みの最新月ページを1枚だけ取得します")
            targets = [latest]

    failures: List[str] = []
    frames: List[pd.DataFrame] = []
    for i, entry in enumerate(targets):
        if i > 0:
            time.sleep(REQUEST_INTERVAL_SEC)
        try:
            frames.append(fetch_month(entry))
        except CollectError as e:
            if fetch_all:
                # バックフィルは手動の一度きりなので、1ページ失敗しても最後まで走って全体を報告する
                logger.error(f"バックフィル中の失敗: {e}")
                failures.append(f"{entry['year']}年{entry['month']}月: {e}")
                continue
            raise

    rows_written = 0
    if frames:
        new_df = pd.concat(frames, ignore_index=True)
        check_consistency(history, new_df)
        history, rows_written = upsert_history(history, new_df)
        logger.info(f"追加・更新された行: {rows_written}行")
    elif not history.empty:
        history = sort_history(history)

    # 取得が無かった日も書き出す（並び順の正規化と、実在しない日付の除去が効くため）。
    # 内容が同じならワークフロー側で差分が出ないので、空コミットにはならない。
    if dry_run:
        logger.info(f"[dry-run] {HISTORY_CSV.name} は書き換えません")
    elif not history.empty:
        save_history(history)

    snow_data = build_snow_data(history, url_map, today)
    latest_observed_date = snow_data["latest"]["observed_date"] if snow_data["latest"] else None
    logger.info(
        f"JSON: season={snow_data['season']} status={snow_data['season_status']} "
        f"latest={latest_observed_date} 日数={len(snow_data['series']['dates'])}"
    )

    if failures:
        raise CollectError(
            f"バックフィル中に {len(failures)}件のページで失敗しました:\n  - " + "\n  - ".join(failures)
        )

    status = build_status("ok", note, rows_written, latest_observed_date)

    if dry_run:
        logger.info("[dry-run] data/ 以下は書き換えません")
        print(json.dumps(snow_data, ensure_ascii=False, indent=2)[:2000])
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        write_json(SNOW_JSON, snow_data)
        write_json(STATUS_JSON, status)

    return status


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="妙高市 降積雪観測データの収集")
    parser.add_argument("--all", action="store_true", dest="fetch_all",
                        help="全期間をバックフィルする（初回に1回だけ）")
    parser.add_argument("--dry-run", action="store_true",
                        help="取得して結果を表示するがファイルは書かない")
    args = parser.parse_args(argv)

    today = datetime.now(JST).date()
    logger.info(f"実行日（JST）: {today} / 当シーズン: {season_label(current_fiscal_year(today))}")

    try:
        status = run(fetch_all=args.fetch_all, dry_run=args.dry_run, today=today)
    except CollectError as e:
        logger.error(f"失敗: {e}")
        if not args.dry_run:
            # 失敗もコミットして残す（Git履歴がそのまま成績表になる）
            write_json(STATUS_JSON, build_status("error", str(e), 0, None))
        return 1
    except Exception as e:  # 想定外は失敗として扱う
        logger.exception("想定外のエラー")
        if not args.dry_run:
            write_json(STATUS_JSON, build_status("error", f"想定外のエラー: {e}", 0, None))
        return 1

    logger.info(f"完了: {status['note']}（追加・更新 {status['rows_written']}行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
