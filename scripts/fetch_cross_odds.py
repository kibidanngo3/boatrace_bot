"""舟券種間の裁定検証用: 3連単以外(2連単・2連複・3連複・単勝・複勝)のオッズを取得する。

odds_result_cache.csv (ホールドアウト期間・実結果つき) と同じレース集合に対して、
他の舟券種のオッズを追加取得し odds_cross_cache.csv に貯める。

使い方:
    python scripts/fetch_cross_odds.py --limit 500
"""
import argparse
import csv
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import BoatRaceScraperV5  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
FIELDS = ["date", "course", "rno", "odds2t_json", "odds2f_json", "odds3f_json", "oddstan_json", "oddsfuku_json"]

_lock = threading.Lock()
_thread_local = threading.local()


def get_scraper():
    if not hasattr(_thread_local, "scraper"):
        _thread_local.scraper = BoatRaceScraperV5()
    return _thread_local.scraper


def _to_float(text, default=0.0):
    m = re.search(r"\d+(?:\.\d+)?", text or "")
    return float(m.group(0)) if m else default


def _boat_num(text):
    m = re.search(r"[1-6]", text or "")
    return int(m.group(0)) if m else None


def _expand_grid(table, ncols):
    """rowspanを展開してグリッド化する(colspanは対象テーブルで常に1)。"""
    rows = table.select("tbody tr")
    grid = []
    pending = {}  # col -> (end_row_exclusive, text, cls)
    for r_i, row in enumerate(rows):
        cells = row.select("td")
        ci = 0
        grid_row = [None] * ncols
        c = 0
        while c < ncols:
            if c in pending and r_i < pending[c][0]:
                grid_row[c] = pending[c][1]
                c += 1
                continue
            if ci < len(cells):
                cell = cells[ci]
                text = cell.get_text(strip=True)
                cls = cell.get("class") or []
                rowspan = int(cell.get("rowspan") or 1)
                grid_row[c] = (text, cls)
                if rowspan > 1:
                    pending[c] = (r_i + rowspan, (text, cls))
                ci += 1
                c += 1
            else:
                c += 1
        grid.append(grid_row)
    return grid


def fetch_odds2tf(scraper, course, rno, date_str):
    """2連単・2連複オッズを取得する。戻り値: {"2t": {"1-2": 8.4, ...}, "2f": {"1-2": 7.0, ...}}"""
    jcd = scraper.COURSE_MAP.get(course, "01")
    url = f"https://www.boatrace.jp/owpc/pc/race/odds2tf?rno={rno}&jcd={jcd}&hd={date_str}"
    soup = scraper._get_soup(url, referer=f"{scraper.INDEX_URL}?hd={date_str}")
    if not soup or "データがありません" in soup.text:
        return {}, {}

    result = {}
    for heading_text, key in (("2連単オッズ", "2t"), ("2連複オッズ", "2f")):
        h = next((h for h in soup.select("h3") if heading_text in h.get_text()), None)
        if not h:
            continue
        table = h.find_next("table")
        if not table:
            continue
        header_ths = table.select("thead th")
        first_boats = [int(t.get_text(strip=True)) for t in header_ths if re.fullmatch(r"[1-6]", t.get_text(strip=True))]
        if len(first_boats) != 6:
            first_boats = list(range(1, 7))

        odds = {}
        for row in table.select("tbody tr"):
            cells = row.select("td")
            pos = 0
            for group_idx, first in enumerate(first_boats):
                if pos + 1 >= len(cells):
                    break
                boat_cell, odd_cell = cells[pos], cells[pos + 1]
                pos += 2
                if "is-disabled" in (boat_cell.get("class") or []):
                    continue
                second = _boat_num(boat_cell.get_text(strip=True))
                odd = _to_float(odd_cell.get_text(strip=True))
                if second and odd > 0:
                    if key == "2f":
                        a, b = sorted([first, second])
                        odds[f"{a}-{b}"] = odd
                    else:
                        odds[f"{first}-{second}"] = odd
        result[key] = odds
    return result.get("2t", {}), result.get("2f", {})


def fetch_odds3f(scraper, course, rno, date_str):
    """3連複オッズを取得する。戻り値: {"1-2-3": 4.2, ...}(艇番は昇順に正規化)"""
    jcd = scraper.COURSE_MAP.get(course, "01")
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3f?rno={rno}&jcd={jcd}&hd={date_str}"
    soup = scraper._get_soup(url, referer=f"{scraper.INDEX_URL}?hd={date_str}")
    if not soup or "データがありません" in soup.text:
        return {}

    odds_cell = soup.select_one("td.oddsPoint")
    if not odds_cell:
        return {}
    table = odds_cell.find_parent("table")
    grid = _expand_grid(table, ncols=18)

    odds = {}
    for grid_row in grid:
        # 行は (p, q) ペア(p<q)を表し、group 0..p-2 が companion=1..p-1 を表す。
        # cell構成: [p, q, odds, p, q, odds, ...] (companion数だけ繰り返し)
        cells = [c for c in grid_row if c is not None]
        n_groups = len(cells) // 3
        for g in range(n_groups):
            p_cell, q_cell, odd_cell = cells[g * 3], cells[g * 3 + 1], cells[g * 3 + 2]
            if "is-disabled" in p_cell[1] or "is-disabled" in odd_cell[1]:
                continue
            p = _boat_num(p_cell[0])
            q = _boat_num(q_cell[0])
            odd = _to_float(odd_cell[0])
            companion = g + 1
            if p and q and odd > 0 and companion not in (p, q):
                a, b, c = sorted([companion, p, q])
                odds[f"{a}-{b}-{c}"] = odd
    return odds


def fetch_oddstf(scraper, course, rno, date_str):
    """単勝・複勝オッズを取得する。戻り値: ({"1": 4.5,...}, {"1": [1.8,5.7],...})"""
    jcd = scraper.COURSE_MAP.get(course, "01")
    url = f"https://www.boatrace.jp/owpc/pc/race/oddstf?rno={rno}&jcd={jcd}&hd={date_str}"
    soup = scraper._get_soup(url, referer=f"{scraper.INDEX_URL}?hd={date_str}")
    if not soup or "データがありません" in soup.text:
        return {}, {}

    tables = [t for t in soup.select("table") if t.get("class") and "is-w495" in t.get("class")]
    tan, fuku = {}, {}
    for table in tables:
        heading_prev = table.find_previous(["h3"])
        is_fuku = heading_prev and "複勝" in heading_prev.get_text()
        for row in table.select("tbody tr"):
            boat_cell = row.select_one("td.is-fBold")
            odd_cell = row.select_one("td.oddsPoint")
            if not boat_cell or not odd_cell:
                continue
            boat = _boat_num(boat_cell.get_text(strip=True))
            text = odd_cell.get_text(strip=True)
            if not boat:
                continue
            if is_fuku:
                nums = re.findall(r"\d+(?:\.\d+)?", text)
                if len(nums) >= 2:
                    fuku[str(boat)] = [float(nums[0]), float(nums[1])]
            else:
                odd = _to_float(text)
                if odd > 0:
                    tan[str(boat)] = odd
    return tan, fuku


def fetch(race, delay):
    scraper = get_scraper()
    try:
        odds2t, odds2f = fetch_odds2tf(scraper, race["course"], race["rno"], race["date"])
        time.sleep(delay)
        odds3f = fetch_odds3f(scraper, race["course"], race["rno"], race["date"])
        time.sleep(delay)
        oddstan, oddsfuku = fetch_oddstf(scraper, race["course"], race["rno"], race["date"])
        time.sleep(delay)
        if not (odds2t and odds2f and odds3f and oddstan and oddsfuku):
            return None
        return {
            "date": race["date"], "course": race["course"], "rno": race["rno"],
            "odds2t_json": json.dumps(odds2t), "odds2f_json": json.dumps(odds2f),
            "odds3f_json": json.dumps(odds3f),
            "oddstan_json": json.dumps(oddstan), "oddsfuku_json": json.dumps(oddsfuku),
        }
    except Exception as e:
        print(f"  ❌ error: {race['course']} {race['rno']}R - {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="odds_result_cache.csv")
    parser.add_argument("--output", default="odds_cross_cache.csv")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    CACHE = BASE_DIR / args.output
    done = set()
    if CACHE.exists():
        with open(CACHE, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done.add((r["date"], r["course"], str(int(r["rno"]))))
    print(f"既存キャッシュ: {len(done)}件")

    with open(BASE_DIR / args.source, encoding="utf-8-sig") as f:
        races = [
            {"date": r["date"], "course": r["course"], "rno": int(r["rno"])}
            for r in csv.DictReader(f)
        ]
    todo = [r for r in races if (r["date"], r["course"], str(r["rno"])) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"対象 {len(races)}レース / 未取得 {len(todo)}件を取得する\n")
    if not todo:
        return

    exists = CACHE.exists()
    f = open(CACHE, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if not exists:
        writer.writeheader()
        f.flush()

    ok = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(fetch, r, args.delay) for r in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r:
                with _lock:
                    writer.writerow(r)
                    f.flush()
                    ok += 1
            if i % 100 == 0:
                rate = i / (time.time() - start)
                print(f"  {i}/{len(todo)} (成功 {ok}) {rate:.1f}件/秒 残り約{(len(todo)-i)/rate/60:.0f}分", flush=True)
    f.close()
    print(f"\n完了: {ok}/{len(todo)}件を追記した")


if __name__ == "__main__":
    main()
