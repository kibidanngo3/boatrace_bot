"""boatrace.jp の直前情報ページから「スタート展示」を過去分収集する。

ここから取れるのは、B/Kファイルには含まれないがレース前に確定している2つの情報:

  1. 展示ST      : スタート展示で各艇が実際に切ったST(.19 / F.07=フライングは負値)
  2. 進入コース  : スタート展示での進入順(前付けがあると枠番と一致しない)

いずれも締切前に公開されるため特徴量として正当。学習に使っていた st_i(Kファイルの
本番ST)とは全くの別物である点に注意(あちらはレース後にしか確定しない=漏洩)。

中断しても再開できるよう、1レース取得するたびにCSVへ追記する。

使い方:
    python scripts/fetch_exhibition.py --workers 8
    python scripts/fetch_exhibition.py --limit 200      # 動作確認用
"""
import argparse
import csv
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from bs4 import BeautifulSoup, SoupStrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import BoatRaceScraperV5  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "exhibition_data.csv"

# 5万ページ取るので、本番の _get_soup は使わない。
# あれは res.apparent_encoding(本文全体を走査する文字コード推定)を毎回呼ぶため
# 1リクエストに数秒かかり、この規模だと十数時間かかってしまう。
# ここでは encoding を utf-8 に固定し、スタート展示のテーブルだけを lxml で読む。
ONLY_ST_TABLE = SoupStrainer("table", class_="is-w238")
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}

FIELDS = (
    ["date", "course", "rno"]
    + [f"ex_st_{i}" for i in range(1, 7)]
    + [f"in_course_{i}" for i in range(1, 7)]
)

_lock = threading.Lock()
_thread_local = threading.local()


COURSE_MAP = BoatRaceScraperV5.COURSE_MAP


def get_session():
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update(HEADERS)
        _thread_local.session = s
    return _thread_local.session


def parse_st(text):
    """'.19' -> 0.19 / 'F.07' -> -0.07(フライングは大時計より前に出ている) / 'L' -> None"""
    text = (text or "").strip()
    if not text:
        return None
    m = re.search(r"(\d?\.\d+)", text)
    if not m:
        return None  # 'L'(出遅れ)など
    value = float("0" + m.group(1)) if m.group(1).startswith(".") else float(m.group(1))
    return -value if text.upper().startswith("F") else value


def fetch(race, delay):
    jcd = COURSE_MAP.get(race["course"])
    if not jcd:
        return None
    url = (f"https://www.boatrace.jp/owpc/pc/race/beforeinfo"
           f"?rno={race['rno']}&jcd={jcd}&hd={race['date']}")
    try:
        res = get_session().get(url, timeout=20)
        res.raise_for_status()
        res.encoding = "utf-8"
        table = BeautifulSoup(res.text, "lxml", parse_only=ONLY_ST_TABLE)
        if not table.find("tbody"):
            return None

        row = {"date": race["date"], "course": race["course"], "rno": race["rno"]}
        found = 0
        # 行の並び順がそのまま進入コース(1コース〜6コース)
        for lane, tr in enumerate(table.select("tbody tr"), start=1):
            num = tr.select_one("span.table1_boatImage1Number")
            st = tr.select_one("span.table1_boatImage1Time")
            if not num or not st:
                continue
            try:
                boat = int(num.get_text(strip=True))
            except ValueError:
                continue
            if not 1 <= boat <= 6:
                continue
            row[f"ex_st_{boat}"] = parse_st(st.get_text(strip=True))
            row[f"in_course_{boat}"] = lane
            found += 1

        if found != 6:
            return None  # 6艇そろわない回は捨てる(中止・欠場など)
        return row
    except Exception:
        return None
    finally:
        time.sleep(delay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training_data.csv")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None, help="動作確認用に件数を絞る")
    args = parser.parse_args()

    done = set()
    if OUT.exists():
        with open(OUT, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done.add((r["date"], r["course"], r["rno"]))
        print(f"取得済み: {len(done)}件(再開)")

    with open(BASE_DIR / args.input, encoding="utf-8-sig") as f:
        races = [
            {"date": r["date"], "course": r["course"], "rno": str(int(r["rno"]))}
            for r in csv.DictReader(f)
        ]
    todo = [r for r in races if (r["date"], r["course"], r["rno"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"全{len(races)}レース / 未取得 {len(todo)}件を取得する\n")
    if not todo:
        return

    exists = OUT.exists()
    f = open(OUT, "a", newline="", encoding="utf-8-sig")
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
            if i % 500 == 0:
                rate = i / (time.time() - start)
                eta = (len(todo) - i) / rate / 60
                print(f"  {i}/{len(todo)} (成功 {ok}) "
                      f"{rate:.1f}件/秒 残り約{eta:.0f}分", flush=True)
    f.close()
    print(f"\n完了: {ok}/{len(todo)}件を {OUT.name} に保存した")


if __name__ == "__main__":
    main()
