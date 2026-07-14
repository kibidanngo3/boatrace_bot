"""ホールドアウト期間の実オッズ(3連単全120通り)と実結果を集めて odds_result_cache.csv に貯める。

これまでのキャッシュは「その時々のバックテストが必要としたレース」だけを取っていたため、
モデルの発動条件で選ばれた偏った標本になっていた(実際、1号艇の1着率が24.7%しかなく、
全国平均55.4%とかけ離れていた)。優位性の有無を判定するには、条件で絞らずホールドアウト
全体のオッズが要る。

使い方:
    python scripts/fetch_odds.py --min-date 20260512
"""
import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import BoatRaceScraperV5  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE = BASE_DIR / "odds_result_cache.csv"
FIELDS = ["date", "course", "rno", "result_ticket", "payout", "odds_json"]

_lock = threading.Lock()
_thread_local = threading.local()


def get_scraper():
    if not hasattr(_thread_local, "scraper"):
        _thread_local.scraper = BoatRaceScraperV5()
    return _thread_local.scraper


def fetch(race, delay):
    scraper = get_scraper()
    try:
        odds_map = scraper.fetch_odds3t(race["course"], race["rno"], race["date"])
        time.sleep(delay)
        if not odds_map:
            return None
        result = scraper.fetch_race_result(race["course"], race["rno"], race["date"])
        time.sleep(delay)
        if not result:
            return None
        return {
            "date": race["date"], "course": race["course"], "rno": race["rno"],
            "result_ticket": result["ticket"], "payout": result["payout"],
            "odds_json": json.dumps(odds_map),
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-date", default="20260512", help="ホールドアウトの開始日")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    done = set()
    if CACHE.exists():
        with open(CACHE, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done.add((r["date"], r["course"], str(int(r["rno"]))))
    print(f"既存キャッシュ: {len(done)}件")

    with open(BASE_DIR / "training_data.csv", encoding="utf-8-sig") as f:
        races = [
            {"date": r["date"], "course": r["course"], "rno": int(r["rno"])}
            for r in csv.DictReader(f)
            if r["date"] >= args.min_date
        ]
    todo = [r for r in races
            if (r["date"], r["course"], str(r["rno"])) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"ホールドアウト {len(races)}レース / 未取得 {len(todo)}件を取得する\n")
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
            if i % 300 == 0:
                rate = i / (time.time() - start)
                print(f"  {i}/{len(todo)} (成功 {ok}) "
                      f"{rate:.1f}件/秒 残り約{(len(todo)-i)/rate/60:.0f}分", flush=True)
    f.close()
    print(f"\n完了: {ok}/{len(todo)}件を追記した")


if __name__ == "__main__":
    main()
