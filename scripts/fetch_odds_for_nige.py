"""イン逃げ条件(1号艇の勝率>=55%と予測)が出たホールドアウトのレースについて、
実オッズと実結果を取得して odds_result_cache.csv に追記する。

既存キャッシュは「イン飛び条件が出たレース」だけを集めたバックテストの副産物なので、
1号艇の勝率が24.7%しかない偏った標本になっている(全国平均は55.4%)。
イン逃げ戦略をそこで評価するのは不公平なため、逆側のレースを補充する。

使い方:
    python scripts/fetch_odds_for_nige.py --sample-size 600
"""
import argparse
import csv
import json
import pickle
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_model import build_features  # noqa: E402
from main import BoatRaceScraperV5, IN_JUMP_THRESHOLD  # noqa: E402

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
    parser.add_argument("--sample-size", type=int, default=600)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--min-date", default="20260512")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    done = set()
    if CACHE.exists():
        with open(CACHE, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done.add((r["date"], r["course"], str(int(r["rno"]))))
    print(f"既存キャッシュ: {len(done)}件")

    with open(BASE_DIR / "training_data.csv", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["date"] >= args.min_date]
    print(f"ホールドアウト: {len(rows)}レース")

    model = pickle.load(open(BASE_DIR / "final_model_v5.pkl", "rb"))
    probs = model.predict(build_features(pd.DataFrame(rows)))

    # イン逃げ条件が出ていて、まだオッズを持っていないレース
    targets = [
        row for row, p in zip(rows, probs)
        if p[0] >= IN_JUMP_THRESHOLD
        and (row["date"], row["course"], str(int(row["rno"]))) not in done
    ]
    print(f"イン逃げ条件を満たし、かつ未取得: {len(targets)}レース")

    random.seed(args.seed)
    random.shuffle(targets)
    targets = targets[: args.sample_size]
    print(f"今回取得する: {len(targets)}レース\n")

    exists = CACHE.exists()
    f = open(CACHE, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if not exists:
        writer.writeheader()

    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(fetch, r, args.delay) for r in targets]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r:
                with _lock:
                    writer.writerow(r)
                    f.flush()
                    ok += 1
            if i % 100 == 0:
                print(f"  progress: {i}/{len(targets)} (取得成功 {ok})", flush=True)
    f.close()
    print(f"\n取得成功: {ok}/{len(targets)}件を追記した")


if __name__ == "__main__":
    main()
