"""締切までの時間ごとに、オッズが最終オッズからどれだけズレているかを実測する。

本番は締切5〜35分前に買い目を決めているが(main.py:1423 の `if 5 <= diff <= 35`)、
その時点のオッズは実測で中央値32%も最終オッズとズレていた
(predictions.csv の賭け時オッズ vs 最終オッズ、96点)。

締切に近づくほどオッズは収束するはずだが、**どのくらい近づけば十分なのかは測っていない**。
勘で窓を変えるのは今日の教訓に反するので、実測する。

レース当日に走らせ、締切前の各タイミングでオッズを記録し、後で最終オッズと比較する。

使い方(レース開催時間中に走らせる):
    python scripts/measure_odds_drift.py --hours 6

出力: odds_drift_samples.csv (date, course, rno, minutes_before, ticket, odds)
     最終オッズは finalize で後から付ける:
    python scripts/measure_odds_drift.py --finalize
"""
import argparse
import csv
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import BoatRaceScraperV5, JST  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "odds_drift_samples.csv"
FIELDS = ["date", "course", "rno", "deadline", "minutes_before", "ticket", "odds"]

# 締切の何分前にオッズを記録するか
OFFSETS = [30, 20, 15, 10, 7, 5, 3, 2]
TOLERANCE = 1  # 目標時刻に対する許容誤差(分)


def snapshot(scraper, race, minutes_before, writer, lock_free=True):
    odds_map = scraper.fetch_odds3t(race["course"], race["rno"], race["date"])
    if not odds_map:
        return 0
    for ticket, odds in odds_map.items():
        writer.writerow({
            "date": race["date"], "course": race["course"], "rno": race["rno"],
            "deadline": race["deadline"], "minutes_before": minutes_before,
            "ticket": ticket, "odds": odds,
        })
    return len(odds_map)


def run_live(args):
    scraper = BoatRaceScraperV5()
    exists = OUT.exists()
    f = open(OUT, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if not exists:
        writer.writeheader()

    end_at = datetime.now(JST) + timedelta(hours=args.hours)
    taken = set()  # (race_id, offset)

    while datetime.now(JST) < end_at:
        now = datetime.now(JST)
        date_str = now.strftime("%Y%m%d")
        try:
            # main.py と同じ形式: {(会場, レース番号): (締切時刻, URL)}
            all_races = scraper.fetch_all_venue_schedules(date_str)
        except Exception as e:
            print(f"スケジュール取得失敗: {e}", flush=True)
            time.sleep(60)
            continue

        for (course, rno), (time_str, _url) in all_races.items():
            try:
                deadline = datetime.strptime(
                    f"{date_str} {time_str}", "%Y%m%d %H:%M").replace(tzinfo=JST)
            except Exception:
                continue
            diff = (deadline - now).total_seconds() / 60
            race_id = f"{date_str}_{course}_{rno}"

            for off in OFFSETS:
                if abs(diff - off) <= TOLERANCE and (race_id, off) not in taken:
                    n = snapshot(scraper, {
                        "date": date_str, "course": course,
                        "rno": rno, "deadline": time_str,
                    }, off, writer)
                    if n:
                        taken.add((race_id, off))
                        f.flush()
                        print(f"  {course}{rno}R 締切{time_str} "
                              f"の{off}分前: {n}点記録", flush=True)
                    time.sleep(0.3)

        time.sleep(45)

    f.close()
    print(f"\n記録完了: {OUT.name}")


def finalize(args):
    """記録済みのレースについて最終オッズを取り、ズレを集計する。"""
    scraper = BoatRaceScraperV5()
    rows = list(csv.DictReader(open(OUT, encoding="utf-8-sig")))
    races = sorted({(r["date"], r["course"], r["rno"]) for r in rows})
    print(f"記録: {len(rows):,}点 / {len(races)}レース\n")

    finals = {}
    for date, course, rno in races:
        m = scraper.fetch_odds3t(course, int(rno), date)
        if m:
            finals[(date, course, rno)] = m
        time.sleep(0.3)
    print(f"最終オッズ取得: {len(finals)}レース\n")

    by_off = {}
    for r in rows:
        key = (r["date"], r["course"], r["rno"])
        if key not in finals:
            continue
        fin = finals[key].get(r["ticket"])
        if not fin:
            continue
        seen = float(r["odds"])
        if seen <= 0:
            continue
        by_off.setdefault(int(r["minutes_before"]), []).append(float(fin) / seen)

    print("締切までの時間ごとの、最終オッズとのズレ")
    print(f"{'締切前':>7} {'点数':>8} {'ズレ幅の中央値':>14} {'25%超ズレた割合':>16} {'比の中央値':>11}")
    for off in sorted(by_off, reverse=True):
        ratios = np.array(by_off[off])
        print(f"{off:5d}分 {len(ratios):8,} "
              f"{np.median(np.abs(ratios - 1)) * 100:13.1f}% "
              f"{(np.abs(ratios - 1) > 0.25).mean() * 100:15.0f}% "
              f"{np.median(ratios):10.3f}")

    print("\n※ ズレ幅が十分小さくなる時刻が、買い目を決めるべきタイミング。")
    print("   現在の本番の窓は締切5〜35分前(main.py の `if 5 <= diff <= 35`)。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=6)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    if args.finalize:
        finalize(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
