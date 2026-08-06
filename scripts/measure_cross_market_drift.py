"""締切前の暫定オッズでも、単勝×3連単プール間の乖離(diff)が機能するかを実測する。

CROSS_MARKET_REPORT.md で見つかった効果(1〜6倍帯・diff>=0.08で回収率116.9%)は
確定後の最終オッズで検証したものだった。本番は締切5〜35分前の暫定オッズで賭けるため、
そこでも同じ効果が残るかは別問題(オッズドリフトは最終オッズと中央値32%ズレることが
既に分かっている)。

measure_odds_drift.py と同じやり方で、締切前の複数タイミングで単勝オッズと3連単オッズを
両方記録し、レース確定後に最終オッズ・実結果と突き合わせて finalize する。

使い方(レース開催時間中、7:00〜21:00 JSTに走らせる):
    python scripts/measure_cross_market_drift.py --hours 6
    (レース終了後)
    python scripts/measure_cross_market_drift.py --finalize
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import BoatRaceScraperV5, JST  # noqa: E402
from scripts.cross_market_arbitrage import marginal_win_from_3t, normalize, BOATS  # noqa: E402
from scripts.fetch_cross_odds import fetch_oddstf  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "cross_market_drift_samples.csv"
FIELDS = ["date", "course", "rno", "deadline", "minutes_before", "odds3t_json", "tan_json"]

OFFSETS = [30, 20, 15, 10, 7, 5, 3, 2]
TOLERANCE = 1


def snapshot(scraper, race, minutes_before, writer):
    odds3t = scraper.fetch_odds3t(race["course"], race["rno"], race["date"])
    if not odds3t:
        return False
    tan, _fuku = fetch_oddstf(scraper, race["course"], race["rno"], race["date"])
    if len(tan) != 6:
        return False
    writer.writerow({
        "date": race["date"], "course": race["course"], "rno": race["rno"],
        "deadline": race["deadline"], "minutes_before": minutes_before,
        "odds3t_json": json.dumps(odds3t), "tan_json": json.dumps(tan),
    })
    return True


def run_live(args):
    scraper = BoatRaceScraperV5()
    exists = OUT.exists()
    f = open(OUT, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if not exists:
        writer.writeheader()

    end_at = datetime.now(JST) + timedelta(hours=args.hours)
    taken = set()

    print(f"開始: {datetime.now(JST):%H:%M} 〜 {end_at:%H:%M} JST まで記録する\n")

    while datetime.now(JST) < end_at:
        now = datetime.now(JST)
        date_str = now.strftime("%Y%m%d")
        try:
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
                    ok = snapshot(scraper, {
                        "date": date_str, "course": course,
                        "rno": rno, "deadline": time_str,
                    }, off, writer)
                    if ok:
                        taken.add((race_id, off))
                        f.flush()
                        print(f"  {course}{rno}R 締切{time_str} の{off}分前: 記録", flush=True)
                    time.sleep(0.3)

        time.sleep(45)

    f.close()
    print(f"\n記録完了: {OUT.name}")


def finalize(args):
    """記録済みレースの実結果を取り、暫定オッズでdiff戦略が機能するか集計する。"""
    scraper = BoatRaceScraperV5()
    rows = list(csv.DictReader(open(OUT, encoding="utf-8-sig")))
    races = sorted({(r["date"], r["course"], r["rno"]) for r in rows})
    print(f"記録: {len(rows):,}件 / {len(races)}レース\n")

    results = {}
    for date, course, rno in races:
        r = scraper.fetch_race_result(course, int(rno), date)
        if r:
            results[(date, course, rno)] = r
        time.sleep(0.2)
    print(f"実結果取得: {len(results)}レース\n")

    by_off = {}  # off -> list of (diff, tan_odds, hit)
    for row in rows:
        key = (row["date"], row["course"], row["rno"])
        if key not in results:
            continue
        winner = int(results[key]["ticket"].split("-")[0])
        odds3t = {t: float(o) for t, o in json.loads(row["odds3t_json"]).items()}
        tan = {int(k): float(v) for k, v in json.loads(row["tan_json"]).items()}
        q_tan = normalize({b: 1 / tan[b] for b in BOATS})
        q_3t = marginal_win_from_3t(odds3t)
        for b in BOATS:
            d = q_3t[b] - q_tan[b]
            by_off.setdefault(int(row["minutes_before"]), []).append((d, tan[b], b == winner))

    print("締切前タイミングごとの diff>=0.08 選抜・単勝ベット回収率")
    print("(最終オッズでの検証結果: 116.9%[103.8〜130.0%] が基準)\n")
    for off in sorted(by_off, reverse=True):
        recs = by_off[off]
        diffs = np.array([r[0] for r in recs])
        odds = np.array([r[1] for r in recs])
        hit = np.array([r[2] for r in recs])
        m = (diffs >= 0.08) & (odds >= 1) & (odds < 6)
        n = int(m.sum())
        if n == 0:
            print(f"  締切{off:3d}分前: 該当なし")
            continue
        roi = odds[m & hit].sum() / n * 100
        print(f"  締切{off:3d}分前: n={n:4d} 回収率={roi:6.1f}%")

    print("\n※ 締切5〜35分前(本番の窓)の回収率が最終オッズ版から大きく崩れていないかを確認する。")


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
