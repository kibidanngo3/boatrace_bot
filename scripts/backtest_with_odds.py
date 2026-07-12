"""v5モデル + 実際の過去オッズ・結果を使って、EV/ケリー戦略込みの回収率をバックテストする。

training_data.csv には勝ち艇番号しか入っていないため、オッズ取得と同時に
boatrace.jpのraceresultページから実際の3連単結果も取得する。
賭け金計算は毎回バンクロール10,000円固定という単純化をしている(累積シミュレーションではない)。

使い方:
    python scripts/backtest_with_odds.py --sample-size 2000
"""
import argparse
import csv
import pickle
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_model import build_features  # noqa: E402
from main import (  # noqa: E402
    BoatRaceScraperV5, build_tickets, add_expected_values, add_kelly_stakes,
    MIN_EXPECTED_VALUE, KELLY_FRACTION, STARTING_BANKROLL,
    IN_JUMP_THRESHOLD, FOCUS_TOP_THRESHOLD, STANDARD_TOP_THRESHOLD,
)

BASE_DIR = Path(__file__).resolve().parent.parent
_thread_local = threading.local()


def get_scraper():
    if not hasattr(_thread_local, "scraper"):
        _thread_local.scraper = BoatRaceScraperV5()
    return _thread_local.scraper


def determine_strategy(probs):
    in_win_prob = probs[0]
    in_jump_prob = 1 - in_win_prob
    all_ranking = sorted({i + 1: p for i, p in enumerate(probs)}.items(), key=lambda x: x[1], reverse=True)
    ranking_without_1 = [r for r in all_ranking if r[0] != 1]
    top1, top2, top3 = ranking_without_1[0], ranking_without_1[1], ranking_without_1[2]

    strategy = ""
    if in_jump_prob >= IN_JUMP_THRESHOLD:
        if top1[1] >= FOCUS_TOP_THRESHOLD:
            strategy = "FOCUS"
        elif top1[1] >= STANDARD_TOP_THRESHOLD:
            strategy = "STANDARD"
        else:
            strategy = "WIDE"
    return strategy, top1, top2, top3


def process_race(race, probs, delay):
    scraper = get_scraper()
    strategy, top1, top2, top3 = determine_strategy(probs)
    if not strategy:
        return None

    tickets = build_tickets(strategy, top1, top2, top3)
    odds_map = scraper.fetch_odds3t(race["course"], race["rno"], race["date"])
    time.sleep(delay)
    if not odds_map:
        return None

    value_tickets = add_expected_values(tickets, probs, odds_map, strategy)
    if not value_tickets:
        return {"strategy": strategy, "stake": 0, "return": 0, "hit": False, "has_bet": False}

    value_tickets = add_kelly_stakes(value_tickets, STARTING_BANKROLL, KELLY_FRACTION)
    staked = [t for t in value_tickets if t["stake"] > 0]
    if not staked:
        return {"strategy": strategy, "stake": 0, "return": 0, "hit": False, "has_bet": False}

    result = scraper.fetch_race_result(race["course"], race["rno"], race["date"])
    time.sleep(delay)
    if not result:
        return None

    stake = sum(t["stake"] for t in staked)
    matched = next((t for t in staked if t["ticket"] == result["ticket"]), None)
    is_hit = matched is not None
    return_amount = int(matched["stake"] / 100 * result["payout"]) if is_hit else 0

    return {
        "strategy": strategy,
        "stake": stake,
        "return": return_amount,
        "hit": is_hit,
        "has_bet": True,
        "course": race["course"],
        "rno": race["rno"],
        "date": race["date"],
        "odds": matched["odds"] if is_hit else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training_data.csv")
    parser.add_argument("--model", default="final_model_v5.pkl")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-date", default=None, help="この日付以降のみ対象にする(学習データとの重複=過学習を避けるため)")
    args = parser.parse_args()

    with open(BASE_DIR / args.model, "rb") as f:
        model = pickle.load(f)

    with open(BASE_DIR / args.input, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if args.min_date:
        rows = [r for r in rows if r["date"] >= args.min_date]
    print(f"総レース数: {len(rows)}" + (f" ({args.min_date}以降)" if args.min_date else ""))

    df = pd.DataFrame(rows)
    X = build_features(df)
    all_probs = model.predict(X)

    triggering = []
    for row, probs in zip(rows, all_probs):
        strategy, *_ = determine_strategy(probs)
        if strategy:
            row["rno"] = int(row["rno"])
            triggering.append((row, probs))
    print(f"賭け対象レース数: {len(triggering)}")

    by_month = defaultdict(list)
    for item in triggering:
        by_month[item[0]["date"][:6]].append(item)

    random.seed(args.seed)
    per_month = max(1, args.sample_size // max(1, len(by_month)))
    sample = []
    for month, items in by_month.items():
        random.shuffle(items)
        sample.extend(items[:per_month])
    random.shuffle(sample)
    sample = sample[: args.sample_size]
    print(f"サンプル数: {len(sample)} ({len(by_month)}ヶ月に分散)")

    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_race, row, probs, args.delay) for row, probs in sample]
        for future in as_completed(futures):
            r = future.result()
            if r:
                results.append(r)
            done += 1
            if done % 200 == 0:
                print(f"  progress: {done}/{len(sample)}")

    print(f"\n有効件数(オッズ取得成功): {len(results)}")

    def summarize(rs, label):
        bet_rs = [r for r in rs if r["has_bet"]]
        n = len(bet_rs)
        if n == 0:
            print(f"{label}: 賭け成立0件")
            return
        hits = sum(1 for r in bet_rs if r["hit"])
        stake = sum(r["stake"] for r in bet_rs)
        ret = sum(r["return"] for r in bet_rs)
        roi = (ret / stake * 100) if stake else 0
        print(
            f"{label}: 賭け成立={n}/{len(rs)}(EV/Kelly条件クリア) 的中={hits} "
            f"的中率={hits/n*100:.1f}% 投資={stake:,}円 払戻={ret:,}円 "
            f"収支={ret-stake:+,}円 回収率={roi:.1f}%"
        )

    print("\n--- 的中した賭けの内訳 ---")
    for r in sorted((r for r in results if r["hit"]), key=lambda r: -r["return"]):
        print(f"  {r['date']} {r['course']}{r['rno']}R odds={r['odds']} stake={r['stake']} return={r['return']:,}")

    print("\n--- 全体 ---")
    summarize(results, "全体")
    print("\n--- 戦略別 ---")
    for strategy in ["FOCUS", "STANDARD", "WIDE"]:
        rs = [r for r in results if r["strategy"] == strategy]
        summarize(rs, strategy)


if __name__ == "__main__":
    main()
