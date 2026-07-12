"""v6-all(1着+着順)モデルで実オッズ・実結果バックテスト。

オッズと確定結果(モデル非依存)は odds_result_cache.csv にキャッシュし、
今後のEV閾値スイープ等で再取得せず使い回せるようにする。

使い方:
    python scripts/backtest_v6.py --sample-size 2000 --min-date 20260512
    python scripts/backtest_v6.py --min-ev 1.10  # 閾値だけ変えてキャッシュから即再評価
"""
import argparse
import csv
import json
import pickle
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_model_v6 import build_features_v6
from main import (
    BoatRaceScraperV5, build_tickets, add_kelly_stakes,
    KELLY_FRACTION, STARTING_BANKROLL,
    IN_JUMP_THRESHOLD, FOCUS_TOP_THRESHOLD, STANDARD_TOP_THRESHOLD, MAX_TICKET_COUNT,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_PATH = BASE_DIR / "odds_result_cache.csv"
_thread_local = threading.local()
_cache_lock = threading.Lock()


def get_scraper():
    if not hasattr(_thread_local, "scraper"):
        _thread_local.scraper = BoatRaceScraperV5()
    return _thread_local.scraper


def load_odds_cache():
    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                cache[(r["date"], r["course"], r["rno"])] = {
                    "result_ticket": r["result_ticket"] or None,
                    "payout": int(r["payout"]) if r["payout"] else 0,
                    "odds": json.loads(r["odds_json"]) if r["odds_json"] else {},
                }
    return cache


def fetch_odds_result(race, delay):
    scraper = get_scraper()
    odds = scraper.fetch_odds3t(race["course"], race["rno"], race["date"])
    time.sleep(delay)
    if not odds:
        return None
    result = scraper.fetch_race_result(race["course"], race["rno"], race["date"])
    time.sleep(delay)
    if not result:
        return None
    return {"result_ticket": result["ticket"], "payout": result["payout"], "odds": odds}


def determine_strategy(win_probs):
    in_jump = 1 - win_probs[0]
    ranking = sorted(((i + 1, win_probs[i]) for i in range(6) if i != 0), key=lambda x: -x[1])
    top1, top2, top3 = ranking[0], ranking[1], ranking[2]
    strat = ""
    if in_jump >= IN_JUMP_THRESHOLD:
        if top1[1] >= FOCUS_TOP_THRESHOLD:
            strat = "FOCUS"
        elif top1[1] >= STANDARD_TOP_THRESHOLD:
            strat = "STANDARD"
    return strat, top1, top2, top3


def trifecta_prob(first, second, third, win_probs, feat_row, m2, m3, cfg):
    p1 = win_probs[first - 1]
    x2 = feat_row.copy(); x2["given_1st"] = first
    raw2 = np.asarray(m2.predict(x2[cfg["features_2nd"]]), dtype=float)[0]
    raw2[first - 1] = 0
    p2 = raw2[second - 1] / raw2.sum() if raw2.sum() > 0 else 0
    x3 = feat_row.copy(); x3["given_1st"] = first; x3["given_2nd"] = second
    raw3 = np.asarray(m3.predict(x3[cfg["features_3rd"]]), dtype=float)[0]
    raw3[first - 1] = 0; raw3[second - 1] = 0
    p3 = raw3[third - 1] / raw3.sum() if raw3.sum() > 0 else 0
    return p1 * p2 * p3


def evaluate_race(race, win_probs, feat_row, cache_entry, m2, m3, cfg, min_ev):
    strat, top1, top2, top3 = determine_strategy(win_probs)
    if not strat:
        return None
    odds_map = cache_entry["odds"]
    tickets = build_tickets(strat, top1, top2, top3)
    enriched = []
    for t in tickets:
        odds = odds_map.get(t)
        if not odds:
            continue
        f, s, th = [int(x) for x in t.split("-")]
        prob = trifecta_prob(f, s, th, win_probs, feat_row, m2, m3, cfg)
        ev = prob * odds
        if ev >= min_ev:
            enriched.append({"ticket": t, "odds": odds, "probability": prob, "expected_value": ev})
    enriched.sort(key=lambda x: x["expected_value"], reverse=True)
    enriched = enriched[:MAX_TICKET_COUNT.get(strat, 8)]
    if not enriched:
        return {"strategy": strat, "stake": 0, "return": 0, "hit": False, "has_bet": False}
    enriched = add_kelly_stakes(enriched, STARTING_BANKROLL, KELLY_FRACTION)
    staked = [t for t in enriched if t["stake"] > 0]
    if not staked:
        return {"strategy": strat, "stake": 0, "return": 0, "hit": False, "has_bet": False}
    stake = sum(t["stake"] for t in staked)
    matched = next((t for t in staked if t["ticket"] == cache_entry["result_ticket"]), None)
    is_hit = matched is not None
    ret = int(matched["stake"] / 100 * cache_entry["payout"]) if is_hit else 0
    return {"strategy": strat, "stake": stake, "return": ret, "hit": is_hit, "has_bet": True,
            "odds": matched["odds"] if is_hit else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training_data.csv")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--min-date", default="20260512")
    parser.add_argument("--min-ev", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(BASE_DIR / "model_v6_all.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "model_v6_all_config.pkl", "rb") as f:
        model_cfg = pickle.load(f)
    with open(BASE_DIR / "order_model_2nd_v6.pkl", "rb") as f:
        m2 = pickle.load(f)
    with open(BASE_DIR / "order_model_3rd_v6.pkl", "rb") as f:
        m3 = pickle.load(f)
    with open(BASE_DIR / "order_model_config_v6.pkl", "rb") as f:
        order_cfg = pickle.load(f)

    df = pd.read_csv(BASE_DIR / args.input, dtype=str, encoding="utf-8-sig")
    df = df[df["date"] >= args.min_date].reset_index(drop=True)
    feats = build_features_v6(df.copy(), "all")
    for c in model_cfg.get("categorical", []):
        feats[c] = feats[c].astype("category")
    all_probs = model.predict(feats[model_cfg["features"]])

    triggering = []
    for idx, (_, row) in enumerate(df.iterrows()):
        strat, *_ = determine_strategy(all_probs[idx])
        if strat:
            triggering.append((row["date"], row["course"], str(row["rno"]), idx))
    print(f"賭け対象レース数: {len(triggering)}")

    by_month = defaultdict(list)
    for item in triggering:
        by_month[item[0][:6]].append(item)
    random.seed(args.seed)
    per_month = max(1, args.sample_size // max(1, len(by_month)))
    sample = []
    for items in by_month.values():
        random.shuffle(items)
        sample.extend(items[:per_month])
    random.shuffle(sample)
    sample = sample[: args.sample_size]
    print(f"サンプル数: {len(sample)}")

    cache = load_odds_cache()
    print(f"オッズキャッシュ: {len(cache)}件")

    # キャッシュに無いレースだけオッズ・結果を取得
    to_fetch = [item for item in sample if (item[0], item[1], item[2]) not in cache]
    print(f"新規取得: {len(to_fetch)}件")
    if to_fetch:
        cache_exists = CACHE_PATH.exists()
        cf = open(CACHE_PATH, "a", newline="", encoding="utf-8-sig")
        writer = csv.DictWriter(cf, fieldnames=["date", "course", "rno", "result_ticket", "payout", "odds_json"])
        if not cache_exists:
            writer.writeheader(); cf.flush()

        def worker(item):
            date, course, rno, idx = item
            race = {"date": date, "course": course, "rno": int(rno)}
            return item, fetch_odds_result(race, args.delay)

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(worker, item) for item in to_fetch]
            for fut in as_completed(futures):
                item, res = fut.result()
                done += 1
                if res:
                    key = (item[0], item[1], item[2])
                    cache[key] = res
                    with _cache_lock:
                        writer.writerow({"date": item[0], "course": item[1], "rno": item[2],
                                         "result_ticket": res["result_ticket"], "payout": res["payout"],
                                         "odds_json": json.dumps(res["odds"])})
                        cf.flush()
                if done % 200 == 0:
                    print(f"  fetch: {done}/{len(to_fetch)}")
        cf.close()

    # 評価(全てローカル・モデル計算のみ)
    results = []
    for date, course, rno, idx in sample:
        entry = cache.get((date, course, rno))
        if not entry:
            continue
        feat_row = feats.iloc[[idx]]
        r = evaluate_race(df.iloc[idx], all_probs[idx], feat_row, entry, m2, m3, order_cfg, args.min_ev)
        if r:
            results.append(r)

    def summarize(rs, label):
        bet = [r for r in rs if r["has_bet"]]
        n = len(bet)
        if not n:
            print(f"{label}: 賭け成立0件"); return
        hits = sum(1 for r in bet if r["hit"])
        stake = sum(r["stake"] for r in bet)
        ret = sum(r["return"] for r in bet)
        roi = ret / stake * 100 if stake else 0
        print(f"{label}: 賭け成立={n} 的中={hits} 的中率={hits/n*100:.1f}% "
              f"投資={stake:,}円 払戻={ret:,}円 収支={ret-stake:+,}円 回収率={roi:.1f}%")

    print(f"\n=== v6-all バックテスト (min-ev={args.min_ev}) ===")
    summarize(results, "全体")
    for strat in ["FOCUS", "STANDARD"]:
        summarize([r for r in results if r["strategy"] == strat], strat)


if __name__ == "__main__":
    main()
