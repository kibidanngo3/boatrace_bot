"""買い目構築方式の比較: 現行の絞り込み vs 全120通りEV順。

現行の build_tickets は top1/top2/top3 から機械的に組んだ4〜8点だけを候補にする。
これを「全120通りの3連単からEV順に選ぶ」方式に変えると、上位予測に入らないが
高EVな買い目を拾えるかもしれない。キャッシュ済みオッズで3方式を同一レース比較する。

方式:
  A. narrow    : 現行 build_tickets（top1/2/3 絞り込み）
  B. full120   : 全120通りからEV順
  C. full120_nolead1: 全120通りだが1号艇1着(イン逃げ)の買い目は除外（イン飛び戦略の趣旨に合わせる）

いずれも strategy トリガー・MAX_TICKET_COUNT上限・EV>=1.0・ケリーは共通。
"""
import csv
import json
import pickle
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_model import build_features
from main import (
    build_tickets, add_kelly_stakes, estimate_ticket_probability,
    KELLY_FRACTION, STARTING_BANKROLL, MAX_TICKET_COUNT, MIN_EXPECTED_VALUE,
    IN_JUMP_THRESHOLD, FOCUS_TOP_THRESHOLD, STANDARD_TOP_THRESHOLD,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_PATH = BASE_DIR / "odds_result_cache.csv"
ALL_TRIFECTA = [f"{a}-{b}-{c}" for a, b, c in permutations(range(1, 7), 3)]


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


def candidate_tickets(mode, strat, top1, top2, top3, odds_map):
    if mode == "narrow":
        return build_tickets(strat, top1, top2, top3)
    if mode == "full120":
        return [t for t in ALL_TRIFECTA if t in odds_map]
    if mode == "full120_nolead1":
        return [t for t in ALL_TRIFECTA if t in odds_map and not t.startswith("1-")]
    raise ValueError(mode)


def max_drawdown(sorted_profits):
    cum = peak = max_dd = 0
    for p in sorted_profits:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return max_dd


def evaluate(mode, per_race):
    bets = hits = stake_sum = ret_sum = 0
    pnls = []
    for pr in per_race:
        picked = [t for t in pr[mode] if t["expected_value"] >= MIN_EXPECTED_VALUE]
        picked.sort(key=lambda x: x["expected_value"], reverse=True)
        picked = picked[:MAX_TICKET_COUNT.get(pr["strategy"], 8)]
        if not picked:
            continue
        picked = add_kelly_stakes(picked, STARTING_BANKROLL, KELLY_FRACTION)
        staked = [t for t in picked if t["stake"] > 0]
        if not staked:
            continue
        stake = sum(t["stake"] for t in staked)
        matched = next((t for t in staked if t["ticket"] == pr["result_ticket"]), None)
        ret = int(matched["stake"] / 100 * pr["payout"]) if matched else 0
        bets += 1
        hits += 1 if matched else 0
        stake_sum += stake
        ret_sum += ret
        pnls.append((pr["date"], ret - stake))
    roi = ret_sum / stake_sum * 100 if stake_sum else 0
    hit_rate = hits / bets * 100 if bets else 0
    pnls.sort(key=lambda x: x[0])
    dd = max_drawdown([p for _, p in pnls])
    return bets, hits, hit_rate, stake_sum, ret_sum, roi, dd


def main_run():
    with open(BASE_DIR / "final_model_v5.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "model_config_v5.pkl", "rb") as f:
        config = pickle.load(f)

    cache = {}
    with open(CACHE_PATH, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cache[(r["date"], r["course"], r["rno"])] = {
                "result_ticket": r["result_ticket"] or None,
                "payout": int(r["payout"]) if r["payout"] else 0,
                "odds": json.loads(r["odds_json"]) if r["odds_json"] else {},
            }

    df = pd.read_csv(BASE_DIR / "training_data.csv", dtype=str, encoding="utf-8-sig")
    df["key"] = list(zip(df["date"], df["course"], df["rno"].astype(str)))
    df = df[df["key"].isin(cache.keys())].reset_index(drop=True)
    feats = build_features(df.copy())[config["features"]]
    win_probs_all = np.asarray(model.predict(feats), dtype=float)

    modes = ["narrow", "full120", "full120_nolead1"]
    per_race = []
    for i, row in df.iterrows():
        probs = win_probs_all[i]
        strat, top1, top2, top3 = determine_strategy(probs)
        if not strat:
            continue
        entry = cache[(row["date"], row["course"], str(row["rno"]))]
        odds_map = entry["odds"]
        input_df = feats.iloc[[i]]
        rec = {"strategy": strat, "date": row["date"], "result_ticket": entry["result_ticket"], "payout": entry["payout"]}
        for mode in modes:
            ev_tickets = []
            for t in candidate_tickets(mode, strat, top1, top2, top3, odds_map):
                odds = odds_map.get(t)
                if not odds:
                    continue
                prob = estimate_ticket_probability(t, probs, input_df)
                ev_tickets.append({"ticket": t, "odds": odds, "probability": prob, "expected_value": prob * odds})
            rec[mode] = ev_tickets
        per_race.append(rec)
    print(f"戦略トリガーレース: {len(per_race)}\n")

    print(f"{'方式':<18} | {'賭数':>4} {'的中':>3} {'的中率':>6} | {'投資':>9} {'払戻':>10} {'収支':>10} {'回収率':>7} {'最大DD':>9}")
    print("-" * 92)
    for mode in modes:
        bets, hits, hr, stake, ret, roi, dd = evaluate(mode, per_race)
        print(f"{mode:<18} | {bets:>4} {hits:>3} {hr:>5.1f}% | "
              f"{stake:>8,}円 {ret:>9,}円 {ret-stake:>+9,}円 {roi:>6.1f}% {dd:>7,}円")


if __name__ == "__main__":
    main_run()
