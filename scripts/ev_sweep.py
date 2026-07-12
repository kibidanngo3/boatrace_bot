"""本番モデル(v5 + order_v1)でEV閾値をスイープし、最適な MIN_EXPECTED_VALUE を探る。

odds_result_cache.csv のキャッシュ済みオッズ・結果を使うため再取得不要。
全閾値を同一レース集合で評価するので、閾値間の相対比較として妥当。
main.py の estimate_ticket_probability をそのまま使う（order_v1モデルを自動ロード）ので本番挙動に忠実。

追加で、決着日順に資金を累積したときの最大ドローダウンも算出する
（これまで未計測だった資金推移リスクの指標）。

使い方:
    python scripts/ev_sweep.py
"""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_model import build_features
import main
from main import (
    build_tickets, add_kelly_stakes, estimate_ticket_probability,
    KELLY_FRACTION, STARTING_BANKROLL, MAX_TICKET_COUNT,
    IN_JUMP_THRESHOLD, FOCUS_TOP_THRESHOLD, STANDARD_TOP_THRESHOLD,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_PATH = BASE_DIR / "odds_result_cache.csv"
THRESHOLDS = [1.00, 1.05, 1.10, 1.15, 1.20]


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


def max_drawdown(sorted_profits):
    """決着順のper-race損益リストから、累積資金の最大ドローダウン(円)を返す。"""
    cum = 0
    peak = 0
    max_dd = 0
    for p in sorted_profits:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return max_dd


def main_run():
    with open(BASE_DIR / "final_model_v5.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "model_config_v5.pkl", "rb") as f:
        config = pickle.load(f)

    # キャッシュ読み込み
    cache = {}
    with open(CACHE_PATH, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cache[(r["date"], r["course"], r["rno"])] = {
                "result_ticket": r["result_ticket"] or None,
                "payout": int(r["payout"]) if r["payout"] else 0,
                "odds": json.loads(r["odds_json"]) if r["odds_json"] else {},
            }
    print(f"キャッシュ: {len(cache)}レース")

    # 特徴量入力用に training_data を読み、キャッシュ対象レースだけ抽出
    df = pd.read_csv(BASE_DIR / "training_data.csv", dtype=str, encoding="utf-8-sig")
    df["key"] = list(zip(df["date"], df["course"], df["rno"].astype(str)))
    df = df[df["key"].isin(cache.keys())].reset_index(drop=True)
    feats = build_features(df.copy())[config["features"]]
    win_probs_all = np.asarray(model.predict(feats), dtype=float)

    # レースごとに (strategy, tickets, EV付きチケット) を1回だけ計算しておく
    per_race = []  # {date, ev_tickets: [{ticket,odds,ev}], result_ticket, payout}
    for i, row in df.iterrows():
        probs = win_probs_all[i]
        strat, top1, top2, top3 = determine_strategy(probs)
        if not strat:
            continue
        entry = cache[(row["date"], row["course"], str(row["rno"]))]
        odds_map = entry["odds"]
        input_df = feats.iloc[[i]]
        ev_tickets = []
        for t in build_tickets(strat, top1, top2, top3):
            odds = odds_map.get(t)
            if not odds:
                continue
            prob = estimate_ticket_probability(t, probs, input_df)
            ev_tickets.append({"ticket": t, "odds": odds, "probability": prob, "expected_value": prob * odds})
        if ev_tickets:
            per_race.append({
                "date": row["date"], "strategy": strat, "ev_tickets": ev_tickets,
                "result_ticket": entry["result_ticket"], "payout": entry["payout"],
            })
    print(f"戦略トリガー&オッズ有りレース: {len(per_race)}\n")

    print(f"{'EV閾値':>6} | {'賭数':>4} {'的中':>3} {'的中率':>6} | {'投資':>9} {'払戻':>10} {'収支':>10} {'回収率':>7} {'最大DD':>9}")
    print("-" * 88)
    for th in THRESHOLDS:
        race_pnls = []  # (date, profit)
        bets = hits = stake_sum = ret_sum = 0
        for pr in per_race:
            picked = [t for t in pr["ev_tickets"] if t["expected_value"] >= th]
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
            race_pnls.append((pr["date"], ret - stake))

        roi = ret_sum / stake_sum * 100 if stake_sum else 0
        hit_rate = hits / bets * 100 if bets else 0
        race_pnls.sort(key=lambda x: x[0])
        dd = max_drawdown([p for _, p in race_pnls])
        print(f"{th:>6.2f} | {bets:>4} {hits:>3} {hit_rate:>5.1f}% | "
              f"{stake_sum:>8,}円 {ret_sum:>9,}円 {ret_sum-stake_sum:>+9,}円 {roi:>6.1f}% {dd:>7,}円")


if __name__ == "__main__":
    main_run()
