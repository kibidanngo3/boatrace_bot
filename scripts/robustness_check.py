"""full120_nolead1 の改善が「まぐれ」でないかを検証する。

(1) 高配当依存チェック: 的中を払戻の大きい順に並べ、上位N件を除いても
    現行(narrow)より優位が残るか。
(2) 期間分割チェック: ホールドアウト60日を前半/後半に割り、両方で改善するか。
(3) 的中内訳: full120_nolead1で当たった買い目を全部並べる。

キャッシュ(odds_result_cache.csv)を使うので再取得なし。
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
    if mode == "full120_nolead1":
        return [t for t in ALL_TRIFECTA if t in odds_map and not t.startswith("1-")]
    raise ValueError(mode)


def simulate(mode, per_race):
    """各賭けレースの (date, stake, return, hit) リストを返す。"""
    out = []
    for pr in per_race:
        ev_tickets = []
        for t in candidate_tickets(mode, pr["strategy"], pr["top1"], pr["top2"], pr["top3"], pr["odds"]):
            odds = pr["odds"].get(t)
            if not odds:
                continue
            prob = estimate_ticket_probability(t, pr["probs"], pr["input_df"])
            ev = prob * odds
            if ev >= MIN_EXPECTED_VALUE:
                ev_tickets.append({"ticket": t, "odds": odds, "probability": prob, "expected_value": ev})
        ev_tickets.sort(key=lambda x: x["expected_value"], reverse=True)
        ev_tickets = ev_tickets[:MAX_TICKET_COUNT.get(pr["strategy"], 8)]
        if not ev_tickets:
            continue
        ev_tickets = add_kelly_stakes(ev_tickets, STARTING_BANKROLL, KELLY_FRACTION)
        staked = [t for t in ev_tickets if t["stake"] > 0]
        if not staked:
            continue
        stake = sum(t["stake"] for t in staked)
        matched = next((t for t in staked if t["ticket"] == pr["result_ticket"]), None)
        ret = int(matched["stake"] / 100 * pr["payout"]) if matched else 0
        out.append({"date": pr["date"], "stake": stake, "return": ret, "hit": matched is not None,
                    "payout": pr["payout"] if matched else 0, "ticket": pr["result_ticket"] if matched else None})
    return out


def roi_of(sims):
    stake = sum(s["stake"] for s in sims)
    ret = sum(s["return"] for s in sims)
    return (ret / stake * 100 if stake else 0), stake, ret, len(sims), sum(1 for s in sims if s["hit"])


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

    per_race = []
    for i, row in df.iterrows():
        probs = win_probs_all[i]
        strat, top1, top2, top3 = determine_strategy(probs)
        if not strat:
            continue
        entry = cache[(row["date"], row["course"], str(row["rno"]))]
        per_race.append({
            "date": row["date"], "strategy": strat, "top1": top1, "top2": top2, "top3": top3,
            "probs": probs, "input_df": feats.iloc[[i]],
            "odds": entry["odds"], "result_ticket": entry["result_ticket"], "payout": entry["payout"],
        })

    narrow = simulate("narrow", per_race)
    full = simulate("full120_nolead1", per_race)

    # 全体日付の中央値で前半/後半に分割
    dates = sorted(set(pr["date"] for pr in per_race))
    mid = dates[len(dates) // 2]
    print(f"ホールドアウト: {dates[0]}〜{dates[-1]} / 分割点: {mid}\n")

    print("=" * 70)
    print("【全体】")
    for name, sims in [("narrow(現行)", narrow), ("full120_nolead1", full)]:
        roi, stake, ret, n, hits = roi_of(sims)
        print(f"  {name:<18}: 賭数={n} 的中={hits} 投資={stake:,} 払戻={ret:,} 収支={ret-stake:+,} 回収率={roi:.1f}%")

    print("\n【(1) 高配当依存チェック: full120_nolead1 で的中を払戻大きい順に除外】")
    full_hits = sorted([s for s in full if s["hit"]], key=lambda s: -s["return"])
    total_stake = sum(s["stake"] for s in full)
    total_ret = sum(s["return"] for s in full)
    print(f"  除外0件(全体): 回収率={total_ret/total_stake*100:.1f}% (収支{total_ret-total_stake:+,})")
    for n_ex in [1, 3, 5, 10]:
        ret_ex = total_ret - sum(s["return"] for s in full_hits[:n_ex])
        roi_ex = ret_ex / total_stake * 100 if total_stake else 0
        print(f"  上位{n_ex:>2}件除外: 回収率={roi_ex:.1f}% (収支{ret_ex-total_stake:+,}) ← 現行(除外なし)は335.8%")

    print("\n【(2) 期間分割チェック: 前半/後半で両方改善するか】")
    for half, cond in [("前半", lambda d: d < mid), ("後半", lambda d: d >= mid)]:
        nr = roi_of([s for s in narrow if cond(s["date"])])
        fu = roi_of([s for s in full if cond(s["date"])])
        print(f"  {half}: narrow 回収率={nr[0]:.1f}%(賭{nr[3]} 的中{nr[4]}) → "
              f"full120_nolead1 回収率={fu[0]:.1f}%(賭{fu[3]} 的中{fu[4]})")

    print("\n【(3) full120_nolead1 の的中内訳(払戻順)】")
    for s in full_hits:
        print(f"  {s['date']} {s['ticket']} 払戻={s['payout']:,} stake={s['stake']} return={s['return']:,}")


if __name__ == "__main__":
    main_run()
