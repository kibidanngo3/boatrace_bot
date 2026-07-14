"""買い目候補をオッズ帯で絞ったとき、実戦形式(EV+ケリー)の収支がどうなるかを検証する。

market_bias.py で分かったこと: 実効控除率はオッズ帯で全く違う(1-20倍は約20%だが、
1000倍超は67%)。EV = 確率 × オッズ なので、EVフィルタは構造的に大穴へ吸い寄せられ、
モデルの選別能力が市場構造の不利に飲み込まれていた。

そこで候補をオッズ帯で先に絞り、その中でEV順に選ぶ。帯ごとの成績を比較する。

使い方:
    python scripts/band_strategy.py --model final_model_v8.pkl --order-suffix v8 --v8
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_model import (  # noqa: E402
    build_features, merge_exhibition, FEATURES, FEATURES_NO_ST, FEATURES_V8,
)
from scripts.nige_vs_jump import ALL_120, race_ticket_probs, ORDER  # noqa: E402
from main import add_kelly_stakes, KELLY_FRACTION, STARTING_BANKROLL  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent

# 検証する「候補をどのオッズ帯に絞るか」
BANDS = {
    "1-10倍": (1, 10),
    "1-20倍": (1, 20),
    "1-50倍": (1, 50),
    "10-50倍": (10, 50),
    "20-100倍": (20, 100),
    "制限なし(現行)": (1, 10 ** 9),
}


def simulate(band, ev_min, cap, records):
    """1レースずつ、帯内の買い目からEV順に選び、ケリーで賭ける。"""
    lo, hi = band
    bets = []
    for rec in records:
        cands = [
            {"ticket": t, "odds": o, "probability": p, "expected_value": p * o}
            for t, o, p in rec["tickets"]
            if lo <= o < hi and p * o >= ev_min
        ]
        if not cands:
            continue
        cands.sort(key=lambda x: x["expected_value"], reverse=True)
        cands = cands[:cap]

        staked = [t for t in add_kelly_stakes(cands, STARTING_BANKROLL, KELLY_FRACTION)
                  if t["stake"] > 0]
        if not staked:
            continue

        stake = sum(t["stake"] for t in staked)
        won = next((t for t in staked if t["ticket"] == rec["won"]), None)
        ret = int(won["stake"] / 100 * rec["payout"]) if won else 0
        bets.append({
            "stake": stake, "return": ret, "hit": won is not None,
            "date": rec["date"], "n": len(staked),
            "avg_odds": sum(t["odds"] for t in staked) / len(staked),
        })
    return bets


def report(label, bets, n_races):
    if not bets:
        print(f"{label:16s} 賭け成立0件")
        return
    stake = sum(b["stake"] for b in bets)
    ret = sum(b["return"] for b in bets)
    hits = sum(1 for b in bets if b["hit"])
    roi = ret / stake * 100 if stake else 0
    avg_odds = sum(b["avg_odds"] for b in bets) / len(bets)

    by_date = sorted(bets, key=lambda b: b["date"])
    mid = len(by_date) // 2
    halves = []
    for half in (by_date[:mid], by_date[mid:]):
        s = sum(b["stake"] for b in half)
        v = sum(b["return"] for b in half)
        halves.append(v / s * 100 if s else 0)

    print(f"{label:16s} 賭け{len(bets):5,}/{n_races}  平均{avg_odds:6.1f}倍  "
          f"的中{hits:4d}({hits/len(bets)*100:4.1f}%)  "
          f"収支{ret-stake:>+9,}  回収率 {roi:5.1f}%  "
          f"[前半{halves[0]:.0f}% 後半{halves[1]:.0f}%]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_model_v8.pkl")
    parser.add_argument("--order-suffix", default="v8")
    parser.add_argument("--drop-st", action="store_true")
    parser.add_argument("--v8", action="store_true")
    parser.add_argument("--ev-min", type=float, nargs="+", default=[1.0, 1.15, 1.3])
    parser.add_argument("--cap", type=int, default=8)
    parser.add_argument("--calibrator", default=None,
                        help="学習期間で作った較正関数(ticket_calibrator_v8.pkl)を適用する")
    args = parser.parse_args()

    features = FEATURES_V8 if args.v8 else (FEATURES_NO_ST if args.drop_st else FEATURES)
    model = pickle.load(open(BASE_DIR / args.model, "rb"))
    ORDER["2nd"] = pickle.load(open(BASE_DIR / f"order_model_2nd_{args.order_suffix}.pkl", "rb"))
    ORDER["3rd"] = pickle.load(open(BASE_DIR / f"order_model_3rd_{args.order_suffix}.pkl", "rb"))
    ORDER["cfg"] = pickle.load(open(BASE_DIR / f"order_model_config_{args.order_suffix}.pkl", "rb"))

    with open(BASE_DIR / "odds_result_cache.csv", encoding="utf-8-sig") as f:
        cache = {(r["date"], r["course"], str(int(r["rno"]))): r for r in csv.DictReader(f)}
    with open(BASE_DIR / "training_data.csv", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)
                if (r["date"], r["course"], str(int(r["rno"]))) in cache]

    df = pd.DataFrame(rows)
    if args.v8:
        df = merge_exhibition(df, BASE_DIR / "exhibition_data.csv")
        rows = df.to_dict("records")
    X = build_features(df, features)
    all_probs = model.predict(X)

    iso = None
    if args.calibrator:
        iso = pickle.load(open(BASE_DIR / args.calibrator, "rb"))
        print(f"較正関数を適用: {args.calibrator}")

    records = []
    for i, (row, probs) in enumerate(zip(rows, all_probs)):
        c = cache[(row["date"], row["course"], str(int(row["rno"])))]
        odds_map = json.loads(c["odds_json"])
        tp = race_ticket_probs(probs, X.iloc[[i]].reset_index(drop=True))
        if iso is not None:
            keys = list(tp.keys())
            vals = iso.predict(np.array([tp[k] for k in keys]))
            tp = dict(zip(keys, vals))
        records.append({
            "date": row["date"],
            "won": c["result_ticket"],
            "payout": int(c["payout"]),
            "tickets": [(t, float(odds_map[t]), tp[t]) for t in ALL_120 if t in odds_map],
        })

    n = len(records)
    print(f"\n対象: {n:,}レース (ホールドアウト全体)")
    print(f"1/4ケリー・バンクロール{STARTING_BANKROLL:,}円固定・最大{args.cap}点\n")

    for ev_min in args.ev_min:
        print(f"--- EV >= {ev_min} ---")
        for label, band in BANDS.items():
            report(label, simulate(band, ev_min, args.cap, records), n)
        print()


if __name__ == "__main__":
    main()
