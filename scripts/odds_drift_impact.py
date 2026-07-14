"""「締切前の暫定オッズで選び、最終オッズで払い戻される」という本番の条件を再現する。

バックテストは最終オッズ(レース確定後のページ)で買い目を選んでいるが、
本番は締切5〜35分前の暫定オッズしか見られない。実測すると、両者のズレは
中央値32%、64%の舟券が25%以上動く(predictions.csv の賭け時オッズと最終オッズを比較)。

払戻はパリミュチュエルなので最終オッズ基準で正しい。問題は「選抜」の側で、
バックテストは賭けた時点では知り得ないオッズで買い目を選んでいる。

ここでは最終オッズに実測どおりのノイズを乗せて「暫定オッズ」を作り、
  - 選抜は暫定オッズで行う(EV計算・点数選び・ケリー)
  - 払戻は最終オッズで行う
として、回収率がどれだけ落ちるかを測る。

使い方:
    python scripts/odds_drift_impact.py --model final_model_v8.pkl --order-suffix v8 --v8
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

# predictions.csv の賭け時オッズ vs 最終オッズ の実測(96点)から、ズレ幅の中央値は32%。
# ただしノイズの入れ方には注意が要る。単純に seen = final / exp(N(0, sigma)) とすると
# E[final/seen] = exp(sigma^2/2) = 1.11 となり、「選んだ舟券が平均11%高い配当で返ってくる」
# という存在しないボーナスが発生し、ノイズを入れたほうが成績が良く見えてしまう。
# ここでは E[final|seen] = seen となるよう平均を -sigma^2/2 だけずらし、払戻中立にする。
# こうすると「暫定オッズで選ぶと選抜が乱れる」効果だけを分離して測れる。
DRIFT_SIGMA = 0.45

BANDS = {
    "1-10倍": (1, 10),
    "1-20倍": (1, 20),
    "1-50倍": (1, 50),
    "制限なし(現行)": (1, 10 ** 9),
}


def simulate(records, band, ev_min, cap, rng, use_drift):
    lo, hi = band
    bets = []
    for rec in records:
        cands = []
        for t, final_odds, p in rec["tickets"]:
            if use_drift:
                # 賭けた時点に見えていたであろうオッズ。
                # log(final) = log(seen) + eta, eta ~ N(-sigma^2/2, sigma^2) とすることで
                # E[final|seen] = seen (払戻中立)になる。
                eta = rng.normal(-DRIFT_SIGMA ** 2 / 2, DRIFT_SIGMA)
                seen = final_odds / np.exp(eta)
            else:
                seen = final_odds
            if not (lo <= seen < hi):
                continue
            if p * seen < ev_min:
                continue
            cands.append({
                "ticket": t, "odds": seen, "final_odds": final_odds,
                "probability": p, "expected_value": p * seen,
            })
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
        # 払戻は最終オッズ基準(パリミュチュエル)
        ret = int(won["stake"] / 100 * rec["payout"]) if won else 0
        bets.append({"stake": stake, "return": ret, "hit": won is not None})
    return bets


def roi_of(bets):
    if not bets:
        return None
    s = sum(b["stake"] for b in bets)
    r = sum(b["return"] for b in bets)
    return (len(bets), sum(1 for b in bets if b["hit"]), r / s * 100 if s else 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_model_v8.pkl")
    parser.add_argument("--order-suffix", default="v8")
    parser.add_argument("--drop-st", action="store_true")
    parser.add_argument("--v8", action="store_true")
    parser.add_argument("--ev-min", type=float, default=1.15)
    parser.add_argument("--cap", type=int, default=8)
    parser.add_argument("--trials", type=int, default=5, help="ノイズを変えて何回試すか")
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

    records = []
    for i, (row, probs) in enumerate(zip(rows, all_probs)):
        c = cache[(row["date"], row["course"], str(int(row["rno"])))]
        odds_map = json.loads(c["odds_json"])
        tp = race_ticket_probs(probs, X.iloc[[i]].reset_index(drop=True))
        records.append({
            "won": c["result_ticket"],
            "payout": int(c["payout"]),
            "tickets": [(t, float(odds_map[t]), tp[t]) for t in ALL_120 if t in odds_map],
        })

    print(f"\n対象: {len(records):,}レース / EV>={args.ev_min} / 最大{args.cap}点")
    print(f"暫定オッズのノイズ: log正規 sigma={DRIFT_SIGMA} (実測: ズレ幅の中央値32%)\n")

    print(f"{'オッズ帯':>16} {'最終オッズで選抜':>18} {'暫定オッズで選抜(本番相当)':>30}")
    print("-" * 70)
    for label, band in BANDS.items():
        clean = roi_of(simulate(records, band, args.ev_min, args.cap,
                                np.random.default_rng(0), use_drift=False))
        dirty = []
        for seed in range(args.trials):
            r = roi_of(simulate(records, band, args.ev_min, args.cap,
                                np.random.default_rng(seed), use_drift=True))
            if r:
                dirty.append(r)

        if not clean:
            continue
        c_txt = f"{clean[2]:6.1f}% ({clean[0]:,}賭)"
        if dirty:
            rois = [d[2] for d in dirty]
            n_avg = int(np.mean([d[0] for d in dirty]))
            d_txt = f"{np.mean(rois):6.1f}% ± {np.std(rois):4.1f} ({n_avg:,}賭)"
        else:
            d_txt = "賭け成立なし"
        print(f"{label:>16} {c_txt:>18} {d_txt:>30}")

    print("\n※ 右列が本番に近い条件。左列(従来のバックテスト)がどれだけ楽観的だったかが分かる。")


if __name__ == "__main__":
    main()
