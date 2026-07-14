"""EV順に選ぶ行為そのものが生む「勝者の呪い」を定量化する。

モデルの舟券確率はホールドアウト全体では正しく較正されている
(予測0.08% → 実際0.08%)。ところが EV>=1.15 で選んだ舟券の回収率は66%しかない。
確率が正しいなら115%返るはずで、これは矛盾に見える。

種明かしは選択バイアス。EV = 確率 × オッズ で上位を取ると、
「たまたま確率を高く見積もってしまった舟券」ばかりが選ばれる(勝者の呪い)。
全体として較正されていても、選ばれた集合の中では過大評価になっている。

ここではEV帯ごとに「モデルの予測確率の平均」と「実際の的中率」を直接比べる。

使い方:
    python scripts/selection_bias.py --model final_model_v8.pkl --order-suffix v8 --v8
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

BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_model_v8.pkl")
    parser.add_argument("--order-suffix", default="v8")
    parser.add_argument("--drop-st", action="store_true")
    parser.add_argument("--v8", action="store_true")
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

    p, o, hit = [], [], []
    for i, (row, probs) in enumerate(zip(rows, all_probs)):
        c = cache[(row["date"], row["course"], str(int(row["rno"])))]
        odds_map = json.loads(c["odds_json"])
        won = c["result_ticket"]
        tp = race_ticket_probs(probs, X.iloc[[i]].reset_index(drop=True))
        for t in ALL_120:
            od = odds_map.get(t)
            if not od:
                continue
            p.append(tp[t])
            o.append(float(od))
            hit.append(t == won)

    p = np.array(p)
    o = np.array(o)
    hit = np.array(hit)
    ev = p * o
    print(f"\n対象: {len(rows):,}レース / {len(p):,}点\n")

    print("=== 全体の較正(選ばない場合) ===")
    print(f"  モデルの予測確率の平均 {p.mean()*100:.3f}%  /  実際の的中率 {hit.mean()*100:.3f}%")
    print(f"  → 比 {hit.mean()/p.mean():.2f} (1.0なら正しい)\n")

    print("=== EVで選抜したあとの較正 ===")
    print(f"{'EV閾値':>7} {'点数':>9} {'予測確率':>9} {'実際':>9} {'比':>6} "
          f"{'期待回収率':>10} {'実際の回収率':>12}")
    for thr in [1.0, 1.15, 1.3, 1.5, 2.0, 3.0]:
        m = ev >= thr
        if m.sum() < 100:
            continue
        pred = p[m].mean()
        actual = hit[m].mean()
        expected_roi = ev[m].mean() * 100
        actual_roi = o[m & hit].sum() / m.sum() * 100
        print(f"{thr:7.2f} {m.sum():9,} {pred*100:8.3f}% {actual*100:8.3f}% "
              f"{actual/pred:5.2f} {expected_roi:9.1f}% {actual_roi:11.1f}%")

    print("\n※ 比が1.0を大きく下回るほど、選抜によって確率が水増しされている(勝者の呪い)。")
    print("   全体では較正されているのに選抜後に崩れるなら、原因はモデルではなく選抜方法。")

    print("\n=== オッズ帯ごとに見た「選抜後の水増し」 ===")
    print(f"{'オッズ帯':>12} {'EV>=1.15の点数':>14} {'予測確率':>9} {'実際':>9} {'比':>6}")
    for lo, hi in [(1, 10), (10, 20), (20, 50), (50, 100), (100, 300), (300, 10**9)]:
        m = (ev >= 1.15) & (o >= lo) & (o < hi)
        if m.sum() < 50:
            continue
        pred = p[m].mean()
        actual = hit[m].mean()
        label = f"{lo}-{hi if hi < 10**9 else '∞'}倍"
        print(f"{label:>12} {m.sum():14,} {pred*100:8.3f}% {actual*100:8.3f}% {actual/pred:5.2f}")


if __name__ == "__main__":
    main()
