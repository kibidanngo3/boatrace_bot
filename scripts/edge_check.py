"""漏洩なしモデルに、そもそも市場(オッズ)に対する優位性があるのかを確認する。

EV閾値を振って黒字(回収率>100%)になる領域が存在するかを見る。
控除率が約25%なので、無作為に賭ければ回収率は約75%に収束する。
モデルに優位性がないなら、どの閾値でも75%前後かそれ以下にとどまるはず。

使い方:
    python scripts/edge_check.py --model final_model_v7.pkl --order-suffix v7 --drop-st
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
from scripts.train_model import build_features, FEATURES, FEATURES_NO_ST  # noqa: E402
from scripts.nige_vs_jump import ALL_120, race_ticket_probs, ORDER  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_model_v7.pkl")
    parser.add_argument("--order-suffix", default="v7")
    parser.add_argument("--drop-st", action="store_true")
    args = parser.parse_args()

    features = FEATURES_NO_ST if args.drop_st else FEATURES
    model = pickle.load(open(BASE_DIR / args.model, "rb"))
    ORDER["2nd"] = pickle.load(open(BASE_DIR / f"order_model_2nd_{args.order_suffix}.pkl", "rb"))
    ORDER["3rd"] = pickle.load(open(BASE_DIR / f"order_model_3rd_{args.order_suffix}.pkl", "rb"))
    ORDER["cfg"] = pickle.load(open(BASE_DIR / f"order_model_config_{args.order_suffix}.pkl", "rb"))

    with open(BASE_DIR / "odds_result_cache.csv", encoding="utf-8-sig") as f:
        cache = {(r["date"], r["course"], str(int(r["rno"]))): r for r in csv.DictReader(f)}
    with open(BASE_DIR / "training_data.csv", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)
                if (r["date"], r["course"], str(int(r["rno"]))) in cache]

    X = build_features(pd.DataFrame(rows), features)
    all_probs = model.predict(X)
    print(f"対象: {len(rows)}レース ({min(r['date'] for r in rows)}〜{max(r['date'] for r in rows)})\n")

    # 全レース・全120通りについて (モデル確率, オッズ, 当たったか) を集める
    recs = []
    for i, (row, probs) in enumerate(zip(rows, all_probs)):
        c = cache[(row["date"], row["course"], str(int(row["rno"])))]
        odds_map = json.loads(c["odds_json"])
        won = c["result_ticket"]
        tp = race_ticket_probs(probs, X.iloc[[i]].reset_index(drop=True))
        for t in ALL_120:
            o = odds_map.get(t)
            if not o:
                continue
            recs.append((tp[t], float(o), t == won))

    p = np.array([r[0] for r in recs])
    o = np.array([r[1] for r in recs])
    hit = np.array([r[2] for r in recs])
    ev = p * o
    print(f"評価した舟券: {len(recs):,}点\n")

    print("=== EV閾値ごとの回収率(100円ずつ均等賭け) ===")
    print("控除率25%のため、優位性ゼロなら約75%に収束する\n")
    print(f"{'EV閾値':>7} {'買う点数':>9} {'的中':>6} {'的中率':>7} {'回収率':>8} {'平均オッズ':>9}")
    for thr in [1.0, 1.1, 1.15, 1.2, 1.5, 2.0, 3.0, 5.0]:
        m = ev >= thr
        if m.sum() == 0:
            continue
        roi = (o[m & hit].sum() / m.sum()) * 100
        print(f"{thr:7.2f} {m.sum():9,} {hit[m].sum():6d} "
              f"{hit[m].mean()*100:6.2f}% {roi:7.1f}% {o[m].mean():8.1f}")

    print("\n=== モデル確率の較正(舟券レベル) ===")
    print("モデルが「この舟券は◯%で当たる」と言った時、実際に何%当たったか\n")
    print(f"{'予測確率帯':>14} {'点数':>9} {'予測平均':>9} {'実際':>8}")
    for lo, hi in [(0, .002), (.002, .005), (.005, .01), (.01, .02), (.02, .05), (.05, 1.0)]:
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        print(f"{lo*100:5.1f}%-{hi*100:5.1f}% {m.sum():9,} "
              f"{p[m].mean()*100:8.2f}% {hit[m].mean()*100:7.2f}%")

    print("\n=== 市場(オッズ)との比較 ===")
    # オッズから市場の暗黙確率を出す(控除率25%で正規化前の生の値)
    implied = 1 / o
    better = p > implied
    print(f"モデルが市場より高い確率を付けた舟券: {better.sum():,}点 "
          f"(実際の的中率 {hit[better].mean()*100:.2f}% / 市場の暗黙確率平均 {implied[better].mean()*100:.2f}%)")
    print(f"  → この集合に均等賭けした回収率: {(o[better & hit].sum() / better.sum())*100:.1f}%")


if __name__ == "__main__":
    main()
