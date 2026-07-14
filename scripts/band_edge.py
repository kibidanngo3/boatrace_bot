"""オッズ帯を固定した上で、モデルに選別能力があるかを検証する。

EV = 確率 × オッズ で選ぶと、オッズが大きい舟券ほどEVが高く出やすく、
選抜が構造的に大穴(=実効控除率が最大67%の墓場)へ吸い寄せられる。
そこでオッズ帯を固定し、「同じ帯の中で」モデルが当たる舟券を選べているかを見る。

帯を固定すれば、比較対象は「その帯を無作為に買った場合の回収率」になる。
モデルに選別能力があるなら、モデル上位の回収率がそれを上回るはず。

使い方:
    python scripts/band_edge.py --model final_model_v8.pkl --order-suffix v8 --v8
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

BANDS = [(1, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 500)]


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
    print(f"\n対象: {len(rows):,}レース / {len(p):,}点\n")

    print("各オッズ帯の中で、モデルの確率が高い順に上位X%だけ買ったらどうなるか")
    print("「無作為」列を上回れば、その帯でモデルに選別能力がある\n")
    header = f"{'オッズ帯':>12} {'点数':>8} {'無作為':>8} {'上位50%':>9} {'上位20%':>9} {'上位10%':>9} {'上位5%':>9}"
    print(header)
    print("-" * len(header))

    for lo, hi in BANDS:
        m = (o >= lo) & (o < hi)
        n = int(m.sum())
        if n < 500:
            continue
        pb, ob, hb = p[m], o[m], hit[m]
        base_roi = ob[hb].sum() / n * 100

        cells = []
        order = np.argsort(-pb)  # モデル確率の高い順
        for frac in (0.5, 0.2, 0.1, 0.05):
            k = max(1, int(n * frac))
            sel = order[:k]
            roi = ob[sel][hb[sel]].sum() / k * 100
            cells.append(f"{roi:8.1f}%")

        label = f"{lo}-{hi}倍"
        print(f"{label:>12} {n:8,} {base_roi:7.1f}% " + " ".join(cells))

    print("\n※ 100%を超える帯があれば、そこが勝負できる土俵になる。")


if __name__ == "__main__":
    main()
