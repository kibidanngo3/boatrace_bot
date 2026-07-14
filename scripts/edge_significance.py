"""高EV帯で回収率100%超が出たとき、それが実力か偶然かを検定する。

平均オッズ800倍の舟券は、1本の的中で回収率が大きく動く。的中55本程度では
「たまたま当たった」と「優位性がある」が見分けられないので、以下で確かめる。

  - ブートストラップ信頼区間 : 下限が100%を割るなら、黒字と言い切れない
  - 期間分割               : 前半・後半の両方で黒字か
  - 上位配当を除外          : 数本の大穴に支えられていないか
  - 帰無仮説の下でのp値     : 「優位性ゼロ(期待回収率75%)」でこの結果が出る確率

使い方:
    python scripts/edge_significance.py --model final_model_v8.pkl --order-suffix v8 --v8
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
TAKEOUT_ROI = 75.0  # 控除率25% = 優位性ゼロなら期待回収率75%


def bootstrap_roi(payouts, n_tickets, n_boot=20000, seed=42):
    """舟券を復元抽出し直して回収率の分布を作る(1点100円・均等賭け)。"""
    rng = np.random.default_rng(seed)
    # 各舟券の払戻(外れは0)を並べた配列から復元抽出する
    returns = np.zeros(n_tickets)
    returns[: len(payouts)] = payouts
    idx = rng.integers(0, n_tickets, size=(n_boot, n_tickets))
    sampled = returns[idx].sum(axis=1) / n_tickets * 100
    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_model_v8.pkl")
    parser.add_argument("--order-suffix", default="v8")
    parser.add_argument("--drop-st", action="store_true")
    parser.add_argument("--v8", action="store_true")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[1.5, 2.0, 3.0])
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

    recs = []  # (ev, odds, hit, date)
    for i, (row, probs) in enumerate(zip(rows, all_probs)):
        c = cache[(row["date"], row["course"], str(int(row["rno"])))]
        odds_map = json.loads(c["odds_json"])
        won = c["result_ticket"]
        tp = race_ticket_probs(probs, X.iloc[[i]].reset_index(drop=True))
        for t in ALL_120:
            o = odds_map.get(t)
            if not o:
                continue
            o = float(o)
            recs.append((tp[t] * o, o, t == won, row["date"]))

    ev = np.array([r[0] for r in recs])
    odds = np.array([r[1] for r in recs])
    hit = np.array([r[2] for r in recs])
    dates = np.array([r[3] for r in recs])
    print(f"\n評価した舟券: {len(recs):,}点\n")

    for thr in args.thresholds:
        m = ev >= thr
        n = int(m.sum())
        if n == 0:
            continue
        payouts = odds[m & hit]
        roi = payouts.sum() / n * 100

        print(f"===== EV >= {thr} =====")
        print(f"  買う点数 {n:,} / 的中 {len(payouts)} / 回収率 {roi:.1f}%")

        boot = bootstrap_roi(payouts, n)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  ブートストラップ95%信頼区間: {lo:.1f}% 〜 {hi:.1f}%")
        p_break_even = (boot < 100).mean()
        print(f"  この賭けが赤字(<100%)である確率: {p_break_even:.1%}")

        # 帰無仮説(優位性ゼロ=期待回収率75%)の下で、この回収率以上が出る確率
        # 的中を独立なポアソン試行とみなし、払戻の分布は実際の高EV券のオッズ分布から取る
        rng = np.random.default_rng(7)
        null_hit_rate = TAKEOUT_ROI / 100 / odds[m].mean()  # 期待回収率75%になる的中率
        null_rois = []
        for _ in range(5000):
            k = rng.binomial(n, null_hit_rate)
            if k == 0:
                null_rois.append(0.0)
                continue
            sampled_odds = rng.choice(odds[m], size=k, replace=True)
            null_rois.append(sampled_odds.sum() / n * 100)
        p_value = (np.array(null_rois) >= roi).mean()
        print(f"  優位性ゼロでもこの回収率が出る確率(p値): {p_value:.3f}")

        # 上位配当を除いたら
        top = np.sort(payouts)[::-1]
        for k in (1, 3):
            if len(top) > k:
                print(f"  高配当上位{k}本を除くと: {(payouts.sum() - top[:k].sum()) / n * 100:.1f}%")

        # 期間分割
        mid = np.median(dates[m].astype(int))
        first = m & (dates.astype(int) <= mid)
        second = m & (dates.astype(int) > mid)
        for label, mm in (("前半", first), ("後半", second)):
            nn = int(mm.sum())
            if nn:
                rr = odds[mm & hit].sum() / nn * 100
                print(f"  {label}: {rr:.1f}% ({nn:,}点 / 的中{int((mm & hit).sum())})")
        print()


if __name__ == "__main__":
    main()
