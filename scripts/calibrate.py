"""舟券レベルの確率を較正する(Isotonic Regression)。

edge_check で分かったとおり、モデルの舟券確率は過信気味だった
(「7.62%当たる」と言った舟券が実際は6.07%しか当たらない)。
EV = 確率 × オッズ で買い目を選ぶ以上、確率が歪んでいれば選抜そのものが歪む。

較正関数は **学習期間のデータだけ** から作る。ホールドアウトで作れば、
それは「答えを見て調整した」ことになり、今日2度踏んだ罠と同じになる。

使い方:
    python scripts/calibrate.py --model final_model_v8.pkl --order-suffix v8 --v8
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_model import (  # noqa: E402
    build_features, merge_exhibition, FEATURES, FEATURES_NO_ST, FEATURES_V8,
)
from scripts.nige_vs_jump import ALL_120  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent

TICKETS = [tuple(int(x) for x in t.split("-")) for t in ALL_120]
PAIRS = [(a, b) for a in range(1, 7) for b in range(1, 7) if a != b]
PAIR_INDEX = {p: i for i, p in enumerate(PAIRS)}


def ticket_probs_batch(probs, X, model_2nd, model_3rd, cfg, chunk=2000):
    """レースをまとめて、120通りの舟券確率を一括計算する(1レースずつ回すと遅すぎる)。"""
    n = len(X)
    out = np.zeros((n, 120), dtype=np.float32)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = X.iloc[start:end].reset_index(drop=True)
        m = len(block)

        # 2着モデル: given_1st = 1..6 の6通りを全レース分まとめて推論
        x2 = pd.concat([block] * 6, ignore_index=True)
        x2["given_1st"] = np.repeat(np.arange(1, 7), m)
        raw2 = np.asarray(model_2nd.predict(x2[cfg["features_2nd"]]), dtype=np.float32)
        raw2 = raw2.reshape(6, m, 6).transpose(1, 0, 2)  # (レース, given_1st, 艇)

        # 3着モデル: (given_1st, given_2nd) の30通り
        x3 = pd.concat([block] * len(PAIRS), ignore_index=True)
        x3["given_1st"] = np.repeat([a for a, _ in PAIRS], m)
        x3["given_2nd"] = np.repeat([b for _, b in PAIRS], m)
        raw3 = np.asarray(model_3rd.predict(x3[cfg["features_3rd"]]), dtype=np.float32)
        raw3 = raw3.reshape(len(PAIRS), m, 6).transpose(1, 0, 2)  # (レース, ペア, 艇)

        p1 = probs[start:end]
        for ti, (a, b, c) in enumerate(TICKETS):
            r2 = raw2[:, a - 1, :].copy()
            r2[:, a - 1] = 0
            s2 = r2.sum(axis=1)
            p2 = np.divide(r2[:, b - 1], s2, out=np.zeros(m, np.float32), where=s2 > 0)

            r3 = raw3[:, PAIR_INDEX[(a, b)], :].copy()
            r3[:, a - 1] = 0
            r3[:, b - 1] = 0
            s3 = r3.sum(axis=1)
            p3 = np.divide(r3[:, c - 1], s3, out=np.zeros(m, np.float32), where=s3 > 0)

            out[start:end, ti] = p1[:, a - 1] * p2 * p3

        print(f"  {end}/{n} レース", flush=True)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_model_v8.pkl")
    parser.add_argument("--order-suffix", default="v8")
    parser.add_argument("--drop-st", action="store_true")
    parser.add_argument("--v8", action="store_true")
    parser.add_argument("--cutoff", default="20260512", help="この日より前だけで較正関数を作る")
    parser.add_argument("--out", default="ticket_calibrator_v8.pkl")
    args = parser.parse_args()

    features = FEATURES_V8 if args.v8 else (FEATURES_NO_ST if args.drop_st else FEATURES)
    model = pickle.load(open(BASE_DIR / args.model, "rb"))
    model_2nd = pickle.load(open(BASE_DIR / f"order_model_2nd_{args.order_suffix}.pkl", "rb"))
    model_3rd = pickle.load(open(BASE_DIR / f"order_model_3rd_{args.order_suffix}.pkl", "rb"))
    cfg = pickle.load(open(BASE_DIR / f"order_model_config_{args.order_suffix}.pkl", "rb"))

    df = pd.read_csv(BASE_DIR / "training_data.csv", dtype=str, encoding="utf-8-sig")
    df = df.dropna(subset=["label", "label_2nd", "label_3rd"])
    for col in ("label", "label_2nd", "label_3rd"):
        df[col] = df[col].astype(int)
    df = df[df["label"].between(1, 6) & df["label_2nd"].between(1, 6) & df["label_3rd"].between(1, 6)]
    if args.v8:
        df = merge_exhibition(df, BASE_DIR / "exhibition_data.csv")

    train = df[df["date"] < args.cutoff].reset_index(drop=True)
    print(f"較正に使う学習期間: {len(train):,}レース ({train['date'].min()}〜{train['date'].max()})")
    print("※ ホールドアウトは一切使わない\n")

    X = build_features(train.copy(), features)
    probs = model.predict(X)
    tp = ticket_probs_batch(probs, X, model_2nd, model_3rd, cfg)

    # 実際に当たった舟券に1を立てる
    won_index = {}
    for ti, t in enumerate(TICKETS):
        won_index[t] = ti
    y = np.zeros_like(tp, dtype=np.int8)
    for i, (a, b, c) in enumerate(zip(train["label"], train["label_2nd"], train["label_3rd"])):
        y[i, won_index[(a, b, c)]] = 1

    p_flat = tp.ravel().astype(np.float64)
    y_flat = y.ravel()
    print(f"\n較正に使う舟券: {len(p_flat):,}点 / 的中 {y_flat.sum():,}点")

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_flat, y_flat)

    print("\n=== 学習期間での較正前後 ===")
    print(f"{'予測確率帯':>14} {'点数':>10} {'較正前':>9} {'較正後':>9} {'実際':>9}")
    cal = iso.predict(p_flat)
    for lo, hi in [(0, .002), (.002, .005), (.005, .01), (.01, .02), (.02, .05), (.05, 1.0)]:
        m = (p_flat >= lo) & (p_flat < hi)
        if m.sum() == 0:
            continue
        print(f"{lo*100:5.1f}%-{hi*100:5.1f}% {m.sum():10,} "
              f"{p_flat[m].mean()*100:8.2f}% {cal[m].mean()*100:8.2f}% {y_flat[m].mean()*100:8.2f}%")

    with open(BASE_DIR / args.out, "wb") as f:
        pickle.dump(iso, f)
    print(f"\n保存しました: {args.out}")


if __name__ == "__main__":
    main()
