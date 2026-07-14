"""out-of-fold 予測を使って縮小モデルを学習し、「市場を知った上でモデルは追加情報を持つか」を
正しくやり直す。

前回(shrinkage.py)は学習期間に対する v8 の in-sample 予測を入力にしていたため、
縮小モデルが「モデルを強く信じろ」と誤学習し、ホールドアウトで破綻した
(LogLoss が市場のみより10.8%悪化)。その結論は無効。

ここでは oof_predictions.py が作った out-of-sample の舟券確率を使う。
本番と同じ条件の確率なので、これで初めて公平な比較になる。

比較する3つ:
    市場のみ      : オッズだけから P(的中) を学習(競艇の知識ゼロ)
    モデルのみ    : モデルの確率だけ
    市場 + モデル : 両方

「市場のみ」を上回れなければ、モデルは市場に対して無価値。

使い方:
    python scripts/shrinkage_oof.py --v8
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_model import (  # noqa: E402
    build_features, merge_exhibition, FEATURES, FEATURES_NO_ST, FEATURES_V8,
)
from scripts.calibrate import ticket_probs_batch, TICKETS  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
EPS = 1e-9

PARAMS = {
    "objective": "binary", "metric": "binary_logloss", "verbosity": -1,
    "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 500,
    "random_state": 42,
}

SETS = {
    "市場のみ": ["logit_market", "log_odds"],
    "モデルのみ": ["logit_model"],
    "市場 + モデル": ["logit_model", "logit_market", "log_odds", "diff"],
}


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def make_frame(ticket_probs, labels, keys, cache):
    """(モデル確率, 市場確率, 的中) を舟券単位に展開する。オッズが無いレースは捨てる。"""
    model_p, odds, y = [], [], []
    for i, key in enumerate(keys):
        date, course, rno = key.split("_")
        c = cache.get((date, course, rno))
        if not c:
            continue
        odds_map = json.loads(c["odds_json"])
        for ti, t in enumerate(TICKETS):
            o = odds_map.get(f"{t[0]}-{t[1]}-{t[2]}")
            if not o:
                continue
            model_p.append(ticket_probs[i, ti])
            odds.append(float(o))
            y.append(labels[i, ti])

    model_p = np.asarray(model_p, dtype=np.float64)
    odds = np.asarray(odds, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    market_p = 1.0 / odds

    X = pd.DataFrame({
        "logit_model": logit(model_p),
        "logit_market": logit(market_p),
        "log_odds": np.log(odds),
        "diff": logit(model_p) - logit(market_p),
    })
    return X, y, odds, model_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8", action="store_true")
    parser.add_argument("--drop-st", action="store_true")
    parser.add_argument("--model", default="final_model_v8.pkl")
    parser.add_argument("--order-suffix", default="v8")
    parser.add_argument("--oof", default="oof_ticket_probs.npz")
    args = parser.parse_args()

    features = FEATURES_V8 if args.v8 else (FEATURES_NO_ST if args.drop_st else FEATURES)

    def load_cache(path):
        with open(BASE_DIR / path, encoding="utf-8-sig") as f:
            return {(r["date"], r["course"], str(int(r["rno"]))): r for r in csv.DictReader(f)}

    train_cache = load_cache("odds_train_cache.csv")
    hold_cache = load_cache("odds_result_cache.csv")

    # --- 学習側: out-of-fold の確率 ---
    z = np.load(BASE_DIR / args.oof, allow_pickle=True)
    Xtr, ytr, _, p_oof = make_frame(z["probs"], z["labels"], list(z["keys"]), train_cache)
    print(f"学習(out-of-fold): {len(ytr):,}点 / 的中 {ytr.sum():,}")

    m = p_oof < 0.002
    print(f"  健全性チェック: 予測<0.2%の舟券で 予測 {p_oof[m].mean()*100:.3f}% / "
          f"実際 {ytr[m].mean()*100:.3f}%  ← 近ければ out-of-sample\n")

    # --- 検証側: ホールドアウト(本番モデルの予測。これは元から out-of-sample) ---
    model = pickle.load(open(BASE_DIR / args.model, "rb"))
    m2 = pickle.load(open(BASE_DIR / f"order_model_2nd_{args.order_suffix}.pkl", "rb"))
    m3 = pickle.load(open(BASE_DIR / f"order_model_3rd_{args.order_suffix}.pkl", "rb"))
    cfg = pickle.load(open(BASE_DIR / f"order_model_config_{args.order_suffix}.pkl", "rb"))

    df = pd.read_csv(BASE_DIR / "training_data.csv", dtype=str, encoding="utf-8-sig")
    df = df.dropna(subset=["label", "label_2nd", "label_3rd"])
    for c in ("label", "label_2nd", "label_3rd"):
        df[c] = df[c].astype(int)
    df = df[df["label"].between(1, 6) & df["label_2nd"].between(1, 6) & df["label_3rd"].between(1, 6)]
    if args.v8:
        df = merge_exhibition(df, BASE_DIR / "exhibition_data.csv")
    ho = df[df["date"] >= "20260512"].reset_index(drop=True)

    Xho_feat = build_features(ho.copy(), features)
    tp_ho = ticket_probs_batch(model.predict(Xho_feat), Xho_feat, m2, m3, cfg)
    idx = {t: i for i, t in enumerate(TICKETS)}
    y_ho = np.zeros((len(ho), 120), dtype=np.int8)
    for i, (a, b, c) in enumerate(zip(ho["label"], ho["label_2nd"], ho["label_3rd"])):
        y_ho[i, idx[(a, b, c)]] = 1
    keys_ho = [f"{d}_{c}_{int(r)}" for d, c, r in zip(ho["date"], ho["course"], ho["rno"])]

    Xho, yho, odds_ho, _ = make_frame(tp_ho, y_ho, keys_ho, hold_cache)
    print(f"検証(ホールドアウト): {len(yho):,}点 / 的中 {yho.sum():,}\n")

    print("=== 市場を知った上で、モデルは追加情報を持つか(OOFでやり直し) ===")
    print(f"{'使う情報':>14} {'LogLoss':>10} {'EV>=1.15の点数':>14} {'回収率':>9} {'的中':>6}")
    results = {}
    for name, feats in SETS.items():
        b = lgb.train(PARAMS, lgb.Dataset(Xtr[feats], label=ytr), num_boost_round=300)
        p = b.predict(Xho[feats])
        ll = log_loss(yho, p, labels=[0, 1])
        ev = p * odds_ho
        sel = ev >= 1.15
        n = int(sel.sum())
        roi = odds_ho[sel & (yho == 1)].sum() / n * 100 if n else 0
        hits = int((sel & (yho == 1)).sum())
        results[name] = (ll, n, roi)
        print(f"{name:>14} {ll:10.6f} {n:14,} {roi:8.1f}% {hits:6,}")

    base = results["市場のみ"][0]
    full = results["市場 + モデル"][0]
    print(f"\n市場のみ → 市場+モデル: LogLoss {(base - full) / base * 100:+.2f}% (プラスなら改善)")
    print("→ 改善していれば、モデルは市場に対して確かに追加情報を持っている")


if __name__ == "__main__":
    main()
