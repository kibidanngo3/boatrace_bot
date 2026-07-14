"""学習期間について out-of-fold の舟券確率を作る。

なぜ必要か:
縮小モデル(shrinkage.py)は「モデルの確率」と「市場の確率」から本当の的中確率を学習する。
ところが最初の実装では、学習期間のレースに対する **v8 の in-sample 予測** を入力にしていた。
v8 はその学習期間で学習されたモデルなので、学習期間での予測は異常に鋭い:

    学習期間     : 予測 0.08% → 実際 0.02%  (勝者を記憶している)
    ホールドアウト: 予測 0.08% → 実際 0.08%  (正常)

この汚染された入力で縮小モデルを作ると「モデルを強く信じろ」と学習してしまい、
ホールドアウトで過信して破綻する。実際 LogLoss が市場のみより10.8%悪化した。
「モデルは市場に対して無価値」という結論は、この汚染された実験の産物であり、まだ証明されていない。

ここでは時系列で fold を切り、各 fold を **それ以前のデータだけで学習したモデル** で予測する。
本番と同じ out-of-sample 条件の確率が得られる。

使い方:
    python scripts/oof_predictions.py --v8
出力: oof_ticket_probs.npz (race_key, 120通りの確率, 的中ラベル)
"""
import argparse
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_model import (  # noqa: E402
    build_features, merge_exhibition, FEATURES, FEATURES_NO_ST, FEATURES_V8,
)
from scripts.calibrate import ticket_probs_batch, TICKETS  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent

PARAMS = {
    "objective": "multiclass", "num_class": 6, "metric": "multi_logloss",
    "verbosity": -1, "learning_rate": 0.03, "num_leaves": 31, "random_state": 42,
}

# 学習期間(20250711〜20260511)を、最初の4ヶ月をウォームアップにして3つのfoldに切る
FOLDS = [
    ("20251111", "20260111"),
    ("20260111", "20260311"),
    ("20260311", "20260512"),
]


def train_fold_models(fit_df, features, valid_days=20):
    """fold より前のデータだけで、1着モデルと2着・3着モデルを学習する。"""
    dates = sorted(fit_df["date"].unique())
    cut = dates[-valid_days]
    tr = fit_df[fit_df["date"] < cut].reset_index(drop=True)
    va = fit_df[fit_df["date"] >= cut].reset_index(drop=True)

    Xtr = build_features(tr.copy(), features)
    Xva = build_features(va.copy(), features)

    def fit(y_tr, y_va, extra_tr=None, extra_va=None, feats=None):
        A, B = Xtr.copy(), Xva.copy()
        if extra_tr:
            for k, v in extra_tr.items():
                A[k] = v
            for k, v in extra_va.items():
                B[k] = v
        f = feats or features
        return lgb.train(
            PARAMS,
            lgb.Dataset(A[f], label=y_tr),
            num_boost_round=2000,
            valid_sets=[lgb.Dataset(B[f], label=y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

    model = fit(tr["label"].values - 1, va["label"].values - 1)

    f2 = features + ["given_1st"]
    m2 = fit(tr["label_2nd"].values - 1, va["label_2nd"].values - 1,
             {"given_1st": tr["label"].values}, {"given_1st": va["label"].values}, f2)

    f3 = features + ["given_1st", "given_2nd"]
    m3 = fit(tr["label_3rd"].values - 1, va["label_3rd"].values - 1,
             {"given_1st": tr["label"].values, "given_2nd": tr["label_2nd"].values},
             {"given_1st": va["label"].values, "given_2nd": va["label_2nd"].values}, f3)

    return model, m2, m3, {"features_2nd": f2, "features_3rd": f3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8", action="store_true")
    parser.add_argument("--drop-st", action="store_true")
    parser.add_argument("--out", default="oof_ticket_probs.npz")
    args = parser.parse_args()

    features = FEATURES_V8 if args.v8 else (FEATURES_NO_ST if args.drop_st else FEATURES)

    df = pd.read_csv(BASE_DIR / "training_data.csv", dtype=str, encoding="utf-8-sig")
    df = df.dropna(subset=["label", "label_2nd", "label_3rd"])
    for c in ("label", "label_2nd", "label_3rd"):
        df[c] = df[c].astype(int)
    df = df[df["label"].between(1, 6) & df["label_2nd"].between(1, 6) & df["label_3rd"].between(1, 6)]
    if args.v8:
        df = merge_exhibition(df, BASE_DIR / "exhibition_data.csv")
    df = df.sort_values("date").reset_index(drop=True)

    idx = {t: i for i, t in enumerate(TICKETS)}
    keys, probs_all, labels_all = [], [], []

    for start, end in FOLDS:
        fit_df = df[df["date"] < start].reset_index(drop=True)
        fold_df = df[(df["date"] >= start) & (df["date"] < end)].reset_index(drop=True)
        print(f"\n=== fold {start}〜{end} ===")
        print(f"  学習に使う: {len(fit_df):,}レース (〜{start}) / 予測する: {len(fold_df):,}レース")
        if fold_df.empty:
            continue

        model, m2, m3, cfg = train_fold_models(fit_df, features)
        X = build_features(fold_df.copy(), features)
        p1 = model.predict(X)
        tp = ticket_probs_batch(p1, X, m2, m3, cfg)

        y = np.zeros((len(fold_df), 120), dtype=np.int8)
        for i, (a, b, c) in enumerate(zip(fold_df["label"], fold_df["label_2nd"], fold_df["label_3rd"])):
            y[i, idx[(a, b, c)]] = 1

        keys.extend(
            f"{d}_{c}_{int(r)}"
            for d, c, r in zip(fold_df["date"], fold_df["course"], fold_df["rno"])
        )
        probs_all.append(tp)
        labels_all.append(y)

        # このfoldでの較正を確認(out-of-sampleなら予測≒実際になるはず)
        pf = tp.ravel().astype(float)
        yf = y.ravel()
        m = pf < 0.002
        print(f"  較正チェック(予測<0.2%の舟券): 予測平均 {pf[m].mean()*100:.3f}% / "
              f"実際 {yf[m].mean()*100:.3f}%  ← 近ければ out-of-sample として健全")

    probs = np.vstack(probs_all)
    labels = np.vstack(labels_all)
    np.savez_compressed(
        BASE_DIR / args.out,
        keys=np.array(keys), probs=probs.astype(np.float32), labels=labels,
    )
    print(f"\n保存しました: {args.out} ({len(keys):,}レース)")


if __name__ == "__main__":
    main()
