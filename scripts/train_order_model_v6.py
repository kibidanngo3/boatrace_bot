"""v6-all特徴量で2着・3着モデルを学習する（train_order_model.pyのv6版）。

1着モデル(model_v6_all.pkl)と同じ特徴量エンジニアリング(build_features_v6の"all")を使い、
2着モデルは実1着艇番、3着モデルは実1・2着艇番を追加入力として直接学習する。

使い方:
    python scripts/train_order_model_v6.py --input training_data.csv --valid-days 60
"""
import argparse
import pickle
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from train_model_v6 import build_features_v6, FEATURE_SETS, V6B_CATEGORICAL

BASE_DIR = Path(__file__).resolve().parent.parent

PARAMS = {
    "objective": "multiclass", "num_class": 6, "metric": "multi_logloss",
    "verbosity": -1, "learning_rate": 0.03, "num_leaves": 31, "random_state": 42,
}


def train_one(X, y, X_valid, y_valid, categorical):
    train_set = lgb.Dataset(X, label=y, categorical_feature=categorical or "auto")
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
    model = lgb.train(
        PARAMS, train_set, num_boost_round=2000,
        valid_sets=[train_set, valid_set], valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=200)],
    )
    probs = model.predict(X_valid, num_iteration=model.best_iteration)
    print(f"  Accuracy: {accuracy_score(y_valid, probs.argmax(axis=1)):.3f}")
    print(f"  LogLoss:  {log_loss(y_valid, probs, labels=list(range(6))):.4f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training_data.csv")
    parser.add_argument("--valid-days", type=int, default=60)
    parser.add_argument("--feature-set", default="all")
    args = parser.parse_args()

    base_features = FEATURE_SETS[args.feature_set]
    categorical = V6B_CATEGORICAL if args.feature_set in ("v6b", "all") else []

    df = pd.read_csv(BASE_DIR / args.input, dtype=str, encoding="utf-8-sig")
    df = df.dropna(subset=["label", "label_2nd", "label_3rd"])
    for col in ("label", "label_2nd", "label_3rd"):
        df[col] = df[col].astype(int)
    df = df[df["label"].between(1, 6) & df["label_2nd"].between(1, 6) & df["label_3rd"].between(1, 6)]
    df = df.sort_values("date")

    dates = sorted(df["date"].unique())
    cutoff = dates[-args.valid_days]
    train_df = df[df["date"] < cutoff].reset_index(drop=True)
    valid_df = df[df["date"] >= cutoff].reset_index(drop=True)
    print(f"学習: {len(train_df)}件 / 検証: {len(valid_df)}件 (cutoff={cutoff}, feature-set={args.feature_set})")

    base_train = build_features_v6(train_df.copy(), args.feature_set)
    base_valid = build_features_v6(valid_df.copy(), args.feature_set)

    features_2nd = base_features + ["given_1st"]
    features_3rd = base_features + ["given_1st", "given_2nd"]

    print("\n=== 2着モデル(v6) ===")
    X2t = base_train.copy(); X2t["given_1st"] = train_df["label"].values
    X2v = base_valid.copy(); X2v["given_1st"] = valid_df["label"].values
    for c in categorical:
        X2t[c] = X2t[c].astype("category"); X2v[c] = X2v[c].astype("category")
    model_2nd = train_one(X2t[features_2nd], train_df["label_2nd"].values - 1,
                          X2v[features_2nd], valid_df["label_2nd"].values - 1, categorical)

    print("\n=== 3着モデル(v6) ===")
    X3t = base_train.copy(); X3t["given_1st"] = train_df["label"].values; X3t["given_2nd"] = train_df["label_2nd"].values
    X3v = base_valid.copy(); X3v["given_1st"] = valid_df["label"].values; X3v["given_2nd"] = valid_df["label_2nd"].values
    for c in categorical:
        X3t[c] = X3t[c].astype("category"); X3v[c] = X3v[c].astype("category")
    model_3rd = train_one(X3t[features_3rd], train_df["label_3rd"].values - 1,
                          X3v[features_3rd], valid_df["label_3rd"].values - 1, categorical)

    with open(BASE_DIR / "order_model_2nd_v6.pkl", "wb") as f:
        pickle.dump(model_2nd, f)
    with open(BASE_DIR / "order_model_3rd_v6.pkl", "wb") as f:
        pickle.dump(model_3rd, f)
    with open(BASE_DIR / "order_model_config_v6.pkl", "wb") as f:
        pickle.dump({"features_2nd": features_2nd, "features_3rd": features_3rd,
                     "categorical": categorical, "feature_set": args.feature_set}, f)
    print("\n保存しました: order_model_2nd_v6.pkl, order_model_3rd_v6.pkl, order_model_config_v6.pkl")


if __name__ == "__main__":
    main()
