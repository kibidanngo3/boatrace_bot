"""2着・3着を直接予測するモデルを学習する。

main.py の estimate_ticket_probability() は現状、1着確率ベクトルから
Plackett-Luceの簡易近似(条件付き確率の再正規化)で2着・3着を推定している。
これは「1着になりやすい艇=2着にもなりやすい」という前提の粗い近似で、
実際のコース取り・展開による着順の偏りを反映できていない。

このスクリプトは「1着が誰か」を追加の入力特徴量として与えた上で、
実際の2着・3着ラベルから直接学習することでこの近似を置き換える。

使い方:
    python scripts/train_order_model.py --input training_data.csv --valid-days 60
"""
import argparse
import pickle
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from train_model import build_features, FEATURES, BASE_DIR  # noqa: E402

FEATURES_2ND = FEATURES + ["given_1st"]
FEATURES_3RD = FEATURES + ["given_1st", "given_2nd"]

PARAMS = {
    "objective": "multiclass",
    "num_class": 6,
    "metric": "multi_logloss",
    "verbosity": -1,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "random_state": 42,
}


def train_one(X, y, X_valid, y_valid, num_boost_round=2000, early_stopping_rounds=50):
    train_set = lgb.Dataset(X, label=y)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
    model = lgb.train(
        PARAMS,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(period=100)],
    )
    probs = model.predict(X_valid, num_iteration=model.best_iteration)
    print(f"  Accuracy: {accuracy_score(y_valid, probs.argmax(axis=1)):.3f}")
    print(f"  LogLoss:  {log_loss(y_valid, probs, labels=list(range(6))):.4f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training_data.csv")
    parser.add_argument("--valid-days", type=int, default=60)
    args = parser.parse_args()

    df = pd.read_csv(BASE_DIR / args.input, dtype=str, encoding="utf-8-sig")
    df = df.dropna(subset=["label", "label_2nd", "label_3rd"])
    for col in ("label", "label_2nd", "label_3rd"):
        df[col] = df[col].astype(int)
    df = df[df["label"].between(1, 6) & df["label_2nd"].between(1, 6) & df["label_3rd"].between(1, 6)]
    df = df.sort_values("date")

    dates = sorted(df["date"].unique())
    cutoff_date = dates[-args.valid_days]
    train_df = df[df["date"] < cutoff_date].reset_index(drop=True)
    valid_df = df[df["date"] >= cutoff_date].reset_index(drop=True)
    print(f"学習: {len(train_df)}件 / 検証: {len(valid_df)}件 (cutoff={cutoff_date})")

    base_train = build_features(train_df.copy())
    base_valid = build_features(valid_df.copy())

    # --- 2着モデル: 実際の1着艇番を追加特徴量として与える ---
    print("\n=== 2着モデル学習 ===")
    X2_train = base_train.copy()
    X2_train["given_1st"] = train_df["label"].values
    X2_valid = base_valid.copy()
    X2_valid["given_1st"] = valid_df["label"].values
    y2_train = train_df["label_2nd"].values - 1
    y2_valid = valid_df["label_2nd"].values - 1
    model_2nd = train_one(X2_train[FEATURES_2ND], y2_train, X2_valid[FEATURES_2ND], y2_valid)

    # --- 3着モデル: 実際の1着・2着艇番を追加特徴量として与える ---
    print("\n=== 3着モデル学習 ===")
    X3_train = base_train.copy()
    X3_train["given_1st"] = train_df["label"].values
    X3_train["given_2nd"] = train_df["label_2nd"].values
    X3_valid = base_valid.copy()
    X3_valid["given_1st"] = valid_df["label"].values
    X3_valid["given_2nd"] = valid_df["label_2nd"].values
    y3_train = train_df["label_3rd"].values - 1
    y3_valid = valid_df["label_3rd"].values - 1
    model_3rd = train_one(X3_train[FEATURES_3RD], y3_train, X3_valid[FEATURES_3RD], y3_valid)

    with open(BASE_DIR / "order_model_2nd_v1.pkl", "wb") as f:
        pickle.dump(model_2nd, f)
    with open(BASE_DIR / "order_model_3rd_v1.pkl", "wb") as f:
        pickle.dump(model_3rd, f)
    with open(BASE_DIR / "order_model_config_v1.pkl", "wb") as f:
        pickle.dump({"features_2nd": FEATURES_2ND, "features_3rd": FEATURES_3RD, "params": PARAMS}, f)
    print("\n保存しました: order_model_2nd_v1.pkl, order_model_3rd_v1.pkl, order_model_config_v1.pkl")


if __name__ == "__main__":
    main()
