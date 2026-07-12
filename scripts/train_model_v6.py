"""v5のFEATURESに特徴量を追加してアブレーション比較する(v6-A/B/C/all)。

一度に全部足すと何が効いたか分からなくなるため、カテゴリごとに単独で試せるようにする。
評価は v5 と全く同じホールドアウト条件(直近60日)・同じ指標(Accuracy/LogLoss/較正/
賭け対象局面での1着的中率)で行い、model_evaluation_log.md に転記できる形で出力する。

使い方:
    python scripts/train_model_v6.py --feature-set v6a
    python scripts/train_model_v6.py --feature-set v6b
    python scripts/train_model_v6.py --feature-set v6c
    python scripts/train_model_v6.py --feature-set all
"""
import argparse
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from train_model import RANK_MAP, to_float, calibration_report  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent

BASE_FEATURES = ["wind_speed", "wave", "is_debuff_1"]
for _i in range(1, 7):
    BASE_FEATURES.extend([
        f"rank_val_{_i}", f"win_rate_{_i}", f"ex_time_{_i}",
        f"ex_diff_{_i}", f"ex_rank_{_i}", f"st_{_i}",
    ])

V6A_FEATURES = []
for _i in range(1, 7):
    V6A_FEATURES.extend([
        f"national_2_rate_{_i}", f"local_win_rate_{_i}", f"local_2_rate_{_i}",
        f"motor_2_rate_{_i}", f"boat_2_rate_{_i}", f"weight_{_i}",
    ])

V6B_FEATURES = ["course_id", "race_no", "month_sin", "month_cos", "wind_dir_id"]
V6B_CATEGORICAL = ["course_id", "wind_dir_id"]

V6C_FEATURES = []
for _i in range(1, 7):
    V6C_FEATURES.extend([
        f"win_rate_diff_{_i}", f"st_diff_{_i}", f"local_win_diff_{_i}",
        f"motor_diff_{_i}", f"boat_diff_{_i}", f"weight_diff_{_i}",
    ])

FEATURE_SETS = {
    "baseline": BASE_FEATURES,
    "v6a": BASE_FEATURES + V6A_FEATURES,
    "v6b": BASE_FEATURES + V6B_FEATURES,
    "v6c": BASE_FEATURES + V6C_FEATURES,
    "all": BASE_FEATURES + V6A_FEATURES + V6B_FEATURES + V6C_FEATURES,
}

COURSE_ID_MAP = {
    "桐生": 1, "戸田": 2, "江戸川": 3, "平和島": 4, "多摩川": 5, "浜名湖": 6, "蒲郡": 7,
    "常滑": 8, "津": 9, "三国": 10, "びわこ": 11, "住之江": 12, "尼崎": 13, "鳴門": 14,
    "丸亀": 15, "児島": 16, "宮島": 17, "徳山": 18, "下関": 19, "若松": 20, "芦屋": 21,
    "福岡": 22, "唐津": 23, "大村": 24,
}
WIND_DIR_MAP = {d: i for i, d in enumerate(
    ["無風", "北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東",
     "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"]
)}


def build_features_v6(df, feature_set):
    ex_cols = [f"ex_time_{i}" for i in range(1, 7)]
    for c in ex_cols:
        df[c] = df[c].apply(to_float)
    ex_mean = df[ex_cols].mean(axis=1)
    ex_ranks = df[ex_cols].rank(axis=1, method="min")

    out = pd.DataFrame(index=df.index)
    out["wind_speed"] = df["wind_speed"].apply(to_float)
    out["wave"] = df["wave"].apply(to_float)

    for i in range(1, 7):
        out[f"rank_val_{i}"] = df[f"rank_{i}"].map(RANK_MAP).fillna(2)
        out[f"win_rate_{i}"] = df[f"win_rate_{i}"].apply(to_float)
        out[f"ex_time_{i}"] = df[f"ex_time_{i}"].apply(to_float)
        out[f"ex_diff_{i}"] = out[f"ex_time_{i}"] - ex_mean
        out[f"ex_rank_{i}"] = ex_ranks[f"ex_time_{i}"]
        out[f"st_{i}"] = df[f"st_{i}"].apply(to_float)
    out["is_debuff_1"] = ((out["rank_val_1"] <= 2) & (out["ex_rank_1"] >= 4)).astype(int)

    needed = FEATURE_SETS[feature_set]

    if any(f.startswith(("national_2_rate", "local_win_rate", "local_2_rate", "motor_2_rate", "boat_2_rate", "weight_")) for f in needed):
        for i in range(1, 7):
            out[f"national_2_rate_{i}"] = df[f"national_2_rate_{i}"].apply(to_float)
            out[f"local_win_rate_{i}"] = df[f"local_win_rate_{i}"].apply(to_float)
            out[f"local_2_rate_{i}"] = df[f"local_2_rate_{i}"].apply(to_float)
            out[f"motor_2_rate_{i}"] = df[f"motor_2_rate_{i}"].apply(to_float)
            out[f"boat_2_rate_{i}"] = df[f"boat_2_rate_{i}"].apply(to_float)
            out[f"weight_{i}"] = df[f"weight_{i}"].apply(to_float)

    if "course_id" in needed:
        out["course_id"] = df["course"].map(COURSE_ID_MAP).fillna(0).astype(int)
        out["race_no"] = df["rno"].astype(int)
        month = df["date"].str[4:6].astype(int)
        out["month_sin"] = np.sin(2 * np.pi * month / 12)
        out["month_cos"] = np.cos(2 * np.pi * month / 12)
        out["wind_dir_id"] = df["wind_dir"].map(WIND_DIR_MAP).fillna(-1).astype(int)

    if any(f.startswith(("win_rate_diff", "st_diff", "local_win_diff", "motor_diff", "boat_diff", "weight_diff")) for f in needed):
        local_win = pd.DataFrame({i: df[f"local_win_rate_{i}"].apply(to_float) for i in range(1, 7)})
        motor2 = pd.DataFrame({i: df[f"motor_2_rate_{i}"].apply(to_float) for i in range(1, 7)})
        boat2 = pd.DataFrame({i: df[f"boat_2_rate_{i}"].apply(to_float) for i in range(1, 7)})
        weight = pd.DataFrame({i: df[f"weight_{i}"].apply(to_float) for i in range(1, 7)})
        win_rate_df = pd.DataFrame({i: out[f"win_rate_{i}"] for i in range(1, 7)})
        st_df = pd.DataFrame({i: out[f"st_{i}"] for i in range(1, 7)})

        local_win_mean, motor2_mean = local_win.mean(axis=1), motor2.mean(axis=1)
        boat2_mean, weight_mean = boat2.mean(axis=1), weight.mean(axis=1)
        win_rate_mean, st_mean = win_rate_df.mean(axis=1), st_df.mean(axis=1)

        for i in range(1, 7):
            out[f"win_rate_diff_{i}"] = win_rate_df[i] - win_rate_mean
            out[f"st_diff_{i}"] = st_df[i] - st_mean
            out[f"local_win_diff_{i}"] = local_win[i] - local_win_mean
            out[f"motor_diff_{i}"] = motor2[i] - motor2_mean
            out[f"boat_diff_{i}"] = boat2[i] - boat2_mean
            out[f"weight_diff_{i}"] = weight[i] - weight_mean

    return out[needed]


def betting_regime_accuracy(probs, label, in_jump_threshold=0.55, focus_th=0.35, standard_th=0.25):
    """main.pyのFOCUS/STANDARD判定と同じロジックで、賭け対象局面でのtop1的中率を出す。"""
    counts = {"FOCUS": [0, 0], "STANDARD": [0, 0]}
    for p, lbl in zip(probs, label):
        in_jump = 1 - p[0]
        ranking = sorted(((i + 1, p[i]) for i in range(6) if i != 0), key=lambda x: -x[1])
        top1_boat, top1_prob = ranking[0]
        if in_jump >= in_jump_threshold:
            if top1_prob >= focus_th:
                strat = "FOCUS"
            elif top1_prob >= standard_th:
                strat = "STANDARD"
            else:
                continue
            counts[strat][1] += 1
            if lbl == top1_boat:
                counts[strat][0] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training_data.csv")
    parser.add_argument("--feature-set", choices=list(FEATURE_SETS.keys()), required=True)
    parser.add_argument("--valid-days", type=int, default=60)
    parser.add_argument("--num-boost-round", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--save", action="store_true", help="モデルをmodel_v6_<feature-set>.pklとして保存する")
    parser.add_argument("--num-threads", type=int, default=4, help="並列実行時にCPUを食い合わないよう制限する")
    args = parser.parse_args()

    features = FEATURE_SETS[args.feature_set]
    categorical = V6B_CATEGORICAL if args.feature_set in ("v6b", "all") else []

    df = pd.read_csv(BASE_DIR / args.input, dtype=str, encoding="utf-8-sig")
    df = df.dropna(subset=["label", "label_2nd", "label_3rd"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].between(1, 6)]
    df = df.sort_values("date")

    dates = sorted(df["date"].unique())
    cutoff_date = dates[-args.valid_days]
    train_df = df[df["date"] < cutoff_date].reset_index(drop=True)
    valid_df = df[df["date"] >= cutoff_date].reset_index(drop=True)
    print(f"特徴量セット: {args.feature_set} ({len(features)}個)")
    print(f"学習: {len(train_df)}件 / 検証: {len(valid_df)}件 (cutoff={cutoff_date})")

    X_train = build_features_v6(train_df.copy(), args.feature_set)
    X_valid = build_features_v6(valid_df.copy(), args.feature_set)
    y_train = train_df["label"].values - 1
    y_valid = valid_df["label"].values - 1

    for col in categorical:
        X_train[col] = X_train[col].astype("category")
        X_valid[col] = X_valid[col].astype("category")

    params = {
        "objective": "multiclass", "num_class": 6, "metric": "multi_logloss",
        "verbosity": -1, "learning_rate": 0.03, "num_leaves": 31, "random_state": 42,
        "num_threads": args.num_threads,
    }
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical or "auto")
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

    model = lgb.train(
        params, train_set, num_boost_round=args.num_boost_round,
        valid_sets=[train_set, valid_set], valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(args.early_stopping_rounds), lgb.log_evaluation(period=100)],
    )

    valid_probs = model.predict(X_valid, num_iteration=model.best_iteration)
    valid_pred = valid_probs.argmax(axis=1)
    print(f"\n=== {args.feature_set} 検証結果 ===")
    print(f"Accuracy: {accuracy_score(y_valid, valid_pred):.3f}")
    print(f"LogLoss:  {log_loss(y_valid, valid_probs, labels=list(range(6))):.4f}")

    p_boat1 = valid_probs[:, 0]
    y_boat1 = (y_valid == 0).astype(int)
    calibration_report(y_boat1, p_boat1, f"{args.feature_set} 検証データ")

    counts = betting_regime_accuracy(valid_probs, valid_df["label"].astype(int).values)
    for strat, (hit, n) in counts.items():
        if n:
            print(f"  賭け対象局面 {strat}: {hit}/{n} = {hit/n*100:.1f}%")
    total_hit = sum(c[0] for c in counts.values())
    total_n = sum(c[1] for c in counts.values())
    if total_n:
        print(f"  賭け対象局面 合計: {total_hit}/{total_n} = {total_hit/total_n*100:.1f}%")

    if args.save:
        model_path = BASE_DIR / f"model_v6_{args.feature_set}.pkl"
        config_path = BASE_DIR / f"model_v6_{args.feature_set}_config.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(config_path, "wb") as f:
            pickle.dump({"features": features, "categorical": categorical, "params": params}, f)
        print(f"保存しました: {model_path.name}, {config_path.name}")


if __name__ == "__main__":
    main()
