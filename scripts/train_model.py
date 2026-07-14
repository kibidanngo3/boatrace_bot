"""training_data.csv から本番と同じ特徴量を再現してLightGBM多クラスモデルを再学習する。

main.py の predict_single() と全く同じ特徴量エンジニアリングをここでも行うことで、
学習時と推論時のズレ（ラベルエンコーディングのバグなど）を無くす。

使い方:
    python scripts/train_model.py
    python scripts/train_model.py --input training_data.csv --valid-days 60
"""
import argparse
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

BASE_DIR = Path(__file__).resolve().parent.parent

RANK_MAP = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}

FEATURES = ["wind_speed", "wave", "is_debuff_1"]
for boat_no in range(1, 7):
    FEATURES.extend([
        f"rank_val_{boat_no}", f"win_rate_{boat_no}", f"ex_time_{boat_no}",
        f"ex_diff_{boat_no}", f"ex_rank_{boat_no}", f"st_{boat_no}",
    ])

# st_i は K ファイル(競走成績)の「スタートタイミング」= 本番レースで実際に切ったST。
# レース前には存在しない情報なので、学習に使うとデータ漏洩になる。
# 実際、本番の main.py はこの列に常に定数 0.15 を入れており、学習時と別物を食わせていた。
FEATURES_NO_ST = [f for f in FEATURES if not f.startswith("st_")]

# v8: スタート展示から取れる「締切前に確定している」情報を追加する(exhibition_data.csv)。
#   ex_st_i     : 展示ST(フライングは負値)。当日の出足の実測値。
#   ex_st_rank_i: レース内での展示STの速い順(1〜6)
#   in_course_i : 進入コース。前付けがあると枠番と一致せず、1号艇が外に飛ばされると
#                 勝率が激減するため、枠番だけを見ていた従来モデルには無い情報。
#   is_wakunari : 6艇とも枠番どおりの進入(枠なり)なら1
# いずれも本番の直前情報ページから締切前に取得できるので漏洩ではない。
FEATURES_V8 = list(FEATURES_NO_ST)
for boat_no in range(1, 7):
    FEATURES_V8.extend([
        f"ex_st_{boat_no}", f"ex_st_rank_{boat_no}", f"in_course_{boat_no}",
    ])
FEATURES_V8.append("is_wakunari")


def to_float(value, default=0.0):
    # float("nan") は例外にならないので、明示的に欠損を弾かないと default に落ちない
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(result) else result


def build_features(df, features=None):
    """main.py predict_single() と同じロジックで特徴量を作る(1行=1レース)。

    features に FEATURES_NO_ST を渡すと、漏洩する st_i を除いた特徴量だけを返す。
    """
    ex_cols = [f"ex_time_{i}" for i in range(1, 7)]
    for c in ex_cols:
        df[c] = df[c].apply(to_float)
    ex_mean = df[ex_cols].mean(axis=1)

    # ex_rank_i: レースごとに6艇のex_timeを method='min' で順位付け(main.pyのpandas rankと同じ)
    ex_ranks = df[ex_cols].rank(axis=1, method="min")

    features = features or FEATURES
    # st_i を要求されていない限り触らない。本番(main.py)はもうSTを取得しないため、
    # ライブ由来のデータには st_ 列が存在せず、無条件に読むと KeyError になる。
    want_st = any(f == f"st_{i}" for f in features for i in range(1, 7))
    want_ex_st = any(f.startswith("ex_st_") for f in features)

    out = pd.DataFrame(index=df.index)
    out["wind_speed"] = df["wind_speed"].apply(to_float)
    out["wave"] = df["wave"].apply(to_float)

    for i in range(1, 7):
        out[f"rank_val_{i}"] = df[f"rank_{i}"].map(RANK_MAP).fillna(2)
        out[f"win_rate_{i}"] = df[f"win_rate_{i}"].apply(to_float)
        out[f"ex_time_{i}"] = df[f"ex_time_{i}"].apply(to_float)
        out[f"ex_diff_{i}"] = out[f"ex_time_{i}"] - ex_mean
        out[f"ex_rank_{i}"] = ex_ranks[f"ex_time_{i}"]
        if want_st:
            out[f"st_{i}"] = df[f"st_{i}"].apply(to_float)

    out["is_debuff_1"] = ((out["rank_val_1"] <= 2) & (out["ex_rank_1"] >= 4)).astype(int)

    if want_ex_st:
        ex_st_cols = [f"ex_st_{i}" for i in range(1, 7)]
        # 展示STが欠けている場合(取得失敗・L=出遅れ)は 0.16 前後の平均的な値で埋める
        ex_st = df[ex_st_cols].apply(lambda c: c.map(lambda v: to_float(v, 0.16)))
        ex_st_ranks = ex_st.rank(axis=1, method="min")
        for i in range(1, 7):
            out[f"ex_st_{i}"] = ex_st[f"ex_st_{i}"]
            out[f"ex_st_rank_{i}"] = ex_st_ranks[f"ex_st_{i}"]
            out[f"in_course_{i}"] = df[f"in_course_{i}"].apply(lambda v: to_float(v, i))
        # 枠なり進入(全艇が枠番どおり)かどうか。前付けがあるレースは別物になる。
        out["is_wakunari"] = (
            sum((out[f"in_course_{i}"] == i).astype(int) for i in range(1, 7)) == 6
        ).astype(int)

    return out[features]


def merge_exhibition(df, path):
    """exhibition_data.csv(展示ST・進入コース)を training_data と突合する。

    取得できなかったレースは学習から落とす。定数で埋めると、本番で同じ列に別の値が
    来たときに学習と推論がズレる(v5 の st_i で実際に起きた事故)。
    """
    ex = pd.read_csv(path, dtype={"date": str}, encoding="utf-8-sig")
    ex["rno"] = ex["rno"].astype(str).astype(int)
    df = df.copy()
    df["rno"] = df["rno"].astype(str).astype(int)
    before = len(df)
    df = df.merge(ex, on=["date", "course", "rno"], how="inner")
    print(f"スタート展示を突合: {before} → {len(df)}件 ({len(df)/before:.1%} が取得済み)")
    return df


def calibration_report(y_true_boat1_win, p_boat1_win, label):
    """モデルが「1号艇は負ける」と自信を持って言った局面で、実際の1号艇勝率と
    比較する(会話で見つかった較正の崩れを定量的に追跡するため)。"""
    mask = p_boat1_win < 0.45  # イン飛び率55%以上で賭ける、に相当
    n = mask.sum()
    if n == 0:
        print(f"  [{label}] イン飛び率55%以上のサンプルなし")
        return
    predicted_win_rate = p_boat1_win[mask].mean()
    actual_win_rate = y_true_boat1_win[mask].mean()
    print(
        f"  [{label}] イン飛び予測局面(n={n}): "
        f"モデルの平均1号艇勝率予測={predicted_win_rate:.1%} / 実際の1号艇勝率={actual_win_rate:.1%}"
    )


def main():
    parser = argparse.ArgumentParser(description="training_data.csv からモデルを再学習する")
    parser.add_argument("--input", default="training_data.csv")
    parser.add_argument("--valid-days", type=int, default=60, help="末尾何日分を検証用に切り出すか")
    parser.add_argument("--model-out", default="final_model_v5.pkl")
    parser.add_argument("--config-out", default="model_config_v5.pkl")
    parser.add_argument("--num-boost-round", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--drop-st", action="store_true", help="漏洩する st_i を特徴量から外す")
    parser.add_argument("--v8", action="store_true",
                        help="スタート展示(展示ST・進入コース)を特徴量に加える。exhibition_data.csv が必要")
    parser.add_argument("--exhibition", default="exhibition_data.csv")
    args = parser.parse_args()

    input_path = BASE_DIR / args.input
    print(f"読み込み中: {input_path}")
    df = pd.read_csv(input_path, dtype=str, encoding="utf-8-sig")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].between(1, 6)]
    print(f"総レース数: {len(df)}")

    # 突合は train/valid に分ける前に済ませる(後だと分割済みDataFrameに反映されない)
    if args.v8:
        features = FEATURES_V8
        df = merge_exhibition(df, BASE_DIR / args.exhibition)
        print("※ スタート展示(展示ST・進入コース)を特徴量に追加して学習する")
    elif args.drop_st:
        features = FEATURES_NO_ST
        print("※ st_i(本番STによる漏洩特徴量)を除外して学習する")
    else:
        features = FEATURES

    df = df.sort_values("date")
    dates = df["date"].unique()
    if len(dates) <= args.valid_days:
        raise SystemExit(f"日数が足りません(全{len(dates)}日 <= valid-days {args.valid_days})")
    cutoff_date = sorted(dates)[-args.valid_days]
    train_df = df[df["date"] < cutoff_date]
    valid_df = df[df["date"] >= cutoff_date]
    print(f"学習: {len(train_df)}件 ({train_df['date'].min()}〜{train_df['date'].max()})")
    print(f"検証: {len(valid_df)}件 ({valid_df['date'].min()}〜{valid_df['date'].max()}) [直近{args.valid_days}日をホールドアウト]")

    X_train = build_features(train_df.reset_index(drop=True), features)
    X_valid = build_features(valid_df.reset_index(drop=True), features)
    y_train = train_df["label"].values - 1  # LightGBM multiclass は 0始まりが必須
    y_valid = valid_df["label"].values - 1

    params = {
        "objective": "multiclass",
        "num_class": 6,
        "metric": "multi_logloss",
        "verbosity": -1,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "random_state": 42,
    }

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=features)
    valid_set = lgb.Dataset(X_valid, label=y_valid, feature_name=features, reference=train_set)

    print("\n学習開始...")
    model = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds),
            lgb.log_evaluation(period=100),
        ],
    )

    print("\n--- 検証結果 ---")
    valid_probs = model.predict(X_valid, num_iteration=model.best_iteration)
    valid_pred = valid_probs.argmax(axis=1)
    print(f"Accuracy (勝ち艇の完全一致): {accuracy_score(y_valid, valid_pred):.3f}")
    print(f"LogLoss: {log_loss(y_valid, valid_probs, labels=list(range(6))):.4f}")

    p_boat1_win_valid = valid_probs[:, 0]
    y_boat1_win_valid = (y_valid == 0).astype(int)
    calibration_report(y_boat1_win_valid, p_boat1_win_valid, "検証データ(新モデル)")

    train_probs = model.predict(X_train, num_iteration=model.best_iteration)
    p_boat1_win_train = train_probs[:, 0]
    y_boat1_win_train = (y_train == 0).astype(int)
    calibration_report(y_boat1_win_train, p_boat1_win_train, "学習データ(新モデル・参考値)")

    model_path = BASE_DIR / args.model_out
    config_path = BASE_DIR / args.config_out
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(config_path, "wb") as f:
        pickle.dump({"features": features, "params": params, "date_trained": pd.Timestamp.now().strftime("%Y%m%d")}, f)
    print(f"\n保存しました: {model_path.name}, {config_path.name}")
    print("(本番が読むのは main.py の MODEL_PATH。切り替えるにはそこを書き換える)")


if __name__ == "__main__":
    main()
