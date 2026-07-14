"""漏洩なしのベースライン(v7)の上で、既に収集済みなのに使っていない特徴量を測り直す。

v6のアブレーションは「漏洩ありのv5」をベースラインに測ったため、
本番ST(st_i)という極めて強い漏洩特徴量が全てを説明してしまい、
当地成績やモーターの寄与が誤差に埋もれて「効果なし」と結論されていた。
漏洩を除いた今なら評価が変わりうるので、同じ特徴量群を測り直す。

v6になかった追加分:
  avg_st_i : 期別平均ST。選手のスタート力そのもの。漏洩する本番STがあった頃は
             使う動機がなく、一度も特徴量に入っていなかった。

使い方:
    python scripts/ablation_v9.py
    python scripts/ablation_v9.py --sets base A B C all
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_model import (  # noqa: E402
    build_features, to_float, FEATURES_NO_ST, BASE_DIR,
)

# A: 選手の実力・当地適性・機力(既に training_data.csv にある)
GROUP_A = []
for _i in range(1, 7):
    GROUP_A.extend([
        f"avg_st_{_i}",            # 期別平均ST(v6では未使用)
        f"national_2_rate_{_i}", f"local_win_rate_{_i}", f"local_2_rate_{_i}",
        f"motor_2_rate_{_i}", f"boat_2_rate_{_i}", f"weight_{_i}",
    ])

# B: 会場・レース番号・季節・風向
GROUP_B = ["course_id", "race_no", "month_sin", "month_cos"]

# C: レース内での相対値(絶対値より「この面子の中で強いか」が効くはず)
GROUP_C = []
for _i in range(1, 7):
    GROUP_C.extend([
        f"win_rate_diff_{_i}", f"avg_st_diff_{_i}",
        f"local_win_diff_{_i}", f"motor_diff_{_i}",
    ])

COURSE_ID_MAP = {
    "桐生": 1, "戸田": 2, "江戸川": 3, "平和島": 4, "多摩川": 5, "浜名湖": 6, "蒲郡": 7,
    "常滑": 8, "津": 9, "三国": 10, "びわこ": 11, "住之江": 12, "尼崎": 13, "鳴門": 14,
    "丸亀": 15, "児島": 16, "宮島": 17, "徳山": 18, "下関": 19, "若松": 20, "芦屋": 21,
    "福岡": 22, "唐津": 23, "大村": 24,
}

SETS = {
    "base": FEATURES_NO_ST,
    "A": FEATURES_NO_ST + GROUP_A,
    "B": FEATURES_NO_ST + GROUP_B,
    "C": FEATURES_NO_ST + GROUP_A + GROUP_C,  # 差分は元の値とセットで意味を持つ
    "all": FEATURES_NO_ST + GROUP_A + GROUP_B + GROUP_C,
}

PARAMS = {
    "objective": "multiclass", "num_class": 6, "metric": "multi_logloss",
    "verbosity": -1, "learning_rate": 0.03, "num_leaves": 31, "random_state": 42,
}


def build_extended(df, features):
    """build_features(漏洩なし)に、追加群の列を足す。"""
    out = build_features(df.copy(), FEATURES_NO_ST)

    want = set(features)
    need_a = any(f in want for f in GROUP_A)
    need_b = any(f in want for f in GROUP_B)
    need_c = any(f in want for f in GROUP_C)

    if need_a or need_c:
        for i in range(1, 7):
            for col in ("avg_st", "national_2_rate", "local_win_rate",
                        "local_2_rate", "motor_2_rate", "boat_2_rate", "weight"):
                out[f"{col}_{i}"] = df[f"{col}_{i}"].apply(to_float)

    if need_b:
        out["course_id"] = df["course"].map(COURSE_ID_MAP).fillna(0).astype(int)
        out["race_no"] = df["rno"].apply(lambda v: to_float(v, 0))
        month = df["date"].str[4:6].apply(lambda v: to_float(v, 1))
        out["month_sin"] = np.sin(2 * np.pi * month / 12)
        out["month_cos"] = np.cos(2 * np.pi * month / 12)

    if need_c:
        # レース内平均からの差。絶対的な勝率より「この6艇の中での相対的な強さ」が効く想定。
        for base in ("win_rate", "avg_st", "local_win_rate", "motor_2_rate"):
            cols = [f"{base}_{i}" for i in range(1, 7)]
            mean = out[cols].mean(axis=1)
            alias = {"win_rate": "win_rate_diff", "avg_st": "avg_st_diff",
                     "local_win_rate": "local_win_diff", "motor_2_rate": "motor_diff"}[base]
            for i in range(1, 7):
                out[f"{alias}_{i}"] = out[f"{base}_{i}"] - mean

    return out[features]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-days", type=int, default=60)
    parser.add_argument("--sets", nargs="+", default=["base", "A", "B", "C", "all"])
    args = parser.parse_args()

    df = pd.read_csv(BASE_DIR / "training_data.csv", dtype=str, encoding="utf-8-sig")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].between(1, 6)].sort_values("date")

    cutoff = sorted(df["date"].unique())[-args.valid_days]
    train_df = df[df["date"] < cutoff].reset_index(drop=True)
    valid_df = df[df["date"] >= cutoff].reset_index(drop=True)
    y_train = train_df["label"].values - 1
    y_valid = valid_df["label"].values - 1
    print(f"学習 {len(train_df)}件 / 検証 {len(valid_df)}件 (cutoff={cutoff})")
    print(f"ベースライン(常に1号艇): Accuracy {(y_valid == 0).mean():.3f}\n")

    results = []
    for name in args.sets:
        features = SETS[name]
        X_train = build_extended(train_df, features)
        X_valid = build_extended(valid_df, features)

        model = lgb.train(
            PARAMS,
            lgb.Dataset(X_train, label=y_train, feature_name=features),
            num_boost_round=2000,
            valid_sets=[lgb.Dataset(X_valid, label=y_valid, feature_name=features)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        probs = model.predict(X_valid, num_iteration=model.best_iteration)
        acc = accuracy_score(y_valid, probs.argmax(axis=1))
        ll = log_loss(y_valid, probs, labels=list(range(6)))

        # 較正: モデルが「1号艇は負ける」と言った局面での実測
        p1 = probs[:, 0]
        mask = p1 < 0.45
        cal_pred = p1[mask].mean() if mask.sum() else float("nan")
        cal_true = (y_valid[mask] == 0).mean() if mask.sum() else float("nan")

        results.append((name, len(features), acc, ll, mask.sum(), cal_pred, cal_true))
        print(f"{name:5s} 特徴量{len(features):3d}個  Accuracy {acc:.4f}  LogLoss {ll:.4f}  "
              f"較正: 予測{cal_pred:.1%} / 実際{cal_true:.1%} (n={mask.sum()})")

    print("\n=== まとめ (baseは漏洩なしv7と同じ特徴量) ===")
    base_ll = next(r[3] for r in results if r[0] == "base") if any(r[0] == "base" for r in results) else None
    for name, nf, acc, ll, n, cp, ct in results:
        delta = f"{(base_ll - ll) / base_ll * 100:+.2f}%" if base_ll else "-"
        print(f"  {name:5s} Accuracy {acc:.4f}  LogLoss {ll:.4f} (base比 {delta})")


if __name__ == "__main__":
    main()
