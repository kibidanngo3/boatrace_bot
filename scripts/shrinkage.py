"""モデルの確率を市場の確率に向けて縮め、EV選抜の「勝者の呪い」を打ち消す。

selection_bias.py で分かったこと:
モデルの舟券確率は全体では正しく較正されているが、EVで上位を選んだ瞬間に
2〜4倍の水増しになる。EV = 確率 × オッズ で選ぶ行為が、モデルが上振れした
舟券だけを拾ってしまうため(勝者の呪い)。

対策は、モデルの確率をそのまま信じず、市場の暗黙確率(1/オッズ)と突き合わせて
「本当の的中確率」を学習し直すこと。式は決め打ちせず、

    P(的中) = f( logit(モデル確率), logit(市場確率), オッズ )

を学習期間のデータから学ぶ。市場をどれだけ信じるかをデータに決めさせる。

学習期間のオッズは odds_train_cache.csv(scripts/fetch_odds.py --odds-only)。
ホールドアウトは学習に一切使わない。

使い方:
    python scripts/shrinkage.py --model final_model_v8.pkl --order-suffix v8 --v8
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_model import (  # noqa: E402
    build_features, merge_exhibition, FEATURES, FEATURES_NO_ST, FEATURES_V8,
)
from scripts.calibrate import ticket_probs_batch, TICKETS  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
EPS = 1e-9

SHRINK_FEATURES = ["logit_model", "logit_market", "log_odds", "diff"]


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def build_pairs(df, cache, model, m2, m3, cfg, features, label):
    """(モデル確率, 市場確率, 的中) の組を舟券単位で作る。"""
    keep = [
        i for i, r in enumerate(df.itertuples())
        if (r.date, r.course, str(int(r.rno))) in cache
    ]
    df = df.iloc[keep].reset_index(drop=True)
    print(f"{label}: オッズと突合できたのは {len(df):,}レース")

    X = build_features(df.copy(), features)
    probs = model.predict(X)
    tp = ticket_probs_batch(probs, X, m2, m3, cfg)

    idx = {t: i for i, t in enumerate(TICKETS)}
    n = len(df)
    model_p = np.zeros(n * 120, dtype=np.float64)
    market_o = np.zeros(n * 120, dtype=np.float64)
    y = np.zeros(n * 120, dtype=np.int8)

    pos = 0
    for i, row in enumerate(df.itertuples()):
        odds_map = json.loads(cache[(row.date, row.course, str(int(row.rno)))]["odds_json"])
        won = (row.label, row.label_2nd, row.label_3rd)
        for ti, t in enumerate(TICKETS):
            o = odds_map.get(f"{t[0]}-{t[1]}-{t[2]}")
            if not o:
                continue
            model_p[pos] = tp[i, ti]
            market_o[pos] = float(o)
            y[pos] = 1 if t == won else 0
            pos += 1

    model_p = model_p[:pos]
    market_o = market_o[:pos]
    y = y[:pos]

    # 市場の暗黙確率。1/オッズ の合計は控除率のぶん1を超えるので、レース内で正規化はせず
    # 生の 1/オッズ を使い、控除率の効果は log_odds 側に吸収させる。
    market_p = 1.0 / market_o

    feats = pd.DataFrame({
        "logit_model": logit(model_p),
        "logit_market": logit(market_p),
        "log_odds": np.log(market_o),
        "diff": logit(model_p) - logit(market_p),
    })
    return feats, y, model_p, market_o


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_model_v8.pkl")
    parser.add_argument("--order-suffix", default="v8")
    parser.add_argument("--drop-st", action="store_true")
    parser.add_argument("--v8", action="store_true")
    parser.add_argument("--cutoff", default="20260512")
    parser.add_argument("--train-odds", default="odds_train_cache.csv")
    parser.add_argument("--holdout-odds", default="odds_result_cache.csv")
    parser.add_argument("--out", default="shrink_model_v8.pkl")
    args = parser.parse_args()

    features = FEATURES_V8 if args.v8 else (FEATURES_NO_ST if args.drop_st else FEATURES)
    model = pickle.load(open(BASE_DIR / args.model, "rb"))
    m2 = pickle.load(open(BASE_DIR / f"order_model_2nd_{args.order_suffix}.pkl", "rb"))
    m3 = pickle.load(open(BASE_DIR / f"order_model_3rd_{args.order_suffix}.pkl", "rb"))
    cfg = pickle.load(open(BASE_DIR / f"order_model_config_{args.order_suffix}.pkl", "rb"))

    def load_cache(path):
        with open(BASE_DIR / path, encoding="utf-8-sig") as f:
            return {(r["date"], r["course"], str(int(r["rno"]))): r for r in csv.DictReader(f)}

    train_cache = load_cache(args.train_odds)
    hold_cache = load_cache(args.holdout_odds)
    print(f"学習期間のオッズ: {len(train_cache):,}レース / ホールドアウト: {len(hold_cache):,}レース\n")

    df = pd.read_csv(BASE_DIR / "training_data.csv", dtype=str, encoding="utf-8-sig")
    df = df.dropna(subset=["label", "label_2nd", "label_3rd"])
    for c in ("label", "label_2nd", "label_3rd"):
        df[c] = df[c].astype(int)
    df = df[df["label"].between(1, 6) & df["label_2nd"].between(1, 6) & df["label_3rd"].between(1, 6)]
    if args.v8:
        df = merge_exhibition(df, BASE_DIR / "exhibition_data.csv")

    train_df = df[df["date"] < args.cutoff].reset_index(drop=True)
    hold_df = df[df["date"] >= args.cutoff].reset_index(drop=True)

    # early stopping の検証にホールドアウトを使うと、そこにモデルを合わせ込むことになる
    # (今日2度踏んだ罠と同じ)。学習期間の末尾を切り出して検証に使う。
    inner_cutoff = sorted(train_df["date"].unique())[-30]
    fit_df = train_df[train_df["date"] < inner_cutoff].reset_index(drop=True)
    val_df = train_df[train_df["date"] >= inner_cutoff].reset_index(drop=True)
    print(f"縮小モデルの学習: 〜{inner_cutoff} / 検証: {inner_cutoff}〜 (ホールドアウトは触らない)\n")

    Xfit, yfit, _, _ = build_pairs(fit_df, train_cache, model, m2, m3, cfg, features, "学習(内側)")
    Xval, yval, _, _ = build_pairs(val_df, train_cache, model, m2, m3, cfg, features, "検証(内側)")
    Xho, yho, p_model_ho, odds_ho = build_pairs(
        hold_df, hold_cache, model, m2, m3, cfg, features, "ホールドアウト")

    print(f"\n学習 {len(yfit):,}点(的中 {yfit.sum():,}) / "
          f"内側検証 {len(yval):,}点 / ホールドアウト {len(yho):,}点(的中 {yho.sum():,})")

    shrink = lgb.train(
        {"objective": "binary", "metric": "binary_logloss", "verbosity": -1,
         "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 500,
         "random_state": 42},
        lgb.Dataset(Xfit[SHRINK_FEATURES], label=yfit),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(Xval[SHRINK_FEATURES], label=yval)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    p_shrunk = shrink.predict(Xho[SHRINK_FEATURES], num_iteration=shrink.best_iteration)

    print("\n=== 選抜後の較正: 縮小前 vs 縮小後 ===")
    print(f"{'EV閾値':>7} {'':>4} {'点数':>9} {'予測確率':>9} {'実際':>9} {'比':>6} {'回収率':>9}")
    for thr in [1.0, 1.15, 1.3, 1.5, 2.0]:
        for name, p in (("縮小前", p_model_ho), ("縮小後", p_shrunk)):
            ev = p * odds_ho
            m = ev >= thr
            if m.sum() < 50:
                print(f"{thr:7.2f} {name:>4} {m.sum():9,}  (点数不足)")
                continue
            roi = odds_ho[m & (yho == 1)].sum() / m.sum() * 100
            print(f"{thr:7.2f} {name:>4} {m.sum():9,} {p[m].mean()*100:8.3f}% "
                  f"{yho[m].mean()*100:8.3f}% {yho[m].mean()/p[m].mean():5.2f} {roi:8.1f}%")
        print()

    with open(BASE_DIR / args.out, "wb") as f:
        pickle.dump({"model": shrink, "features": SHRINK_FEATURES}, f)
    print(f"保存しました: {args.out}")
    print("\n※ 「比」が1.0に近づけば勝者の呪いは消えている。")
    print("   回収率が100%を超える閾値が現れるかどうかが本番。")


if __name__ == "__main__":
    main()
