"""「イン逃げ(1号艇1着)」に賭ける戦略と、現行の「イン飛び(1号艇を1着にしない)」戦略を
まったく同じレース・同じEV/ケリー条件で比較する。

odds_result_cache.csv (実オッズ全120通り + 実結果) だけを使うのでネットワーク不要。
買い目の母集団と発動条件だけを差し替え、それ以外(EV閾値・点数上限・ケリー)は現行と揃える。

使い方:
    python scripts/nige_vs_jump.py
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
from scripts.train_model import build_features, FEATURES, FEATURES_NO_ST  # noqa: E402
from main import (  # noqa: E402
    add_kelly_stakes,
    MIN_EXPECTED_VALUE, KELLY_FRACTION, STARTING_BANKROLL,
    IN_JUMP_THRESHOLD,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE = BASE_DIR / "odds_result_cache.csv"
TICKET_CAP = 8   # 3アームで揃える(現行STANDARDと同じ)

ALL_120 = [
    f"{a}-{b}-{c}"
    for a in range(1, 7) for b in range(1, 7) for c in range(1, 7)
    if len({a, b, c}) == 3
]


def all_tickets(lead_is_one):
    """lead_is_one=True なら 1-x-y の20通り、False なら 1着が1号艇以外の100通り。"""
    return [t for t in ALL_120 if (t[0] == "1") == lead_is_one]


ORDER = {}  # 着順モデル(2着/3着)とその設定を入れる


def race_ticket_probs(probs, input_df):
    """1レース分の120通りの確率をまとめて計算する。

    main.estimate_ticket_probability は買い目ごとにモデルを呼ぶため 1レース240回の
    推論になり遅い。ここでは given_1st の6通り・(given_1st, given_2nd) の30通りを
    それぞれ1回のバッチ推論にまとめる(結果は同じ)。
    """
    model_2nd, model_3rd, cfg = ORDER["2nd"], ORDER["3rd"], ORDER["cfg"]

    x2 = pd.concat([input_df] * 6, ignore_index=True)
    x2["given_1st"] = list(range(1, 7))
    raw2 = np.asarray(model_2nd.predict(x2[cfg["features_2nd"]]), dtype=float)  # (6, 6)

    pairs = [(a, b) for a in range(1, 7) for b in range(1, 7) if a != b]
    x3 = pd.concat([input_df] * len(pairs), ignore_index=True)
    x3["given_1st"] = [a for a, _ in pairs]
    x3["given_2nd"] = [b for _, b in pairs]
    raw3 = np.asarray(model_3rd.predict(x3[cfg["features_3rd"]]), dtype=float)  # (30, 6)
    raw3_by_pair = {pair: raw3[i] for i, pair in enumerate(pairs)}

    out = {}
    for t in ALL_120:
        a, b, c = (int(x) for x in t.split("-"))
        r2 = raw2[a - 1].copy()
        r2[a - 1] = 0
        p2 = r2[b - 1] / r2.sum() if r2.sum() > 0 else 0

        r3 = raw3_by_pair[(a, b)].copy()
        r3[a - 1] = r3[b - 1] = 0
        p3 = r3[c - 1] / r3.sum() if r3.sum() > 0 else 0

        out[t] = float(probs[a - 1]) * p2 * p3
    return out


def evaluate(tickets, ticket_probs, odds_map, result_ticket, payout):
    """EV閾値・点数上限・ケリーを現行と同じ条件で適用し、1レース分の収支を返す。"""
    enriched = []
    for t in tickets:
        odds = odds_map.get(t)
        if not odds:
            continue
        p = ticket_probs[t]
        ev = p * odds
        if ev >= MIN_EXPECTED_VALUE:
            enriched.append({"ticket": t, "odds": odds, "probability": p, "expected_value": ev})
    enriched.sort(key=lambda x: x["expected_value"], reverse=True)
    enriched = enriched[:TICKET_CAP]
    if not enriched:
        return None

    staked = [t for t in add_kelly_stakes(enriched, STARTING_BANKROLL, KELLY_FRACTION) if t["stake"] > 0]
    if not staked:
        return None

    stake = sum(t["stake"] for t in staked)
    matched = next((t for t in staked if t["ticket"] == result_ticket), None)
    ret = int(matched["stake"] / 100 * payout) if matched else 0
    return {"stake": stake, "return": ret, "hit": matched is not None,
            "n_tickets": len(staked), "odds": matched["odds"] if matched else None,
            "avg_odds": sum(t["odds"] for t in staked) / len(staked)}


def summarize(label, rs, n_races):
    if not rs:
        print(f"{label:22s} 賭け成立0件")
        return
    stake = sum(r["stake"] for r in rs)
    ret = sum(r["return"] for r in rs)
    hits = sum(1 for r in rs if r["hit"])
    roi = ret / stake * 100 if stake else 0
    avg_t = sum(r["n_tickets"] for r in rs) / len(rs)
    print(f"{label:22s} 賭け {len(rs):4d}/{n_races}レース  平均{avg_t:.1f}点  "
          f"的中 {hits:3d} ({hits/len(rs)*100:4.1f}%)  "
          f"投資 {stake:>8,}  払戻 {ret:>8,}  収支 {ret-stake:>+9,}  回収率 {roi:5.1f}%")


def robustness(label, rs):
    """回収率が一部の大穴的中に依存していないか、期間で安定しているかを見る。"""
    if not rs:
        return
    stake = sum(r["stake"] for r in rs)
    ret = sum(r["return"] for r in rs)
    roi = ret / stake * 100 if stake else 0

    hits = sorted((r for r in rs if r["hit"]), key=lambda r: -r["return"])
    top3 = sum(r["return"] for r in hits[:3])
    roi_ex = (ret - top3) / stake * 100 if stake else 0

    rs_sorted = sorted(rs, key=lambda r: r["date"])
    mid = len(rs_sorted) // 2
    halves = []
    for half in (rs_sorted[:mid], rs_sorted[mid:]):
        s = sum(r["stake"] for r in half)
        v = sum(r["return"] for r in half)
        halves.append(v / s * 100 if s else 0)

    print(f"  {label}: 回収率 {roi:.1f}%  / 高配当上位3本を除くと {roi_ex:.1f}%  "
          f"/ 前半 {halves[0]:.1f}% 後半 {halves[1]:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_model_v5.pkl")
    parser.add_argument("--order-suffix", default="v1")
    parser.add_argument("--drop-st", action="store_true",
                        help="st_i を除いた特徴量で推論する(漏洩なしモデル用)")
    args = parser.parse_args()

    features = FEATURES_NO_ST if args.drop_st else FEATURES
    leak = "st_除外(漏洩なし)" if args.drop_st else "st_あり(漏洩)"
    print(f"モデル: {args.model} / 着順モデル: {args.order_suffix} / 特徴量: {leak}\n")

    with open(BASE_DIR / args.model, "rb") as f:
        model = pickle.load(f)
    ORDER["2nd"] = pickle.load(open(BASE_DIR / f"order_model_2nd_{args.order_suffix}.pkl", "rb"))
    ORDER["3rd"] = pickle.load(open(BASE_DIR / f"order_model_3rd_{args.order_suffix}.pkl", "rb"))
    ORDER["cfg"] = pickle.load(open(BASE_DIR / f"order_model_config_{args.order_suffix}.pkl", "rb"))

    with open(CACHE, encoding="utf-8-sig") as f:
        cache = {(r["date"], r["course"], str(int(r["rno"]))): r for r in csv.DictReader(f)}
    print(f"オッズ・結果が揃っているレース: {len(cache)}件")

    with open(BASE_DIR / "training_data.csv", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)
                if (r["date"], r["course"], str(int(r["rno"]))) in cache]
    print(f"特徴量と突合できたレース: {len(rows)}件")
    print(f"期間: {min(r['date'] for r in rows)} 〜 {max(r['date'] for r in rows)}\n")

    df = pd.DataFrame(rows)
    X = build_features(df, features)
    all_probs = model.predict(X)

    jump, nige, both = [], [], []
    n_jump_gate = n_nige_gate = 0
    boat1_wins = 0
    boat1_wins_in_jump_gate = 0

    for i, (row, probs) in enumerate(zip(rows, all_probs)):
        key = (row["date"], row["course"], str(int(row["rno"])))
        c = cache[key]
        result_ticket, payout = c["result_ticket"], int(c["payout"])
        odds_map = {k: float(v) for k, v in json.loads(c["odds_json"]).items()}
        input_df = X.iloc[[i]].reset_index(drop=True)

        in_win = float(probs[0])
        in_jump = 1 - in_win
        actual_boat1_won = result_ticket.startswith("1-")
        boat1_wins += actual_boat1_won

        tp = race_ticket_probs(probs, input_df)

        # 現行: イン飛びが55%以上と読めたレースだけ、1号艇を1着にしない買い目を買う
        if in_jump >= IN_JUMP_THRESHOLD:
            n_jump_gate += 1
            boat1_wins_in_jump_gate += actual_boat1_won
            r = evaluate(all_tickets(False), tp, odds_map, result_ticket, payout)
            if r:
                jump.append({**r, "date": row["date"]})

        # 対案: イン逃げが55%以上と読めたレースだけ、1-x-y を買う
        if in_win >= IN_JUMP_THRESHOLD:
            n_nige_gate += 1
            r = evaluate(all_tickets(True), tp, odds_map, result_ticket, payout)
            if r:
                nige.append({**r, "date": row["date"]})

        # 参考: 発動条件なし・120通り全部からEVで選ぶ
        r = evaluate(ALL_120, tp, odds_map, result_ticket, payout)
        if r:
            both.append({**r, "date": row["date"]})

        if (i + 1) % 200 == 0:
            print(f"  progress: {i+1}/{len(rows)}", flush=True)

    n = len(rows)
    print("=== 前提の確認 ===")
    print(f"1号艇が実際に1着だったレース: {boat1_wins}/{n} ({boat1_wins/n*100:.1f}%)")
    print(f"イン飛び条件(1号艇の勝率<45%と予測)を満たしたレース: {n_jump_gate}件")
    print(f"  → うち実際に1号艇が勝ってしまった: {boat1_wins_in_jump_gate}件 "
          f"({boat1_wins_in_jump_gate/n_jump_gate*100:.1f}%) ※ここは構造上100%ハズレ")
    print(f"イン逃げ条件(1号艇の勝率>=55%と予測)を満たしたレース: {n_nige_gate}件\n")

    print("=== 同一レース・同一EV/ケリー条件での比較 ===")
    print(f"(EV>={MIN_EXPECTED_VALUE} / 最大{TICKET_CAP}点 / {KELLY_FRACTION}ケリー / バンクロール{STARTING_BANKROLL:,}円固定)\n")
    summarize("イン飛び(現行)", jump, n)
    summarize("イン逃げ(対案)", nige, n)
    summarize("両方(条件なし120点)", both, n)

    print("\n=== 頑健性(まぐれ当たり依存・期間安定性) ===")
    robustness("イン飛び(現行)", jump)
    robustness("イン逃げ(対案)", nige)

    for label, rs in (("イン飛び(現行)", jump), ("イン逃げ(対案)", nige)):
        if rs:
            avg = sum(r["avg_odds"] for r in rs) / len(rs)
            print(f"  {label}: 買った舟券の平均オッズ {avg:.1f}倍")


if __name__ == "__main__":
    main()
