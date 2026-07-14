"""市場(オッズ)そのものの偏りを調べる。モデルは一切使わない。

競馬・競艇では「大穴が過剰に買われ、本命が過小評価される」(favorite-longshot bias)
という現象が知られている。もし本当なら、オッズ帯ごとに無作為に賭けたときの回収率は
一定(=控除率どおりの約75%)にならず、低オッズほど高く、高オッズほど低くなるはず。

これは我々のモデルの成否とは無関係に成り立つ市場の性質なので、
「どのオッズ帯なら勝負になるのか」という土俵選びの材料になる。

使い方:
    python scripts/market_bias.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

BUCKETS = [
    (1, 10), (10, 20), (20, 50), (50, 100),
    (100, 200), (200, 500), (500, 1000), (1000, 10 ** 9),
]


def main():
    stats = defaultdict(lambda: {"n": 0, "hits": 0, "ret": 0.0})
    n_races = 0

    with open(BASE_DIR / "odds_result_cache.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            n_races += 1
            won = r["result_ticket"]
            for ticket, odds in json.loads(r["odds_json"]).items():
                odds = float(odds)
                for lo, hi in BUCKETS:
                    if lo <= odds < hi:
                        s = stats[(lo, hi)]
                        s["n"] += 1
                        if ticket == won:
                            s["hits"] += 1
                            s["ret"] += odds
                        break

    print(f"対象: {n_races:,}レース\n")
    print("オッズ帯ごとに、その帯の舟券を全部100円ずつ買ったらどうなるか")
    print("(モデルは使っていない。純粋な市場の性質)\n")
    print(f"{'オッズ帯':>14} {'点数':>9} {'的中':>6} {'実際の的中率':>11} "
          f"{'市場の暗黙確率':>13} {'回収率':>8}")

    for lo, hi in BUCKETS:
        s = stats[(lo, hi)]
        if s["n"] == 0:
            continue
        actual = s["hits"] / s["n"]
        roi = s["ret"] / s["n"] * 100
        # その帯の平均オッズから市場が暗黙に置いている確率
        label = f"{lo}-{hi if hi < 10**9 else '∞'}倍"
        # 暗黙確率は 1/odds の平均で近似する
        implied = np.mean([1 / o for o in [(lo + min(hi, lo * 3)) / 2]])
        print(f"{label:>14} {s['n']:9,} {s['hits']:6,} {actual*100:10.3f}% "
              f"{implied*100:12.3f}% {roi:7.1f}%")

    print("\n※ 回収率が低オッズ帯で高く、高オッズ帯で低いなら favorite-longshot bias。")
    print("   その場合、大穴を選ぶ戦略は市場構造そのものに逆らっていることになる。")


if __name__ == "__main__":
    main()
