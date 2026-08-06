"""舟券種間(プール間)の価格整合性を検証する。

パリミュチュエル方式では舟券の種類ごとに資金プールが分かれる(単勝プール・3連単プールは別物)。
同じレース結果に対する賭け手の「読み」がプールごとに系統的に異なれば、
一方のプールの情報をもう一方のプールの賭けに使うことで市場平均(回収率70〜79%)を
上回れる可能性がある。これはモデルの予測力に頼らない、算数的な整合性チェックである。

検証する2つの整合性:
  1. 単勝プール vs 3連単プールから周辺化した「1着になる確率」
  2. 2連複プール vs 3連複プールから周辺化した「その2艇が上位に絡む確率」

使い方:
    python scripts/cross_market_arbitrage.py
"""
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
TAKEOUT_ROI = 75.0  # 控除率25% = 優位性ゼロなら期待回収率75%
BOATS = [1, 2, 3, 4, 5, 6]
PAIRS = list(itertools.combinations(BOATS, 2))


def normalize(d):
    s = sum(d.values())
    if s <= 0:
        return {k: 0.0 for k in d}
    return {k: v / s for k, v in d.items()}


def marginal_win_from_3t(odds3t):
    """3連単オッズから「その艇が1着になる確率」を周辺化する(正規化済み)。"""
    raw = {}
    for b in BOATS:
        s = 0.0
        for x, y in itertools.permutations([n for n in BOATS if n != b], 2):
            o = odds3t.get(f"{b}-{x}-{y}")
            if o:
                s += 1 / float(o)
        raw[b] = s
    return normalize(raw)


def marginal_pair_from_3f(odds3f):
    """3連複オッズから「その2艇が(3艇の組に)絡む確率」を周辺化する(正規化済み)。"""
    raw = {}
    for a, b in PAIRS:
        s = 0.0
        for c in BOATS:
            if c in (a, b):
                continue
            key = "-".join(str(n) for n in sorted([a, b, c]))
            o = odds3f.get(key)
            if o:
                s += 1 / float(o)
        raw[(a, b)] = s
    return normalize(raw)


def load_data():
    with open(BASE_DIR / "odds_result_cache.csv", encoding="utf-8-sig") as f:
        base = {
            (r["date"], r["course"], str(int(r["rno"]))): r
            for r in csv.DictReader(f) if r.get("date")
        }
    with open(BASE_DIR / "odds_cross_cache.csv", encoding="utf-8-sig") as f:
        cross = {
            (r["date"], r["course"], str(int(r["rno"]))): r
            for r in csv.DictReader(f) if r.get("date")
        }
    keys = sorted(set(base) & set(cross))
    races = []
    for k in keys:
        b, c = base[k], cross[k]
        try:
            odds3t = {t: float(o) for t, o in json.loads(b["odds_json"]).items()}
            tan = {int(k2): float(v) for k2, v in json.loads(c["oddstan_json"]).items()}
            odds2f = {t: float(o) for t, o in json.loads(c["odds2f_json"]).items()}
            odds3f = {t: float(o) for t, o in json.loads(c["odds3f_json"]).items()}
        except (json.JSONDecodeError, ValueError):
            continue
        if len(tan) != 6 or len(odds2f) != 15 or len(odds3f) != 20:
            continue
        won = b["result_ticket"].split("-")
        races.append({
            "date": b["date"], "course": b["course"], "rno": b["rno"],
            "odds3t": odds3t, "tan": tan, "odds2f": odds2f, "odds3f": odds3f,
            "winner": int(won[0]), "placed2": {int(won[0]), int(won[1])},
        })
    return races


def bootstrap_roi(payouts, n, n_boot=20000, seed=42):
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    returns[: len(payouts)] = payouts
    idx = rng.integers(0, n, size=(n_boot, n))
    return returns[idx].sum(axis=1) / n * 100


def report_bucket(label, payouts, n, dates=None):
    if n == 0:
        print(f"  {label}: 該当なし")
        return
    roi = sum(payouts) / n * 100
    print(f"  {label}: n={n:,} 的中={len(payouts)} 回収率={roi:.1f}%", end="")
    if n >= 20:
        boot = bootstrap_roi(payouts, n)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p_break_even = (boot < 100).mean()
        print(f"  [95%CI {lo:.1f}%〜{hi:.1f}%, 赤字確率{p_break_even:.1%}]", end="")
    print()


def main():
    races = load_data()
    print(f"対象: {len(races):,}レース (odds_result_cache × odds_cross_cache の共通分)\n")
    if len(races) < 30:
        print("サンプルが少なすぎるため参考値のみ。")

    # ---------- 1. 単勝プール vs 3連単プールの周辺化確率 ----------
    print("=" * 70)
    print("【1】単勝プール vs 3連単プール周辺化: 1着確率の較正")
    print("=" * 70)

    logloss_tan, logloss_3t = [], []
    calib_tan = defaultdict(lambda: {"n": 0, "hit": 0})
    calib_3t = defaultdict(lambda: {"n": 0, "hit": 0})
    diffs = []  # (diff, tan_odds, boat, hit, date)

    for race in races:
        q_tan = normalize({b: 1 / race["tan"][b] for b in BOATS})
        q_3t = marginal_win_from_3t(race["odds3t"])
        winner = race["winner"]

        eps = 1e-9
        logloss_tan.append(-np.log(max(q_tan[winner], eps)))
        logloss_3t.append(-np.log(max(q_3t[winner], eps)))

        for b in BOATS:
            calib_tan[round(q_tan[b], 1)]["n"] += 1
            calib_tan[round(q_tan[b], 1)]["hit"] += int(b == winner)
            calib_3t[round(q_3t[b], 1)]["n"] += 1
            calib_3t[round(q_3t[b], 1)]["hit"] += int(b == winner)
            diffs.append((q_3t[b] - q_tan[b], race["tan"][b], b == winner, race["date"]))

    print(f"\nLogLoss: 単勝プール={np.mean(logloss_tan):.4f} / 3連単プール周辺化={np.mean(logloss_3t):.4f}")
    print("(値が小さいほど的中確率の予測が正確。差が大きいほどプール間の情報の質に差がある)\n")

    print("較正(予測確率帯ごとの実際の的中率):")
    print(f"{'確率帯':>8} {'単勝:予測':>10} {'単勝:実際':>10} {'n':>6}   {'3連単周辺:予測':>14} {'3連単周辺:実際':>14} {'n':>6}")
    for p in sorted(set(list(calib_tan.keys()) + list(calib_3t.keys()))):
        t, m = calib_tan.get(p, {"n": 0, "hit": 0}), calib_3t.get(p, {"n": 0, "hit": 0})
        t_actual = t["hit"] / t["n"] if t["n"] else float("nan")
        m_actual = m["hit"] / m["n"] if m["n"] else float("nan")
        print(f"{p*100:6.1f}%  {p*100:9.1f}% {t_actual*100:9.1f}% {t['n']:6}   "
              f"{p*100:13.1f}% {m_actual*100:13.1f}% {m['n']:6}")

    # ---------- 2. 乖離ベースの裁定ベット ----------
    print("\n" + "=" * 70)
    print("【2】プール間の乖離を使った単勝ベット")
    print("=" * 70)
    print("q_3t(3連単周辺化) が q_tan(単勝) より大きく上振れしている艇に単勝で賭けたら?\n")

    diff_arr = np.array([d[0] for d in diffs])
    odds_arr = np.array([d[1] for d in diffs])
    hit_arr = np.array([d[2] for d in diffs])
    date_arr = np.array([d[3] for d in diffs])

    for thr in (0.03, 0.05, 0.08, 0.12):
        m = diff_arr >= thr
        n = int(m.sum())
        payouts = odds_arr[m & hit_arr]
        report_bucket(f"diff>={thr:.2f} (3連単周辺確率が単勝より+{thr*100:.0f}pt以上高い)", payouts, n)

    print("\n逆方向(単勝プールの方が3連単周辺化より強気)も参考に見る:")
    for thr in (0.03, 0.05, 0.08):
        m = diff_arr <= -thr
        n = int(m.sum())
        payouts = odds_arr[m & hit_arr]
        report_bucket(f"diff<=-{thr:.2f}", payouts, n)

    # ---------- 3. 2連複プール vs 3連複プール周辺化 ----------
    print("\n" + "=" * 70)
    print("【3】2連複プール vs 3連複プール周辺化: ペア確率の較正")
    print("=" * 70)

    logloss_2f, logloss_3f = [], []
    pair_diffs = []  # (diff, odds2f, hit(両艇が3着以内に絡む), date)

    for race in races:
        q_2f = normalize({p: 1 / race["odds2f"]["-".join(map(str, p))] for p in PAIRS})
        q_3f = marginal_pair_from_3f(race["odds3f"])
        placed = race["placed2"]  # 1・2着の艇番セット(3連複的中の必要条件の一部として簡易化)

        eps = 1e-9
        for p in PAIRS:
            hit = p[0] in placed and p[1] in placed  # 両艇が1・2着を占めたか(2連複的中条件と同一)
            if hit:
                logloss_2f.append(-np.log(max(q_2f[p], eps)))
                logloss_3f.append(-np.log(max(q_3f[p], eps)))
            key = "-".join(map(str, p))
            pair_diffs.append((q_3f[p] - q_2f[p], race["odds2f"][key], hit, race["date"]))

    print(f"\n的中ペアに対する平均驚き度(-log尤度、小さいほど的中を高く予測できていた):")
    print(f"  2連複プール={np.mean(logloss_2f):.4f} / 3連複プール周辺化={np.mean(logloss_3f):.4f}\n")

    pd_arr = np.array([d[0] for d in pair_diffs])
    po_arr = np.array([d[1] for d in pair_diffs])
    ph_arr = np.array([d[2] for d in pair_diffs])

    print("乖離ベースの2連複ベット (3連複周辺化が2連複より強気なペアを買う):")
    for thr in (0.02, 0.04, 0.06):
        m = pd_arr >= thr
        n = int(m.sum())
        payouts = po_arr[m & ph_arr]
        report_bucket(f"diff>={thr:.2f}", payouts, n)

    print(f"\n(参考: 母集団サイズ 単勝分析={len(diffs):,}艇 / 2連複分析={len(pair_diffs):,}ペア)")
    print(f"控除率25%の場合、優位性ゼロなら期待回収率は約{TAKEOUT_ROI:.0f}%に収束する。")


if __name__ == "__main__":
    main()
