"""predictions.csv を元に成績を分析し、EV閾値ごとの回収率をシミュレーションする。

使い方:
    python scripts/backtest.py
    python scripts/backtest.py --strategy FOCUS
    python scripts/backtest.py --ev-thresholds 1.0 1.5 2.0
"""
import argparse
import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PREDICTION_LOG_FILE = BASE_DIR / "predictions.csv"


def load_rows():
    with open(PREDICTION_LOG_FILE, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_ticket_details(row):
    """{ticket: {odds, probability, expected_value, stake}} のJSONをパースする。"""
    try:
        details = json.loads(row.get("ticket_details") or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return details


def summarize(rows, label):
    total = len(rows)
    if total == 0:
        print(f"{label}: データなし")
        return
    hits = sum(1 for r in rows if r.get("is_hit") == "1")
    stake = sum(int(float(r.get("stake") or 0)) for r in rows)
    ret = sum(int(float(r.get("return_amount") or 0)) for r in rows)
    profit = ret - stake
    roi = (ret / stake * 100) if stake else 0
    hit_rate = (hits / total * 100) if total else 0
    print(
        f"{label}: 件数={total} 的中={hits} 的中率={hit_rate:.1f}% "
        f"投資={stake:,}円 払戻={ret:,}円 収支={profit:+,}円 回収率={roi:.1f}%"
    )


def simulate_ev_threshold(rows, thresholds):
    print("\n--- EV閾値シミュレーション (実際のオッズ・結果を使って再集計、賭け金は100円均等と仮定) ---")
    for th in thresholds:
        stake = 0
        ret = 0
        hits = 0
        count = 0
        for r in rows:
            details = parse_ticket_details(r)
            picked = [t for t, d in details.items() if d.get("expected_value", 0) >= th]
            if not picked:
                continue
            count += 1
            stake += len(picked) * 100
            if r.get("result_ticket") in picked:
                hits += 1
                ret += int(float(r.get("result_payout") or 0))
        roi = (ret / stake * 100) if stake else 0
        print(f"EV>={th:.2f}: 予想数={count} 的中={hits} 投資={stake:,}円 払戻={ret:,}円 回収率={roi:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="predictions.csv の成績分析・EV閾値シミュレーション")
    parser.add_argument("--strategy", help="戦略でフィルタ (FOCUS/STANDARD/WIDE)")
    parser.add_argument("--ev-thresholds", nargs="*", type=float, default=[0.8, 1.0, 1.2, 1.5, 2.0])
    args = parser.parse_args()

    if not PREDICTION_LOG_FILE.exists():
        print(f"predictions.csv が見つかりません: {PREDICTION_LOG_FILE}")
        return

    rows = load_rows()
    settled = [r for r in rows if r.get("settled_at")]
    if args.strategy:
        settled = [r for r in settled if r.get("strategy") == args.strategy]

    print(f"対象件数(決着済): {len(settled)} / 全件: {len(rows)}\n")
    summarize(settled, "全体")

    by_strategy = {}
    for r in settled:
        by_strategy.setdefault(r.get("strategy") or "UNKNOWN", []).append(r)
    print()
    for strategy, srows in sorted(by_strategy.items()):
        summarize(srows, strategy)

    simulate_ev_threshold(settled, args.ev_thresholds)


if __name__ == "__main__":
    main()
