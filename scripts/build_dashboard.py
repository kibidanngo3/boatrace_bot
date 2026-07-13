"""predictions.csv から成績ダッシュボード(単一HTML)を生成する。

Bot実行のたびに呼ばれ、docs/index.html を更新する。GitHub Pagesで公開すれば
スマホからいつでも成績を確認できる。外部CDNに依存しない自己完結型HTML。

使い方:
    python scripts/build_dashboard.py
    python scripts/build_dashboard.py --input predictions.csv --output docs/index.html
"""
import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
JST = timezone(timedelta(hours=9))


def safe_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def load_settled(input_path):
    if not input_path.exists():
        return []
    with open(input_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    settled = [r for r in rows if r.get("settled_at") and r.get("date")]
    settled.sort(key=lambda r: (r.get("date", ""), r.get("deadline", "")))
    return settled


def summarize(rows):
    stake = sum(safe_int(r.get("stake")) for r in rows)
    ret = sum(safe_int(r.get("return_amount")) for r in rows)
    hits = sum(1 for r in rows if r.get("is_hit") == "1")
    n = len(rows)
    return {
        "n": n,
        "hits": hits,
        "hit_rate": (hits / n * 100) if n else 0,
        "stake": stake,
        "return": ret,
        "profit": ret - stake,
        "roi": (ret / stake * 100) if stake else 0,
    }


def group_summary(rows, key_fn):
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    out = []
    for key, items in groups.items():
        s = summarize(items)
        s["key"] = key
        out.append(s)
    out.sort(key=lambda x: -x["profit"])
    return out


def build_equity_curve(rows, starting_bankroll=10000):
    """決着順に資金推移を作り、最大ドローダウンも返す。"""
    points = []
    cum = starting_bankroll
    peak = starting_bankroll
    max_dd = 0
    for r in rows:
        cum += safe_int(r.get("profit"))
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
        points.append({
            "label": f"{r.get('date','')[4:6]}/{r.get('date','')[6:8]} {r.get('course','')}{r.get('rno','')}R",
            "value": cum,
            "hit": r.get("is_hit") == "1",
        })
    return points, max_dd


def render(rows, generated_at):
    overall = summarize(rows)
    by_course = group_summary(rows, lambda r: r.get("course", "?"))
    by_strategy = group_summary(rows, lambda r: r.get("strategy", "?"))
    by_date = group_summary(rows, lambda r: r.get("date", "?"))
    by_date.sort(key=lambda x: x["key"], reverse=True)
    curve, max_dd = build_equity_curve(rows)

    history = []
    for r in reversed(rows):
        history.append({
            "date": r.get("date", ""),
            "course": r.get("course", ""),
            "rno": r.get("rno", ""),
            "strategy": r.get("strategy", ""),
            "tickets": r.get("tickets", ""),
            "result": r.get("result_ticket", ""),
            "payout": safe_int(r.get("result_payout")),
            "hit": r.get("is_hit") == "1",
            "stake": safe_int(r.get("stake")),
            "ret": safe_int(r.get("return_amount")),
            "profit": safe_int(r.get("profit")),
        })

    data = {
        "overall": overall,
        "maxDd": max_dd,
        "curve": curve,
        "byCourse": by_course,
        "byStrategy": by_strategy,
        "byDate": by_date,
        "history": history,
        "generatedAt": generated_at,
    }

    return TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))


TEMPLATE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Boatrace Bot 成績</title>
<style>
  :root {
    --bg: #0f1218; --card: #181d26; --line: #262d3a;
    --fg: #e7ecf3; --muted: #94a3b8;
    --up: #34d399; --down: #f87171; --accent: #60a5fa;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #f6f8fb; --card: #fff; --line: #e3e8ef;
      --fg: #111827; --muted: #6b7280;
      --up: #059669; --down: #dc2626; --accent: #2563eb;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 16px; background: var(--bg); color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Hiragino Sans", "Noto Sans JP", sans-serif;
    line-height: 1.6;
  }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 16px; }
  .grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); margin-bottom: 20px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }
  .card .label { color: var(--muted); font-size: 12px; }
  .card .value { font-size: 22px; font-weight: 700; margin-top: 2px; font-variant-numeric: tabular-nums; }
  .up { color: var(--up); } .down { color: var(--down); }
  section { margin-bottom: 24px; }
  h2 { font-size: 15px; margin: 0 0 10px; color: var(--muted); font-weight: 600; }
  .scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; min-width: 520px; }
  th, td { padding: 8px 10px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
  th { color: var(--muted); font-weight: 600; text-align: right; position: sticky; top: 0; background: var(--card); }
  th:first-child, td:first-child { text-align: left; }
  tbody tr:hover { background: rgba(96,165,250,.06); }
  .num { font-variant-numeric: tabular-nums; }
  .pill { display: inline-block; padding: 1px 7px; border-radius: 999px; font-size: 11px; font-weight: 600; }
  .pill.hit { background: rgba(52,211,153,.15); color: var(--up); }
  .pill.miss { background: rgba(148,163,184,.15); color: var(--muted); }
  svg { display: block; width: 100%; height: 220px; }
  .empty { color: var(--muted); padding: 24px; text-align: center; }
  .tickets { color: var(--muted); font-size: 12px; max-width: 260px; overflow: hidden; text-overflow: ellipsis; }
</style>
</head>
<body>
<h1>🚤 Boatrace Bot 成績</h1>
<div class="sub" id="sub"></div>
<div id="app"></div>

<script>
const D = __DATA__;

const yen = n => (n < 0 ? "-" : "") + "¥" + Math.abs(n).toLocaleString();
const signed = n => (n >= 0 ? "+" : "-") + "¥" + Math.abs(n).toLocaleString();
const cls = n => n >= 0 ? "up" : "down";

document.getElementById("sub").textContent = "最終更新 " + D.generatedAt + "（決着済 " + D.overall.n + " レース）";

if (D.overall.n === 0) {
  document.getElementById("app").innerHTML = '<div class="empty">まだ決着したレースがありません。</div>';
} else {
  const o = D.overall;
  const cards = `
    <div class="grid">
      <div class="card"><div class="label">収支</div><div class="value ${cls(o.profit)}">${signed(o.profit)}</div></div>
      <div class="card"><div class="label">回収率</div><div class="value ${o.roi >= 100 ? "up" : "down"}">${o.roi.toFixed(1)}%</div></div>
      <div class="card"><div class="label">的中率</div><div class="value">${o.hit_rate.toFixed(1)}%<span style="font-size:13px;color:var(--muted)"> (${o.hits}/${o.n})</span></div></div>
      <div class="card"><div class="label">投資 / 払戻</div><div class="value" style="font-size:16px">${yen(o.stake)} / ${yen(o.return)}</div></div>
      <div class="card"><div class="label">最大ドローダウン</div><div class="value down">${yen(D.maxDd)}</div></div>
    </div>`;

  // 資金推移(自前SVG。外部ライブラリ不要)
  const c = D.curve;
  const W = 800, H = 220, P = 34;
  const vals = c.map(p => p.value);
  const lo = Math.min(...vals, 10000), hi = Math.max(...vals, 10000);
  const pad = (hi - lo) * 0.12 || 1000;
  const yMin = lo - pad, yMax = hi + pad;
  const x = i => P + (W - P * 2) * (c.length === 1 ? 0.5 : i / (c.length - 1));
  const y = v => H - P - (H - P * 2) * ((v - yMin) / (yMax - yMin));
  const line = c.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.value).toFixed(1)}`).join(" ");
  const area = line + ` L${x(c.length - 1).toFixed(1)},${y(yMin)} L${x(0).toFixed(1)},${y(yMin)} Z`;
  const base = y(10000);
  const dots = c.filter(p => p.hit).map((p, _i) => {
    const i = c.indexOf(p);
    return `<circle cx="${x(i).toFixed(1)}" cy="${y(p.value).toFixed(1)}" r="3.5" fill="var(--up)"><title>${p.label}</title></circle>`;
  }).join("");
  const chart = `
    <section>
      <h2>資金推移（元手 ¥10,000 / 緑点は的中）</h2>
      <div class="card">
        <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
          <defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="var(--accent)" stop-opacity=".28"/>
            <stop offset="100%" stop-color="var(--accent)" stop-opacity="0"/>
          </linearGradient></defs>
          <line x1="${P}" y1="${base}" x2="${W - P}" y2="${base}" stroke="var(--line)" stroke-dasharray="4 4"/>
          <text x="${P}" y="${base - 6}" fill="var(--muted)" font-size="11">元手 ¥10,000</text>
          <path d="${area}" fill="url(#g)"/>
          <path d="${line}" fill="none" stroke="var(--accent)" stroke-width="2"/>
          ${dots}
          <text x="${P}" y="16" fill="var(--muted)" font-size="11">${yen(Math.round(yMax))}</text>
          <text x="${P}" y="${H - 8}" fill="var(--muted)" font-size="11">${yen(Math.round(yMin))}</text>
        </svg>
      </div>
    </section>`;

  const table = (title, rows, keyLabel) => `
    <section>
      <h2>${title}</h2>
      <div class="card scroll">
        <table>
          <thead><tr>
            <th>${keyLabel}</th><th>賭数</th><th>的中</th><th>的中率</th><th>投資</th><th>払戻</th><th>収支</th><th>回収率</th>
          </tr></thead>
          <tbody>
            ${rows.map(r => `
              <tr>
                <td>${r.key}</td>
                <td class="num">${r.n}</td>
                <td class="num">${r.hits}</td>
                <td class="num">${r.hit_rate.toFixed(1)}%</td>
                <td class="num">${yen(r.stake)}</td>
                <td class="num">${yen(r.return)}</td>
                <td class="num ${cls(r.profit)}">${signed(r.profit)}</td>
                <td class="num ${r.roi >= 100 ? "up" : "down"}">${r.roi.toFixed(0)}%</td>
              </tr>`).join("")}
          </tbody>
        </table>
      </div>
    </section>`;

  const history = `
    <section>
      <h2>全履歴（新しい順）</h2>
      <div class="card scroll">
        <table>
          <thead><tr>
            <th>日付</th><th>レース</th><th>戦略</th><th>買い目</th><th>結果</th><th>払戻</th><th>収支</th><th></th>
          </tr></thead>
          <tbody>
            ${D.history.map(h => `
              <tr>
                <td class="num">${h.date.slice(4,6)}/${h.date.slice(6,8)}</td>
                <td>${h.course} ${h.rno}R</td>
                <td>${h.strategy}</td>
                <td class="tickets">${h.tickets}</td>
                <td class="num">${h.result}</td>
                <td class="num">${h.payout ? yen(h.payout) : "-"}</td>
                <td class="num ${cls(h.profit)}">${signed(h.profit)}</td>
                <td><span class="pill ${h.hit ? "hit" : "miss"}">${h.hit ? "的中" : "-"}</span></td>
              </tr>`).join("")}
          </tbody>
        </table>
      </div>
    </section>`;

  document.getElementById("app").innerHTML =
    cards + chart +
    table("日別", D.byDate, "日付") +
    table("会場別", D.byCourse, "会場") +
    table("戦略別", D.byStrategy, "戦略") +
    history;
}
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="predictions.csv")
    parser.add_argument("--output", default="docs/index.html")
    args = parser.parse_args()

    input_path = BASE_DIR / args.input
    output_path = BASE_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_settled(input_path)
    generated_at = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    output_path.write_text(render(rows, generated_at), encoding="utf-8")
    print(f"ダッシュボードを生成しました: {output_path} (決着済 {len(rows)}レース)")


if __name__ == "__main__":
    main()
