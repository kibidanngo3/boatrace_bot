import os
import csv
import json
import pandas as pd
import numpy as np
import pickle
import re
import requests
import time
import traceback
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime, timedelta, timezone

# ==========================================
# 設定
# ==========================================
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
JST = timezone(timedelta(hours=9), 'JST')

# 通知先チャンネルの分割。未設定のものは DISCORD_WEBHOOK_URL にフォールバックするため、
# Secretsを追加しなければ従来どおり全て同じチャンネルに流れる。
#   predict  : 投資チャンス(買い目)の通知
#   stats    : 日次・週次・月次の成績サマリー
#   system   : クラッシュ・スケジュール取得失敗・モデル更新
#   schedule : 締切順ダイジェスト
WEBHOOKS = {
    "predict": os.environ.get("DISCORD_WEBHOOK_URL_PREDICT") or DISCORD_WEBHOOK_URL,
    "stats": os.environ.get("DISCORD_WEBHOOK_URL_STATS") or DISCORD_WEBHOOK_URL,
    "system": os.environ.get("DISCORD_WEBHOOK_URL_SYSTEM") or DISCORD_WEBHOOK_URL,
    "schedule": os.environ.get("DISCORD_WEBHOOK_URL_SCHEDULE") or DISCORD_WEBHOOK_URL,
}

# パスの自動解決：GitHub Actions等の環境でも確実にファイルを見つける
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_model_v5.pkl"
CONFIG_PATH = BASE_DIR / "model_config_v5.pkl"

# 通知済みログファイル (スクリプトと同じ場所に作成)
LOG_FILE = BASE_DIR / "notified_races.log"
PREDICTION_LOG_FILE = BASE_DIR / "predictions.csv"
STATE_FILE = BASE_DIR / "bot_state.json"
STAKE_PER_TICKET = 100  # 舟券の購入単位 (100円単位) / ケリー計算後の丸め単位
NOTIFIED_LOG_KEEP_DAYS = 2
OPERATING_HOUR_START = 7   # 07:00 JST (モーニング競走を考慮)
OPERATING_HOUR_END = 22    # 22:00 JST (ナイター開催を考慮)
SCHEDULE_FAILURE_ALERT_THRESHOLD = 3
SCHEDULE_FAILURE_ALERT_INTERVAL = 3

# ヘルスチェック: 「Actionsはsuccessなのに実は壊れている」サイレント故障を検知する。
# 過去に bot-state が一度も作られず実績が蓄積されない障害や、連敗カウントが
# 残り続けてケリー係数が半減しっぱなしになる障害を、誰も気づかないまま踏んでいる。
HEALTH_NO_BET_HOURS = 6          # 稼働時間中にこの時間ベットが1件も出なければ警告
HEALTH_UNSETTLED_HOURS = 3       # 締切からこの時間が過ぎても決着しない予想が滞留したら警告
HEALTH_UNSETTLED_COUNT = 3       # 上記の滞留がこの件数を超えたら警告
HEALTH_ALERT_COOLDOWN_HOURS = 6  # 同じ種類の警告を連投しない間隔

STARTING_BANKROLL = 10000  # 元手資金 (円)
KELLY_FRACTION = 0.25      # 1/4ケリー (モデル誤差を考慮して保守的に)
MAX_RACE_STAKE_RATIO = 0.10  # 1レースあたりの賭け金上限 (バンクロールに対する比率)
DAILY_LOSS_LIMIT_RATIO = 0.20    # その日の損失がバンクロールの20%に達したら以降ベット停止

PREDICTION_LOG_FIELDS = [
    "run_at",
    "race_id",
    "date",
    "course",
    "rno",
    "deadline",
    "strategy",
    "in_win_prob",
    "in_jump_prob",
    "top1_boat",
    "top1_prob",
    "top2_boat",
    "top2_prob",
    "top3_boat",
    "top3_prob",
    "ticket_count",
    "max_expected_value",
    "tickets",
    "ticket_details",
    "bankroll_at_bet",
    "reason",
    "discord_message_id",  # 確定後にその予想通知へ結果を追記するために保持する
    "result_ticket",
    "result_payout",
    "is_hit",
    "stake",
    "return_amount",
    "profit",
    "roi",
    "settled_at",
]

EXTRA_FEATURE_FIELDS = []
for boat_no in range(1, 7):
    EXTRA_FEATURE_FIELDS.extend([
        f"avg_st_{boat_no}",
        f"national_win_rate_{boat_no}",
        f"national_2_rate_{boat_no}",
        f"national_3_rate_{boat_no}",
        f"local_win_rate_{boat_no}",
        f"local_2_rate_{boat_no}",
        f"local_3_rate_{boat_no}",
        f"motor_no_{boat_no}",
        f"motor_2_rate_{boat_no}",
        f"motor_3_rate_{boat_no}",
        f"boat_no_{boat_no}",
        f"boat_2_rate_{boat_no}",
        f"boat_3_rate_{boat_no}",
        f"weight_{boat_no}",
        f"tilt_{boat_no}",
        f"parts_count_{boat_no}",
        f"parts_{boat_no}",
    ])
PREDICTION_LOG_FIELDS.extend(EXTRA_FEATURE_FIELDS)

IN_JUMP_THRESHOLD = 0.55
FOCUS_TOP_THRESHOLD = 0.35
STANDARD_TOP_THRESHOLD = 0.25
MIN_EXPECTED_VALUE = 1.15
MAX_TICKET_COUNT = {
    "FOCUS": 4,
    "STANDARD": 8,
    "WIDE": 12,
}

# ==========================================
# 重複通知防止ロジック
# ==========================================
def is_already_notified(race_id):
    if not LOG_FILE.exists():
        return False
    with open(LOG_FILE, "r") as f:
        notified_races = f.read().splitlines()
    return race_id in notified_races

def save_notified_race(race_id):
    with open(LOG_FILE, "a") as f:
        f.write(race_id + "\n")

def prune_notified_races(now_jst, keep_days=NOTIFIED_LOG_KEEP_DAYS):
    """race_id (YYYYMMDD_場名_R番号) の日付部分を見て、古いレースIDを間引く"""
    if not LOG_FILE.exists():
        return
    cutoff = (now_jst.date() - timedelta(days=keep_days)).strftime("%Y%m%d")
    with open(LOG_FILE, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    kept = [line for line in lines if line[:8] >= cutoff]
    if len(kept) != len(lines):
        with open(LOG_FILE, "w") as f:
            f.write("\n".join(kept) + ("\n" if kept else ""))
        print(f"  Pruned notified_races.log: {len(lines)} -> {len(kept)}")

# ==========================================
# 実行間で持ち越す状態 (連続失敗カウント・サマリー送信済みフラグ)
# ==========================================
def load_state():
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)

def save_prediction_log(race_id, race, result, run_at, discord_message_id=None):
    row = {
        "discord_message_id": discord_message_id or "",
        "run_at": run_at.isoformat(),
        "race_id": race_id,
        "date": run_at.strftime("%Y%m%d"),
        "course": result["場名"],
        "rno": race["rno"],
        "deadline": result["締切"],
        "strategy": result["戦略"],
        "in_win_prob": f"{result['1号艇勝率']:.6f}",
        "in_jump_prob": f"{result['イン飛び率']:.6f}",
        "top1_boat": result["1位"][0],
        "top1_prob": f"{result['1位'][1]:.6f}",
        "top2_boat": result["2位"][0],
        "top2_prob": f"{result['2位'][1]:.6f}",
        "top3_boat": result["3位"][0],
        "top3_prob": f"{result['3位'][1]:.6f}",
        "ticket_count": result["点数"],
        "max_expected_value": "" if result["期待値MAX"] is None else f"{result['期待値MAX']:.6f}",
        "tickets": result["買い目"],
        "ticket_details": json.dumps(result.get("買い目内訳", {}), ensure_ascii=False),
        "bankroll_at_bet": result.get("バンクロール", ""),
        "reason": result["根拠"],
    }
    feature_data = result.get("特徴量", {})
    for field in EXTRA_FEATURE_FIELDS:
        value = feature_data.get(field, "")
        if isinstance(value, float):
            row[field] = f"{value:.6f}"
        else:
            row[field] = value

    normalize_prediction_log_header()
    file_exists = PREDICTION_LOG_FILE.exists()
    with open(PREDICTION_LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def normalize_prediction_log_header():
    if not PREDICTION_LOG_FILE.exists():
        return

    with open(PREDICTION_LOG_FILE, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    with open(PREDICTION_LOG_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_LOG_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def settle_prediction_logs(scraper, now_jst, state):
    if not PREDICTION_LOG_FILE.exists():
        return 0

    with open(PREDICTION_LOG_FILE, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return 0

    result_cache = {}
    settled_count = 0
    for row in rows:
        if row.get("settled_at"):
            continue

        race_dt = _prediction_deadline_datetime(row)
        if race_dt and now_jst < race_dt + timedelta(minutes=10):
            continue

        course = row.get("course", "")
        try:
            rno = int(row.get("rno", "0"))
        except ValueError:
            continue
        date_str = row.get("date", "")
        cache_key = (course, rno, date_str)

        if cache_key not in result_cache:
            result_cache[cache_key] = scraper.fetch_race_result(course, rno, date_str)
        result = result_cache[cache_key]
        if not result:
            continue

        try:
            ticket_details = json.loads(row.get("ticket_details") or "{}")
        except (json.JSONDecodeError, TypeError):
            ticket_details = {}

        stake = sum(detail.get("stake", 0) for detail in ticket_details.values())
        is_hit = result["ticket"] in ticket_details
        matched_stake = ticket_details.get(result["ticket"], {}).get("stake", 0)
        return_amount = int(matched_stake / STAKE_PER_TICKET * result["payout"]) if is_hit else 0
        profit = return_amount - stake
        roi = (return_amount / stake) if stake else 0

        state["current_bankroll"] = state.get("current_bankroll", STARTING_BANKROLL) + profit

        row.update({
            "result_ticket": result["ticket"],
            "result_payout": result["payout"],
            "is_hit": "1" if is_hit else "0",
            "stake": stake,
            "return_amount": return_amount,
            "profit": profit,
            "roi": f"{roi:.6f}",
            "settled_at": now_jst.isoformat(),
        })
        settled_count += 1

        # 元の予想通知に結果を追記する(通知を見返すだけで的中/ハズレが分かるように)
        append_result_to_prediction_notice(row, result, is_hit, stake, return_amount, profit)

    if settled_count:
        with open(PREDICTION_LOG_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=PREDICTION_LOG_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Settled prediction logs: {settled_count}")
    return settled_count

def _alert_on_cooldown(state, key, now_jst):
    """同じ警告を連投しないためのクールダウン判定。出してよければFalseを返す。"""
    last = state.get(f"health_alert_{key}")
    if not last:
        return False
    try:
        last_dt = datetime.fromisoformat(last)
    except ValueError:
        return False
    return (now_jst - last_dt) < timedelta(hours=HEALTH_ALERT_COOLDOWN_HOURS)


def run_health_checks(state, now_jst, scrape_ok):
    """Actionsがsuccessでも中身が壊れているケースを検知して #システム に警告する。

    ここで見ているのは「静かに壊れる」種類の異常だけで、クラッシュ系は既に
    fatal error 通知でカバーされている。
    """
    alerts = []
    rows = _load_prediction_rows()

    # (1) 実績の消失: predictions.csv の行数が前回より減った
    #     = 状態の永続化が壊れて履歴がリセットされた疑い(過去に実際に起きた障害)
    prev_count = state.get("health_prediction_rows")
    current_count = len(rows)
    if prev_count is not None and current_count < prev_count:
        alerts.append(
            f"🗂️ **予想ログが減っています**（{prev_count}行 → {current_count}行）\n"
            f"bot-stateブランチからの復元に失敗し、実績が失われている可能性があります。"
        )
    state["health_prediction_rows"] = current_count

    # (2) 稼働時間中なのに長時間ベットが出ていない
    #     = スクレイピング破損・モデル異常・条件が厳しすぎる等の兆候
    if scrape_ok and OPERATING_HOUR_START <= now_jst.hour < OPERATING_HOUR_END:
        last_bet = state.get("health_last_bet_at")
        if last_bet:
            try:
                last_bet_dt = datetime.fromisoformat(last_bet)
                idle_hours = (now_jst - last_bet_dt).total_seconds() / 3600
                if idle_hours >= HEALTH_NO_BET_HOURS and not _alert_on_cooldown(state, "no_bet", now_jst):
                    alerts.append(
                        f"🕳️ **{idle_hours:.1f}時間ベットが出ていません**\n"
                        f"最終ベット: {last_bet_dt.strftime('%m/%d %H:%M')}\n"
                        f"スクレイピングの破損やモデルの異常が疑われます（相場次第で正常な場合もあります）。"
                    )
                    state["health_alert_no_bet"] = now_jst.isoformat()
            except ValueError:
                pass

    # (3) 決着待ちの滞留: 締切を大きく過ぎても settled_at が埋まらない
    #     = 結果ページの構造変更などで決着処理が回っていない疑い
    stale = []
    for row in rows:
        if row.get("settled_at"):
            continue
        deadline = _prediction_deadline_datetime(row)
        if deadline and (now_jst - deadline) > timedelta(hours=HEALTH_UNSETTLED_HOURS):
            stale.append(row)
    if len(stale) >= HEALTH_UNSETTLED_COUNT and not _alert_on_cooldown(state, "unsettled", now_jst):
        sample = "、".join(f"{r.get('course')}{r.get('rno')}R" for r in stale[:3])
        alerts.append(
            f"⏳ **決着処理が滞留しています**（{len(stale)}件）\n"
            f"例: {sample}\n"
            f"結果ページの取得に失敗している可能性があります。"
        )
        state["health_alert_unsettled"] = now_jst.isoformat()

    # (4) バンクロールの枯渇
    bankroll = state.get("current_bankroll", STARTING_BANKROLL)
    if bankroll < STAKE_PER_TICKET and not _alert_on_cooldown(state, "bankroll", now_jst):
        alerts.append(
            f"💸 **バンクロールが枯渇しました**（残り {bankroll:,}円）\n"
            f"最小賭け金 {STAKE_PER_TICKET}円 を下回ったため、以降ベットできません。"
        )
        state["health_alert_bankroll"] = now_jst.isoformat()

    for alert in alerts:
        send_discord_message(f"⚠️ **ヘルスチェック警告**\n\n{alert}", "health check", channel="system")

    if alerts:
        print(f"  ⚠️ Health check raised {len(alerts)} alert(s)")
    return len(alerts)


RESULT_COLOR_HIT = 0x2ECC71
RESULT_COLOR_MISS = 0x7F8C8D

def append_result_to_prediction_notice(row, result, is_hit, stake, return_amount, profit):
    """確定した予想の元通知(Embed)を編集し、結果を追記する。

    予想時にWebhookから受け取ったメッセージIDを使って元メッセージを書き換えるので、
    タイムラインを見返すだけで的中/ハズレと収支が分かる。IDが無い(旧データや
    送信失敗)場合は何もしない。
    """
    message_id = row.get("discord_message_id")
    if not message_id:
        return

    try:
        ticket_details = json.loads(row.get("ticket_details") or "{}")
    except (json.JSONDecodeError, TypeError):
        ticket_details = {}

    if is_hit:
        head = f"✅ **的中！** `{result['ticket']}` {result['payout']:,}円"
        color = RESULT_COLOR_HIT
    else:
        head = f"❌ ハズレ（結果 `{result['ticket']}`）"
        color = RESULT_COLOR_MISS

    profit_text = f"+{profit:,}" if profit >= 0 else f"{profit:,}"
    result_value = (
        f"{head}\n"
        f"投資 `{stake:,}円` / 払戻 `{return_amount:,}円` / 収支 **`{profit_text}円`**"
    )

    ticket_lines = []
    for ticket, detail in ticket_details.items():
        mark = "🎯" if ticket == result["ticket"] else "・"
        ticket_lines.append(
            f"{mark} `{ticket}` ¥{detail.get('stake', 0):,}｜{detail.get('odds', 0):.1f}倍"
        )

    embed = {
        "title": f"{'✅' if is_hit else '❌'} {row.get('course')} {row.get('rno')}R｜締切 {row.get('deadline')}",
        "description": "\n".join(ticket_lines),
        "color": color,
        "fields": [
            {"name": "結果", "value": result_value, "inline": False},
        ],
        "footer": {"text": f"{row.get('strategy')}｜確定済み"},
    }
    edit_discord_embed(message_id, embed, f"result {row.get('race_id')}")

def _load_prediction_rows():
    if not PREDICTION_LOG_FILE.exists():
        return []
    with open(PREDICTION_LOG_FILE, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def _summarize_rows(settled_rows, title):
    if not settled_rows:
        return None

    total_predictions = len(settled_rows)
    hits = sum(1 for row in settled_rows if row.get("is_hit") == "1")
    stake = sum(_safe_int(row.get("stake")) for row in settled_rows)
    return_amount = sum(_safe_int(row.get("return_amount")) for row in settled_rows)
    profit = return_amount - stake
    roi = (return_amount / stake * 100) if stake else 0
    hit_rate = (hits / total_predictions * 100) if total_predictions else 0

    by_strategy = {}
    for row in settled_rows:
        strategy = row.get("strategy") or "UNKNOWN"
        item = by_strategy.setdefault(strategy, {"count": 0, "hits": 0, "stake": 0, "return": 0})
        item["count"] += 1
        item["hits"] += 1 if row.get("is_hit") == "1" else 0
        item["stake"] += _safe_int(row.get("stake"))
        item["return"] += _safe_int(row.get("return_amount"))

    strategy_lines = []
    for strategy, item in sorted(by_strategy.items()):
        strategy_roi = (item["return"] / item["stake"] * 100) if item["stake"] else 0
        strategy_lines.append(
            f"{strategy}: {item['hits']}/{item['count']} 回収率 {strategy_roi:.1f}%"
        )

    profit_text = f"+{profit:,}" if profit >= 0 else f"{profit:,}"
    content = (
        f"**{title}**\n"
        f"予想数: `{total_predictions}` / 的中: `{hits}` / 的中率: `{hit_rate:.1f}%`\n"
        f"投資: `{stake:,}円` / 払戻: `{return_amount:,}円` / 収支: `{profit_text}円`\n"
        f"回収率: `{roi:.1f}%`"
    )
    if strategy_lines:
        content += "\n\n**戦略別**\n" + "\n".join(strategy_lines)
    return content

def build_performance_summary(now_jst):
    rows = _load_prediction_rows()
    if not rows:
        return None
    today = now_jst.strftime("%Y%m%d")
    settled_rows = [row for row in rows if row.get("date") == today and row.get("settled_at")]
    return _summarize_rows(settled_rows, "本日の成績サマリー")

def build_weekly_summary(now_jst):
    rows = _load_prediction_rows()
    if not rows:
        return None
    today = now_jst.date()
    week_start = today - timedelta(days=today.weekday())
    week_start_str = week_start.strftime("%Y%m%d")
    today_str = today.strftime("%Y%m%d")
    settled_rows = [
        row for row in rows
        if row.get("settled_at") and week_start_str <= row.get("date", "") <= today_str
    ]
    title = f"今週の成績サマリー ({week_start.strftime('%m/%d')}〜{today.strftime('%m/%d')})"
    return _summarize_rows(settled_rows, title)

def build_monthly_summary(now_jst):
    rows = _load_prediction_rows()
    if not rows:
        return None
    month_prefix = now_jst.strftime("%Y%m")
    settled_rows = [
        row for row in rows
        if row.get("settled_at") and row.get("date", "").startswith(month_prefix)
    ]
    title = f"今月の成績サマリー ({now_jst.strftime('%Y年%m月')})"
    return _summarize_rows(settled_rows, title)

def _is_last_day_of_month(d):
    return (d + timedelta(days=1)).month != d.month

def _safe_int(value):
    try:
        return int(float(value or 0))
    except Exception:
        return 0

def send_discord_message(content, label, channel="predict"):
    webhook = WEBHOOKS.get(channel) or DISCORD_WEBHOOK_URL
    if not webhook:
        print(f"    ⚠️ Webhook for '{channel}' is not set ({label})")
        return False
    try:
        response = requests.post(
            webhook,
            json={"content": content},
            timeout=15
        )
        if 200 <= response.status_code < 300:
            print(f"    Discord message sent: {label}")
            return True
        print(f"    ❌ Discord Error ({label}): status={response.status_code}")
        print(f"    Response: {response.text[:300]}")
    except Exception as e:
        print(f"    ❌ Discord Exception ({label}): {e}")
    return False

def send_discord_embed(embed, label, channel="predict", return_message_id=False):
    """Embedを送る。return_message_id=True なら送信したメッセージのIDを返す
    (?wait=true を付けるとWebhookのレスポンスに作成されたメッセージが入る)。
    レース確定後にそのメッセージへ結果を追記するために使う。"""
    webhook = WEBHOOKS.get(channel) or DISCORD_WEBHOOK_URL
    if not webhook:
        print(f"    ⚠️ Webhook for '{channel}' is not set ({label})")
        return None if return_message_id else False
    try:
        url = webhook + ("?wait=true" if return_message_id else "")
        response = requests.post(url, json={"embeds": [embed]}, timeout=15)
        if 200 <= response.status_code < 300:
            print(f"    Discord embed sent: {label}")
            if return_message_id:
                try:
                    return response.json().get("id")
                except ValueError:
                    return None
            return True
        print(f"    ❌ Discord Error ({label}): status={response.status_code}")
        print(f"    Response: {response.text[:300]}")
    except Exception as e:
        print(f"    ❌ Discord Exception ({label}): {e}")
    return None if return_message_id else False

def edit_discord_embed(message_id, embed, label, channel="predict"):
    """既に送ったEmbedを編集する(結果の追記に使う)。"""
    webhook = WEBHOOKS.get(channel) or DISCORD_WEBHOOK_URL
    if not webhook or not message_id:
        return False
    try:
        response = requests.patch(
            f"{webhook}/messages/{message_id}",
            json={"embeds": [embed]},
            timeout=15,
        )
        if 200 <= response.status_code < 300:
            print(f"    Discord embed edited: {label}")
            return True
        print(f"    ❌ Discord edit error ({label}): status={response.status_code}")
    except Exception as e:
        print(f"    ❌ Discord edit exception ({label}): {e}")
    return False

STRATEGY_COLORS = {
    "IN_JUMP": 0xE74C3C,
    "FOCUS": 0x3498DB,
    "STANDARD": 0xF1C40F,
    "WIDE": 0x9B59B6,
}

def format_ticket_formation(ticket_details):
    """{"2-1-3": {...}, "2-1-4": {...}} を 1着→2着→3着候補 のフォーメーション表示にまとめる。
    戻り値は (サマリー行のリスト, 買い目ごとの詳細行のリスト)。詳細行はサマリーと同じ並び順。
    """
    groups = {}
    for ticket in ticket_details:
        first, second, third = ticket.split("-")
        groups.setdefault((int(first), int(second)), []).append(int(third))

    summary_lines = []
    detail_lines = []
    for (first, second), thirds in sorted(groups.items()):
        thirds = sorted(thirds)
        third_text = "・".join(str(t) for t in thirds)
        summary_lines.append(f"**{first} → {second} → {third_text}**（{len(thirds)}点）")
        for third in thirds:
            ticket = f"{first}-{second}-{third}"
            detail = ticket_details[ticket]
            detail_lines.append(
                f"`{ticket}` ¥{detail['stake']:,}｜{detail['odds']:.1f}倍｜EV{detail['expected_value']:.2f}"
            )
    return summary_lines, detail_lines

def build_schedule_digest(now_jst):
    """まだ締め切っていない本日の予想レースを、締切順に並べたEmbedを返す。

    Botは15分おきに走り、EV条件を満たさず見送ったレースは通知済みにならないため、
    後の実行で条件を満たすと「締切が前のレース」が後から通知される。その結果
    Discordのタイムライン上では締切が前後して見えるので、都度この一覧を出して
    「今どれをどの順で見ればいいか」を一目で分かるようにする。
    """
    rows = _load_prediction_rows()
    today = now_jst.strftime("%Y%m%d")

    pending = []
    for row in rows:
        if row.get("date") != today or row.get("settled_at"):
            continue
        deadline = _prediction_deadline_datetime(row)
        if not deadline or deadline <= now_jst:
            continue
        pending.append((deadline, row))

    if not pending:
        return None

    pending.sort(key=lambda x: x[0])
    lines = []
    for deadline, row in pending:
        minutes = int((deadline - now_jst).total_seconds() // 60)
        stake = _safe_int(row.get("stake"))
        if not stake:
            try:
                details = json.loads(row.get("ticket_details") or "{}")
                stake = sum(d.get("stake", 0) for d in details.values())
            except (json.JSONDecodeError, TypeError):
                stake = 0
        count = row.get("ticket_count") or "?"
        lines.append(
            f"`{deadline.strftime('%H:%M')}` **{row.get('course')} {row.get('rno')}R**"
            f"｜あと{minutes}分｜{count}点 ¥{stake:,}"
        )

    return {
        "title": f"⏰ 締切スケジュール（未締切 {len(pending)}件）",
        "description": "\n".join(lines),
        "color": 0x2ECC71,
        "footer": {"text": f"{now_jst.strftime('%H:%M')} 時点｜上から順に締め切ります"},
        "timestamp": now_jst.isoformat(),
    }

def _prediction_deadline_datetime(row):
    try:
        return datetime.strptime(
            f"{row.get('date')} {row.get('deadline')}",
            "%Y%m%d %H:%M",
        ).replace(tzinfo=JST)
    except Exception:
        return None

# ==========================================
# ==========================================
# 1. スクレイパー (v5: 指紋偽装・Referer強化版)
# ==========================================
class BoatRaceScraperV5:
    BASE_URL = "https://www.boatrace.jp/owpc/pc/race/beforeinfo"
    LIST_URL = "https://www.boatrace.jp/owpc/pc/race/racelist"
    INDEX_URL = "https://www.boatrace.jp/owpc/pc/race/index"
    ODDS3T_URL = "https://www.boatrace.jp/owpc/pc/race/odds3t"
    RESULT_URL = "https://www.boatrace.jp/owpc/pc/race/raceresult"
    
    COURSE_MAP = {
        "桐生": "01", "戸田": "02", "江戸川": "03", "平和島": "04", "多摩川": "05",
        "浜名湖": "06", "蒲郡": "07", "常滑": "08", "津": "09", "三国": "10",
        "びわこ": "11", "住之江": "12", "尼崎": "13", "鳴門": "14", "丸亀": "15",
        "児島": "16", "宮島": "17", "徳山": "18", "下関": "19", "若松": "20",
        "芦屋": "21", "福岡": "22", "唐津": "23", "大村": "24"
    }

    def __init__(self):
        # よりブラウザに近いヘッダー設定
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.course_links = {} # {course_name: list_url}
        self.date_str = ""
        
        # セッションの初期化 (Warm-up)
        try:
            self.session.get("https://www.boatrace.jp/", timeout=15)
        except: pass

    def _get_soup(self, url, referer=None, retries=3):
        for i in range(retries):
            try:
                headers = {"Referer": referer} if referer else {}
                res = self.session.get(url, headers=headers, timeout=20)
                res.raise_for_status()
                # 文字化け対策: 明示的にエンコーディングを設定
                res.encoding = res.apparent_encoding or "utf-8"
                return BeautifulSoup(res.content, "html.parser")
            except Exception as e:
                wait = (i + 1) * 3
                print(f"[{datetime.now(JST).strftime('%H:%M:%S')}] ⚠️ Retry {i+1}/{retries}: {url} - {e}")
                time.sleep(wait)
        return None

    @staticmethod
    def _to_float(value, default=0.0):
        m = re.search(r"\d+(?:\.\d+)?", value or "")
        return float(m.group(0)) if m else default

    @staticmethod
    def _to_signed_float(value, default=0.0):
        m = re.search(r"[+-]?\d+(?:\.\d+)?", value or "")
        return float(m.group(0)) if m else default

    @staticmethod
    def _parse_entry_features(text):
        text = " ".join((text or "").split())
        m = re.search(r"kg\s+F\d+\s+L\d+\s+(.+)", text)
        if not m:
            return {}

        # 新人選手などは平均STが "-" (未計測) で表示される。
        # 数値だけを正規表現で拾うと "-" が読み飛ばされ後続の値が1つずつズレるため、
        # 空白区切りのトークン単位でパースし、数値化できないものは0.0として扱う。
        tokens = m.group(1).split()
        if len(tokens) < 13:
            return {}

        def to_float(token):
            try:
                return float(token)
            except ValueError:
                return 0.0

        vals = [to_float(t) for t in tokens[:13]]

        return {
            "avg_st": vals[0],
            "national_win_rate": vals[1],
            "national_2_rate": vals[2],
            "national_3_rate": vals[3],
            "local_win_rate": vals[4],
            "local_2_rate": vals[5],
            "local_3_rate": vals[6],
            "motor_no": int(vals[7]),
            "motor_2_rate": vals[8],
            "motor_3_rate": vals[9],
            "boat_no": int(vals[10]),
            "boat_2_rate": vals[11],
            "boat_3_rate": vals[12],
        }

    def fetch_all_venue_schedules(self, date_str):
        """全会場の1R〜12Rスケジュールを網羅的に取得する (最速・確実版)"""
        print(f"[{datetime.now(JST).strftime('%H:%M:%S')}] 🏟️  Retrieving daily schedules for all venues...")
        index_url = f"{self.INDEX_URL}?hd={date_str}"
        soup_index = self._get_soup(index_url, referer="https://www.boatrace.jp/")
        if not soup_index: return {}

        # 1. 開催されている会場を特定
        active_venues = {} # jcd -> course_name
        inv_map = {v: k for k, v in self.COURSE_MAP.items()}
        for link in soup_index.select("a[href*='jcd=']"):
            href = link.get('href', '')
            m_jcd = re.search(r"jcd=(\d{2})", href)
            if m_jcd:
                jcd = m_jcd.group(1)
                if jcd in inv_map:
                    active_venues[jcd] = inv_map[jcd]

        venue_list = sorted(list(set(active_venues.values())))
        print(f"  - Active Venues ({len(venue_list)}): {', '.join(venue_list)}")
        
        all_schedules = {} # (course, rno) -> (time_str, race_url)
        processed_links = set() # 重複リンク排除用
        
        # 2. 各会場の「本日の一覧(raceindex)」から全12レースを取得
        for jcd, course in active_venues.items():
            venue_url = f"https://www.boatrace.jp/owpc/pc/race/raceindex?jcd={jcd}&hd={date_str}"
            soup_v = self._get_soup(venue_url, referer=index_url)
            if not soup_v:
                print(f"    ⚠️ Failed to load venue page: {course}")
                continue
            
            # 手動フィルタリングで確実に抽出
            links = soup_v.find_all("a")
            v_count = 0
            for link in links:
                href = link.get('href', '')
                if 'racelist' not in href or 'rno=' not in href:
                    continue
                
                # 同一レースの重複リンク（ボタンなど）をスキップ
                if href in processed_links:
                    continue
                processed_links.add(href)

                m_rno = re.search(r"rno=(\d{1,2})", href)
                if not m_rno: continue
                rno = int(m_rno.group(1))
                
                # 時刻は親の tr 全体から探す（会場によって隣の td にある場合があるため）
                container = link.find_parent("tr")
                txt = container.get_text(separator=' ').strip().replace('\n', ' ') if container else ""
                
                # HH:MM を探す
                m_time = re.search(r"(\d{1,2}:\d{2})", txt)
                if m_time:
                    time_str = m_time.group(1).zfill(5)
                    full_url = "https://www.boatrace.jp" + href if href.startswith("/") else href
                    all_schedules[(course, rno)] = (time_str, full_url)
                    v_count += 1
            
            # print(f"    ✅ {course}: {v_count} races found")
            time.sleep(0.3) # 負荷軽減
            
        print(f"  - Total unique races logged for today: {len(all_schedules)}")
        return all_schedules

    def fetch_race_data(self, course, rno, date_str, race_url=None, deadline=None):
        """出走表(詳細)と直前情報を取得"""
        jcd = self.COURSE_MAP.get(course, "01")
        # 直接URLが指定されていない場合は構築する
        race_list_url = race_url if race_url else f"{self.LIST_URL}?rno={rno}&jcd={jcd}&hd={date_str}"
        
        try:
            soup_list = self._get_soup(race_list_url, referer=f"{self.INDEX_URL}?hd={date_str}")
            if not soup_list: return None
            
            # 締切時刻が引数で渡されていない場合はページから抽出を試みる
            deadline_str = deadline if deadline else "00:00"
            if not deadline:
                m_time = re.search(r"(\d{1,2}:\d{2})", soup_list.get_text(separator=' '))
                if m_time: deadline_str = m_time.group(1).zfill(5)
            
            # 直前情報のURL
            info_url = f"{self.BASE_URL}?rno={rno}&jcd={jcd}&hd={date_str}"
            soup_info = self._get_soup(info_url, referer=race_list_url)
            if not soup_info or "データがありません" in soup_info.text:
                print(f"  ⚠️ No beforeinfo data: {course} {rno}R")
                return None
            
            bodies = soup_list.select("tbody.is-fs12") or soup_list.select("tbody")
            
            boat_info = {}
            for i in range(1, 7):
                rank, win_rate = "B2", 0.0
                entry_features = {}
                candidate_bodies = [bodies[i - 1]] if len(bodies) >= 6 else bodies
                for b in candidate_bodies:
                    is_boat_row = len(bodies) >= 6 or b.select_one(f".is-ladder{i}") or str(i) in b.text[:5]
                    if is_boat_row:
                        row_text = b.get_text(" ", strip=True)
                        r_m = re.search(r"([AB][12])", row_text)
                        if r_m: rank = r_m.group(1)
                        rates = re.findall(r"(\d\.\d{2})", row_text)
                        if rates: win_rate = float(rates[0])
                        entry_features = self._parse_entry_features(row_text)
                        break
                boat_info[i] = {"rank": rank, "win_rate": win_rate, **entry_features}

            weather = soup_info.select_one(".weather1")
            wind_speed, wave = 0, 0
            if weather:
                txt = weather.text
                w_m = re.search(r"風速.*?(\d+)m", txt)
                h_m = re.search(r"波高.*?(\d+)cm", txt)
                if w_m: wind_speed = int(w_m.group(1))
                if h_m: wave = int(h_m.group(1))

            table = soup_info.select_one(".is-w748")
            if not table:
                print(f"  ⚠️ Beforeinfo table not found: {course} {rno}R")
                return None
            rows = table.select("tbody")
            if len(rows) < 6:
                print(f"  ⚠️ Beforeinfo rows are incomplete: {course} {rno}R rows={len(rows)}")
                return None
            
            data = {"wind_speed": wind_speed, "wave": wave, "deadline": deadline_str}
            for i in range(1, 7):
                tds = rows[i-1].select("td")
                if len(tds) < 5:
                    print(f"  ⚠️ Beforeinfo columns are incomplete: {course} {rno}R boat={i}")
                    return None
                ex_val = tds[4].text.strip()
                data[f"ex_time_{i}"] = self._to_float(ex_val, 6.80)
                data[f"weight_{i}"] = self._to_float(tds[3].get_text(" ", strip=True), 0.0) if len(tds) > 3 else 0.0
                data[f"tilt_{i}"] = self._to_signed_float(tds[5].get_text(" ", strip=True), 0.0) if len(tds) > 5 else 0.0
                parts_text = tds[7].get_text(" ", strip=True) if len(tds) > 7 else ""
                data[f"parts_{i}"] = parts_text
                data[f"parts_count_{i}"] = len(tds[7].select(".label4")) if len(tds) > 7 else 0
                st_text = tds[2].select_one(".is-fs11").text.strip() if tds[2].select_one(".is-fs11") else ".15"
                data[f"st_{i}"] = float("0"+re.search(r"(\.\d+)", st_text).group(1)) if re.search(r"(\.\d+)", st_text) else 0.15
                data[f"rank_{i}"] = boat_info[i]["rank"]
                data[f"win_rate_{i}"] = boat_info[i]["win_rate"]
                for key in [
                    "avg_st",
                    "national_win_rate",
                    "national_2_rate",
                    "national_3_rate",
                    "local_win_rate",
                    "local_2_rate",
                    "local_3_rate",
                    "motor_no",
                    "motor_2_rate",
                    "motor_3_rate",
                    "boat_no",
                    "boat_2_rate",
                    "boat_3_rate",
                ]:
                    data[f"{key}_{i}"] = boat_info[i].get(key, "")

            return data
        except Exception as e:
            print(f"  ❌ fetch_race_data error: {course} {rno}R - {e}")
            traceback.print_exc()
            return None

    def fetch_odds3t(self, course, rno, date_str):
        """3連単オッズを {\"1-2-3\": 12.3} の形で取得する。"""
        jcd = self.COURSE_MAP.get(course, "01")
        odds_url = f"{self.ODDS3T_URL}?rno={rno}&jcd={jcd}&hd={date_str}"

        try:
            soup = self._get_soup(odds_url, referer=f"{self.INDEX_URL}?hd={date_str}")
            if not soup or "データがありません" in soup.text:
                print(f"  ⚠️ No odds data: {course} {rno}R")
                return {}

            odds_cell = soup.select_one("td.oddsPoint")
            if not odds_cell:
                print(f"  ⚠️ Odds table not found: {course} {rno}R")
                return {}

            table = odds_cell.find_parent("table")
            header_cells = table.select("thead th")
            first_boats = [
                int(th.get_text(strip=True))
                for th in header_cells
                if re.fullmatch(r"[1-6]", th.get_text(strip=True))
            ]
            if len(first_boats) != 6:
                first_boats = list(range(1, 7))

            odds = {}
            current_second = [None] * 6
            for row in table.select("tbody tr"):
                cells = row.select("td")
                pos = 0
                for group_idx, first in enumerate(first_boats):
                    if pos >= len(cells):
                        break

                    if pos + 1 < len(cells) and "oddsPoint" in cells[pos + 1].get("class", []):
                        second = current_second[group_idx]
                        third = self._cell_boat_number(cells[pos])
                        odd_text = cells[pos + 1].get_text(strip=True)
                        pos += 2
                    elif pos + 2 < len(cells):
                        second = self._cell_boat_number(cells[pos])
                        third = self._cell_boat_number(cells[pos + 1])
                        odd_text = cells[pos + 2].get_text(strip=True)
                        current_second[group_idx] = second
                        pos += 3
                    else:
                        break

                    odd = self._to_float(odd_text, 0.0)
                    if first and second and third and odd > 0:
                        odds[f"{first}-{second}-{third}"] = odd

            if len(odds) < 100:
                print(f"  ⚠️ Odds parsed incompletely: {course} {rno}R count={len(odds)}")
            return odds
        except Exception as e:
            print(f"  ❌ fetch_odds3t error: {course} {rno}R - {e}")
            traceback.print_exc()
            return {}

    @staticmethod
    def _cell_boat_number(cell):
        m = re.search(r"[1-6]", cell.get_text(strip=True))
        return int(m.group(0)) if m else None

    def fetch_race_result(self, course, rno, date_str):
        """3連単の確定結果を {ticket, payout} で取得する。"""
        jcd = self.COURSE_MAP.get(course, "01")
        result_url = f"{self.RESULT_URL}?rno={rno}&jcd={jcd}&hd={date_str}"

        try:
            soup = self._get_soup(result_url, referer=f"{self.INDEX_URL}?hd={date_str}")
            if not soup or "データがありません" in soup.text:
                return None

            for tbody in soup.select("tbody"):
                first_row = tbody.select_one("tr")
                if not first_row or "3連単" not in first_row.get_text(" ", strip=True):
                    continue

                nums = [
                    n.get_text(strip=True)
                    for n in first_row.select(".numberSet1_number")
                    if re.fullmatch(r"[1-6]", n.get_text(strip=True))
                ]
                payout_cell = first_row.select_one(".is-payout1")
                payout = self._payout_to_int(payout_cell.get_text(strip=True) if payout_cell else "")

                if len(nums) >= 3 and payout:
                    return {
                        "ticket": "-".join(nums[:3]),
                        "payout": payout,
                    }
            return None
        except Exception as e:
            print(f"  ❌ fetch_race_result error: {course} {rno}R - {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def _payout_to_int(value):
        digits = re.sub(r"\D", "", value or "")
        return int(digits) if digits else 0

def predict_probs(model, input_df):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(input_df)[0]
    return model.predict(input_df)[0]

def build_tickets(strategy, top1, top2, top3):
    """買い目候補を「全120通りの3連単から1号艇1着(イン逃げ)を除いたもの」で返す。

    以前は top1/top2/top3 から機械的に4〜8点だけを候補にしていたが、上位予測に
    入らない高EVな買い目を取りこぼしていた。バックテスト(model_evaluation_log.md参照)で
    全120通り・1号艇1着除外をEV順に選ぶ方が回収率が高いと確認されたため変更。
    最終的な購入点数は add_expected_values 側で EV順ソート後 MAX_TICKET_COUNT[strategy] に絞られる。
    top1/top2/top3 は互換性のため引数に残す(未使用)。
    """
    return [
        f"{a}-{b}-{c}"
        for a in range(2, 7)   # 1号艇は1着にしない(イン飛び戦略の趣旨)
        for b in range(1, 7)
        for c in range(1, 7)
        if len({a, b, c}) == 3
    ]

ORDER_MODEL_2ND_PATH = BASE_DIR / "order_model_2nd_v1.pkl"
ORDER_MODEL_3RD_PATH = BASE_DIR / "order_model_3rd_v1.pkl"
ORDER_MODEL_CONFIG_PATH = BASE_DIR / "order_model_config_v1.pkl"
_order_models_cache = None

def _load_order_models():
    """2着・3着専用モデルを遅延ロードする。存在しなければ (None, None, None) を返し、
    estimate_ticket_probability は Plackett-Luce近似にフォールバックする。"""
    global _order_models_cache
    if _order_models_cache is not None:
        return _order_models_cache
    if not (ORDER_MODEL_2ND_PATH.exists() and ORDER_MODEL_3RD_PATH.exists() and ORDER_MODEL_CONFIG_PATH.exists()):
        _order_models_cache = (None, None, None)
        return _order_models_cache
    with open(ORDER_MODEL_2ND_PATH, "rb") as f: model_2nd = pickle.load(f)
    with open(ORDER_MODEL_3RD_PATH, "rb") as f: model_3rd = pickle.load(f)
    with open(ORDER_MODEL_CONFIG_PATH, "rb") as f: order_config = pickle.load(f)
    _order_models_cache = (model_2nd, model_3rd, order_config)
    return _order_models_cache

def estimate_ticket_probability(ticket, probs, input_df=None):
    first, second, third = [int(x) for x in ticket.split("-")]
    p_first = probs[first - 1]

    model_2nd, model_3rd, order_config = _load_order_models()
    if model_2nd is not None and input_df is not None:
        x2 = input_df.copy()
        x2["given_1st"] = first
        raw2 = np.asarray(model_2nd.predict(x2[order_config["features_2nd"]]), dtype=float)[0]
        raw2[first - 1] = 0
        total2 = raw2.sum()
        p_second = (raw2[second - 1] / total2) if total2 > 0 else 0

        x3 = input_df.copy()
        x3["given_1st"] = first
        x3["given_2nd"] = second
        raw3 = np.asarray(model_3rd.predict(x3[order_config["features_3rd"]]), dtype=float)[0]
        raw3[first - 1] = 0
        raw3[second - 1] = 0
        total3 = raw3.sum()
        p_third = (raw3[third - 1] / total3) if total3 > 0 else 0
    else:
        remaining_after_first = [i for i in range(1, 7) if i != first]
        second_base = sum(probs[i - 1] for i in remaining_after_first)
        p_second = probs[second - 1] / second_base if second_base > 0 else 0

        remaining_after_second = [i for i in remaining_after_first if i != second]
        third_base = sum(probs[i - 1] for i in remaining_after_second)
        p_third = probs[third - 1] / third_base if third_base > 0 else 0

    return p_first * p_second * p_third

def add_expected_values(tickets, probs, odds_map, strategy, input_df=None):
    enriched = []
    for ticket in tickets:
        odds = odds_map.get(ticket)
        if not odds:
            continue
        probability = estimate_ticket_probability(ticket, probs, input_df)
        expected_value = probability * odds
        if expected_value >= MIN_EXPECTED_VALUE:
            enriched.append({
                "ticket": ticket,
                "odds": odds,
                "probability": probability,
                "expected_value": expected_value,
            })

    enriched.sort(key=lambda x: x["expected_value"], reverse=True)
    return enriched[:MAX_TICKET_COUNT.get(strategy, 8)]

def add_kelly_stakes(value_tickets, bankroll, kelly_fraction=KELLY_FRACTION):
    """複数の排反な買い目に同時に賭ける場合のケリー基準 (Thorpの一般化式)。
    odds はグロス配当 (100円が odds*100円になる) なので、ネットオッズ b = odds - 1 を使う。
    f_i = p_i - (1 - ΣP) / b_i
    """
    if not value_tickets or bankroll <= 0:
        for item in value_tickets:
            item["stake"] = 0
        return value_tickets

    total_prob = sum(item["probability"] for item in value_tickets)
    for item in value_tickets:
        net_odds = item["odds"] - 1
        if net_odds <= 0:
            item["stake"] = 0
            continue
        edge = item["probability"] - (1 - total_prob) / net_odds
        fraction = max(edge, 0.0) * kelly_fraction
        raw_stake = bankroll * fraction
        item["stake"] = round(raw_stake / STAKE_PER_TICKET) * STAKE_PER_TICKET

    total_stake = sum(item["stake"] for item in value_tickets)
    max_total = bankroll * MAX_RACE_STAKE_RATIO
    if total_stake > max_total > 0:
        scale = max_total / total_stake
        for item in value_tickets:
            item["stake"] = round(item["stake"] * scale / STAKE_PER_TICKET) * STAKE_PER_TICKET

    return value_tickets

# 2. 予測ロジック
# ==========================================
def predict_single(model, config, scraper, course, rno, date_str, bankroll, kelly_fraction=KELLY_FRACTION, race_url=None, deadline=None):
    try:
        data = scraper.fetch_race_data(course, rno, date_str, race_url=race_url, deadline=deadline)
        if not data: 
            print(f"  ⚠️ Failed to fetch detail data for {course} {rno}R")
            return None, -1
        
        ex_cols = [f"ex_time_{i}" for i in range(1, 7)]
        ex_vals = [data[c] for c in ex_cols]
        ex_mean = np.mean(ex_vals)
        rank_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
        
        input_dict = {"wind_speed": data["wind_speed"], "wave": data["wave"]}
        ex_ranks = pd.Series(ex_vals).rank(method="min").tolist()
        
        for i in range(1, 7):
            idx = i - 1
            rv = rank_map.get(data[f"rank_{i}"], 2)
            input_dict[f"rank_val_{i}"] = rv
            input_dict[f"win_rate_{i}"] = data[f"win_rate_{i}"]
            input_dict[f"ex_time_{i}"] = data[f"ex_time_{i}"]
            input_dict[f"ex_diff_{i}"] = data[f"ex_time_{i}"] - ex_mean
            input_dict[f"ex_rank_{i}"] = ex_ranks[idx]
            input_dict[f"st_{i}"] = data[f"st_{i}"]
            
        input_dict["is_debuff_1"] = 1 if (input_dict["rank_val_1"] <= 2 and input_dict["ex_rank_1"] >= 4) else 0
        
        input_df = pd.DataFrame([input_dict])[config["features"]]
        probs = np.asarray(predict_probs(model, input_df), dtype=float)

        # 1号艇の勝率を直接取得
        in_win_prob = probs[0]
        in_jump_prob = 1 - in_win_prob
        
        # 全艇のAI勝率ランキング
        all_ranking = sorted(
            {i + 1: p for i, p in enumerate(probs)}.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 1号艇を除いたランキング
        ranking_without_1 = [r for r in all_ranking if r[0] != 1]
        
        top1, top2, top3 = ranking_without_1[0], ranking_without_1[1], ranking_without_1[2]
        lower_ranking = ranking_without_1[3:]
        
        strategy = ""
        if in_jump_prob >= IN_JUMP_THRESHOLD:
            if top1[1] >= FOCUS_TOP_THRESHOLD:
                strategy = "FOCUS"
            elif top1[1] >= STANDARD_TOP_THRESHOLD:
                strategy = "STANDARD"
            # WIDE (top1 < STANDARD_TOP_THRESHOLD) is disabled: backtests showed 0 hits
        
        if not strategy:
            return None, 0

        tickets = build_tickets(strategy, top1, top2, top3)
        odds_map = scraper.fetch_odds3t(course, rno, date_str)
        value_tickets = add_expected_values(tickets, probs, odds_map, strategy, input_df) if odds_map else []
        if odds_map and not value_tickets:
            print(f"  - {course} {rno}R: No tickets over EV {MIN_EXPECTED_VALUE:.2f}")
            return None, 0

        if not value_tickets:
            # オッズ取得失敗など。全120通り候補をそのまま通知するとスパムになるうえ
            # EV/賭け金を計算できないため、ベットせずスキップする。
            print(f"  - {course} {rno}R: No odds / no value tickets (skip)")
            return None, 0

        value_tickets = add_kelly_stakes(value_tickets, bankroll, kelly_fraction)
        staked_tickets = [item for item in value_tickets if item["stake"] > 0]
        if not staked_tickets:
            print(f"  - {course} {rno}R: Kelly stake is 0 for all tickets (skip)")
            return None, 0

        ticket_text = " / ".join(
            f"{item['ticket']}({item['odds']:.1f}倍/EV{item['expected_value']:.2f}/¥{item['stake']:,})"
            for item in staked_tickets
        )
        ticket_count = len(staked_tickets)
        max_ev = max(item["expected_value"] for item in staked_tickets)
        ticket_details = {
            item["ticket"]: {
                "odds": item["odds"],
                "probability": item["probability"],
                "expected_value": item["expected_value"],
                "stake": item["stake"],
            }
            for item in staked_tickets
        }

        res_dict = {
            "場名": course,
            "レース": f"{rno}R",
            "締切": data['deadline'],
            "1号艇勝率": in_win_prob,
            "イン飛び率": in_jump_prob,
            "戦略": strategy,
            "1位": top1,
            "2位": top2,
            "3位": top3,
            "4位以下": lower_ranking,
            "全体ランキング": all_ranking,
            "根拠": f"1号艇:{data['rank_1']} / 展示:{int(input_dict['ex_rank_1'])}位",
            "買い目": ticket_text,
            "買い目内訳": ticket_details,
            "バンクロール": bankroll,
            "点数": ticket_count,
            "期待値MAX": max_ev,
            "特徴量": {
                field: data.get(field, "")
                for field in EXTRA_FEATURE_FIELDS
            }
        }
        return res_dict, 1
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, -2

# ==========================================
# 3. メイン実行 (パトロール)
# ==========================================
def scan_and_notify(model, config, scraper, now_jst, date_str, run_at, state):
    bankroll = state.get("current_bankroll", STARTING_BANKROLL)
    if bankroll < STAKE_PER_TICKET:
        print(f"  💸 Bankroll too low to bet ({bankroll}円). Skipping.")
        return 0

    day_start_bankroll = state.get("day_start_bankroll", bankroll)
    if day_start_bankroll > 0:
        daily_loss_ratio = (day_start_bankroll - bankroll) / day_start_bankroll
        if daily_loss_ratio >= DAILY_LOSS_LIMIT_RATIO:
            print(f"  🛑 Daily loss limit reached ({daily_loss_ratio:.1%} >= {DAILY_LOSS_LIMIT_RATIO:.0%}). Skipping for today.")
            return 0

    kelly_fraction = KELLY_FRACTION

    # 1. 1日の全スケジュールを取得 (初回、または1時間ごとに更新すると効率的)
    all_races = scraper.fetch_all_venue_schedules(date_str)

    if not all_races:
        state["consecutive_schedule_failures"] = state.get("consecutive_schedule_failures", 0) + 1
        failures = state["consecutive_schedule_failures"]
        last_alert = state.get("last_schedule_alert_count", 0)
        print(f"  ⚠️ Schedule fetch returned no venues ({failures} consecutive failures)")
        if failures >= SCHEDULE_FAILURE_ALERT_THRESHOLD and failures - last_alert >= SCHEDULE_FAILURE_ALERT_INTERVAL:
            send_discord_message(
                f"⚠️ **スケジュール取得が{failures}回連続で失敗しています**\n"
                f"boatrace.jpの構造変更やアクセス制限の可能性があります。",
                "schedule fetch failure",
                channel="system",
            )
            state["last_schedule_alert_count"] = failures
    else:
        state["consecutive_schedule_failures"] = 0
        state["last_schedule_alert_count"] = 0
    save_state(state)

    # 2. 現在のターゲット (5分〜35分前) を抽出
    targets = []
    print(f"[{datetime.now(JST).strftime('%H:%M:%S')}] 🔍 Filtering targets from schedule...")
    for (course, rno), (time_str, race_url) in sorted(all_races.items(), key=lambda kv: (kv[1][0], kv[0])):
        try:
            race_dt = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H:%M").replace(tzinfo=JST)
            diff = (race_dt - now_jst).total_seconds() / 60

            # デバッグ表示: 窓に近いものを出す
            if 0 <= diff <= 45:
                print(f"  - {course} {rno}R: {time_str} (in {diff:.1f}m)")

            if 5 <= diff <= 35:
                # 重複通知チェック
                race_id = f"{date_str}_{course}_{rno}"
                if not is_already_notified(race_id):
                    targets.append({"course": course, "rno": rno, "time": time_str, "url": race_url, "id": race_id})
        except Exception as e:
            print(f"  ⚠️ Failed to evaluate schedule item {course} {rno}R: {e}")

    hit_count = 0
    if not targets:
        print("  (No new target races in the 5-35 min window)")

    for race in targets:
        course = race['course']
        rno = race['rno']
        race_id = race['id']

        print(f"  - {course} {rno}R: Analyzing... (Deadline: {race['time']})")
        res, status = predict_single(model, config, scraper, course, rno, date_str, bankroll, kelly_fraction=kelly_fraction, race_url=race['url'], deadline=race['time'])

        if status == 1:
            hit_count += 1
            # Discord通知処理 (Embed表示)
            ticket_details = res.get("買い目内訳", {})
            summary_lines, detail_lines = format_ticket_formation(ticket_details)
            total_stake = sum(detail["stake"] for detail in ticket_details.values())

            ranking_lines = [
                f"🥇 **{res['1位'][0]}号艇** `{res['1位'][1]:.1%}`",
                f"🥈 **{res['2位'][0]}号艇** `{res['2位'][1]:.1%}`",
                f"🥉 **{res['3位'][0]}号艇** `{res['3位'][1]:.1%}`",
            ]
            for idx, item in enumerate(res["4位以下"], start=4):
                ranking_lines.append(f"{idx}位 **{item[0]}号艇** `{item[1]:.1%}`")

            evaluation_lines = [
                f"イン飛び率 `{res['イン飛び率']:.1%}`",
                f"1号艇勝率 `{res['1号艇勝率']:.1%}`",
            ]
            if res["期待値MAX"] is not None:
                evaluation_lines.append(f"最大期待値 `{res['期待値MAX']:.2f}`")

            embed = {
                "title": f"🎯 {res['場名']} {res['レース']}｜締切 {res['締切']}",
                "description": "\n".join(summary_lines) + "\n\n" + "\n".join(detail_lines),
                "color": STRATEGY_COLORS.get(res["戦略"], 0x95A5A6),
                "fields": [
                    {"name": "レース評価", "value": "\n".join(evaluation_lines), "inline": True},
                    {"name": "AI勝率ランキング", "value": "\n".join(ranking_lines), "inline": True},
                    {"name": "判断根拠", "value": res["根拠"], "inline": False},
                ],
                "footer": {
                    "text": f"{res['戦略']}｜合計 {total_stake:,}円｜バンクロール {res['バンクロール']:,}円"
                },
                "timestamp": datetime.now(JST).isoformat(),
            }

            # メッセージIDを控えておき、レース確定後にこの通知へ結果を追記する
            message_id = send_discord_embed(embed, race_id, return_message_id=True)

            save_prediction_log(race_id, race, res, run_at, discord_message_id=message_id)

            # 通知済みリストに保存
            save_notified_race(race_id)
            # ヘルスチェックの「長時間ベットが出ていない」判定に使う
            state["health_last_bet_at"] = datetime.now(JST).isoformat()
        time.sleep(1)

    # 新たに予想を出した回だけ、未締切レースの締切順ダイジェストを更新する
    if hit_count:
        digest = build_schedule_digest(datetime.now(JST))
        if digest:
            send_discord_embed(digest, "schedule digest", channel="schedule")

    return hit_count

def run_live_patrol():
    run_at = datetime.now(JST)
    print(f"👮 Smart Patrol Start: {run_at.strftime('%Y-%m-%d %H:%M:%S')}")

    prune_notified_races(run_at)
    state = load_state()

    if "current_bankroll" not in state:
        state["current_bankroll"] = STARTING_BANKROLL

    scraper = BoatRaceScraperV5()
    settled_count = settle_prediction_logs(scraper, run_at, state)

    now_jst = datetime.now(JST)
    date_str = now_jst.strftime("%Y%m%d")

    if state.get("bankroll_day") != date_str:
        state["day_start_bankroll"] = state["current_bankroll"]
        state["bankroll_day"] = date_str

    save_state(state)

    hit_count = 0
    if OPERATING_HOUR_START <= now_jst.hour < OPERATING_HOUR_END:
        if not MODEL_PATH.exists():
            print(f"❌ Error: Model file not found at {MODEL_PATH}")
        else:
            with open(MODEL_PATH, "rb") as f: model = pickle.load(f)
            with open(CONFIG_PATH, "rb") as f: config = pickle.load(f)
            print("✅ Model loaded successfully.")

            if state.get("notified_model_version") != MODEL_PATH.name:
                send_discord_message(
                    f"🔄 予測モデルを更新しました: `{MODEL_PATH.name}`\n"
                    f"学習日: {config.get('date_trained', '不明')}\n"
                    f"（このメッセージ以降の予測は新モデルによるものです）",
                    "model version update",
                    channel="system",
                )
                state["notified_model_version"] = MODEL_PATH.name
                save_state(state)

            hit_count = scan_and_notify(model, config, scraper, now_jst, date_str, run_at, state)
    else:
        print(f"  💤 Outside operating hours ({OPERATING_HOUR_START}:00-{OPERATING_HOUR_END}:00 JST). Skipping schedule scan.")

    if settled_count:
        summary = build_performance_summary(run_at)
        if summary:
            send_discord_message(summary, "daily performance summary", channel="stats")

    current_week = run_at.strftime("%G-W%V")
    if run_at.weekday() == 6 and run_at.hour == 23 and state.get("last_weekly_summary") != current_week:
        weekly = build_weekly_summary(run_at)
        if weekly:
            send_discord_message(weekly, "weekly performance summary", channel="stats")
        state["last_weekly_summary"] = current_week
        save_state(state)

    current_month = run_at.strftime("%Y-%m")
    if _is_last_day_of_month(run_at.date()) and run_at.hour == 23 and state.get("last_monthly_summary") != current_month:
        monthly = build_monthly_summary(run_at)
        if monthly:
            send_discord_message(monthly, "monthly performance summary", channel="stats")
        state["last_monthly_summary"] = current_month
        save_state(state)

    # Actionsがsuccessでも静かに壊れているケースを検知する。
    # スケジュール取得が失敗している最中は「ベットが出ない」のは当然なので、
    # 二重に警告しないよう scrape_ok を渡して抑制する。
    scrape_ok = state.get("consecutive_schedule_failures", 0) == 0
    run_health_checks(state, datetime.now(JST), scrape_ok)
    save_state(state)

    print(f"👮 Patrol Finished: Found {hit_count} hits.")

if __name__ == "__main__":
    try:
        run_live_patrol()
    except Exception as e:
        error_text = traceback.format_exc()
        print(f"❌ Fatal error: {e}")
        print(error_text)
        send_discord_message(
            f"🚨 **Boatrace Patrol クラッシュ**\n```\n{error_text[-1500:]}\n```",
            "fatal error",
            channel="system",
        )
        raise
