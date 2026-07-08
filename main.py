import os
import csv
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

# パスの自動解決：GitHub Actions等の環境でも確実にファイルを見つける
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_model_v4.pkl"
CONFIG_PATH = BASE_DIR / "model_config_v4.pkl"

# 通知済みログファイル (スクリプトと同じ場所に作成)
LOG_FILE = BASE_DIR / "notified_races.log"
PREDICTION_LOG_FILE = BASE_DIR / "predictions.csv"

IN_JUMP_THRESHOLD = 0.55
FOCUS_TOP_THRESHOLD = 0.35
STANDARD_TOP_THRESHOLD = 0.25
MIN_EXPECTED_VALUE = 1.00
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

def save_prediction_log(race_id, race, result, run_at):
    fieldnames = [
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
        "reason",
    ]
    row = {
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
        "reason": result["根拠"],
    }

    file_exists = PREDICTION_LOG_FILE.exists()
    with open(PREDICTION_LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ==========================================
# ==========================================
# 1. スクレイパー (v5: 指紋偽装・Referer強化版)
# ==========================================
class BoatRaceScraperV5:
    BASE_URL = "https://www.boatrace.jp/owpc/pc/race/beforeinfo"
    LIST_URL = "https://www.boatrace.jp/owpc/pc/race/racelist"
    INDEX_URL = "https://www.boatrace.jp/owpc/pc/race/index"
    ODDS3T_URL = "https://www.boatrace.jp/owpc/pc/race/odds3t"
    
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
                for b in bodies:
                    is_boat_row = b.select_one(f".is-ladder{i}") or str(i) in b.text[:5]
                    if is_boat_row:
                        r_m = re.search(r"([AB][12])", b.get_text())
                        if r_m: rank = r_m.group(1)
                        rates = re.findall(r"(\d\.\d{2})", b.get_text())
                        if rates: win_rate = float(rates[0])
                        break
                boat_info[i] = {"rank": rank, "win_rate": win_rate}

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
                st_text = tds[2].select_one(".is-fs11").text.strip() if tds[2].select_one(".is-fs11") else ".15"
                data[f"st_{i}"] = float("0"+re.search(r"(\.\d+)", st_text).group(1)) if re.search(r"(\.\d+)", st_text) else 0.15
                data[f"rank_{i}"] = boat_info[i]["rank"]
                data[f"win_rate_{i}"] = boat_info[i]["win_rate"]

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

def predict_probs(model, input_df):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(input_df)[0]
    return model.predict(input_df)[0]

def build_tickets(strategy, top1, top2, top3):
    first = top1[0]
    seconds = [top2[0], top3[0]]
    tickets = []
    for second in seconds:
        thirds = [n for n in range(1, 7) if n not in (first, second)]
        tickets.extend(f"{first}-{second}-{third}" for third in thirds)

    if strategy == "FOCUS":
        tickets = tickets[:4]
    elif strategy == "STANDARD":
        tickets = tickets[:8]
    else:
        tickets = [
            f"{a}-{b}-{c}"
            for a in range(2, 7)
            for b in range(2, 7)
            for c in range(1, 7)
            if len({a, b, c}) == 3
        ]

    return tickets

def estimate_ticket_probability(ticket, probs):
    first, second, third = [int(x) for x in ticket.split("-")]
    p_first = probs[first - 1]

    remaining_after_first = [i for i in range(1, 7) if i != first]
    second_base = sum(probs[i - 1] for i in remaining_after_first)
    p_second = probs[second - 1] / second_base if second_base > 0 else 0

    remaining_after_second = [i for i in remaining_after_first if i != second]
    third_base = sum(probs[i - 1] for i in remaining_after_second)
    p_third = probs[third - 1] / third_base if third_base > 0 else 0

    return p_first * p_second * p_third

def add_expected_values(tickets, probs, odds_map, strategy):
    enriched = []
    for ticket in tickets:
        odds = odds_map.get(ticket)
        if not odds:
            continue
        probability = estimate_ticket_probability(ticket, probs)
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

# 2. 予測ロジック
# ==========================================
def predict_single(model, config, scraper, course, rno, date_str, race_url=None, deadline=None):
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
            else:
                strategy = "WIDE"
        
        if not strategy:
            return None, 0

        tickets = build_tickets(strategy, top1, top2, top3)
        odds_map = scraper.fetch_odds3t(course, rno, date_str)
        value_tickets = add_expected_values(tickets, probs, odds_map, strategy) if odds_map else []
        if odds_map and not value_tickets:
            print(f"  - {course} {rno}R: No tickets over EV {MIN_EXPECTED_VALUE:.2f}")
            return None, 0

        if value_tickets:
            ticket_text = " / ".join(
                f"{item['ticket']}({item['odds']:.1f}倍/EV{item['expected_value']:.2f})"
                for item in value_tickets
            )
            ticket_count = len(value_tickets)
            max_ev = value_tickets[0]["expected_value"]
        else:
            ticket_text = " / ".join(tickets)
            ticket_count = len(tickets)
            max_ev = None

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
            "点数": ticket_count,
            "期待値MAX": max_ev
        }
        return res_dict, 1
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, -2

# ==========================================
# 3. メイン実行 (パトロール)
# ==========================================
def run_live_patrol():
    run_at = datetime.now(JST)
    print(f"👮 Smart Patrol Start: {run_at.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not MODEL_PATH.exists():
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return

    with open(MODEL_PATH, "rb") as f: model = pickle.load(f)
    with open(CONFIG_PATH, "rb") as f: config = pickle.load(f)
    print("✅ Model loaded successfully.")

    scraper = BoatRaceScraperV5()
    now_jst = datetime.now(JST)
    date_str = now_jst.strftime("%Y%m%d")
    
    # 1. 1日の全スケジュールを取得 (初回、または1時間ごとに更新すると効率的)
    all_races = scraper.fetch_all_venue_schedules(date_str)
    
    # 2. 現在のターゲット (5分〜35分前) を抽出
    targets = []
    print(f"[{datetime.now(JST).strftime('%H:%M:%S')}] 🔍 Filtering targets from schedule...")
    for (course, rno), (time_str, race_url) in sorted(all_races.items()):
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
        res, status = predict_single(model, config, scraper, course, rno, date_str, race_url=race['url'], deadline=race['time'])
        
        if status == 1:
            hit_count += 1
            # Discord通知処理 (フォーマットを調整)
            content = f"🎯 **投資チャンス到来！**\n📍 **{res['場名']} {res['レース']}** (締切 {res['締切']})\n"
            content += f"━━━━━━━━━━━━━━━━━━━━\n🔥 戦略: **{res['戦略']}**\n😱 イン飛び率: `{res['イン飛び率']:.1%}`\n\n"
            content += f"🏠 **1号艇勝率**: `{res['1号艇勝率']:.1%}`\n"
            
            content += f"📊 **AI勝率ランキング (1抜き)**\n"
            content += f"🥇 **{res['1位'][0]}号艇**: `{res['1位'][1]:.1%}`\n"
            content += f"🥈 **{res['2位'][0]}号艇**: `{res['2位'][1]:.1%}`\n"
            content += f"🥉 **{res['3位'][0]}号艇**: `{res['3位'][1]:.1%}`\n"
            
            for idx, item in enumerate(res["4位以下"], start=4):
                content += f"{idx}位 **{item[0]}号艇**: `{item[1]:.1%}`\n"
            
            content += "\n"
            if res["期待値MAX"] is not None:
                content += f"📈 最大期待値: `{res['期待値MAX']:.2f}`\n"
            content += f"📝 根拠: {res['根拠']}\n💰 推奨({res['点数']}点): `{res['買い目']}`\n━━━━━━━━━━━━━━━━━━━━"

            if DISCORD_WEBHOOK_URL:
                try:
                    response = requests.post(
                        DISCORD_WEBHOOK_URL,
                        json={"content": content},
                        timeout=15
                    )

                    if 200 <= response.status_code < 300:
                        print(f"    ✅ Notification Sent for {race_id}")
                    else:
                        print(f"    ❌ Discord Error for {race_id}: status={response.status_code}")
                        print(f"    Response: {response.text[:300]}")

                except Exception as e:
                    print(f"    ❌ Discord Exception for {race_id}: {e}")
            else:
                print("    ⚠️ DISCORD_WEBHOOK_URL is not set")

            save_prediction_log(race_id, race, res, run_at)
            
            # 通知済みリストに保存
            save_notified_race(race_id)
        time.sleep(1)

    print(f"👮 Patrol Finished: Found {hit_count} hits.")

if __name__ == "__main__":
    run_live_patrol()
