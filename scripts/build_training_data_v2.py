"""公式データ配信(mbrace.or.jp)のB/Kファイル(1日1アーカイブ・全会場分)から学習データを作る。

boatrace.jpを1レースずつスクレイピングする build_training_data.py に比べて
1日あたりのHTTPリクエスト数を2回(B/K各1回)に減らせるため、桁違いに速い。

使い方:
    python scripts/build_training_data_v2.py --start-date 20250711 --end-date 20260710 --output training_data.csv
"""
import argparse
import csv
import io
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import lhafile
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.build_training_data import FIELDNAMES as _BASE_FIELDNAMES, RAW_FIELDS  # noqa: E402
from main import BoatRaceScraperV5  # noqa: E402

KNOWN_VENUES = sorted(BoatRaceScraperV5.COURSE_MAP.keys(), key=len, reverse=True)
_label_idx = _BASE_FIELDNAMES.index("label")
_wave_idx = _BASE_FIELDNAMES.index("wave")
FIELDNAMES = (
    _BASE_FIELDNAMES[: _wave_idx] + ["wind_dir"] + _BASE_FIELDNAMES[_wave_idx : _label_idx + 1]
    + ["label_2nd", "label_3rd"] + _BASE_FIELDNAMES[_label_idx + 1 :]
)

B_BASE = "https://www1.mbrace.or.jp/od2/B"
K_BASE = "https://www1.mbrace.or.jp/od2/K"

RANK_RE = r"(A1|A2|B1|B2)"
RATE_RE = r"\d{1,3}\.\d{2}"
BOAT_LINE_RE = re.compile(
    r"^(?P<boatno>\d) (?P<no>\d{4})(?P<name>.+?)(?P<age>\d{2})(?P<branch>..)"
    r"(?P<weight>\d{2,3})(?P<rank>" + RANK_RE + r")\s*"
    r"(?P<win_rate>" + RATE_RE + r")\s*(?P<national_2_rate>" + RATE_RE + r")\s*"
    r"(?P<local_win_rate>" + RATE_RE + r")\s*(?P<local_2_rate>" + RATE_RE + r")\s*"
    r"(?P<motor_no>\d+)\s*(?P<motor_2_rate>" + RATE_RE + r")\s*"
    r"(?P<boat_no>\d+)\s*(?P<boat_2_rate>" + RATE_RE + r")"
)
K_BOAT_LINE_RE = re.compile(
    r"^\s*\d{2}\s+(?P<boatno>\d)\s+\d{4}\s+.+?\s+\d+\s+\d+\s+(?P<ex_time>[\d.]+)\s+\d\s+(?P<st>[\d.]+)\s+"
)
RACE_HEADER_RE = re.compile(r"^\s*(\d{1,2})Ｒ")
K_RACE_HEADER_RE = re.compile(r"^\s*(\d{1,2})R\s")
PAYOUT_LINE_RE = re.compile(r"^\s*(\d{1,2})R\s+(\d-\d-\d)\s+(\d+)")
VENUE_RE = re.compile(r"ボートレース(.+)$")
WIND_RE = re.compile(r"風\s*(?P<dir>\S*?)\s*(?P<speed>\d+)m")
WAVE_RE = re.compile(r"波\s*\S*?(\d+)cm")
DEADLINE_RE = re.compile(r"締切予定([0-9０-９]{1,2})[:：]([0-9０-９]{2})")


def z2h(s):
    return unicodedata.normalize("NFKC", s)


def clean_venue(name):
    blob = re.sub(r"\s", "", name)
    for venue in KNOWN_VENUES:
        if blob.startswith(venue):
            return venue
    return None


def fetch_lha_text(url, session):
    res = session.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    if res.status_code != 200 or len(res.content) < 100:
        return None
    lha = lhafile.LhaFile(io.BytesIO(res.content))
    info = lha.infolist()
    if not info:
        return None
    return lha.read(info[0].filename).decode("cp932", errors="replace")


def parse_b_file(text):
    """{venue: {rno: {boatno: {rank, win_rate}}}}"""
    result = {}
    venue = None
    rno = None
    for line in text.splitlines():
        vm = VENUE_RE.search(line)
        if vm and "＊＊＊" not in line and "内容については" not in line:
            venue = clean_venue(vm.group(1))
            if venue:
                result.setdefault(venue, {})
            continue
        rm = RACE_HEADER_RE.match(line)
        if rm:
            rno = int(rm.group(1))
            if venue:
                result[venue].setdefault(rno, {})
            continue
        bm = BOAT_LINE_RE.match(line)
        if bm and venue and rno:
            boatno = int(bm.group("boatno"))
            result[venue][rno][boatno] = {
                "rank": bm.group("rank"),
                "win_rate": bm.group("win_rate"),
                "national_2_rate": bm.group("national_2_rate"),
                "local_win_rate": bm.group("local_win_rate"),
                "local_2_rate": bm.group("local_2_rate"),
                "motor_2_rate": bm.group("motor_2_rate"),
                "boat_2_rate": bm.group("boat_2_rate"),
                "weight": bm.group("weight"),
            }
    return result


def parse_k_file(text):
    """{venue: {rno: {"wind_speed":.., "wave":.., "label":.., "boats": {boatno: {ex_time, st}}}}}"""
    result = {}
    venue = None
    rno = None
    payouts = {}
    for line in text.splitlines():
        vm = VENUE_RE.search(line)
        if vm and "＊＊＊" not in line and "内容については" not in line and "[払戻金]" not in line:
            venue = clean_venue(vm.group(1))
            if venue:
                result.setdefault(venue, {})
                payouts.setdefault(venue, {})
            continue
        pm = PAYOUT_LINE_RE.match(line)
        if pm and venue:
            payouts[venue][int(pm.group(1))] = pm.group(2)
            continue
        rm = K_RACE_HEADER_RE.match(line)
        if rm:
            rno = int(rm.group(1))
            wind = WIND_RE.search(line)
            wave = WAVE_RE.search(line)
            if venue:
                result[venue].setdefault(rno, {"boats": {}})
                result[venue][rno]["wind_speed"] = int(wind.group("speed")) if wind else 0
                result[venue][rno]["wind_dir"] = wind.group("dir") if wind else ""
                result[venue][rno]["wave"] = int(wave.group(1)) if wave else 0
            continue
        bm = K_BOAT_LINE_RE.match(line)
        if bm and venue and rno:
            boatno = int(bm.group("boatno"))
            result[venue][rno]["boats"][boatno] = {
                "ex_time": bm.group("ex_time"),
                "st": bm.group("st"),
            }

    for venue, races in payouts.items():
        for rno, ticket in races.items():
            if venue in result and rno in result[venue]:
                first, second, third = [int(x) for x in ticket.split("-")]
                result[venue][rno]["label"] = first
                result[venue][rno]["label_2nd"] = second
                result[venue][rno]["label_3rd"] = third
    return result


def build_rows(date_str, b_data, k_data):
    rows = []
    for venue, races in k_data.items():
        b_races = b_data.get(venue, {})
        for rno, kinfo in races.items():
            if "label" not in kinfo or len(kinfo["boats"]) < 6:
                continue
            b_boats = b_races.get(rno, {})
            if len(b_boats) < 6:
                continue
            row = {
                "date": date_str,
                "course": venue,
                "rno": rno,
                "deadline": "",
                "wind_speed": kinfo.get("wind_speed", 0),
                "wind_dir": kinfo.get("wind_dir", ""),
                "wave": kinfo.get("wave", 0),
                "label": kinfo["label"],
                "label_2nd": kinfo["label_2nd"],
                "label_3rd": kinfo["label_3rd"],
            }
            ok = True
            for boat_no in range(1, 7):
                for field in RAW_FIELDS:
                    row[f"{field}_{boat_no}"] = ""
                b = b_boats.get(boat_no)
                k = kinfo["boats"].get(boat_no)
                if not b or not k:
                    ok = False
                    break
                row[f"rank_{boat_no}"] = b["rank"]
                row[f"win_rate_{boat_no}"] = b["win_rate"]
                row[f"national_2_rate_{boat_no}"] = b["national_2_rate"]
                row[f"local_win_rate_{boat_no}"] = b["local_win_rate"]
                row[f"local_2_rate_{boat_no}"] = b["local_2_rate"]
                row[f"motor_2_rate_{boat_no}"] = b["motor_2_rate"]
                row[f"boat_2_rate_{boat_no}"] = b["boat_2_rate"]
                row[f"weight_{boat_no}"] = b["weight"]
                row[f"ex_time_{boat_no}"] = k["ex_time"]
                row[f"st_{boat_no}"] = k["st"]
            if ok:
                rows.append(row)
    return rows


def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def load_done_dates(output_path):
    if not output_path.exists():
        return set()
    with open(output_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {r["date"] for r in rows}


def main():
    parser = argparse.ArgumentParser(description="B/Kファイルから学習データセットを作る")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", default="training_data.csv")
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    output_path = Path(args.output)
    done_dates = load_done_dates(output_path)
    print(f"既に取得済み: {len(done_dates)}日分 (スキップされます)")

    file_exists = output_path.exists()
    out_f = open(output_path, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    session = requests.Session()
    start = datetime.strptime(args.start_date, "%Y%m%d").date()
    end = datetime.strptime(args.end_date, "%Y%m%d").date()

    total_races = 0
    total_days = 0
    fail_days = 0

    for d in daterange(start, end):
        date_str = d.strftime("%Y%m%d")
        if date_str in done_dates:
            continue
        yyyymm = date_str[:6]
        yymmdd = date_str[2:]
        b_url = f"{B_BASE}/{yyyymm}/b{yymmdd}.lzh"
        k_url = f"{K_BASE}/{yyyymm}/k{yymmdd}.lzh"
        try:
            b_text = fetch_lha_text(b_url, session)
            time.sleep(args.delay)
            k_text = fetch_lha_text(k_url, session)
            time.sleep(args.delay)
            if not b_text or not k_text:
                fail_days += 1
                print(f"[{date_str}] データなし/取得失敗 (開催なしの可能性)")
                continue
            b_data = parse_b_file(b_text)
            k_data = parse_k_file(k_text)
            rows = build_rows(date_str, b_data, k_data)
            for row in rows:
                writer.writerow(row)
            out_f.flush()
            total_races += len(rows)
            total_days += 1
            print(f"[{date_str}] {len(rows)}レース ok=累計{total_races}")
        except Exception as e:
            fail_days += 1
            print(f"[{date_str}] ERROR: {e}")

    out_f.close()
    print(f"完了: 日数={total_days} 失敗/開催なし={fail_days} 総レース数={total_races}")


if __name__ == "__main__":
    main()
