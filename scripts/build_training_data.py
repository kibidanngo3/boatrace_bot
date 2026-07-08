"""過去のレースから特徴量+正解ラベル(勝った艇)を集めて再学習用データセットを作る。

長時間(数時間〜数十時間)かかるスクレイピングなので、途中で止めても再実行すれば
既に取得済みのレースをスキップして続きから再開できる。

使い方:
    python scripts/build_training_data.py --start-date 20260101 --end-date 20260107
    python scripts/build_training_data.py --start-date 20250701 --end-date 20260701 --output training_data.csv --workers 6
"""
import argparse
import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import BoatRaceScraperV5, JST  # noqa: E402

RAW_FIELDS = [
    "avg_st", "national_win_rate", "national_2_rate", "national_3_rate",
    "local_win_rate", "local_2_rate", "local_3_rate",
    "motor_no", "motor_2_rate", "motor_3_rate",
    "boat_no", "boat_2_rate", "boat_3_rate",
    "weight", "tilt", "parts_count", "parts",
    "ex_time", "st", "rank", "win_rate",
]

FIELDNAMES = ["date", "course", "rno", "deadline", "wind_speed", "wave", "label"]
for boat_no in range(1, 7):
    FIELDNAMES.extend(f"{field}_{boat_no}" for field in RAW_FIELDS)

_thread_local = threading.local()


def get_scraper():
    if not hasattr(_thread_local, "scraper"):
        _thread_local.scraper = BoatRaceScraperV5()
    return _thread_local.scraper


def load_done_race_ids(output_path):
    if not output_path.exists():
        return set()
    with open(output_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {f"{r['date']}_{r['course']}_{r['rno']}" for r in rows}


def flatten_row(date_str, course, rno, deadline, data, label):
    row = {
        "date": date_str,
        "course": course,
        "rno": rno,
        "deadline": deadline,
        "wind_speed": data.get("wind_speed", ""),
        "wave": data.get("wave", ""),
        "label": label,
    }
    for boat_no in range(1, 7):
        for field in RAW_FIELDS:
            row[f"{field}_{boat_no}"] = data.get(f"{field}_{boat_no}", "")
    return row


def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def process_race(date_str, course, rno, time_str, race_url, delay):
    scraper = get_scraper()
    try:
        data = scraper.fetch_race_data(course, rno, date_str, race_url=race_url, deadline=time_str)
        time.sleep(delay)
        if not data:
            return "fail", None

        result = scraper.fetch_race_result(course, rno, date_str)
        time.sleep(delay)
        if not result:
            return "fail", None

        label = int(result["ticket"].split("-")[0])
        row = flatten_row(date_str, course, rno, time_str, data, label)
        return "ok", row
    except Exception as e:
        print(f"  ⚠️ {course} {rno}R failed: {e}")
        return "fail", None


def main():
    parser = argparse.ArgumentParser(description="過去レースの特徴量+結果を収集して学習データセットを作る")
    parser.add_argument("--start-date", required=True, help="YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="YYYYMMDD")
    parser.add_argument("--output", default="training_data.csv")
    parser.add_argument("--delay", type=float, default=0.5, help="リクエスト間の待機秒数(ワーカーごと)")
    parser.add_argument("--workers", type=int, default=1, help="並列ワーカー数")
    parser.add_argument("--limit", type=int, default=None, help="検証用: 処理するレース数の上限")
    args = parser.parse_args()

    output_path = Path(args.output)
    done_ids = load_done_race_ids(output_path)
    print(f"既に取得済み: {len(done_ids)}件 (スキップされます)")

    file_exists = output_path.exists()
    out_f = open(output_path, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
        out_f.flush()

    write_lock = threading.Lock()
    schedule_scraper = BoatRaceScraperV5()

    start = datetime.strptime(args.start_date, "%Y%m%d").date()
    end = datetime.strptime(args.end_date, "%Y%m%d").date()

    total_ok = 0
    total_skip = 0
    total_fail = 0

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for d in daterange(start, end):
                date_str = d.strftime("%Y%m%d")
                print(f"[{datetime.now(JST).strftime('%H:%M:%S')}] === {date_str} ===")

                try:
                    all_races = schedule_scraper.fetch_all_venue_schedules(date_str)
                except Exception as e:
                    print(f"  ❌ schedule fetch failed for {date_str}: {e}")
                    continue

                tasks = []
                for (course, rno), (time_str, race_url) in sorted(all_races.items()):
                    race_id = f"{date_str}_{course}_{rno}"
                    if race_id in done_ids:
                        total_skip += 1
                        continue
                    tasks.append((course, rno, time_str, race_url))

                futures = [
                    executor.submit(process_race, date_str, course, rno, time_str, race_url, args.delay)
                    for course, rno, time_str, race_url in tasks
                ]

                for future in as_completed(futures):
                    status, row = future.result()
                    if status == "ok":
                        with write_lock:
                            writer.writerow(row)
                            out_f.flush()
                        total_ok += 1
                    else:
                        total_fail += 1

                print(f"  progress: ok={total_ok} skip={total_skip} fail={total_fail}")

                if args.limit and (total_ok + total_fail) >= args.limit:
                    print(f"  --limit {args.limit} に達したので終了します")
                    return
    finally:
        out_f.close()

    print(f"完了: ok={total_ok} skip={total_skip} fail={total_fail}")


if __name__ == "__main__":
    main()
