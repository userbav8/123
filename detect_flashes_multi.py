#!/usr/bin/env python3
"""
detect_flashes_multi.py — детектирование LED-вспышек для нескольких сессий.

Запуск из папки с данными:
    python detect_flashes_multi.py --roi 3848 3049 150 150 16 3303 150 150

Скрипт автоматически найдёт в текущей папке:
  - session_events*.jsonl  (лог LED-вспышек)
  - video_01.mp4 .. video_NN.mp4  (видео сессий, сортируются по имени)

Для каждого видео video_NN.mp4 предполагается run_id = NN (число из имени файла).

Результаты сохраняются в ту же папку:
  video_NN_led_log_pulses.csv
  video_NN_video_flashes.csv
  video_NN_flash_alignment.csv

Опциональные аргументы (переопределяют автопоиск):
  --events       путь к jsonl-файлу (если несколько или имя нестандартное)
  --video-dir    папка с видео (если другая)
  --out-dir      папка для результатов (по умолчанию = папка с данными)
  --debug-roi-dir  папка для ROI-кропов
  --run-id-map   переопределение соответствия видео→run_id, например:
                 video_01.mp4=1,video_02.mp4=3
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import struct
import calendar
from bisect import bisect_right
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np

# ── Пороги детектирования ──────────────────────────────────────────────────[...]
LOWER_RED1 = np.array([0, 120, 200], dtype=np.uint8)
UPPER_RED1 = np.array([10, 255, 255], dtype=np.uint8)
LOWER_RED2 = np.array([165, 120, 200], dtype=np.uint8)
UPPER_RED2 = np.array([180, 255, 255], dtype=np.uint8)

HARD_ON_AREA = 5
SOFT_ON_AREA = 3
MIN_LARGEST_BLOB_AREA = 4
PREBUFFER_FRAMES = 20
FLASH_CLEAR_FRAMES_BEFORE = 4
ONSET_NOISE_AREA = 4
FLASH_SPIKE_RATIO = 2

FIRST_FLASH_IGNORE_SEC = 0.3
SEARCH_WINDOW_BEFORE_SEC = 0.45
SEARCH_WINDOW_AFTER_SEC = 1.25

DEFAULT_MORPH_MODE = "none"

VALID_PULSE_TYPES = {
    "video_sync_pulse_on",
    "video_sync_blink",
    "videosyncblink",
}

# ── Утилиты ────────────────────────────────────────────────────────────[...]
def safe_float(v, default=None):
    try:
        if v is None or v == "": return default
        return float(v)
    except Exception:
        return default

def safe_int(v, default=None):
    try:
        if v is None or v == "": return default
        return int(float(v))
    except Exception:
        return default

def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))

def parse_iso_utc(s):
    s = str(s or "").strip()
    if not s: return None
    if s.endswith("Z"): s = s[:-1] + "+00:00"
    m = re.match(r"^(.*T\d\d:\d\d:\d\d)\.(\d+)([+-]\d\d:\d\d)$", s)
    if m:
        s = f"{m.group(1)}.{m.group(2)[:6].ljust(6,'0')}{m.group(3)}"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def iso_utc_from_ts(ts, timespec="milliseconds"):
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec=timespec).replace("+00:00", "Z")

def ns_to_iso(ns):
    if ns is None:
        return ""
    sec = int(ns // 1_000_000_000)
    nsec = int(ns % 1_000_000_000)
    dt = datetime.fromtimestamp(sec, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{nsec:09d}Z"

def datetime_to_ns(year, month, day, hour, minute, second, nano):
    try:
        dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), tzinfo=timezone.utc)
        sec = calendar.timegm(dt.timetuple())
        return sec * 1_000_000_000 + int(nano)
    except Exception:
        return None

def parse_int(x, default=None):
    try:
        if x is None or x == "":
            return default
        return int(float(str(x).strip()))
    except Exception:
        return default

def find_column(fieldnames, candidates, contains_any=None):
    lower_map = {name.lower(): name for name in fieldnames}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if contains_any:
        for name in fieldnames:
            low = name.lower()
            for part in contains_any:
                if part.lower() in low:
                    return name
    return None

def role_from_text(s):
    if s is None:
        return None
    x = str(s).lower()
    if "openimu" in x or "imu" in x:
        return "imu"
    if "gnss0" in x or "gnss_base" in x or "gnssbase" in x or "base" in x:
        return "gnss0"
    if "gnss1" in x or "gnss_rover" in x or "gnssrover" in x or "rover" in x:
        return "gnss1"
    return None

class ChunkIndex:
    def __init__(self):
        self.by_role = {}
        self.starts = {}

    def add(self, role, start, size, host_ns):
        if role is None or start is None or size is None or host_ns is None:
            return
        if size <= 0:
            return
        self.by_role.setdefault(role, []).append((start, start + size, host_ns))

    def finalize(self):
        self.starts = {}
        for role in self.by_role:
            self.by_role[role].sort(key=lambda x: x[0])
            self.starts[role] = [x[0] for x in self.by_role[role]]

    def host_for_offset(self, role, offset):
        arr = self.by_role.get(role)
        if not arr:
            return None
        starts = self.starts.get(role, [])
        i = bisect_right(starts, offset) - 1
        if i < 0 or i >= len(arr):
            return None
        start, end, host_ns = arr[i]
        if start <= offset < end:
            return host_ns
        return None

def load_serial_chunks(path):
    idx = ChunkIndex()
    if not path or not os.path.exists(path):
        return idx

    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        except Exception:
            dialect = csv.excel_tab
        reader = csv.DictReader(f, dialect=dialect)
        fieldnames = reader.fieldnames or []

        source_col = find_column(fieldnames, ["source", "stream", "device", "port", "name"], ["source", "stream"])
        file_path_col = find_column(fieldnames, ["file_path", "file", "path", "filename"], ["file_path", "file"])
        host_col = find_column(fieldnames, ["host_monotonic_ns", "monotonic_ns", "host_ns", "time_ns", "read_end_ns", "read_mid_ns"], ["host_monotonic", "monotonic", "read_end", "read_mid"])
        offset_col = find_column(fieldnames, ["file_offset", "offset", "start_offset", "write_offset"], ["file_offset", "offset"])
        size_col = find_column(fieldnames, ["nbytes", "num_bytes", "bytes", "length", "len", "size"], ["nbytes", "bytes", "size"])

        cumulative = {}
        for row in reader:
            role = None
            if source_col is not None:
                role = role_from_text(row.get(source_col))
            if role is None and file_path_col is not None:
                role = role_from_text(row.get(file_path_col))
            if role is None:
                continue

            host_ns = parse_int(row.get(host_col))
            size = parse_int(row.get(size_col))
            if size is None or host_ns is None:
                continue

            if offset_col is not None:
                start = parse_int(row.get(offset_col))
                if start is None:
                    start = cumulative.get(role, 0)
            else:
                start = cumulative.get(role, 0)

            idx.add(role, start, size, host_ns)
            cumulative[role] = max(cumulative.get(role, 0), start + size)

    idx.finalize()
    return idx

def ubx_checksum(buf):
    ck_a = ck_b = 0
    for b in buf:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return ck_a, ck_b

def iter_ubx(path):
    with open(path, "rb") as f:
        data = f.read()
    i = 0
    n = len(data)
    while i < n:
        j = data.find(b"\xB5\x62", i)
        if j < 0 or j + 6 > n:
            break
        cls_id = data[j + 2]
        msg_id = data[j + 3]
        length = data[j + 4] | (data[j + 5] << 8)
        end = j + 6 + length + 2
        if end > n:
            break
        ck_a_file, ck_b_file = data[end - 2], data[end - 1]
        ck_a_calc, ck_b_calc = ubx_checksum(data[j + 2:end - 2])
        if ck_a_calc != ck_a_file or ck_b_calc != ck_b_file:
            i = j + 2
            continue
        payload = data[j + 6:j + 6 + length]
        yield j, cls_id, msg_id, payload
        i = end

def parse_nav_pvt(payload):
    if len(payload) < 92:
        return None
    i_tow = struct.unpack_from("<I", payload, 0)[0]
    year = struct.unpack_from("<H", payload, 4)[0]
    month = payload[6]
    day = payload[7]
    hour = payload[8]
    minute = payload[9]
    second = payload[10]
    valid = payload[11]
    nano = struct.unpack_from("<i", payload, 16)[0]
    utc_ns = datetime_to_ns(year, month, day, hour, minute, second, nano if nano >= 0 else 0)
    return {"itow_ms": i_tow, "utc_ns": utc_ns, "valid": valid}

def load_gnss_pvt(ubx_path, role, chunks):
    rows = []
    if not ubx_path or not os.path.exists(ubx_path):
        return rows
    for offset, cls_id, msg_id, payload in iter_ubx(ubx_path):
        if cls_id != 0x01 or msg_id != 0x07:
            continue
        host_ns = chunks.host_for_offset(role, offset)
        if host_ns is None:
            continue
        row = parse_nav_pvt(payload)
        if not row or row["utc_ns"] is None:
            continue
        rows.append({"host_ns": host_ns, "utc_ns": row["utc_ns"], "itow_ms": row["itow_ms"], "valid": row["valid"]})
    rows.sort(key=lambda r: r["host_ns"])
    return rows

def fit_utc_from_pvt(pvt_rows):
    good = [r for r in pvt_rows if r.get("host_ns") is not None and r.get("utc_ns") is not None]
    if len(good) < 2:
        return None
    xs = [float(r["host_ns"]) for r in good]
    ys = [float(r["utc_ns"]) for r in good]
    x0 = xs[0]
    y0 = ys[0]
    x = np.array([v - x0 for v in xs], dtype=np.float64)
    y = np.array([v - y0 for v in ys], dtype=np.float64)
    a, b = np.polyfit(x, y, 1)
    def host_to_utc_ns(host_ns):
        return int(round(y0 + (a * (float(host_ns) - x0) + b)))
    return host_to_utc_ns

def load_timebase_from_session(events_path):
    session_dir = os.path.dirname(os.path.abspath(events_path))
    serial_chunks_path = os.path.join(session_dir, "serial_chunks.tsv")
    gnss0_path = os.path.join(session_dir, "gnss0.ubx")
    gnss1_path = os.path.join(session_dir, "gnss1.ubx")

    chunks = load_serial_chunks(serial_chunks_path)
    pvt0 = load_gnss_pvt(gnss0_path, "gnss0", chunks)
    pvt1 = load_gnss_pvt(gnss1_path, "gnss1", chunks)

    if len(pvt0) >= len(pvt1):
        pvt = pvt0
        source = "gnss0_pvt_fit" if pvt0 else ""
    else:
        pvt = pvt1
        source = "gnss1_pvt_fit" if pvt1 else ""

    fn = fit_utc_from_pvt(pvt)
    return fn, source, len(pvt)

def parse_pos_gpst_datetime(date_s, time_s, gps_utc_offset_sec=18.0):
    text = f"{date_s} {time_s}"
    try:
        dt = datetime.strptime(text, "%Y/%m/%d %H:%M:%S.%f")
    except ValueError:
        dt = datetime.strptime(text, "%Y/%m/%d %H:%M:%S")
    dt = dt.replace(tzinfo=timezone.utc)
    utc_dt = dt - timedelta(seconds=gps_utc_offset_sec)
    return dt, utc_dt

def read_pos_file(pos_path, gps_utc_offset_sec=18.0):
    rows = []
    if not pos_path or not os.path.exists(pos_path):
        return rows
    with open(pos_path, "r", encoding="utf-8", errors="replace") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                gpst_dt, utc_dt = parse_pos_gpst_datetime(parts[0], parts[1], gps_utc_offset_sec)
                rows.append({
                    "line_no": line_no,
                    "gpst_datetime": gpst_dt,
                    "utc_datetime": utc_dt,
                    "utc_ts": utc_dt.timestamp(),
                    "lat_deg": float(parts[2]),
                    "lon_deg": float(parts[3]),
                    "height_m": float(parts[4]),
                    "Q": int(float(parts[5])),
                    "ns": int(float(parts[6])),
                    "sdn_m": float(parts[7]),
                    "sde_m": float(parts[8]),
                    "sdu_m": float(parts[9]),
                })
            except Exception:
                continue
    rows.sort(key=lambda r: r["utc_ts"])
    return rows

def enrich_pairs_with_pos_anchor_info(pairs, pos_rows, exact_tol_sec=0.1, interp_max_gap_sec=2.0):
    if not pairs:
        return pairs

    if not pos_rows:
        for p in pairs:
            p["anchor_pos_line_no"] = ""
            p["anchor_pos_exact"] = 0
            p["anchor_pos_interpolated"] = 0
            p["anchor_lat_deg"] = None
            p["anchor_lon_deg"] = None
            p["anchor_height_m"] = None
            p["anchor_Q"] = None
            p["anchor_ns"] = None
            p["anchor_sdn_m"] = None
            p["anchor_sde_m"] = None
            p["anchor_sdu_m"] = None
            p["anchor_is_spatial"] = 0
        return pairs

    utc_ts_list = [r["utc_ts"] for r in pos_rows]

    for p in pairs:
        t = p.get("log_utc_ts")
        if t is None:
            p["anchor_pos_line_no"] = ""
            p["anchor_pos_exact"] = 0
            p["anchor_pos_interpolated"] = 0
            p["anchor_lat_deg"] = None
            p["anchor_lon_deg"] = None
            p["anchor_height_m"] = None
            p["anchor_Q"] = None
            p["anchor_ns"] = None
            p["anchor_sdn_m"] = None
            p["anchor_sde_m"] = None
            p["anchor_sdu_m"] = None
            p["anchor_is_spatial"] = 0
            continue

        i = bisect_right(utc_ts_list, t)

        prev_row = pos_rows[i - 1] if i - 1 >= 0 else None
        next_row = pos_rows[i] if i < len(pos_rows) else None

        nearest = None
        nearest_dt = None
        for row in (prev_row, next_row):
            if row is None:
                continue
            dt = abs(row["utc_ts"] - t)
            if nearest is None or dt < nearest_dt:
                nearest = row
                nearest_dt = dt

        exact = int(nearest is not None and nearest_dt is not None and nearest_dt <= exact_tol_sec)

        interp_used = False
        interp_line_no = ""
        lat = lon = h = None
        q = ns = sdn = sde = sdu = None

        if exact:
            lat = nearest["lat_deg"]
            lon = nearest["lon_deg"]
            h = nearest["height_m"]
            q = nearest["Q"]
            ns = nearest["ns"]
            sdn = nearest["sdn_m"]
            sde = nearest["sde_m"]
            sdu = nearest["sdu_m"]
            interp_line_no = nearest["line_no"]

        elif prev_row is not None and next_row is not None:
            t0 = prev_row["utc_ts"]
            t1 = next_row["utc_ts"]
            gap = t1 - t0

            if gap > 0 and gap <= interp_max_gap_sec and t0 <= t <= t1:
                alpha = (t - t0) / gap

                lat = prev_row["lat_deg"] + alpha * (next_row["lat_deg"] - prev_row["lat_deg"])
                lon = prev_row["lon_deg"] + alpha * (next_row["lon_deg"] - prev_row["lon_deg"])
                h = prev_row["height_m"] + alpha * (next_row["height_m"] - prev_row["height_m"])

                q = max(prev_row["Q"], next_row["Q"])
                ns = min(prev_row["ns"], next_row["ns"])
                sdn = max(prev_row["sdn_m"], next_row["sdn_m"])
                sde = max(prev_row["sde_m"], next_row["sde_m"])
                sdu = max(prev_row["sdu_m"], next_row["sdu_m"])

                interp_used = True
                interp_line_no = f"{prev_row['line_no']}|{next_row['line_no']}"
            elif nearest is not None:
                lat = nearest["lat_deg"]
                lon = nearest["lon_deg"]
                h = nearest["height_m"]
                q = nearest["Q"]
                ns = nearest["ns"]
                sdn = nearest["sdn_m"]
                sde = nearest["sde_m"]
                sdu = nearest["sdu_m"]
                interp_line_no = nearest["line_no"]

        elif nearest is not None:
            lat = nearest["lat_deg"]
            lon = nearest["lon_deg"]
            h = nearest["height_m"]
            q = nearest["Q"]
            ns = nearest["ns"]
            sdn = nearest["sdn_m"]
            sde = nearest["sde_m"]
            sdu = nearest["sdu_m"]
            interp_line_no = nearest["line_no"]

        p["anchor_pos_line_no"] = interp_line_no
        p["anchor_pos_exact"] = exact
        p["anchor_pos_interpolated"] = int(interp_used)
        p["anchor_lat_deg"] = lat
        p["anchor_lon_deg"] = lon
        p["anchor_height_m"] = h
        p["anchor_Q"] = q
        p["anchor_ns"] = ns
        p["anchor_sdn_m"] = sdn
        p["anchor_sde_m"] = sde
        p["anchor_sdu_m"] = sdu

        p["anchor_is_spatial"] = int(
            p.get("detected_on") in ("both", "left", "right") and
            (exact == 1 or interp_used) and
            q == 1 and
            sdn is not None and sde is not None and sdu is not None and
            max(sdn, sde, sdu) <= 0.05
        )

    return pairs
    
# ── ROI ────────────────────────────────────────────────────────────────[...]
def build_wrap_segments(xc, w, frame_w):
    half = int(w // 2)
    raw_x1 = int(xc - half)
    raw_x2 = raw_x1 + int(w)
    if raw_x1 >= 0 and raw_x2 <= frame_w:
        return [(raw_x1, raw_x2)]
    return [(raw_x1 % frame_w, frame_w), (0, raw_x2 % frame_w)]

def build_panorama_roi(xc, yc, w, h, frame_w, frame_h):
    y1 = clamp(int(yc - h // 2), 0, frame_h - 1)
    y2 = clamp(int(yc + h // 2), 0, frame_h)
    if y2 <= y1: y2 = min(frame_h, y1 + 1)
    return {"segments": build_wrap_segments(xc, w, frame_w),
            "y1": y1, "y2": y2, "xc": int(xc), "yc": int(yc), "w": int(w), "h": int(h)}

def extract_panorama_roi(frame, roi_info):
    y1, y2 = roi_info["y1"], roi_info["y2"]
    parts = [frame[y1:y2, sx1:sx2] for sx1, sx2 in roi_info["segments"] if sx2 > sx1]
    if not parts: raise RuntimeError("Пустой ROI")
    return parts[0] if len(parts) == 1 else np.hstack(parts)

def analyze_roi(frame, roi_info, kernel, morph_mode=DEFAULT_MORPH_MODE):
    roi = extract_panorama_roi(frame, roi_info)
    hsv = cv2.cvtColor(cv2.GaussianBlur(roi, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, LOWER_RED1, UPPER_RED1),
                          cv2.inRange(hsv, LOWER_RED2, UPPER_RED2))
    if (morph_mode or DEFAULT_MORPH_MODE) == "none":
        mask = cv2.dilate(mask, kernel, iterations=1)
    else:
        mask = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1), kernel, iterations=1)
    total = int(cv2.countNonZero(mask))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest = max((int(stats[l, cv2.CC_STAT_AREA]) for l in range(1, len(stats))), default=0)
    return total, largest

# ── Логика вспышек ──────────────────────────────────────────────────────[...]
def is_clean_onset(history, detected_on):
    if len(history) < 2: return True
    last = history[-1]
    cur_area = max(
        last["left_area"] if detected_on in ("both", "left") else 0,
        last["right_area"] if detected_on in ("both", "right") else 0
    )
    if cur_area == 0: return False
    prev_max = 0
    for i in range(1, min(FLASH_CLEAR_FRAMES_BEFORE, len(history) - 1) + 1):
        item = history[-(i + 1)]
        if detected_on in ("both", "left"): prev_max = max(prev_max, item["left_area"])
        if detected_on in ("both", "right"): prev_max = max(prev_max, item["right_area"])
    return not (prev_max >= ONSET_NOISE_AREA and cur_area < prev_max * FLASH_SPIKE_RATIO)

def refine_flash_from_history(history):
    last = history[-1]
    ls = last["left_area"] >= SOFT_ON_AREA and last["left_blob"] >= 1
    rs = last["right_area"] >= SOFT_ON_AREA and last["right_blob"] >= 1
    detected_on = "both" if (ls and rs) else ("left" if ls else ("right" if rs else "unknown"))
    best = history[-1]
    for item in reversed(history):
        ls2 = item["left_area"] >= SOFT_ON_AREA and item["left_blob"] >= 1
        rs2 = item["right_area"] >= SOFT_ON_AREA and item["right_blob"] >= 1
        if   detected_on == "both"  and (ls2 and rs2): best = item
        elif detected_on == "left"  and ls2:            best = item
        elif detected_on == "right" and rs2:            best = item
        else: break
    return {"time_sec": best["time_sec"], "frame_idx": best["frame_idx"],
            "detected_on": detected_on, "left_area": best["left_area"],
            "right_area": best["right_area"], "left_blob": best["left_blob"],
            "right_blob": best["right_blob"]}

def _save_roi_debug(history, result, left_roi, right_roi, debug_dir, flash_index, label):
    os.makedirs(debug_dir, exist_ok=True)
    ffi = result["frame_idx"]
    frame = next((item.get("_frame") for item in history if item["frame_idx"] == ffi), None)
    if frame is None and history: frame = history[-1].get("_frame")
    if frame is None: return
    t_str = f"{result['time_sec']:.3f}".replace(".", "_")
    prefix = f"flash_{flash_index:03d}_{label}_f{ffi}_t{t_str}"
    for roi, side in [(left_roi, "LEFT"), (right_roi, "RIGHT")]:
        try:
            cv2.imwrite(os.path.join(debug_dir, f"{prefix}_{side}.jpg"),
                        extract_panorama_roi(frame, roi))
        except Exception as e:
            print(f"  WARNING: ROI {side}: {e}")

def find_first_flash(cap, fps, frame_count, left_roi, right_roi, kernel, debug_dir=None):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    history = []
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret: break
        t = frame_idx / fps
        la, lb = analyze_roi(frame, left_roi, kernel)
        ra, rb = analyze_roi(frame, right_roi, kernel)
        history.append({"frame_idx": frame_idx, "time_sec": t,
                        "left_area": la, "right_area": ra,
                        "left_blob": lb, "right_blob": rb, "_frame": frame})
        if len(history) > PREBUFFER_FRAMES: history.pop(0)
        if t < FIRST_FLASH_IGNORE_SEC: continue
        lh = la >= HARD_ON_AREA and lb >= MIN_LARGEST_BLOB_AREA
        rh = ra >= HARD_ON_AREA and rb >= MIN_LARGEST_BLOB_AREA
        if lh or rh:
            det = "both" if (lh and rh) else ("left" if lh else "right")
            if not is_clean_onset(history, det): continue
            if debug_dir:
                tmp = {"time_sec": t, "frame_idx": frame_idx, "detected_on": det,
                       "left_area": la, "right_area": ra, "left_blob": lb, "right_blob": rb}
                _save_roi_debug(history, tmp, left_roi, right_roi, debug_dir, 1, "first")
            result = refine_flash_from_history(history)
            for item in history: item.pop("_frame", None)
            return result
    return None

def find_flash_in_window(cap, fps, frame_count, left_roi, right_roi, kernel,
                         start_sec, end_sec, debug_dir=None, flash_index=None, debug_area=False):
    start_frame = max(0, int(math.floor(start_sec * fps)))
    end_frame = min(frame_count - 1, int(math.ceil(end_sec * fps)))
    if end_frame <= start_frame: return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    history = []
    while True:
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) > end_frame: break
        ret, frame = cap.read()
        if not ret: break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        t = frame_idx / fps
        la, lb = analyze_roi(frame, left_roi, kernel)
        ra, rb = analyze_roi(frame, right_roi, kernel)
        history.append({"frame_idx": frame_idx, "time_sec": t,
                        "left_area": la, "right_area": ra,
                        "left_blob": lb, "right_blob": rb, "_frame": frame})
        if len(history) > PREBUFFER_FRAMES: history.pop(0)
        lh = la >= HARD_ON_AREA and lb >= MIN_LARGEST_BLOB_AREA
        rh = ra >= HARD_ON_AREA and rb >= MIN_LARGEST_BLOB_AREA
        if debug_area and (la > 0 or ra > 0):
            print(f"  [dbg] t={t:.3f} f={frame_idx} L={la}(b{lb}) R={ra}(b{rb})")
        if lh or rh:
            det = "both" if (lh and rh) else ("left" if lh else "right")
            if not is_clean_onset(history, det): continue
            result = refine_flash_from_history(history)
            if debug_dir:
                _save_roi_debug(history, result, left_roi, right_roi, debug_dir, flash_index, "window")
            for item in history: item.pop("_frame", None)
            return result
    return None

def expected_pattern_time(index_zero_based):
    t = 0.0
    for k in range(index_zero_based):
        t += 10.0 if k % 2 == 0 else 11.0
    return t

# ── Чтение лога ────────────────────────────────────────────────────────[...]
def read_led_runs_from_jsonl(events_path):
    runs = {}
    with open(events_path, "r", encoding="utf-8", errors="replace") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line: continue
            try: ev = json.loads(line)
            except Exception: continue
            if not isinstance(ev, dict): continue
            if str(ev.get("type", "")).strip().lower() not in VALID_PULSE_TYPES: continue
            run_id = safe_int(ev.get("video_sync_run_id", ev.get("run_id", ev.get("led_run_id")))) or 1
            row = {
                "run_id": run_id,
                "blink_index_global": safe_int(ev.get("blink_index_global",
                    ev.get("blink_index", ev.get("global_index"))), None),
                "blink_index_in_run": safe_int(ev.get("blink_index_in_run",
                    ev.get("pulse_index_in_run", ev.get("blink_index", ev.get("index")))), None),
                "host_monotonic_ns": safe_int(ev.get("host_monotonic_ns")),
                "host_wall_time_utc": ev.get("host_wall_time_utc", ev.get("time_utc", "")),
                "scheduled_time_from_start_sec": safe_float(ev.get("scheduled_time_from_start_sec",
                    ev.get("scheduled_rel_sec", ev.get("time_from_run_start_sec"))), None),
                "source_line_no": line_no,
            }
            runs.setdefault(run_id, []).append(row)
    for run_id, rows in runs.items():
        rows.sort(key=lambda r: (
            r["blink_index_in_run"] if r["blink_index_in_run"] is not None else 10**12,
            r["host_monotonic_ns"] if r["host_monotonic_ns"] is not None else 10**18,
            r["source_line_no"],
        ))
        for i, r in enumerate(rows):
            if r["blink_index_in_run"] is None: r["blink_index_in_run"] = i + 1
            if r["blink_index_global"] is None: r["blink_index_global"] = r["blink_index_in_run"]
        base = rows[0]["scheduled_time_from_start_sec"] or 0.0
        for i, r in enumerate(rows):
            r["scheduled_rel_to_first_sec"] = (
                expected_pattern_time(i) if r["scheduled_time_from_start_sec"] is None
                else r["scheduled_time_from_start_sec"] - base
            )
    return runs

# ── Детектирование ──────────────────────────────────────────────────────[...]
def detect_video_flashes(video_path, log_pulses, roi_values, debug_dir=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"Не удалось открыть: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps
    lx, ly, lw, lh, rx, ry, rw, rh = roi_values
    left_roi = build_panorama_roi(lx, ly, lw, lh, frame_w, frame_h)
    right_roi = build_panorama_roi(rx, ry, rw, rh, frame_w, frame_h)
    kernel = np.ones((3, 3), np.uint8)

    first = find_first_flash(cap, fps, frame_count, left_roi, right_roi, kernel, debug_dir)
    if first is None:
        cap.release()
        raise RuntimeError("Первая вспышка не найдена")
    first.update({"video_flash_index": 1, "expected_video_time_sec": first["time_sec"],
                  "search_start_sec": 0.0, "search_end_sec": duration_sec,
                  "matched_log_index": 0,
                  "matched_log_blink_index_in_run": log_pulses[0]["blink_index_in_run"],
                  "matched_log_blink_index_global": log_pulses[0]["blink_index_global"]})
    video_flashes = [first]
    first_time = first["time_sec"]
    print(f"  Первая вспышка: t={first_time:.3f}s frame={first['frame_idx']}")

    for i in range(1, len(log_pulses)):
        rel = log_pulses[i]["scheduled_rel_to_first_sec"]
        expected_t = first_time + rel
        search_start = expected_t - SEARCH_WINDOW_BEFORE_SEC
        search_end = expected_t + SEARCH_WINDOW_AFTER_SEC
        if search_start > duration_sec: break
        found = find_flash_in_window(cap, fps, frame_count, left_roi, right_roi, kernel,
                                     search_start, search_end, debug_dir, i + 1)
        if found is None:
            print(f"  WARNING: вспышка #{i+1} не найдена около t={expected_t:.3f}")
            find_flash_in_window(cap, fps, frame_count, left_roi, right_roi, kernel,
                                 search_start, search_end, debug_area=True)
            found = {"time_sec": expected_t, "frame_idx": round(expected_t * fps),
                     "detected_on": "missing", "left_area": 0, "right_area": 0,
                     "left_blob": 0, "right_blob": 0, "interpolated": True}
            print(f"  -> Интерполирована вспышка #{len(video_flashes)+1}: t={expected_t:.3f}")
        else:
            print(f"  Вспышка #{len(video_flashes)+1}: t={found['time_sec']:.3f} "
                  f"expected={expected_t:.3f} dt={found['time_sec']-expected_t:+.3f} "
                  f"on={found['detected_on']}")
        found.update({"video_flash_index": len(video_flashes) + 1,
                      "expected_video_time_sec": expected_t,
                      "search_start_sec": search_start, "search_end_sec": search_end,
                      "matched_log_index": i,
                      "matched_log_blink_index_in_run": log_pulses[i]["blink_index_in_run"],
                      "matched_log_blink_index_global": log_pulses[i]["blink_index_global"]})
        video_flashes.append(found)
    cap.release()
    return video_flashes, fps, frame_count, duration_sec

def build_pairs(video_flashes, log_pulses, host_to_utc_ns=None, timebase_source=""):
    pairs = []
    for vf in video_flashes:
        lp = log_pulses[vf["matched_log_index"]]

        log_utc_ts = None
        time_source = ""
        time_warning = ""

        host_ns = lp.get("host_monotonic_ns")
        if host_to_utc_ns is not None and host_ns is not None:
            utc_ns = host_to_utc_ns(host_ns)
            log_utc_ts = utc_ns / 1e9
            time_source = timebase_source or "gnss_pvt_fit"
        else:
            utc_dt = parse_iso_utc(lp["host_wall_time_utc"])
            if utc_dt is not None:
                log_utc_ts = utc_dt.timestamp()
                time_source = "host_wallclock_fallback"
                time_warning = "wallclock_used_as_fallback"

        if log_utc_ts is None:
            print(f" WARNING: нет UTC у лог-вспышки #{vf['matched_log_index']+1}, пропущена")
            continue

        detected_on = vf["detected_on"]
        if time_source in ("gnss0_pvt_fit", "gnss1_pvt_fit", "gnss_pvt_fit"):
            if detected_on == "both":
                anchor_quality = "good"
                anchor_reason = "both+gnss_pvt_fit"
            elif detected_on in ("left", "right"):
                anchor_quality = "single_side_gnss"
                anchor_reason = f"{detected_on}+gnss_pvt_fit"
            else:
                anchor_quality = "weak"
                anchor_reason = f"{detected_on or 'unknown'}+gnss_pvt_fit"
        elif time_source == "host_wallclock_fallback":
            anchor_quality = "fallback"
            anchor_reason = "host_wallclock_fallback"
        else:
            anchor_quality = "weak"
            anchor_reason = "unknown_time_source"

        pairs.append({
            "pair_index": len(pairs) + 1,
            "video_flash_index": vf["video_flash_index"],
            "video_time_sec": vf["time_sec"],
            "video_frame_idx": vf["frame_idx"],
            "detected_on": detected_on,
            "log_run_id": lp["run_id"],
            "blink_index_global": lp["blink_index_global"],
            "blink_index_in_run": lp["blink_index_in_run"],
            "host_monotonic_ns": lp["host_monotonic_ns"],
            "host_wall_time_utc": lp["host_wall_time_utc"],
            "log_utc_ts": log_utc_ts,
            "time_source": time_source,
            "time_warning": time_warning,
            "anchor_quality": anchor_quality,
            "anchor_reason": anchor_reason,
        })

    if len(pairs) < 2:
        raise RuntimeError("Нужно минимум 2 пары с UTC")

    pairs.sort(key=lambda p: p["video_time_sec"])
    for i, p in enumerate(pairs):
        p["pair_index"] = i + 1
    return pairs

# ── Запись CSV ──────────────────────────────────────────────────────────[...]
def write_video_flashes_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["video_flash_index","time_sec","frame_idx","detected_on",
                    "left_area","right_area","left_blob","right_blob",
                    "expected_video_time_sec","dt_from_expected_sec",
                    "search_start_sec","search_end_sec",
                    "matched_log_blink_index_global","matched_log_blink_index_in_run"])
        for r in rows:
            dt = r["time_sec"] - r["expected_video_time_sec"]
            w.writerow([r["video_flash_index"], f"{r['time_sec']:.6f}", r["frame_idx"],
                        r["detected_on"], r["left_area"], r["right_area"],
                        r["left_blob"], r["right_blob"],
                        f"{r['expected_video_time_sec']:.6f}", f"{dt:.6f}",
                        f"{r['search_start_sec']:.6f}", f"{r['search_end_sec']:.6f}",
                        r["matched_log_blink_index_global"], r["matched_log_blink_index_in_run"]])

def write_alignment_csv(path, pairs):
    with open(path, "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "pair_index","video_flash_index","video_time_sec","video_frame_idx",
            "detected_on","log_run_id","blink_index_global","blink_index_in_run",
            "host_monotonic_ns","host_wall_time_utc","log_utc_datetime",
            "time_source","time_warning",
            "anchor_quality","anchor_reason",
            "anchor_pos_line_no","anchor_pos_exact","anchor_pos_interpolated",
            "anchor_lat_deg","anchor_lon_deg","anchor_height_m",
            "anchor_Q","anchor_ns","anchor_sdn_m","anchor_sde_m","anchor_sdu_m",
            "anchor_is_spatial"
        ])
        for p in pairs:
            w.writerow([
                p["pair_index"], p["video_flash_index"], f"{p['video_time_sec']:.6f}",
                p["video_frame_idx"], p["detected_on"], p["log_run_id"],
                p["blink_index_global"], p["blink_index_in_run"],
                "" if p["host_monotonic_ns"] is None else p["host_monotonic_ns"],
                p["host_wall_time_utc"], iso_utc_from_ts(p["log_utc_ts"]),
                p.get("time_source", ""), p.get("time_warning", ""),
                p.get("anchor_quality", ""), p.get("anchor_reason", ""),
                p.get("anchor_pos_line_no", ""),
                p.get("anchor_pos_exact", 0),
                p.get("anchor_pos_interpolated", 1),
                "" if p.get("anchor_lat_deg") is None else f"{p['anchor_lat_deg']:.11f}",
                "" if p.get("anchor_lon_deg") is None else f"{p['anchor_lon_deg']:.11f}",
                "" if p.get("anchor_height_m") is None else f"{p['anchor_height_m']:.4f}",
                "" if p.get("anchor_Q") is None else p["anchor_Q"],
                "" if p.get("anchor_ns") is None else p["anchor_ns"],
                "" if p.get("anchor_sdn_m") is None else f"{p['anchor_sdn_m']:.4f}",
                "" if p.get("anchor_sde_m") is None else f"{p['anchor_sde_m']:.4f}",
                "" if p.get("anchor_sdu_m") is None else f"{p['anchor_sdu_m']:.4f}",
                p.get("anchor_is_spatial", 0),
            ])

def write_log_pulses_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["run_id","blink_index_global","blink_index_in_run","host_monotonic_ns",
                    "host_wall_time_utc","scheduled_time_from_start_sec",
                    "scheduled_rel_to_first_sec","source_line_no"])
        for r in rows:
            w.writerow([r["run_id"], r["blink_index_global"], r["blink_index_in_run"],
                        "" if r["host_monotonic_ns"] is None else r["host_monotonic_ns"],
                        r["host_wall_time_utc"],
                        "" if r["scheduled_time_from_start_sec"] is None
                           else f"{r['scheduled_time_from_start_sec']:.6f}",
                        f"{r['scheduled_rel_to_first_sec']:.6f}", r["source_line_no"]])

# ── Автопоиск файлов ─────────────────────────────────────────────────────[...]
def auto_find_events(work_dir):
    """Найти session_events*.jsonl в рабочей папке."""
    patterns = ["session_events*.jsonl", "*.jsonl"]
    for pat in patterns:
        found = sorted(glob.glob(os.path.join(work_dir, pat)))
        if found:
            if len(found) > 1:
                print(f"Найдено несколько jsonl-файлов, использую первый: {found[0]}")
                print(f"  Остальные: {found[1:]}")
                print(f"  Если нужен другой — укажите --events <путь>")
            return found[0]
    return None

def auto_find_videos(work_dir):
    """Найти video_NN.mp4 (или любые .mp4) в рабочей папке, отсортировать по имени."""
    # Сначала ищем video_NN.mp4
    specific = sorted(glob.glob(os.path.join(work_dir, "video_*.mp4")))
    if specific:
        return specific
    # Fallback: все .mp4
    all_mp4 = sorted(glob.glob(os.path.join(work_dir, "*.mp4")))
    return all_mp4

def extract_run_id_from_filename(path):
    """Извлечь число из имени файла: video_01.mp4 -> 1, video_03.mp4 -> 3."""
    name = Path(path).stem  # video_01
    m = re.search(r'(\d+)$', name)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)', name)
    if m:
        return int(m.group(1))
    return None

# ── main ──────────────────────────────────────────────────────────────[...]
def main():
    parser = argparse.ArgumentParser(
        description="Детектирование LED-вспышек. Запускается из папки с данными.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--roi", nargs=8, type=int, required=True,
                        metavar=("LX","LY","LW","LH","RX","RY","RW","RH"),
                        help="ROI: left_x left_y left_w left_h right_x right_y right_w right_h")
    parser.add_argument("--events", default="",
                        help="Путь к session_events.jsonl (по умолчанию — автопоиск в текущей папке)")
    parser.add_argument("--video-dir", default="",
                        help="Папка с видео (по умолчанию — текущая папка)")
    parser.add_argument("--out-dir", default="",
                        help="Папка для результатов (по умолчанию — папка с видео)")
    parser.add_argument("--debug-roi-dir", default="",
                        help="Папка для ROI-кропов (необязательно)")
    parser.add_argument("--run-id-map", default="",
                        help="Переопределение run_id: 'video_01.mp4=1,video_02.mp4=3'")
    args = parser.parse_args()

    work_dir = os.path.abspath(args.video_dir) if args.video_dir else os.getcwd()
    print(f"Рабочая папка: {work_dir}")

    # ── Поиск jsonl ──────────────────────────────────────────────────────────[...]
    events_path = args.events
    if not events_path:
        events_path = auto_find_events(work_dir)
        if not events_path:
            print("ERROR: не найден session_events*.jsonl в текущей папке.", file=sys.stderr)
            print("Укажите --events <путь>", file=sys.stderr)
            sys.exit(1)
    print(f"Лог LED: {events_path}")

    # ── Чтение лога ───────────────────────────────────────────────────────[...]
    print("Чтение LED-журнала...")
    all_runs = read_led_runs_from_jsonl(events_path)
    print(f"Найдено run_id: {sorted(all_runs.keys())}")
    print()

    host_to_utc_ns, timebase_source, pvt_count = load_timebase_from_session(events_path)
    if host_to_utc_ns is not None:
        print(f"Timebase: {timebase_source}, NAV-PVT epochs={pvt_count}")
    else:
        print("Timebase: GNSS fit недоступен, будет использован host_wall_time_utc как fallback")
    print()
    
    pos_rows = read_pos_file(os.path.join(os.path.dirname(os.path.abspath(events_path)), "gnss0.pos"), 18.0)

    # ── Поиск видео ───────────────────────────────────────────────────────[...]
    videos = auto_find_videos(work_dir)
    if not videos:
        print(f"ERROR: не найдено .mp4-файлов в {work_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Найдено видео ({len(videos)}):")
    for v in videos:
        print(f"  {os.path.basename(v)}")
    print()

    # ── Разбор run_id_map ─────────────────────────────────────────────────────
    run_id_override = {}
    if args.run_id_map:
        for part in args.run_id_map.split(","):
            part = part.strip()
            if "=" in part:
                fname, rid = part.split("=", 1)
                run_id_override[fname.strip()] = int(rid.strip())

    # ── Папка для результатов ────────────────────────────────────────────────[...]
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else work_dir
    os.makedirs(out_dir, exist_ok=True)

    roi_values = args.roi
    debug_dir_base = args.debug_roi_dir

    # ── Обработка каждого видео ─────────────────────────────────────────────
    success_count = 0
    failed_sessions = []

    for sess_idx, video_path in enumerate(videos, start=1):
        vname = os.path.basename(video_path)
        vstem = Path(video_path).stem  # video_01

        # Определяем run_id
        if vname in run_id_override:
            run_id = run_id_override[vname]
        else:
            run_id = extract_run_id_from_filename(video_path)
            if run_id is None:
                print(f"  WARNING: не удалось определить run_id из имени '{vname}', пропускаю")
                failed_sessions.append((sess_idx, vname, "не удалось определить run_id"))
                continue

        print(f"{'='*60}")
        print(f"Сессия {sess_idx}/{len(videos)}: {vname}  run_id={run_id}")

        debug_dir = None
        if debug_dir_base:
            debug_dir = os.path.join(debug_dir_base, f"s{sess_idx:02d}_{vstem}")

        try:
            if run_id not in all_runs:
                raise RuntimeError(f"run_id={run_id} не найден. Доступные: {sorted(all_runs.keys())}")
            log_pulses = all_runs[run_id]
            if len(log_pulses) < 2:
                raise RuntimeError(f"В run_id={run_id} меньше 2 вспышек")

            print(f"  Вспышек в логе: {len(log_pulses)}")
            print(f"  LED UTC: {log_pulses[0]['host_wall_time_utc']}  ..  {log_pulses[-1]['host_wall_time_utc']}")

            video_flashes, fps, frame_count, duration_sec = detect_video_flashes(
                video_path, log_pulses, roi_values, debug_dir)

            print(f"  FPS={fps:.4f} кадров={frame_count} длит={duration_sec:.1f}s "
                  f"вспышек={len(video_flashes)}")

            pairs = build_pairs(video_flashes, log_pulses, host_to_utc_ns, timebase_source)
            enrich_pairs_with_pos_anchor_info(pairs, pos_rows)
            print(f"  Пар синхронизации: {len(pairs)}")
            print(f"  UTC пар: {iso_utc_from_ts(pairs[0]['log_utc_ts'])} .. {iso_utc_from_ts(pairs[-1]['log_utc_ts'])}")

            # Сохраняем с префиксом из имени видео
            prefix = os.path.join(out_dir, vstem)
            write_log_pulses_csv(prefix + "_led_log_pulses.csv", log_pulses)
            write_video_flashes_csv(prefix + "_video_flashes.csv", video_flashes)
            write_alignment_csv(prefix + "_flash_alignment.csv", pairs)

            print(f"  -> {vstem}_led_log_pulses.csv")
            print(f"  -> {vstem}_video_flashes.csv")
            print(f"  -> {vstem}_flash_alignment.csv")
            success_count += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            failed_sessions.append((sess_idx, vname, str(e)))
        print()

    # ── Итого ────────────────────────────────────────────────────────────[...]
    print(f"{'='*60}")
    print(f"Готово. Успешно: {success_count}/{len(videos)}")
    if failed_sessions:
        print(f"Ошибки ({len(failed_sessions)}):")
        for idx, name, err in failed_sessions:
            print(f"  [{idx}] {name}: {err}")

if __name__ == "__main__":
    main()