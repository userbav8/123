#!/usr/bin/env python3
"""
detect_flashes_multi.py — Обнаружение LED-вспышек по видео и синхронизация с логом.

Основное использование:
    python detect_flashes_multi.py --roi 3848 3049 150 150 16 3303 150 150

Требуемые входные файлы:
  - session_events*.jsonl  (лог LED-вспышек)
  - video_01.mp4 .. video_NN.mp4  (видеофайлы по синхро, каждый содержит run_id = NN (извлекается из имени)).

Выходные файлы (для каждого видео_NN.mp4):
  - video_NN_led_log_pulses.csv
  - video_NN_video_flashes.csv
  - video_NN_flash_alignment.csv

Дополнительные опции:
  --events         путь к jsonl-файлу (по умолчанию session_events.jsonl из текущей папки)
  --video-dir      папка с видео (по умолчанию текущая папка)
  --out-dir        папка для результатов (по умолчанию текущая папка)
  --debug-roi-dir  папка для ROI-отладки
  --run-id-map     переопределяет соответствие между run_id и видеофайлом (напр.: video_01.mp4=1,video_02.mp4=3)
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

# ── Константы для поиска вспышек ──────────────────────────────────────────
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

# ── Утилиты ──────────────────────────────────────────────────────────────
def safe_float(v, default=None):
    try:
        if v is None or v == "": return default
        return float(v)
    except Exception: return default

def safe_int(v, default=None):
    try:
        if v is None or v == "": return default
        return int(float(v))
    except Exception: return default

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
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
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
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(',')
                if len(parts) < 5:
                    continue
                role = role_from_text(parts[0])
                start = parse_int(parts[1])
                size = parse_int(parts[2])
                host_ns = parse_int(parts[3])
                idx.add(role, start, size, host_ns)
    except Exception:
        pass
    idx.finalize()
    return idx

def ubx_checksum(buf):
    ck_a, ck_b = 0, 0
    for byte in buf:
        ck_a = (ck_a + byte) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return ck_a, ck_b

def iter_ubx(path):
    try:
        with open(path, "rb") as fp:
            while True:
                sync1 = fp.read(1)
                if not sync1:
                    break
                if sync1 != b'\xb5':
                    continue
                sync2 = fp.read(1)
                if sync2 != b'\x62':
                    continue
                cls_byte = fp.read(1)
                if not cls_byte:
                    break
                id_byte = fp.read(1)
                if not id_byte:
                    break
                len_bytes = fp.read(2)
                if len(len_bytes) < 2:
                    break
                length = struct.unpack('<H', len_bytes)[0]
                payload = fp.read(length)
                if len(payload) < length:
                    break
                ck_a_exp, ck_b_exp = fp.read(2)
                if len(ck_a_exp) < 2:
                    break
                ck_a_exp, ck_b_exp = ord(ck_a_exp) if isinstance(ck_a_exp, str) else ck_a_exp, ord(ck_b_exp) if isinstance(ck_b_exp, str) else ck_b_exp
                ck_a, ck_b = ubx_checksum(cls_byte + id_byte + len_bytes + payload)
                if ck_a == ck_a_exp and ck_b == ck_b_exp:
                    yield cls_byte, id_byte, payload
    except Exception:
        pass

def parse_nav_pvt(payload):
    if len(payload) < 92:
        return None
    try:
        (iTOW, year, month, day, hour, minute, second, valid, tAcc, nano, fixType,
         flags, flagsVersion, numSV, lon, lat, height, hMSL, hAcc, vAcc) = struct.unpack(
            '<IHBBBBBBIiBBBBiiiii', payload[:92])
        return {
            "iTOW": iTOW,
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "second": second,
            "nano": nano,
            "valid": valid,
            "fixType": fixType,
            "numSV": numSV,
            "lon": lon * 1e-7,
            "lat": lat * 1e-7,
            "height": height / 1000.0,
            "hAcc": hAcc / 1000.0,
            "vAcc": vAcc / 1000.0,
        }
    except Exception:
        return None

def load_gnss_pvt(ubx_path, role, chunks):
    rows = []
    try:
        for cls_byte, id_byte, payload in iter_ubx(ubx_path):
            cls_byte = ord(cls_byte) if isinstance(cls_byte, str) else cls_byte
            id_byte = ord(id_byte) if isinstance(id_byte, str) else id_byte
            if cls_byte == 0x01 and id_byte == 0x07:
                pvt = parse_nav_pvt(payload)
                if pvt:
                    rows.append(pvt)
    except Exception:
        pass
    return rows

def fit_utc_from_pvt(pvt_rows):
    def host_to_utc_ns(host_ns):
        if not pvt_rows:
            return host_ns
        sorted_pvt = sorted(pvt_rows, key=lambda r: r["iTOW"])
        ts = [datetime_to_ns(r["year"], r["month"], r["day"], r["hour"], r["minute"], r["second"], r["nano"]) for r in sorted_pvt]
        itow = [r["iTOW"] * 1_000_000 for r in sorted_pvt]
        if len(ts) < 2 or len(itow) < 2:
            return host_ns
        a, b = np.polyfit(itow, ts, 1)
        return int(a * host_ns + b)
    return host_to_utc_ns

def load_timebase_from_session(events_path):
    timebase_source = ""
    try:
        with open(events_path, "r", encoding="utf-8", errors="replace") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("event_type") == "timebase_sync":
                        timebase_source = obj.get("source", "")
                        break
                except Exception:
                    pass
    except Exception:
        pass
    return timebase_source

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
    try:
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
                    row = {
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
                    }
                    rows.append(row)
                except Exception:
                    continue
    except Exception:
        pass
    rows.sort(key=lambda r: r["utc_ts"])
    return rows

def enrich_pairs_with_pos_anchor_info(pairs, pos_rows, exact_tol_sec=0.1, interp_max_gap_sec=2.0):
    if not pos_rows:
        return
    pos_start = pos_rows[0]["utc_ts"]
    pos_end = pos_rows[-1]["utc_ts"]
    for pair in pairs:
        anchor_utc = pair.get("log_utc_ts")
        if anchor_utc is None:
            continue
        if anchor_utc < pos_start or anchor_utc > pos_end:
            pair["anchor_quality"] = ""
            pair["anchor_reason"] = "outside_pos_range"
            continue
        xs = [r["utc_ts"] for r in pos_rows]
        idx = bisect_right(xs, anchor_utc) - 1
        if idx < 0 or idx >= len(pos_rows):
            pair["anchor_quality"] = ""
            pair["anchor_reason"] = "invalid_index"
            continue
        r0 = pos_rows[idx]
        if abs(anchor_utc - r0["utc_ts"]) <= exact_tol_sec:
            pair["anchor_quality"] = "good"
            pair["anchor_reason"] = "exact_match"
            pair["anchor_pos_exact"] = 1
            pair["anchor_pos_interpolated"] = 0
            pair["anchor_lat_deg"] = r0["lat_deg"]
            pair["anchor_lon_deg"] = r0["lon_deg"]
            pair["anchor_height_m"] = r0["height_m"]
            pair["anchor_Q"] = r0["Q"]
            pair["anchor_ns"] = r0["ns"]
            pair["anchor_sdn_m"] = r0["sdn_m"]
            pair["anchor_sde_m"] = r0["sde_m"]
            pair["anchor_sdu_m"] = r0["sdu_m"]
            pair["anchor_is_spatial"] = 1 if r0["Q"] == 1 else 0
        elif idx + 1 < len(pos_rows):
            r1 = pos_rows[idx + 1]
            gap = r1["utc_ts"] - r0["utc_ts"]
            if gap > 0 and gap <= interp_max_gap_sec:
                alpha = (anchor_utc - r0["utc_ts"]) / gap
                pair["anchor_quality"] = "good"
                pair["anchor_reason"] = "interpolated"
                pair["anchor_pos_exact"] = 0
                pair["anchor_pos_interpolated"] = 1
                pair["anchor_lat_deg"] = r0["lat_deg"] + alpha * (r1["lat_deg"] - r0["lat_deg"])
                pair["anchor_lon_deg"] = r0["lon_deg"] + alpha * (r1["lon_deg"] - r0["lon_deg"])
                pair["anchor_height_m"] = r0["height_m"] + alpha * (r1["height_m"] - r0["height_m"])
                pair["anchor_Q"] = r0["Q"]
                pair["anchor_ns"] = r0["ns"]
                pair["anchor_sdn_m"] = r0["sdn_m"]
                pair["anchor_sde_m"] = r0["sde_m"]
                pair["anchor_sdu_m"] = r0["sdu_m"]
                pair["anchor_is_spatial"] = 1 if r0["Q"] == 1 else 0
            else:
                pair["anchor_quality"] = ""
                pair["anchor_reason"] = "gap_too_large"
        else:
            pair["anchor_quality"] = ""
            pair["anchor_reason"] = "no_next_row"

# ── НОВАЯ ФУНКЦИЯ: Якорная коррекция времени ──────────────────────────────
def correct_flash_timing_with_anchors(pairs, pos_rows):
    """
    Использовать высококачественные якоря для коррекции дрейфа FPS видео.
    
    Якоря с Q=1 (точные координаты) служат опорными точками для пересчёта
    реального FPS и устранения накопленного дрейфа времени.
    """
    # Найти якоря высокого качества (anchor_is_spatial=1)
    good_anchors = [p for p in pairs if p.get("anchor_is_spatial") == 1]
    
    if len(good_anchors) < 2:
        print(f"  ⚠️  Недостаточно якорей высокого качества для коррекции FPS (найдено {len(good_anchors)}, нужно ≥2)")
        return pairs
    
    # Линейная регрессия: пересчитаем реальный FPS
    video_times = np.array([a["video_time_sec"] for a in good_anchors], dtype=np.float64)
    utc_times = np.array([a["log_utc_ts"] for a in good_anchors], dtype=np.float64)
    
    # Полином 1-й степени: найти реальное соотношение
    # UTC = a * video_time + b
    a, b = np.polyfit(video_times, utc_times, 1)
    
    print(f"\n  === Якорная коррекция времени ===")
    print(f"  Якорей качества (Q=1): {len(good_anchors)}")
    print(f"  Соотношение: UTC = {a:.10f} * video_t + {b:.1f}")
    print(f"  Эффективный FPS: {a:.6f} (заявленный 30.0)")
    fps_error_pct = (a - 30.0) / 30.0 * 100
    print(f"  Дрейф FPS: {fps_error_pct:+.4f}%")
    
    # Пересчитать все вспышки
    for p in pairs:
        corrected_utc = a * p["video_time_sec"] + b
        old_utc = p["log_utc_ts"]
        delta_ms = (corrected_utc - old_utc) * 1000
        
        p["log_utc_ts_original"] = old_utc
        p["log_utc_ts"] = corrected_utc
        p["utc_correction_ms"] = delta_ms
    
    print(f"  Коррекция применена ко всем парам\n")
    
    return pairs

# ── Остальные функции ────────────────────────────────────────────────────
def build_wrap_segments(xc, w, frame_w):
    if xc - w // 2 >= 0 and xc + w // 2 < frame_w:
        return [(xc - w // 2, xc + w // 2)]
    segs = []
    left = (xc - w // 2) % frame_w
    right = (xc + w // 2) % frame_w
    if left < right:
        segs.append((left, right))
    else:
        segs.append((left, frame_w))
        segs.append((0, right))
    return segs

def build_panorama_roi(xc, yc, w, h, frame_w, frame_h):
    x_segs = build_wrap_segments(xc, w, frame_w)
    y_seg = (max(0, yc - h // 2), min(frame_h, yc + h // 2))
    return {"x_segments": x_segs, "y_segment": y_seg}

def extract_panorama_roi(frame, roi_info):
    rois = []
    y_start, y_end = roi_info["y_segment"]
    for x_start, x_end in roi_info["x_segments"]:
        roi = frame[y_start:y_end, x_start:x_end]
        rois.append(roi)
    if len(rois) > 1:
        return np.hstack(rois)
    return rois[0] if rois else None

def analyze_roi(frame, roi_info, kernel, morph_mode=DEFAULT_MORPH_MODE):
    roi = extract_panorama_roi(frame, roi_info)
    if roi is None or roi.size == 0:
        return {"hard_on": 0, "soft_on": 0, "largest_blob_area": 0}
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask = cv2.bitwise_or(mask1, mask2)
    if morph_mode in ("open", "close", "grad"):
        if morph_mode == "open":
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif morph_mode == "close":
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        elif morph_mode == "grad":
            mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    num_labels, labels = cv2.connectedComponents(mask)
    areas = {}
    for label in range(1, num_labels):
        area = np.sum(labels == label)
        areas[label] = area
    hard_on = 1 if np.sum(mask > 0) >= HARD_ON_AREA else 0
    soft_on = 1 if np.sum(mask > 0) >= SOFT_ON_AREA else 0
    largest_blob_area = max(areas.values()) if areas else 0
    return {"hard_on": hard_on, "soft_on": soft_on, "largest_blob_area": largest_blob_area}

def is_clean_onset(history, detected_on):
    if len(history) < FLASH_CLEAR_FRAMES_BEFORE:
        return False
    for i in range(1, FLASH_CLEAR_FRAMES_BEFORE + 1):
        if history[-i].get("hard_on") > 0:
            return False
    return True

def refine_flash_from_history(history):
    for i in range(len(history) - 1, -1, -1):
        if history[i]["hard_on"] > 0:
            return i
    for i in range(len(history) - 1, -1, -1):
        if history[i]["soft_on"] > 0:
            return i
    return len(history) - 1

def _save_roi_debug(history, result, left_roi, right_roi, debug_dir, flash_index, label):
    if not debug_dir:
        return
    try:
        import os
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"flash_{flash_index:04d}_{label}.png")
        debug_frames = len(history)
        col_width = 200
        row_height = 100
        canvas = np.ones((row_height * 3, col_width * min(debug_frames, 8), 3), dtype=np.uint8) * 255
        for idx, frame_data in enumerate(history[:8]):
            y_offset = 0
            left_img = frame_data.get("left_roi_img")
            right_img = frame_data.get("right_roi_img")
            if left_img is not None:
                left_resized = cv2.resize(left_img, (col_width, row_height))
                canvas[y_offset:y_offset+row_height, idx*col_width:(idx+1)*col_width] = left_resized
            y_offset += row_height
            if right_img is not None:
                right_resized = cv2.resize(right_img, (col_width, row_height))
                canvas[y_offset:y_offset+row_height, idx*col_width:(idx+1)*col_width] = right_resized
        cv2.imwrite(debug_path, canvas)
    except Exception:
        pass

def find_first_flash(cap, fps, frame_count, left_roi, right_roi, kernel, debug_dir=None):
    search_frames = int(fps * (FIRST_FLASH_IGNORE_SEC + SEARCH_WINDOW_AFTER_SEC))
    history = []
    for frame_idx in range(min(search_frames, frame_count)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        left_result = analyze_roi(frame, left_roi, kernel)
        right_result = analyze_roi(frame, right_roi, kernel)
        combined_hard = left_result["hard_on"] or right_result["hard_on"]
        combined_soft = left_result["soft_on"] or right_result["soft_on"]
        history.append({
            "frame_idx": frame_idx,
            "hard_on": combined_hard,
            "soft_on": combined_soft,
            "left_result": left_result,
            "right_result": right_result,
        })
        if combined_hard > 0 and frame_idx > fps * FIRST_FLASH_IGNORE_SEC and is_clean_onset(history, "both"):
            flash_frame = refine_flash_from_history(history[-PREBUFFER_FRAMES:])
            flash_frame_abs = frame_idx - len(history[-PREBUFFER_FRAMES:]) + flash_frame
            return flash_frame_abs, "both"
    return None, None

def find_flash_in_window(cap, fps, frame_count, left_roi, right_roi, kernel,
                        search_start_sec, search_end_sec, debug_dir=None):
    start_frame = int(search_start_sec * fps)
    end_frame = int(search_end_sec * fps)
    start_frame = max(0, start_frame)
    end_frame = min(end_frame, frame_count - 1)
    history = []
    for frame_idx in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        left_result = analyze_roi(frame, left_roi, kernel)
        right_result = analyze_roi(frame, right_roi, kernel)
        combined_hard = left_result["hard_on"] or right_result["hard_on"]
        combined_soft = left_result["soft_on"] or right_result["soft_on"]
        history.append({
            "frame_idx": frame_idx,
            "hard_on": combined_hard,
            "soft_on": combined_soft,
            "left_result": left_result,
            "right_result": right_result,
        })
        if combined_hard > 0 and is_clean_onset(history, "both"):
            flash_frame = refine_flash_from_history(history[-PREBUFFER_FRAMES:])
            flash_frame_abs = frame_idx - len(history[-PREBUFFER_FRAMES:]) + flash_frame
            detected_on = "both" if (left_result["hard_on"] and right_result["hard_on"]) else (
                "left" if left_result["hard_on"] else "right" if right_result["hard_on"] else "unknown")
            return flash_frame_abs, detected_on
    return None, None

def expected_pattern_time(index_zero_based):
    interval = 10.0
    return interval * index_zero_based

def read_led_runs_from_jsonl(events_path):
    runs = {}
    try:
        with open(events_path, "r", encoding="utf-8", errors="replace") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("event_type") == "led_pulse_run":
                        run_id = obj.get("run_id")
                        if run_id is not None:
                            runs[run_id] = obj
                except Exception:
                    pass
    except Exception:
        pass
    return runs

def detect_video_flashes(video_path, log_pulses, roi_values, debug_dir=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    if len(roi_values) < 8:
        raise RuntimeError(f"ROI должен содержать 8 значений, получено {len(roi_values)}")
    left_roi = build_panorama_roi(roi_values[0], roi_values[1], roi_values[2], roi_values[3], roi_values[4], roi_values[3])
    right_roi = build_panorama_roi(roi_values[5], roi_values[1], roi_values[6], roi_values[7], roi_values[4], roi_values[3])
    
    first_flash_frame, _ = find_first_flash(cap, fps, frame_count, left_roi, right_roi, kernel, debug_dir)
    if first_flash_frame is None:
        cap.release()
        return []
    first_flash_time = first_flash_frame / fps
    video_flashes = [{"frame_idx": first_flash_frame, "time_sec": first_flash_time, "video_flash_index": 0}]
    
    for pulse_idx, pulse in enumerate(log_pulses[1:], start=1):
        expected_time = expected_pattern_time(pulse_idx) + first_flash_time
        search_start = expected_time - SEARCH_WINDOW_BEFORE_SEC
        search_end = expected_time + SEARCH_WINDOW_AFTER_SEC
        detected_frame, detected_on = find_flash_in_window(cap, fps, frame_count, left_roi, right_roi, kernel,
                                                           search_start, search_end, debug_dir)
        if detected_frame is not None:
            detected_time = detected_frame / fps
            video_flashes.append({
                "frame_idx": detected_frame,
                "time_sec": detected_time,
                "video_flash_index": pulse_idx,
                "detected_on": detected_on,
            })
    
    cap.release()
    return video_flashes

def build_pairs(video_flashes, log_pulses, host_to_utc_ns=None, timebase_source=""):
    pairs = []
    for vf_idx, vf in enumerate(video_flashes):
        if vf_idx >= len(log_pulses):
            break
        pulse = log_pulses[vf_idx]
        pair = {
            "pair_index": vf_idx,
            "video_flash_index": vf["video_flash_index"],
            "video_time_sec": vf["time_sec"],
            "video_frame_idx": vf["frame_idx"],
            "detected_on": vf.get("detected_on", "unknown"),
            "log_run_id": pulse.get("run_id"),
            "blink_index_global": pulse.get("blink_index_global"),
            "blink_index_in_run": pulse.get("blink_index_in_run"),
            "host_monotonic_ns": pulse.get("host_monotonic_ns"),
            "host_wall_time_utc": pulse.get("host_wall_time_utc", ""),
        }
        if host_to_utc_ns:
            utc_ns = host_to_utc_ns(pulse.get("host_monotonic_ns"))
            if utc_ns:
                pair["log_utc_ts"] = utc_ns / 1_000_000_000
        if "log_utc_ts" not in pair:
            dt = parse_iso_utc(pulse.get("host_wall_time_utc"))
            if dt:
                pair["log_utc_ts"] = dt.timestamp()
        pairs.append(pair)
    
    return pairs

def write_video_flashes_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["video_flash_index", "frame_idx", "video_time_sec", "detected_on"])
        for r in rows:
            w.writerow([r["video_flash_index"], r["frame_idx"], f"{r['time_sec']:.6f}", r.get("detected_on", "")])

def write_alignment_csv(path, pairs):
    with open(path, "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "pair_index", "video_flash_index", "video_time_sec", "video_frame_idx",
            "detected_on", "log_run_id", "blink_index_global", "blink_index_in_run",
            "host_monotonic_ns", "host_wall_time_utc", "log_utc_datetime",
            "log_utc_ts_original", "utc_correction_ms",
            "anchor_quality", "anchor_reason",
            "anchor_pos_line_no", "anchor_pos_exact", "anchor_pos_interpolated",
            "anchor_lat_deg", "anchor_lon_deg", "anchor_height_m",
            "anchor_Q", "anchor_ns", "anchor_sdn_m", "anchor_sde_m", "anchor_sdu_m",
            "anchor_is_spatial"
        ])
        for p in pairs:
            w.writerow([
                p["pair_index"],
                p["video_flash_index"],
                f"{p['video_time_sec']:.6f}",
                p["video_frame_idx"],
                p.get("detected_on", ""),
                p.get("log_run_id", ""),
                p.get("blink_index_global", ""),
                p.get("blink_index_in_run", ""),
                "" if p.get("host_monotonic_ns") is None else p["host_monotonic_ns"],
                p.get("host_wall_time_utc", ""),
                iso_utc_from_ts(p["log_utc_ts"]) if p.get("log_utc_ts") else "",
                "" if p.get("log_utc_ts_original") is None else f"{p['log_utc_ts_original']:.3f}",
                "" if p.get("utc_correction_ms") is None else f"{p['utc_correction_ms']:+.3f}",
                p.get("anchor_quality", ""),
                p.get("anchor_reason", ""),
                p.get("anchor_pos_line_no", ""),
                p.get("anchor_pos_exact", 0),
                p.get("anchor_pos_interpolated", 0),
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
        w.writerow(["pulse_index", "run_id", "blink_index_global", "blink_index_in_run",
                    "host_monotonic_ns", "host_wall_time_utc", "scheduled_rel_to_first_sec"])
        for i, r in enumerate(rows):
            w.writerow([i, r.get("run_id", ""), r.get("blink_index_global", ""),
                       r.get("blink_index_in_run", ""), r.get("host_monotonic_ns", ""),
                       r.get("host_wall_time_utc", ""), expected_pattern_time(i)])

def auto_find_events(work_dir):
    patterns = [
        os.path.join(work_dir, "session_events.jsonl"),
        os.path.join(work_dir, "session_events_*.jsonl"),
    ]
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return sorted(files)[0]
    return None

def auto_find_videos(work_dir):
    return sorted(glob.glob(os.path.join(work_dir, "video_*.mp4")))

def extract_run_id_from_filename(path):
    match = re.search(r'video_(\d+)', Path(path).name)
    return int(match.group(1)) if match else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", nargs=8, type=int, required=True,
                       help="ROI: left_xc left_yc left_w left_h frame_w right_xc right_w right_h")
    parser.add_argument("--events", default=None,
                       help="Path to session_events.jsonl")
    parser.add_argument("--video-dir", default=None,
                       help="Directory containing video files")
    parser.add_argument("--out-dir", default=None,
                       help="Output directory")
    parser.add_argument("--debug-roi-dir", default=None,
                       help="Directory for ROI debug images")
    parser.add_argument("--run-id-map", default=None,
                       help="Mapping of video files to run_ids (e.g. video_01.mp4=1,video_02.mp4=3)")
    parser.add_argument("--gps-utc-offset-sec", type=float, default=18.0)
    parser.add_argument("--pos", default="gnss0.pos",
                       help="Path to POS file for anchor enrichment")
    args = parser.parse_args()
    
    work_dir = args.video_dir or os.getcwd()
    out_dir = args.out_dir or work_dir
    events_path = args.events or auto_find_events(work_dir)
    pos_path = os.path.join(work_dir, args.pos)
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Рабочая папка: {work_dir}")
    print(f"Лог LED: {events_path}")
    
    if not events_path or not os.path.exists(events_path):
        raise RuntimeError(f"Не найден файл с логом LED: {events_path}")
    
    print("Чтение LED-журнала...")
    run_to_pulses = {}
    
    try:
        with open(events_path, "r", encoding="utf-8", errors="replace") as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("event_type") not in VALID_PULSE_TYPES:
                        continue
                    run_id = obj.get("run_id")
                    if run_id is None:
                        continue
                    run_to_pulses.setdefault(run_id, []).append(obj)
                except Exception:
                    continue
    except Exception as e:
        raise RuntimeError(f"Ошибка при чтении LED-журнала: {e}")
    
    for run_id in run_to_pulses:
        run_to_pulses[run_id].sort(key=lambda x: (x.get("blink_index_global") or 0, x.get("host_monotonic_ns") or 0))
    
    print(f"Найдено run_id: {sorted(run_to_pulses.keys())}")
    
    timebase_source = load_timebase_from_session(events_path)
    print(f"\nTimebase: {timebase_source or 'unknown'}")
    
    videos = auto_find_videos(work_dir)
    print(f"\nНайдено видео ({len(videos)}):")
    for v in videos:
        print(f"  {Path(v).name}")
    
    run_id_map = {}
    if args.run_id_map:
        for mapping in args.run_id_map.split(","):
            parts = mapping.split("=")
            if len(parts) == 2:
                run_id_map[Path(parts[0]).name] = int(parts[1])
    
    pos_rows = []
    if os.path.exists(pos_path):
        pos_rows = read_pos_file(pos_path, args.gps_utc_offset_sec)
        print(f"Загружено {len(pos_rows)} строк POS")
    
    for sess_idx, video_path in enumerate(videos, start=1):
        video_name = Path(video_path).name
        run_id = run_id_map.get(video_name) or extract_run_id_from_filename(video_path)
        
        if run_id is None or run_id not in run_to_pulses:
            print(f"\nСессия {sess_idx}/{len(videos)}: {video_name}  [ПРОПУЩЕНО: run_id не найден]")
            continue
        
        log_pulses = run_to_pulses[run_id]
        
        print("\n" + "=" * 60)
        print(f"Сессия {sess_idx}/{len(videos)}: {video_name}  run_id={run_id}")
        print(f"  Вспышек в логе: {len(log_pulses)}")
        
        if log_pulses:
            first_pulse = min(log_pulses, key=lambda x: x.get("host_wall_time_utc", ""))
            last_pulse = max(log_pulses, key=lambda x: x.get("host_wall_time_utc", ""))
            first_dt = parse_iso_utc(first_pulse.get("host_wall_time_utc", ""))
            last_dt = parse_iso_utc(last_pulse.get("host_wall_time_utc", ""))
            if first_dt and last_dt:
                print(f"  LED UTC: {first_dt.isoformat()} .. {last_dt.isoformat()}")
        
        try:
            video_flashes = detect_video_flashes(video_path, log_pulses, args.roi, args.debug_roi_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        
        if not video_flashes:
            print(f"  ERROR: не найдено вспышек в видео")
            continue
        
        first_flash_time = video_flashes[0]["time_sec"]
        print(f"  Первая вспышка: t={first_flash_time:.3f}s frame={video_flashes[0]['frame_idx']}")
        
        host_to_utc_ns = None
        if timebase_source == "gnss0_pvt_fit":
            print(f"  Используется GNSS PVT для синхронизации...")
        
        pairs = build_pairs(video_flashes, log_pulses, host_to_utc_ns, timebase_source)
        
        enrich_pairs_with_pos_anchor_info(pairs, pos_rows)
        
        # ← ПРИМЕНЯЕМ ЯКОРНУЮ КОРРЕКЦИЮ
        pairs = correct_flash_timing_with_anchors(pairs, pos_rows)
        
        for i, pair in enumerate(pairs[:40], start=1):
            dt_sec = pair.get("video_time_sec") - expected_pattern_time(i)
            expected_video_sec = expected_pattern_time(i)
            actual_video_sec = pair["video_time_sec"]
            dt_display = (actual_video_sec - expected_video_sec)
            detected = pair.get("detected_on", "")
            print(f"  Вспышка #{i}: t={actual_video_sec:.3f} expected={expected_video_sec:.3f} dt={dt_display:+.3f} on={detected}")
        
        video_stem = Path(video_path).stem
        out_flashes = os.path.join(out_dir, f"{video_stem}_video_flashes.csv")
        out_pairs = os.path.join(out_dir, f"{video_stem}_flash_alignment.csv")
        out_pulses = os.path.join(out_dir, f"{video_stem}_led_log_pulses.csv")
        
        write_video_flashes_csv(out_flashes, video_flashes)
        write_alignment_csv(out_pairs, pairs)
        write_log_pulses_csv(out_pulses, log_pulses)
        
        print(f"  → {out_flashes}")
        print(f"  → {out_pairs}")
        print(f"  → {out_pulses}")

if __name__ == "__main__":
    main()
