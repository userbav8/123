#!/usr/bin/env python3
import argparse
import glob
import csv
import math
import os
import re
from bisect import bisect_right
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')


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


def parse_pos_gpst_datetime(date_s, time_s, gps_utc_offset_sec):
    """[FIX] Обрабатывает время как с дробной частью, так и без."""
    text = f"{date_s} {time_s}"
    try:
        dt = datetime.strptime(text, "%Y/%m/%d %H:%M:%S.%f")
    except ValueError:
        dt = datetime.strptime(text, "%Y/%m/%d %H:%M:%S")
    dt = dt.replace(tzinfo=timezone.utc)
    utc_dt = dt - timedelta(seconds=gps_utc_offset_sec)
    return dt, utc_dt


def utc_to_video_piecewise(utc_ts, pairs):
    pairs_sorted = sorted(pairs, key=lambda p: p["log_utc_ts"])
    xs = [p["log_utc_ts"] for p in pairs_sorted]
    ys = [p["video_time_sec"] for p in pairs_sorted]
    if utc_ts <= xs[0]:
        i0, i1 = 0, 1; mode = "extrap_before"
    elif utc_ts >= xs[-1]:
        i0, i1 = len(xs) - 2, len(xs) - 1; mode = "extrap_after"
    else:
        j = bisect_right(xs, utc_ts) - 1
        i0, i1 = j, j + 1; mode = "interp"
    x0, x1, y0, y1 = xs[i0], xs[i1], ys[i0], ys[i1]
    if x1 == x0: return y0, mode, i0, i1
    return y0 + (utc_ts - x0) / (x1 - x0) * (y1 - y0), mode, i0, i1


# [FIX] Минимальное количество обязательных колонок в .pos (до sdu включительно)
_POS_MIN_COLS = 10  # date time lat lon h Q ns sdn sde sdu

def read_pos_file(pos_path, gps_utc_offset_sec):
    """[FIX] Принимает как полный формат RTKLib (15 колонок), так и короткий (10+)."""
    rows = []
    with open(pos_path, "r", encoding="utf-8", errors="replace") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line or line.startswith("%"): continue
            parts = line.split()
            if len(parts) < _POS_MIN_COLS: continue
            try:
                gpst_dt, utc_dt = parse_pos_gpst_datetime(parts[0], parts[1], gps_utc_offset_sec)
                row = {
                    "line_no":      line_no,
                    "gpst_datetime": gpst_dt,
                    "utc_datetime": utc_dt,
                    "utc_ts":       utc_dt.timestamp(),
                    "lat_deg":      float(parts[2]),
                    "lon_deg":      float(parts[3]),
                    "height_m":     float(parts[4]),
                    "Q":            int(float(parts[5])),
                    "ns":           int(float(parts[6])),
                    "sdn_m":        float(parts[7]),
                    "sde_m":        float(parts[8]),
                    "sdu_m":        float(parts[9]),
                    # [FIX] Необязательные колонки — None если отсутствуют
                    "sdne_m":  float(parts[10]) if len(parts) > 10 else None,
                    "sdeu_m":  float(parts[11]) if len(parts) > 11 else None,
                    "sdun_m":  float(parts[12]) if len(parts) > 12 else None,
                    "age_s":   float(parts[13]) if len(parts) > 13 else None,
                    "ratio":   float(parts[14]) if len(parts) > 14 else None,
                }
                rows.append(row)
            except Exception:
                continue
    rows.sort(key=lambda r: r["utc_ts"])
    if len(rows) < 2:
        raise RuntimeError("В POS-файле найдено меньше 2 валидных строк")
    return rows


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371008.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return r * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def lerp(a, b, alpha):
    return float(a) + alpha * (float(b) - float(a))


def lerp_optional(a, b, alpha):
    if a is None or b is None or a == "" or b == "": return None  # [FIX] None вместо ""
    return lerp(a, b, alpha)


def lerp_angle_deg(a, b, alpha):
    if a is None or b is None or a == "" or b == "": return ""
    d = (float(b) - float(a) + 180.0) % 360.0 - 180.0
    return (float(a) + alpha * d) % 360.0


def wrap_angle_signed_deg(v):
    if v is None or v == "": return ""
    return (float(v) + 180.0) % 360.0 - 180.0


def _fmt_optional(v, fmt):
    """[NEW] Форматировать опциональное числовое поле — пустая строка если None."""
    if v is None: return ""
    return fmt % v


def interpolate_pos_row(r0, r1, alpha):
    utc_ts = lerp(r0["utc_ts"], r1["utc_ts"], alpha)
    utc_dt = datetime.fromtimestamp(utc_ts, tz=timezone.utc)
    gpst_dt = utc_dt + (r0["gpst_datetime"] - r0["utc_datetime"])
    nearest = r0 if alpha < 0.5 else r1
    return {
        "line_no":       nearest["line_no"],
        "gpst_datetime": gpst_dt,
        "utc_datetime":  utc_dt,
        "utc_ts":        utc_ts,
        "lat_deg":       lerp(r0["lat_deg"],  r1["lat_deg"],  alpha),
        "lon_deg":       lerp(r0["lon_deg"],  r1["lon_deg"],  alpha),
        "height_m":      lerp(r0["height_m"], r1["height_m"], alpha),
        "Q":   nearest["Q"],
        "ns":  nearest["ns"],
        "sdn_m": lerp(r0["sdn_m"], r1["sdn_m"], alpha),
        "sde_m": lerp(r0["sde_m"], r1["sde_m"], alpha),
        "sdu_m": lerp(r0["sdu_m"], r1["sdu_m"], alpha),
        # [FIX] lerp_optional возвращает None для отсутствующих колонок
        "sdne_m": lerp_optional(r0["sdne_m"], r1["sdne_m"], alpha),
        "sdeu_m": lerp_optional(r0["sdeu_m"], r1["sdeu_m"], alpha),
        "sdun_m": lerp_optional(r0["sdun_m"], r1["sdun_m"], alpha),
        "age_s":  lerp_optional(r0["age_s"],  r1["age_s"],  alpha),
        "ratio":  lerp_optional(r0["ratio"],  r1["ratio"],  alpha),
    }


def pos_at_utc_ts(pos_rows, utc_ts):
    xs = [r["utc_ts"] for r in pos_rows]
    if utc_ts < xs[0] or utc_ts > xs[-1]:
        raise RuntimeError(
            f"Время {iso_utc_from_ts(utc_ts)} вне диапазона POS: "
            f"{iso_utc_from_ts(xs[0])} .. {iso_utc_from_ts(xs[-1])}")
    if utc_ts == xs[0]: return pos_rows[0]
    if utc_ts == xs[-1]: return pos_rows[-1]
    j = bisect_right(xs, utc_ts) - 1
    r0, r1 = pos_rows[j], pos_rows[j + 1]
    alpha = (utc_ts - r0["utc_ts"]) / (r1["utc_ts"] - r0["utc_ts"]) if r1["utc_ts"] != r0["utc_ts"] else 0.0
    return interpolate_pos_row(r0, r1, alpha)


def build_clipped_pos_path(pos_rows, start_utc_ts, end_utc_ts):
    if end_utc_ts < start_utc_ts:
        raise RuntimeError("end_utc_ts < start_utc_ts")
    pos_start = pos_rows[0]["utc_ts"]
    pos_end   = pos_rows[-1]["utc_ts"]
    clipped_start = max(start_utc_ts, pos_start)
    clipped_end   = min(end_utc_ts,   pos_end)
    if clipped_end < clipped_start:
        raise RuntimeError(
            f"Нет пересечения: sync {iso_utc_from_ts(start_utc_ts)}..{iso_utc_from_ts(end_utc_ts)} "
            f"и POS {iso_utc_from_ts(pos_start)}..{iso_utc_from_ts(pos_end)}")
    if clipped_start != start_utc_ts or clipped_end != end_utc_ts:
        print(f"WARNING: sync-интервал обрезан по POS: "
              f"{iso_utc_from_ts(clipped_start)} .. {iso_utc_from_ts(clipped_end)}")
    path = [pos_at_utc_ts(pos_rows, clipped_start)]
    for r in pos_rows:
        if clipped_start < r["utc_ts"] < clipped_end:
            path.append(r)
    path.append(pos_at_utc_ts(pos_rows, clipped_end))
    path.sort(key=lambda r: r["utc_ts"])
    return path


def read_attitude_csv(path):
    if not path: return []
    rows = []
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as fp:
        reader = csv.DictReader(fp)
        for raw in reader:
            utc_text = (raw.get("time_utc") or raw.get("utc_datetime") or
                        raw.get("utc") or raw.get("host_wall_time_utc") or "")
            dt = parse_iso_utc(utc_text)
            if dt is None: continue
            rows.append({
                "utc_ts":     dt.timestamp(),
                "heading_deg": safe_float(raw.get("heading_deg", raw.get("heading", "")), ""),
                "roll_deg":    safe_float(raw.get("roll_deg",    raw.get("roll",    "")), ""),
                "pitch_deg":   safe_float(raw.get("pitch_deg",   raw.get("pitch",   "")), ""),
                "yaw_deg":     safe_float(raw.get("yaw_deg",     raw.get("yaw",     "")), ""),
            })
    rows.sort(key=lambda r: r["utc_ts"])
    return rows


def attitude_at_utc_ts(att_rows, utc_ts):
    empty = {"heading_deg": "", "roll_deg": "", "pitch_deg": "", "yaw_deg": "", "attitude_mode": "none"}
    if not att_rows: return empty
    # [FIX] Защита от единственной строки
    if len(att_rows) == 1:
        r = att_rows[0]
        return {**r, "attitude_mode": "single_row"}
    xs = [r["utc_ts"] for r in att_rows]
    if utc_ts <= xs[0]:
        i0, i1 = 0, 1; mode = "extrap_before"
    elif utc_ts >= xs[-1]:
        i0, i1 = len(xs) - 2, len(xs) - 1; mode = "extrap_after"
    else:
        j = bisect_right(xs, utc_ts) - 1
        i0, i1 = j, j + 1; mode = "interp"
    r0, r1 = att_rows[i0], att_rows[i1]
    alpha = (utc_ts - r0["utc_ts"]) / (r1["utc_ts"] - r0["utc_ts"]) if r1["utc_ts"] != r0["utc_ts"] else 0.0
    return {
        "heading_deg": lerp_angle_deg(r0["heading_deg"], r1["heading_deg"], alpha),
        "roll_deg":    wrap_angle_signed_deg(lerp_angle_deg(r0["roll_deg"], r1["roll_deg"], alpha)),
        "pitch_deg":   wrap_angle_signed_deg(lerp_angle_deg(r0["pitch_deg"], r1["pitch_deg"], alpha)),
        "yaw_deg":     lerp_angle_deg(r0["yaw_deg"], r1["yaw_deg"], alpha),
        "attitude_mode": mode,
    }


# ── Кусочно-якорная коррекция ────────────────────────────────────────────────
def extract_reliable_anchors(pairs, quality_level="medium", verbose=True):
    """
    quality_level: 
      - "strict": pos_exact=1, Q=1, std<=0.05м
      - "medium": pos_exact=1 OR pos_interpolated=1, Q>=1, std<=0.1м  ← НОВЫЙ
      - "loose":  любые, но Q>=1, std<=0.2м
    """
    if quality_level == "strict":
        criteria = {
            "require_exact": True,
            "allow_interpolated": False,
            "min_q": 1,
            "max_std_m": 0.05,
        }
    elif quality_level == "medium":
        criteria = {
            "require_exact": False,  # ← ПОЗВОЛЯЕМ интерполяцию
            "allow_interpolated": True,
            "min_q": 1,
            "max_std_m": 0.10,
        }
    else:  # loose
        criteria = {
            "require_exact": False,
            "allow_interpolated": True,
            "min_q": 1,
            "max_std_m": 0.20,
        }
    
    anchors = []
    rejected = []
    
    for p in pairs:
        pair_idx = p.get("pair_index", "?")
        rejection_reasons = []
        
        # Проверка качества (good или single_side_gnss)
        quality = p.get("anchor_quality", "")
        if quality not in ("good", "single_side_gnss"):
            rejection_reasons.append(f"quality={quality} (not good/single_side_gnss)")
        
        # Проверка координаты (точная или интерполированная)
        pos_exact = p.get("anchor_pos_exact")
        pos_interp = p.get("anchor_pos_interpolated")
        
        if criteria["require_exact"] and pos_exact != 1:
            rejection_reasons.append(f"pos_exact={pos_exact} (not exact)")
        elif not criteria["allow_interpolated"] and pos_interp == 1:
            rejection_reasons.append(f"pos_interpolated=1 (not allowed in strict mode)")
        elif pos_exact != 1 and pos_interp != 1:
            rejection_reasons.append(f"pos neither exact nor interpolated")
        
        # Проверка Q
        anchor_q = p.get("anchor_Q")
        if anchor_q is None or anchor_q < criteria["min_q"]:
            rejection_reasons.append(f"Q={anchor_q} (need >={criteria['min_q']})")
        
        # Проверка точности (std)
        sdn = safe_float(p.get("anchor_sdn_m"), None)
        sde = safe_float(p.get("anchor_sde_m"), None)
        sdu = safe_float(p.get("anchor_sdu_m"), None)
        
        if sdn is None or sde is None or sdu is None:
            rejection_reasons.append(f"std: sdn={sdn} sde={sde} sdu={sdu}")
        else:
            max_std = max(sdn, sde, sdu)
            if max_std > criteria["max_std_m"]:
                rejection_reasons.append(f"std={max_std:.4f}м > {criteria['max_std_m']:.4f}м")
        
        if rejection_reasons:
            rejected.append({
                "pair_index": pair_idx,
                "reasons": rejection_reasons,
            })
            continue
        
        anchors.append(p)
    
    if verbose:
        print(f"\n=== Отбор якорей (качество={quality_level}) ===")
        print(f"Всего пар: {len(pairs)}, надёжных: {len(anchors)}")
        if rejected:
            print(f"Отклонено: {len(rejected)}")
            for rej in rejected[:10]:  # Первые 10
                print(f"  пара {rej['pair_index']:3d}: {'; '.join(rej['reasons'][:2])}")
            if len(rejected) > 10:
                print(f"  ... и ещё {len(rejected)-10}")
        print()
    
    return anchors

def build_distance_on_trajectory(clipped_all):
    """
    Построить кумулятивное расстояние для каждой точки траектории.
    Возвращает список (utc_ts, cumulative_distance_m).
    """
    distances = [(clipped_all[0]["utc_ts"], 0.0)]
    cum_dist = 0.0
    
    for i in range(1, len(clipped_all)):
        r0 = clipped_all[i - 1]
        r1 = clipped_all[i]
        dist = haversine_m(r0["lat_deg"], r0["lon_deg"], r1["lat_deg"], r1["lon_deg"])
        cum_dist += dist
        distances.append((r1["utc_ts"], cum_dist))
    
    return distances


def find_anchor_distance_on_trajectory(anchor, clipped_all, distances):
    """
    Найти дистанцию якоря на траектории путём интерполяции.
    """
    anchor_utc = anchor["log_utc_ts"]
    utc_list = [d[0] for d in distances]
    dist_list = [d[1] for d in distances]
    
    if anchor_utc <= utc_list[0]:
        return dist_list[0]
    if anchor_utc >= utc_list[-1]:
        return dist_list[-1]
    
    j = bisect_right(utc_list, anchor_utc) - 1
    t0, t1 = utc_list[j], utc_list[j + 1]
    d0, d1 = dist_list[j], dist_list[j + 1]
    
    if t1 == t0:
        return d0
    
    alpha = (anchor_utc - t0) / (t1 - t0)
    return d0 + alpha * (d1 - d0)


def compute_anchor_corrections(reliable_anchors, clipped_all, pairs_sorted_utc):
    """
    Вычислить коррекции по надёжным якорям.
    
    Возвращаем список словарей с полями:
      - trajectory_distance_m
      - video_time_residual_s  (в секундах)
    """
    if not reliable_anchors or not clipped_all:
        return []
    
    distances = build_distance_on_trajectory(clipped_all)
    
    xs = [p["log_utc_ts"] for p in pairs_sorted_utc]
    ys = [p["video_time_sec"] for p in pairs_sorted_utc]
    x0, x1 = xs[0], xs[-1]
    y0, y1 = ys[0], ys[-1]
    
    corrections = []
    
    for anchor in reliable_anchors:
        anchor_utc = anchor["log_utc_ts"]
        anchor_video = anchor["video_time_sec"]
        
        trajectory_dist = find_anchor_distance_on_trajectory(anchor, clipped_all, distances)
        
        if x1 != x0:
            expected_video_time = y0 + (anchor_utc - x0) / (x1 - x0) * (y1 - y0)
        else:
            expected_video_time = y0
        
        # Разница во времени (секундах): положительное — видео позже, отрицательное — раньше
        video_time_delta_s = anchor_video - expected_video_time
        
        corrections.append({
            "pair_index": anchor.get("pair_index"),
            "log_utc_ts": anchor_utc,
            "video_time_sec": anchor_video,
            "trajectory_distance_m": trajectory_dist,
            "video_time_residual_s": video_time_delta_s,
        })
    
    return corrections


def apply_piecewise_correction(distance_m, corrections):
    """
    Применить кусочно-линейную коррекцию по временам (возвращает дельту времени в секундах).
    """
    if not corrections:
        return 0.0
    
    sorted_corr = sorted(corrections, key=lambda c: c["trajectory_distance_m"])
    
    first_dist = sorted_corr[0]["trajectory_distance_m"]
    last_dist = sorted_corr[-1]["trajectory_distance_m"]
    
    if distance_m <= first_dist:
        return 0.0
    
    if distance_m >= last_dist:
        return 0.0
    
    for i in range(len(sorted_corr) - 1):
        c0 = sorted_corr[i]
        c1 = sorted_corr[i + 1]
        d0 = c0["trajectory_distance_m"]
        d1 = c1["trajectory_distance_m"]
        
        if d0 <= distance_m <= d1:
            if d1 == d0:
                return c0["video_time_residual_s"]
            alpha = (distance_m - d0) / (d1 - d0)
            return c0["video_time_residual_s"] + alpha * (
                c1["video_time_residual_s"] - c0["video_time_residual_s"]
            )
    
    return 0.0


def adjusted_video_time_for_distance_original(distance_m, avg_speed_m_per_s, baseline_video_time_m0, corrections):
    """
    Устаревшая функция: оставлена для совместимости (не используется дальше).
    """
    uncorrected_video_time = baseline_video_time_m0 + distance_m / avg_speed_m_per_s
    residual_m = 0.0
    residual_time_s = 0.0
    return uncorrected_video_time + residual_time_s


def build_frame_plan_by_distance(pos_rows, pairs, distance_step_m, fps, frame_count, att_rows,
                                  use_anchor_correction=True):
    pairs_by_utc = sorted(pairs, key=lambda p: p["log_utc_ts"])

    print("\n=== Диагностика синхропар ===")
    warn_count = 0
    for i in range(len(pairs_by_utc) - 1):
        p0, p1 = pairs_by_utc[i], pairs_by_utc[i + 1]
        dt_utc   = p1["log_utc_ts"]    - p0["log_utc_ts"]
        dt_video = p1["video_time_sec"] - p0["video_time_sec"]
        ratio = dt_video / dt_utc if dt_utc > 0 else 0.0
        flag = ""
        if abs(ratio - 1.0) > 0.02:
            flag = "  <<< ПРОБЛЕМА (ratio далёк от 1.0)"; warn_count += 1
        elif abs(ratio - 1.0) > 0.01:
            flag = "  < подозрительно"
        print(f"  пара {p0['pair_index']:4d}->{p1['pair_index']:4d}  "
              f"dt_utc={dt_utc:7.3f}s  dt_video={dt_video:7.3f}s  ratio={ratio:.5f}{flag}")
    print("  Все сегменты в норме" if warn_count == 0 else f"  Проблемных сегментов: {warn_count}")
    print("=" * 44 + "\n")

    # ── Подготовка якорной коррекции ──
    reliable_anchors = []
    corrections = []
    if use_anchor_correction:
        reliable_anchors = extract_reliable_anchors(pairs, verbose=True)  # ← добавьте verbose=True
        print(f"Найдено надёжных якорей: {len(reliable_anchors)}")
        if reliable_anchors:
            for anchor in reliable_anchors:
                sdn = safe_float(anchor.get("anchor_sdn_m"), 0)
                sde = safe_float(anchor.get("anchor_sde_m"), 0)
                sdu = safe_float(anchor.get("anchor_sdu_m"), 0)
                max_std = max(sdn, sde, sdu)
                pair_idx = anchor.get("pair_index", "?")
                anchor_q = anchor.get("anchor_Q", "?")
                print(f"  пара {pair_idx}: Q={anchor_q}, max_std={max_std:.4f}м")

    all_rows = []
    global_distance_m = 0.0
    clipped_all = []

    pos_start_ts = pos_rows[0]["utc_ts"]
    pos_end_ts   = pos_rows[-1]["utc_ts"]
    first_valid_seg = 0
    for _i in range(len(pairs_by_utc) - 1):
        _p0, _p1 = pairs_by_utc[_i], pairs_by_utc[_i + 1]
        if _p0["log_utc_ts"] >= pos_start_ts and _p1["log_utc_ts"] <= pos_end_ts:
            _xs = [r["utc_ts"] for r in pos_rows]
            _i0 = bisect_right(_xs, _p0["log_utc_ts"]) - 1
            _i1 = bisect_right(_xs, _p1["log_utc_ts"])
            if _i1 - _i0 >= 2:
                first_valid_seg = _i; break

    if first_valid_seg > 0:
        skipped = [f"{pairs_by_utc[i]['pair_index']}→{pairs_by_utc[i+1]['pair_index']}"
                   for i in range(first_valid_seg)]
        print(f"INFO: пропущено {first_valid_seg} начальных сегментов без POS: {', '.join(skipped)}")

    for seg_i in range(first_valid_seg, len(pairs_by_utc) - 1):
        p_left  = pairs_by_utc[seg_i]
        p_right = pairs_by_utc[seg_i + 1]
        utc_left  = p_left["log_utc_ts"]
        utc_right = p_right["log_utc_ts"]
        vid_left  = p_left["video_time_sec"]
        vid_right = p_right["video_time_sec"]
        dt_utc    = utc_right - utc_left
        dt_video  = vid_right - vid_left

        try:
            seg_path = build_clipped_pos_path(pos_rows, utc_left, utc_right)
        except RuntimeError as e:
            print(f"WARNING: сегмент {p_left['pair_index']}→{p_right['pair_index']} пропущен: {e}")
            continue

        if seg_i == first_valid_seg:
            clipped_all.extend(seg_path)
        else:
            clipped_all.extend(seg_path[1:])

        seg_cum = [0.0]
        for a, b in zip(seg_path, seg_path[1:]):
            seg_cum.append(seg_cum[-1] + haversine_m(a["lat_deg"], a["lon_deg"],
                                                      b["lat_deg"], b["lon_deg"]))
        seg_total_m = seg_cum[-1]

        # [FIX] Пропуск сегментов-стоянок (seg_total_m ≈ 0) без накопления ошибки
        if seg_total_m < 1e-3:
            print(f"INFO: сегмент {p_left['pair_index']}→{p_right['pair_index']} пропущен (длина={seg_total_m:.4f}м, стоянка?)")
            continue

        if seg_i == first_valid_seg:
            first_local = 0.0
        else:
            next_global = (int(global_distance_m / distance_step_m) + 1) * distance_step_m
            first_local = next_global - global_distance_m

        targets_local = []
        d = first_local
        while d <= seg_total_m + 1e-9:
            targets_local.append(d); d += distance_step_m

        local_seg_idx = 0
        for local_m in targets_local:
            while local_seg_idx < len(seg_cum) - 2 and seg_cum[local_seg_idx + 1] < local_m:
                local_seg_idx += 1
            r0 = seg_path[local_seg_idx]
            r1 = seg_path[min(local_seg_idx + 1, len(seg_path) - 1)]
            d0 = seg_cum[local_seg_idx]
            d1 = seg_cum[min(local_seg_idx + 1, len(seg_cum) - 1)]
            alpha_pos = clamp((local_m - d0) / (d1 - d0) if d1 != d0 else 0.0, 0.0, 1.0)
            pos = interpolate_pos_row(r0, r1, alpha_pos)
            alpha_vid = clamp((pos["utc_ts"] - utc_left) / dt_utc if dt_utc > 0 else 0.0, 0.0, 1.0)
            video_time_sec = vid_left + alpha_vid * dt_video
            att = attitude_at_utc_ts(att_rows, pos["utc_ts"])
            all_rows.append({
                "distance_m":          global_distance_m + local_m,
                "video_time_sec":      video_time_sec,
                "frame_idx":           clamp(int(round(video_time_sec * fps)), 0, frame_count - 1),
                "utc_datetime":        iso_utc_from_ts(pos["utc_ts"]),
                "gpst_datetime":       pos["gpst_datetime"].isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                "lat_deg":    pos["lat_deg"],   "lon_deg":  pos["lon_deg"],
                "height_m":   pos["height_m"],  "Q":        pos["Q"],    "ns": pos["ns"],
                "sdn_m":      pos["sdn_m"],      "sde_m":    pos["sde_m"],
                "sdu_m":      pos["sdu_m"],
                "sdne_m":     pos["sdne_m"],     "sdeu_m":   pos["sdeu_m"],
                "sdun_m":     pos["sdun_m"],     "age_s":    pos["age_s"],
                "ratio":      pos["ratio"],
                "heading_deg":  att["heading_deg"],
                "roll_deg":     att["roll_deg"],
                "pitch_deg":    att["pitch_deg"],
                "yaw_deg":      att["yaw_deg"],
                "video_time_mode":      "interp_flash_segment",
                "attitude_mode":        att["attitude_mode"],
                "seg_left_pair_index":  p_left["pair_index"],
                "seg_right_pair_index": p_right["pair_index"],
                "image_file":           "",
            })

        global_distance_m += seg_total_m

    # ── Применить якорную коррекцию ──
    if use_anchor_correction and reliable_anchors and clipped_all:
        print("\n=== Применение якорной коррекции ===")
        corrections = compute_anchor_corrections(reliable_anchors, clipped_all, pairs_by_utc)
        
        if corrections:
            print(f"Вычислено коррекций: {len(corrections)}")
            for corr in corrections:
                print(f"  якорь пара {corr['pair_index']}: dist={corr['trajectory_distance_m']:.1f}м, "
                      f"residual_time={corr['video_time_residual_s']:.3f}s")
            
            # Переприменить коррекцию к каждой точке плана: прибавляем дельту времени (в секундах)
            for row in all_rows:
                dist_m = row["distance_m"]
                residual_s = apply_piecewise_correction(dist_m, corrections)
                corrected_video_time = row["video_time_sec"] + residual_s
                
                row["video_time_sec_corrected"] = corrected_video_time
                row["frame_idx_corrected"] = clamp(
                    int(round(corrected_video_time * fps)), 0, frame_count - 1)
                row["correction_applied"] = "yes"
            
            print("Коррекция применена ко всем точкам плана\n")
        else:
            print("Коррекции не вычислены (нет валидных якорей)\n")

    return all_rows, global_distance_m, clipped_all



def extract_frames(video_path, frame_rows, out_dir, jpeg_quality):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")
    for i, row in enumerate(frame_rows, start=1):
        # Используем скорректированный индекс кадра, если он явно присутствует
        frame_idx = int(row["frame_idx_corrected"] if row.get("frame_idx_corrected") is not None else row["frame_idx"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"WARNING: не удалось прочитать кадр frame_idx={frame_idx}"); continue
        name = f"frame_{i:06d}_dist_{row['distance_m']:.1f}m_f{frame_idx}.jpg"
        path = out_dir / name
        if cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]):
            row["image_file"] = str(path)
        else:
            print(f"WARNING: не удалось записать {path}")
    cap.release()



def write_frame_plan_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "distance_m", "video_time_sec", "frame_idx", "utc_datetime", "gpst_datetime",
            "lat_deg", "lon_deg", "height_m", "Q", "ns",
            "sdn_m", "sde_m", "sdu_m", "sdne_m", "sdeu_m", "sdun_m",
            "age_s", "ratio", "heading_deg", "roll_deg", "pitch_deg", "yaw_deg",
            "video_time_mode", "attitude_mode",
            "seg_left_pair_index", "seg_right_pair_index", "image_file",
            # добавлены поля коррекции
            "video_time_sec_corrected", "frame_idx_corrected", "correction_applied",
        ])
        for r in rows:
            def fmtf(v): return "" if v is None or v == "" else f"{float(v):.6f}"
            def fmt4(v): return "" if v is None or v == "" else f"{float(v):.4f}"
            def fmt3(v): return "" if v is None or v == "" else f"{float(v):.3f}"
            w.writerow([
                f"{r['distance_m']:.3f}", f"{r['video_time_sec']:.6f}", r["frame_idx"],
                r["utc_datetime"], r["gpst_datetime"],
                f"{r['lat_deg']:.11f}", f"{r['lon_deg']:.11f}", f"{r['height_m']:.4f}",
                r["Q"], r["ns"],
                fmt4(r["sdn_m"]), fmt4(r["sde_m"]), fmt4(r["sdu_m"]),
                fmt4(r["sdne_m"]), fmt4(r["sdeu_m"]), fmt4(r["sdun_m"]),  # [FIX] None→""
                fmt3(r["age_s"]), fmt3(r["ratio"]),
                fmtf(r["heading_deg"]), fmtf(r["roll_deg"]),
                fmtf(r["pitch_deg"]),   fmtf(r["yaw_deg"]),
                r["video_time_mode"], r["attitude_mode"],
                r["seg_left_pair_index"], r["seg_right_pair_index"], r["image_file"],
                fmtf(r.get("video_time_sec_corrected")), r.get("frame_idx_corrected", ""), r.get("correction_applied", ""),
            ])



def write_clipped_pos_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "utc_datetime", "gpst_datetime", "lat_deg", "lon_deg", "height_m",
            "Q", "ns", "sdn_m", "sde_m", "sdu_m",
            "sdne_m", "sdeu_m", "sdun_m", "age_s", "ratio",
        ])
        for r in rows:
            def fmt4(v): return "" if v is None else f"{float(v):.4f}"
            def fmt3(v): return "" if v is None else f"{float(v):.3f}"
            w.writerow([
                iso_utc_from_ts(r["utc_ts"]),
                r["gpst_datetime"].isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                f"{r['lat_deg']:.11f}", f"{r['lon_deg']:.11f}", f"{r['height_m']:.4f}",
                r["Q"], r["ns"],
                fmt4(r["sdn_m"]), fmt4(r["sde_m"]), fmt4(r["sdu_m"]),
                fmt4(r["sdne_m"]), fmt4(r["sdeu_m"]), fmt4(r["sdun_m"]),
                fmt3(r["age_s"]), fmt3(r["ratio"]),
            ])


def load_alignment_csv(path):
    pairs = []
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as fp:
        reader = csv.DictReader(fp)
        for raw in reader:
            dt = None
            utc_field_used = ""
            for field in ["log_utc_datetime", "host_wall_time_utc"]:
                val = raw.get(field, "")
                if val:
                    dt = parse_iso_utc(val)
                if dt is not None:
                    utc_field_used = field
                    break
            if dt is None:
                print(f"WARNING: строка alignment без UTC, пропущена: {dict(raw)}")
                continue
            pairs.append({
                "pair_index":        safe_int(raw.get("pair_index"), len(pairs) + 1),
                "video_flash_index": safe_int(raw.get("video_flash_index"), 0),
                "video_time_sec":    safe_float(raw.get("video_time_sec"), 0.0),
                "video_frame_idx":   safe_int(raw.get("video_frame_idx"), 0),
                "detected_on":       raw.get("detected_on", ""),
                "log_run_id":        safe_int(raw.get("log_run_id"), 0),
                "blink_index_global": safe_int(raw.get("blink_index_global"), 0),
                "blink_index_in_run": safe_int(raw.get("blink_index_in_run"), 0),
                "host_monotonic_ns":  safe_int(raw.get("host_monotonic_ns"), None),
                "host_wall_time_utc": raw.get("host_wall_time_utc", ""),
                "log_utc_ts":         dt.timestamp(),
                # Поля якорной коррекции
                "anchor_quality":     raw.get("anchor_quality", ""),
                "anchor_pos_exact":   safe_int(raw.get("anchor_pos_exact"), 0),
                "anchor_pos_interpolated": safe_int(raw.get("anchor_pos_interpolated"), 0),
                "anchor_lat_deg":     safe_float(raw.get("anchor_lat_deg")),
                "anchor_lon_deg":     safe_float(raw.get("anchor_lon_deg")),
                "anchor_height_m":    safe_float(raw.get("anchor_height_m")),
                "anchor_Q":           safe_int(raw.get("anchor_Q")),
                "anchor_ns":          safe_int(raw.get("anchor_ns")),
                "anchor_sdn_m":       safe_float(raw.get("anchor_sdn_m")),
                "anchor_sde_m":       safe_float(raw.get("anchor_sde_m")),
                "anchor_sdu_m":       safe_float(raw.get("anchor_sdu_m")),
                "anchor_is_spatial":  safe_int(raw.get("anchor_is_spatial"), 0),
            })
    if len(pairs) < 2:
        raise RuntimeError(f"В alignment CSV найдено меньше 2 строк: {path}")
    return pairs




def auto_find_alignments(work_dir):
    return sorted(glob.glob(os.path.join(work_dir, "video_*_flash_alignment.csv")))

def extract_video_id_from_alignment(path):
    name = Path(path).name
    m = re.search(r'(video_\d+)_flash_alignment\.csv$', name, re.IGNORECASE)
    return m.group(1) if m else None

def write_multi_frame_plan_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "session_index", "video_name", "video_file", "alignment_file",
            "distance_m", "video_time_sec", "frame_idx", "utc_datetime", "gpst_datetime",
            "lat_deg", "lon_deg", "height_m", "Q", "ns",
            "sdn_m", "sde_m", "sdu_m", "sdne_m", "sdeu_m", "sdun_m",
            "age_s", "ratio", "heading_deg", "roll_deg", "pitch_deg", "yaw_deg",
            "video_time_mode", "attitude_mode",
            "seg_left_pair_index", "seg_right_pair_index", "image_file",
            "video_time_sec_corrected", "frame_idx_corrected", "correction_applied",
        ])
        for r in rows:
            def fmtf(v): return "" if v is None or v == "" else f"{float(v):.6f}"
            def fmt4(v): return "" if v is None or v == "" else f"{float(v):.4f}"
            def fmt3(v): return "" if v is None or v == "" else f"{float(v):.3f}"
            w.writerow([
                r.get("session_index", ""), r.get("video_name", ""), r.get("video_file", ""), r.get("alignment_file", ""),
                f"{r['distance_m']:.3f}", f"{r['video_time_sec']:.6f}", r["frame_idx"],
                r["utc_datetime"], r["gpst_datetime"],
                f"{r['lat_deg']:.11f}", f"{r['lon_deg']:.11f}", f"{r['height_m']:.4f}",
                r["Q"], r["ns"],
                fmt4(r["sdn_m"]), fmt4(r["sde_m"]), fmt4(r["sdu_m"]),
                fmt4(r["sdne_m"]), fmt4(r["sdeu_m"]), fmt4(r["sdun_m"]),
                fmt3(r["age_s"]), fmt3(r["ratio"]),
                fmtf(r["heading_deg"]), fmtf(r["roll_deg"]), fmtf(r["pitch_deg"]), fmtf(r["yaw_deg"]),
                r["video_time_mode"], r["attitude_mode"],
                r["seg_left_pair_index"], r["seg_right_pair_index"], r["image_file"],
                fmtf(r.get("video_time_sec_corrected")), r.get("frame_idx_corrected", ""), r.get("correction_applied", ""),
            ])

def extract_frames_multi(video_jobs, pos_rows, distance_m, fps_cache, attitude_path, extract_frames_dir, 
                        jpeg_quality, gps_utc_offset_sec, use_anchor_correction=True):
    att_rows = []
    if attitude_path:
        att_rows = read_attitude_csv(attitude_path)
        print(f"Строк attitude: {len(att_rows)}")

    all_rows = []
    image_counter = 0
    out_frames_dir = Path(extract_frames_dir)
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    for sess_idx, job in enumerate(video_jobs, start=1):
        video_path = job["video_path"]
        alignment_path = job["alignment_path"]
        video_name = Path(video_path).name
        video_stem = Path(video_path).stem

        print("\n" + "=" * 60)
        print(f"Сессия {sess_idx}/{len(video_jobs)}: {video_name}")

        pairs = load_alignment_csv(alignment_path)
        print(f"Пар синхронизации: {len(pairs)}")
        print(f"UTC: {iso_utc_from_ts(pairs[0]['log_utc_ts'])} .. {iso_utc_from_ts(pairs[-1]['log_utc_ts'])}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"FPS={fps:.6f} кадров={frame_count}")

        frame_plan, total_m, _ = build_frame_plan_by_distance(
            pos_rows=pos_rows, pairs=pairs, distance_step_m=distance_m,
            fps=fps, frame_count=frame_count, att_rows=att_rows,
            use_anchor_correction=use_anchor_correction)
        print(f"Длина траектории: {total_m:.3f} м")
        print(f"Точек плана: {len(frame_plan)}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {video_path}")
        for row in frame_plan:
            # Использовать скорректированный frame_idx если он явно задан (не полагаться на truthiness)
            frame_idx = int(row["frame_idx_corrected"] if row.get("frame_idx_corrected") is not None else row["frame_idx"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"WARNING: не удалось прочитать кадр frame_idx={frame_idx} из {video_name}")
                continue
            image_counter += 1
            name = f"frame_{image_counter:06d}_{video_stem}_dist_{row['distance_m']:.1f}m_f{frame_idx}.jpg"
            path = out_frames_dir / name
            if cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]):
                row["image_file"] = str(path)
            else:
                print(f"WARNING: не удалось записать {path}")
                row["image_file"] = ""
            row["session_index"] = sess_idx
            row["video_name"] = video_name
            row["video_file"] = str(video_path)
            row["alignment_file"] = str(alignment_path)
            all_rows.append(row)
        cap.release()

    all_rows.sort(key=lambda r: (r["utc_datetime"], r["session_index"], r["frame_idx"]))
    return all_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance-m", type=float, default=50.0)
    parser.add_argument("--gps-utc-offset-sec", type=float, default=18.0)
    parser.add_argument("--pos", default="gnss0.pos", help="gnss0.pos")
    parser.add_argument("--attitude", default="navigation_table.csv", help="navigation_table.csv (опционально)")
    parser.add_argument("--extract-frames-dir", default="frames")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--no-anchor-correction", action="store_true",
                        help="Отключить якорную коррекцию (по умолчанию включена)")
    args = parser.parse_args()

    work_dir = os.getcwd()
    pos_path = os.path.abspath(args.pos)
    attitude_path = os.path.abspath(args.attitude) if args.attitude else ""
    frames_dir = os.path.abspath(args.extract_frames_dir)
    out_csv = os.path.join(work_dir, f"video_frame_plan_{args.distance_m:g}m.csv")
    use_anchor_correction = not args.no_anchor_correction

    alignments = auto_find_alignments(work_dir)
    if not alignments:
        raise RuntimeError("Не найдены video_*_flash_alignment.csv в текущей папке")

    video_jobs = []
    for alignment in alignments:
        video_id = extract_video_id_from_alignment(alignment)
        if not video_id:
            continue
        video_path = os.path.join(work_dir, video_id + ".mp4")
        if not os.path.exists(video_path):
            print(f"WARNING: видео не найдено для {os.path.basename(alignment)}: {video_path}")
            continue
        video_jobs.append({
            "alignment_path": os.path.abspath(alignment),
            "video_path": os.path.abspath(video_path),
            "video_id": video_id,
        })

    if not video_jobs:
        raise RuntimeError("Не найдено ни одной пары video_XX.mp4 + video_XX_flash_alignment.csv")

    print(f"Найдено сессий: {len(video_jobs)}")
    for job in video_jobs:
        print(f" {Path(job['video_path']).name} <-> {Path(job['alignment_path']).name}")

    pos_rows = read_pos_file(pos_path, args.gps_utc_offset_sec)
    print(f"\nСтрок POS: {len(pos_rows)}")
    print(f"POS UTC: {iso_utc_from_ts(pos_rows[0]['utc_ts'])} .. {iso_utc_from_ts(pos_rows[-1]['utc_ts'])}")

    print(f"\nЯкорная коррекция: {'ВКЛЮЧЕНА' if use_anchor_correction else 'ОТКЛЮЧЕНА'}")

    rows = extract_frames_multi(
        video_jobs=video_jobs,
        pos_rows=pos_rows,
        distance_m=args.distance_m,
        fps_cache=None,
        attitude_path=attitude_path if os.path.exists(attitude_path) else "",
        extract_frames_dir=frames_dir,
        jpeg_quality=args.jpeg_quality,
        gps_utc_offset_sec=args.gps_utc_offset_sec,
        use_anchor_correction=use_anchor_correction,
    )

    write_multi_frame_plan_csv(out_csv, rows)
    print(f"\nГотово.")
    print(f" {out_csv}")
    print(f" Frames: {frames_dir}")

if __name__ == "__main__":
    main()