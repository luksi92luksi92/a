#!/usr/bin/env python3
"""EDM musical-layer inference on top of polyphonic note events.

The raw transcriber remains authoritative for note events. This module adds
soft musical interpretation: key/scale, chords, lead, counter-melody,
arpeggios, pads, stabs, one-shots, motifs, pitch expression, stereo context,
and sidechain/envelope evidence.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "B")
MAJOR_PROFILE = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
CHORD_TEMPLATES = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7), "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10),
}

def _name_pc(pc: int) -> str:
    return NOTE_NAMES[int(pc) % 12]

def _safe_key(chroma: np.ndarray) -> Dict[str, Any]:
    v = np.asarray(chroma, dtype=float).reshape(12)
    if not np.any(v):
        return {"key": None, "tonic_pc": None, "mode": None, "confidence": 0.0, "profile_margin": 0.0}
    v /= max(float(np.linalg.norm(v)), 1e-9)
    best = (-1.0, 0, "major")
    second = -1.0
    for tonic in range(12):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            p = np.roll(profile, tonic); p /= max(float(np.linalg.norm(p)), 1e-9)
            s = float(np.dot(v, p))
            if s > best[0]:
                second = best[0]; best = (s, tonic, mode)
            elif s > second:
                second = s
    margin = best[0] - second
    return {
        "key": f"{_name_pc(best[1])} {best[2]}",
        "tonic_pc": int(best[1]),
        "mode": best[2],
        "confidence": float(np.clip((best[0] - 0.2) / 0.6, 0.0, 1.0)),
        "profile_margin": float(margin),
    }

def estimate_key_context(audio: np.ndarray, sample_rate: int, beat_times: np.ndarray, beats_per_bar: int = 4) -> Dict[str, Any]:
    try:
        import librosa
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        chroma = librosa.feature.chroma_cqt(y=x, sr=int(sample_rate), hop_length=512)
        frame_times = librosa.times_like(chroma[0], sr=int(sample_rate), hop_length=512)
        global_key = _safe_key(np.mean(chroma, axis=1))
        local = []
        step = max(1, int(beats_per_bar) * 8)
        for i in range(0, len(beat_times), step):
            start = float(beat_times[i])
            end = float(beat_times[min(i + step, len(beat_times) - 1)])
            if i + step < len(beat_times):
                end += float(np.median(np.diff(beat_times)))
            idx = np.where((frame_times >= start) & (frame_times < end))[0]
            if idx.size >= 4:
                k = _safe_key(np.mean(chroma[:, idx], axis=1))
                k.update({"start_time_s": start, "end_time_s": end})
                local.append(k)
        return {"global": global_key, "local": local, "scale_constraint": "soft", "method": "chroma_cqt_profile_matching"}
    except Exception as exc:
        return {"global": {"key": None, "confidence": 0.0}, "local": [], "scale_constraint": "soft", "error": repr(exc)}

def _local_key(t: float, context: Dict[str, Any]) -> Dict[str, Any]:
    for k in context.get("local", []):
        if float(k["start_time_s"]) <= t < float(k["end_time_s"]):
            return k
    return context.get("global", {})

def key_fit(midi: int, key_info: Dict[str, Any]) -> float:
    tonic = key_info.get("tonic_pc"); mode = key_info.get("mode")
    if tonic is None or mode is None:
        return 0.5
    pcs = {0, 2, 4, 5, 7, 9, 11} if mode == "major" else {0, 2, 3, 5, 7, 8, 10}
    return 1.0 if midi % 12 in {(tonic + p) % 12 for p in pcs} else 0.25

def _event_groups(events: Sequence[Dict[str, Any]], spread: float = 0.08) -> List[List[Dict[str, Any]]]:
    ordered = sorted(events, key=lambda e: (float(e["start_time_s"]), int(e["pitch_midi"])))
    groups: List[List[Dict[str, Any]]] = []
    for ev in ordered:
        if not groups:
            groups.append([ev]); continue
        anchor = float(np.median([float(x["start_time_s"]) for x in groups[-1]]))
        if abs(float(ev["start_time_s"]) - anchor) <= spread:
            groups[-1].append(ev)
        else:
            groups.append([ev])
    return groups

def _chord_score(pcs: Sequence[int], root: int, quality: str) -> float:
    template = {(root + p) % 12 for p in CHORD_TEMPLATES[quality]}
    have = set(int(p) % 12 for p in pcs)
    if not have:
        return 0.0
    hits = len(have & template)
    extra = len(have - template)
    missing = len(template - have)
    return float(np.clip(hits / max(len(template), 1) - 0.12 * extra - 0.06 * missing, 0.0, 1.0))

def infer_chord(events: Sequence[Dict[str, Any]], key_info: Dict[str, Any]) -> Dict[str, Any] | None:
    if len(events) < 2:
        return None
    pcs = [int(e["pitch_midi"]) % 12 for e in events]
    best = None
    for root in range(12):
        for quality in CHORD_TEMPLATES:
            s = _chord_score(pcs, root, quality)
            if key_fit(root, key_info) > 0.5:
                s += 0.08
            cand = (s, root, quality)
            if best is None or cand[0] > best[0]:
                best = cand
    if best is None or best[0] < 0.42:
        return None
    tones = sorted({(best[1] + p) % 12 for p in CHORD_TEMPLATES[best[2]]})
    return {
        "root": _name_pc(best[1]),
        "root_pc": int(best[1]),
        "quality": best[2],
        "confidence": float(np.clip(best[0], 0.0, 1.0)),
        "pitch_classes": sorted(set(pcs)),
        "chord_tones": tones,
    }

def _pitch_support(audio: np.ndarray, sample_rate: int, start: float, end: float, midi: int) -> float:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    f0 = 440.0 * (2.0 ** ((int(midi) - 69) / 12.0))
    center = int(round(((start + end) * 0.5) * sample_rate))
    n = int(max(1024, min(8192, round(max(0.10, min(0.50, end - start)) * sample_rate))))
    a = max(0, center - n // 2); b = min(len(x), a + n)
    seg = x[a:b]
    if len(seg) < 256 or f0 <= 0:
        return 0.0
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sample_rate)
    total = float(np.sum(spec[(freqs >= 40) & (freqs <= 4000)]) + 1e-9)
    weighted = 0.0; weights = 0.0
    for h in range(1, 6):
        f = f0 * h
        if f > 4000: break
        width = max(2.0, 0.012 * f)
        idx = np.abs(freqs - f) <= width
        if np.any(idx):
            weighted += (1.0 / h) * float(np.max(spec[idx])); weights += 1.0 / h
    return float(np.clip((weighted / max(weights, 1e-9)) / total * 20.0, 0.0, 1.0))

def _duration_class(d: float) -> str:
    if d < 0.12: return "short"
    if d < 0.35: return "medium"
    if d < 1.0: return "sustained"
    return "long"

def _expression(bends: Sequence[int] | None) -> Dict[str, Any]:
    vals = [int(v) for v in (bends or [])]
    if not vals:
        return {"has_pitch_expression": False, "type": "none", "range_units": 0}
    r = max(vals) - min(vals)
    delta = np.diff(vals) if len(vals) > 1 else np.zeros(0)
    kind = "stable" if r == 0 else ("glide_or_strong_bend" if np.any(np.abs(delta) >= 2) else "subtle_bend_or_vibrato")
    return {"has_pitch_expression": bool(r), "type": kind, "range_units": int(r)}

def enrich_events(audio: np.ndarray, sample_rate: int, events: Sequence[Dict[str, Any]], key_context: Dict[str, Any], stereo_channels: np.ndarray | None = None) -> List[Dict[str, Any]]:
    out = []
    for ev in events:
        row = dict(ev)
        start = float(row["start_time_s"]); end = float(row["end_time_s"]); pitch = int(row["pitch_midi"])
        dur = max(0.0, end - start)
        support = _pitch_support(audio, sample_rate, start, end, pitch)
        k = _local_key((start + end) * 0.5, key_context)
        fit = key_fit(pitch, k)
        register = float(np.clip((pitch - 48.0) / 36.0, 0.0, 1.0))
        row.update({
            "pitch_name": f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}",
            "pitch_class": pitch % 12,
            "duration_s": dur,
            "duration_class": _duration_class(dur),
            "spectral_support": support,
            "key_fit": fit,
            "register_score": register,
            "pitch_expression": _expression(row.get("pitch_bends")),
        })
        if stereo_channels is not None:
            center = int(round(((start + end) * 0.5) * sample_rate)); n = int(max(512, min(8192, round(max(0.10, dur) * sample_rate))))
            a = max(0, center - n // 2); b = min(stereo_channels.shape[1], a + n); seg = stereo_channels[:, a:b]
            if seg.shape[1] >= 32:
                mid = 0.5 * (seg[0] + seg[1]); side = 0.5 * (seg[0] - seg[1])
                row["stereo_width"] = float(np.clip(np.sqrt(np.mean(side**2) + 1e-12) / max(np.sqrt(np.mean(mid**2) + 1e-12), 1e-9), 0.0, 2.0))
        row["note_confidence"] = float(np.clip(
            0.40 * float(row.get("confidence", row.get("velocity", 0.0))) +
            0.35 * support + 0.15 * fit + 0.10 * min(1.0, dur / 0.35), 0.0, 1.0))
        out.append(row)
    return out

def infer_chords(events: Sequence[Dict[str, Any]], key_context: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    annotated = [dict(e) for e in events]; chords = []
    for idx, group in enumerate(_event_groups(annotated)):
        if len(group) < 2: continue
        k = _local_key(float(np.median([float(e["start_time_s"]) for e in group])), key_context)
        c = infer_chord(group, k)
        if c is None: continue
        start = min(float(e["start_time_s"]) for e in group); end = max(float(e["end_time_s"]) for e in group)
        cid = f"chord_{idx:04d}"
        role = "pad" if end - start >= 0.55 else ("stab" if end - start <= 0.22 else "chord")
        row = dict(c)
        row.update({
            "id": cid, "start_time_s": start, "end_time_s": end, "duration_s": max(0.0, end - start),
            "duration_class": _duration_class(max(0.0, end - start)), "role": role,
            "note_count": len(group), "notes": [int(e["pitch_midi"]) for e in group],
        })
        chords.append(row)
        tones = set(c["chord_tones"])
        for ev in group:
            if int(ev["pitch_midi"]) % 12 in tones:
                ev.setdefault("chord_memberships", []).append(cid)
    return annotated, chords

def _emit(ev: Dict[str, Any]) -> float:
    duration_score = {"short": 0.45, "medium": 0.70, "sustained": 0.72, "long": 0.55}.get(ev.get("duration_class"), 0.5)
    bend = 0.05 if ev.get("pitch_expression", {}).get("has_pitch_expression") else 0.0
    return float(0.25 * ev.get("note_confidence", 0.0) + 0.20 * ev.get("spectral_support", 0.0) +
                 0.20 * ev.get("register_score", 0.0) + 0.15 * ev.get("key_fit", 0.5) +
                 0.10 * duration_score + 0.05 * ev.get("confidence", 0.0) + bend -
                 0.18 * min(1.0, 0.45 * len(ev.get("chord_memberships", []))))

def _transition(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    gap = float(b["start_time_s"]) - float(a["end_time_s"])
    if gap < -0.05: return -0.35
    interval = abs(int(b["pitch_midi"]) - int(a["pitch_midi"]))
    interval_score = 1.0 if interval <= 2 else 0.85 if interval <= 5 else 0.65 if interval <= 7 else 0.45 if interval <= 12 else 0.25
    gap_score = float(np.exp(-max(gap, 0.0) / 0.55)) if gap <= 1.6 else 0.0
    return 0.55 * interval_score + 0.45 * gap_score

def infer_lead(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not events: return []
    groups = _event_groups(events, 0.085)
    states = [max(g, key=_emit) for g in groups if g]
    if not states: return []
    dp = np.full(len(states), -1e9); prev = np.full(len(states), -1, dtype=int)
    for i, ev in enumerate(states):
        dp[i] = _emit(ev)
        for j in range(i):
            cand = dp[j] + _transition(states[j], ev) + 0.35 * _emit(ev)
            if cand > dp[i]:
                dp[i] = cand; prev[i] = j
    idx = int(np.argmax(dp)); path = []
    while idx >= 0:
        path.append(dict(states[idx])); idx = int(prev[idx])
    path.reverse()
    out = []
    for i, ev in enumerate(path):
        cont = 0.0
        if i: cont += _transition(path[i-1], ev)
        if i + 1 < len(path): cont += _transition(ev, path[i+1])
        score = float(np.clip(0.70 * _emit(ev) + 0.30 * min(1.0, cont / 2.0), 0.0, 1.0))
        if score >= 0.38 or ev.get("pitch_expression", {}).get("has_pitch_expression"):
            row = dict(ev); row["role"] = "lead"; row["lead_score"] = score; out.append(row)
    return out

def infer_counter(events: Sequence[Dict[str, Any]], lead: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lead_keys = {(round(float(e["start_time_s"]), 3), int(e["pitch_midi"])) for e in lead}
    cand = [e for e in events if (round(float(e["start_time_s"]), 3), int(e["pitch_midi"])) not in lead_keys and float(e.get("note_confidence", 0.0)) >= 0.32]
    groups = _event_groups(cand, 0.10); out = []; last = None
    for g in groups:
        best = None
        for ev in g:
            score = 0.38 * ev.get("note_confidence", 0.0) + 0.22 * ev.get("spectral_support", 0.0) + 0.18 * ev.get("register_score", 0.0) + 0.12 * ev.get("key_fit", 0.5) + 0.10 * min(1.0, ev.get("duration_s", 0.0) / 0.35)
            if last is not None: score += 0.15 if abs(int(ev["pitch_midi"]) - last) <= 7 else -0.05
            if ev.get("chord_memberships"): score -= 0.18
            if best is None or score > best[0]: best = (score, ev)
        if best and best[0] >= 0.42:
            row = dict(best[1]); row["role"] = "counter_melody"; row["counter_score"] = float(np.clip(best[0], 0, 1)); out.append(row); last = int(row["pitch_midi"])
    return out

def infer_arpeggios(events: Sequence[Dict[str, Any]], chords: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i in range(max(0, len(events) - 3)):
        w = list(events[i:i+4]); starts = [float(e["start_time_s"]) for e in w]; durs = [float(e["duration_s"]) for e in w]
        span = starts[-1] - starts[0]; pcs = {int(e["pitch_midi"]) % 12 for e in w}
        if span > 1.5 or np.mean(durs) > 0.38 or len(pcs) < 3: continue
        nearby = [c for c in chords if c["start_time_s"] - 0.15 <= np.mean(starts) <= c["end_time_s"] + 0.15]
        support = 0.0
        if nearby:
            chord_pcs = set().union(*(set(c.get("pitch_classes", [])) for c in nearby))
            support = len(pcs & chord_pcs) / max(len(pcs), 1)
        rhythm = np.diff(starts)
        reg = 1.0 - min(1.0, float(np.std(rhythm)) / max(float(np.mean(rhythm)), 1e-4))
        score = 0.50 * support + 0.35 * reg + 0.15 * min(1.0, len(pcs) / 4.0)
        if score >= 0.58:
            out.append({"start_time_s": starts[0], "end_time_s": max(float(e["end_time_s"]) for e in w),
                        "pitch_midi": [int(e["pitch_midi"]) for e in w], "pitch_classes": sorted(pcs), "score": float(score), "role": "arpeggio"})
    merged = []
    for item in out:
        if merged and item["start_time_s"] <= merged[-1]["end_time_s"] + 0.06:
            merged[-1]["end_time_s"] = max(merged[-1]["end_time_s"], item["end_time_s"])
            merged[-1]["score"] = max(merged[-1]["score"], item["score"])
            merged[-1]["pitch_midi"] = sorted(set(merged[-1]["pitch_midi"] + item["pitch_midi"]))
        else:
            merged.append(item)
    return merged

def infer_motifs(lead: Sequence[Dict[str, Any]], beat_times: np.ndarray) -> List[Dict[str, Any]]:
    if len(lead) < 8: return []
    starts = np.asarray([float(e["start_time_s"]) for e in lead]); pitches = np.asarray([int(e["pitch_midi"]) for e in lead])
    beat = float(np.median(np.diff(beat_times))) if len(beat_times) > 1 else 0.5
    out = []
    for n in (4, 5, 6, 8):
        for i in range(0, len(pitches) - 2*n + 1):
            a = pitches[i:i+n] - pitches[i]
            for j in range(i+n, len(pitches) - n + 1):
                b = pitches[j:j+n] - pitches[j]
                ps = 1.0 - min(1.0, float(np.mean(np.abs(a-b))) / 12.0)
                ra = np.diff(starts[i:i+n]); rb = np.diff(starts[j:j+n])
                rs = 1.0 - min(1.0, float(np.mean(np.abs(ra-rb))) / max(beat, 1e-3)) if ra.size else 0.5
                score = 0.65 * ps + 0.35 * rs
                if score >= 0.78:
                    out.append({"start_time_s": float(starts[i]), "repeat_time_s": float(starts[j]), "length_notes": int(n), "score": float(score), "role": "lead_motif", "pitch_intervals": a.astype(int).tolist()})
                    break
    return out

def sidechain_evidence(audio: np.ndarray, sample_rate: int, beat_times: np.ndarray) -> Dict[str, Any]:
    if len(beat_times) < 4: return {"detected": False, "strength": 0.0}
    x = np.asarray(audio, dtype=np.float32).reshape(-1); ratios = []
    for t in beat_times:
        c = int(round(float(t) * sample_rate)); r = int(round(0.22 * sample_rate))
        a = max(0, c-r); b = min(len(x), c+r)
        seg = np.abs(x[a:b])
        if len(seg) < 32: continue
        q = max(4, len(seg)//8); pre = float(np.mean(seg[:q])); post = float(np.mean(seg[q:3*q]))
        ratios.append(pre / max(post, 1e-9))
    if not ratios: return {"detected": False, "strength": 0.0}
    strength = float(np.clip(np.median(ratios) - 0.7, 0.0, 0.6) / 0.6)
    return {"detected": bool(strength >= 0.35), "strength": strength, "method": "beat-centered envelope proxy"}

def stereo_context(audio: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(audio)
    if x.ndim != 2: return {"available": False}
    if x.shape[0] != 2 and x.shape[1] == 2: x = x.T
    if x.shape[0] != 2 or x.shape[1] < 32: return {"available": False}
    mid = 0.5 * (x[0] + x[1]); side = 0.5 * (x[0] - x[1])
    width = float(np.sqrt(np.mean(side.astype(float)**2)) / max(np.sqrt(np.mean(mid.astype(float)**2)), 1e-9))
    return {"available": True, "mean_width_ratio": float(np.clip(width, 0, 2)), "interpretation": "wide" if width > 0.55 else "centered"}

def analyze_other_music(audio: np.ndarray, sample_rate: int, beat_times: np.ndarray, events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    x = np.asarray(audio)
    mono = np.mean(x, axis=0) if x.ndim == 2 and x.shape[0] == 2 else (np.mean(x, axis=1) if x.ndim == 2 and x.shape[1] == 2 else x.reshape(-1))
    key = estimate_key_context(mono, sample_rate, np.asarray(beat_times, dtype=float))
    enriched = enrich_events(mono, sample_rate, events, key)
    enriched, chords = infer_chords(enriched, key)
    lead = infer_lead(enriched)
    counter = infer_counter(enriched, lead)
    arps = infer_arpeggios(enriched, chords)
    lead_keys = {(round(float(e["start_time_s"]),3), int(e["pitch_midi"])) for e in lead}
    counter_keys = {(round(float(e["start_time_s"]),3), int(e["pitch_midi"])) for e in counter}
    chord_keys = {(round(float(e["start_time_s"]),3), int(e["pitch_midi"])) for e in enriched if e.get("chord_memberships")}
    one_shots = [dict(e) for e in enriched if (round(float(e["start_time_s"]),3), int(e["pitch_midi"])) not in lead_keys and float(e["duration_s"]) <= 0.18 and not e.get("chord_memberships")]
    for e in enriched:
        k = (round(float(e["start_time_s"]),3), int(e["pitch_midi"]))
        e["inferred_role"] = "lead" if k in lead_keys else "counter_melody" if k in counter_keys else "chord_tone" if k in chord_keys else "ornament_or_one_shot" if float(e["duration_s"]) <= 0.18 else "supporting_pitch"
    pads = [c for c in chords if c["role"] == "pad"]; stabs = [c for c in chords if c["role"] == "stab"]
    return {
        "analysis_version": "edm-musical-input-v1",
        "key_context": key,
        "chord_events": chords,
        "lead_events": lead,
        "counter_melody_events": counter,
        "arpeggio_events": arps,
        "pad_events": pads,
        "stab_events": stabs,
        "one_shot_events": one_shots,
        "motifs": infer_motifs(lead, np.asarray(beat_times, dtype=float)),
        "sidechain": sidechain_evidence(mono, sample_rate, np.asarray(beat_times, dtype=float)),
        "stereo_context": stereo_context(x),
        "all_note_events": enriched,
    }
