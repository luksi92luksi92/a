#!/usr/bin/env python3
"""Post-run quality gates for the EDM reverse-DAW wrapper.

Keeps bad/underconstrained hypotheses out of user-facing summaries without
changing the authoritative physical evidence. In particular:
- estimates tempo from repeated transient intervals instead of trusting a
  single drifting groove state;
- removes one-occurrence foundation patterns from the report;
- removes trailing/too-short section candidates from the report;
- makes pattern-change wording consistent with the confirmed pattern set;
- recalculates groove-derived style state after tempo correction.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def transient_times(world: Any) -> List[float]:
    times = sorted({
        float(e.time)
        for e in getattr(world, "events", [])
        if getattr(e, "event_type", "") == "transient_detected"
    })
    return times


def _tempo_votes(times: Sequence[float], min_bpm: float = 80.0,
                 max_bpm: float = 180.0) -> Tuple[float, float, int]:
    if len(times) < 12:
        return 0.0, 0.0, 0
    arr = np.asarray(times, dtype=float)
    diffs = np.diff(arr)
    diffs = diffs[(diffs >= 0.08) & (diffs <= 2.0)]
    if diffs.size < 10:
        return 0.0, 0.0, int(diffs.size)

    votes: Dict[float, float] = {}
    for d in diffs:
        for multiple in range(1, 9):
            bpm = 60.0 / (float(d) * multiple)
            if not (min_bpm <= bpm <= max_bpm):
                continue
            key = round(bpm * 2.0) / 2.0
            error = abs((60.0 / key) / d - round((60.0 / key) / d))
            weight = 1.0 / (1.0 + 12.0 * error)
            votes[key] = votes.get(key, 0.0) + weight
    if not votes:
        return 0.0, 0.0, int(diffs.size)
    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    bpm, score = ranked[0]
    total = sum(v for _, v in ranked)
    confidence = float(np.clip(score / max(total, 1e-9) * 4.0, 0.0, 1.0))
    return float(bpm), confidence, int(diffs.size)


def _assign_phase(times: Sequence[float], period: float) -> Tuple[float, float]:
    """Find a beat phase that best explains observed onsets on the quarter-beat grid."""
    if not times or period <= 0:
        return 0.0, 0.0
    arr = np.asarray(times, dtype=float)
    phases = np.linspace(0.0, period, 96, endpoint=False)
    best_phase, best_err = 0.0, float("inf")
    subdivision = period / 4.0
    for phase in phases:
        rel = (arr - phase) / subdivision
        err = np.abs(rel - np.round(rel)) * subdivision
        score = float(np.median(err))
        if score < best_err:
            best_err = score
            best_phase = float(phase)
    return best_phase, best_err


def correct_groove(world: Any) -> Dict[str, float]:
    """Replace a visibly unstable tempo state only when repeated evidence supports it."""
    times = transient_times(world)
    bpm, confidence, n_intervals = _tempo_votes(times)
    if bpm <= 0 or confidence < 0.20:
        return {"bpm": float(getattr(world.groove, "tempo_bpm", 0.0)),
                "confidence": float(getattr(world.groove, "tempo_confidence", 0.0)),
                "intervals": float(n_intervals), "corrected": 0.0}

    groove = world.groove
    old_bpm = float(getattr(groove, "tempo_bpm", 0.0))
    old_conf = float(getattr(groove, "tempo_confidence", 0.0))
    period = 60.0 / bpm
    phase, median_err = _assign_phase(times, period)
    now = float(getattr(world, "t", times[-1] if times else 0.0))
    rel = now - phase
    next_beat = phase + math.ceil(rel / period - 1e-9) * period
    if next_beat <= now + 1e-6:
        next_beat += period

    setattr(groove, "tempo_bpm", bpm)
    setattr(groove, "tempo_confidence", max(confidence, 0.75 if n_intervals >= 40 else confidence))
    setattr(groove, "beat_phase", float(((now - phase) / period) % 1.0))
    setattr(groove, "next_beat_time", float(next_beat))
    setattr(groove, "timing_deviation_ms", float(median_err * 1000.0))

    if n_intervals >= 4:
        diffs = np.diff(np.asarray(times, dtype=float))
        clean = diffs[(diffs >= 0.08) & (diffs <= 2.0)]
        if clean.size:
            cv = float(np.std(clean) / max(np.mean(clean), 1e-9))
            setattr(groove, "regularity", float(np.clip(1.0 - cv, 0.0, 1.0)))
            setattr(groove, "event_density", float(len(times) / max(now - times[0], 1e-6)))

    corrected = float(abs(old_bpm - bpm) > 1.0 or old_conf < confidence)
    return {"bpm": bpm, "confidence": float(getattr(groove, "tempo_confidence", confidence)),
            "intervals": float(n_intervals), "corrected": corrected}


def clean_foundation_patterns(world: Any) -> Dict[str, int]:
    patterns = getattr(world, "patterns", None)
    if patterns is None:
        return {"confirmed": 0, "rejected": 0}
    items = list(patterns.items()) if hasattr(patterns, "items") else []
    rejected = 0
    confirmed = 0
    kept = {}
    for pid, pat in items:
        count = int(getattr(pat, "occurrence_count", 0))
        first = float(getattr(pat, "first_seen", 0.0))
        last = float(getattr(pat, "last_seen", 0.0))
        period = int(getattr(pat, "period_bars", 0))
        if count >= 2 and period >= 1 and last >= first:
            kept[pid] = pat
            confirmed += 1
        else:
            rejected += 1
    try:
        patterns.clear()
        patterns.update(kept)
    except Exception:
        pass
    return {"confirmed": confirmed, "rejected": rejected}


def clean_foundation_sections(world: Any) -> Dict[str, int]:
    """Discard unclosed/too-short trailing section candidates.

    A section is not counted merely because a boundary event fired near the
    end of the audio. A completed section must span at least one musical bar;
    an ongoing tail must already contain one full bar of evidence.
    """
    sections = getattr(world, "sections", None)
    if sections is None:
        return {"kept": 0, "rejected": 0}

    bpm = float(getattr(getattr(world, "groove", None), "tempo_bpm", 0.0))
    if bpm <= 0.0:
        bpm = 120.0
    bar_period = 4.0 * (60.0 / bpm)
    now = float(getattr(world, "t", 0.0))

    kept = []
    rejected = 0
    confirmed_patterns = len(getattr(world, "patterns", {}) or {})

    for sec in list(sections):
        start = float(getattr(sec, "start_time", 0.0))
        end_value = getattr(sec, "end_time", None)
        end = now if end_value is None else float(end_value)
        duration = end - start
        if start < 0.0 or start > now + 1e-6 or duration < bar_period:
            rejected += 1
            continue

        reason = str(getattr(sec, "boundary_reason", ""))
        if "pattern_changed=True" in reason:
            if confirmed_patterns:
                reason = reason.replace("pattern_changed=True", "pattern_changed=True (confirmed repeated pattern evidence)")
            else:
                reason = reason.replace("pattern_changed=True", "pattern_changed=False (no confirmed repeated pattern evidence)")
            try:
                sec.boundary_reason = reason
            except Exception:
                pass
        kept.append(sec)

    try:
        sections[:] = kept
    except Exception:
        pass
    return {"kept": len(kept), "rejected": rejected}


def refresh_style(model: Any) -> None:
    world = getattr(model, "world", None)
    if world is not None:
        clean_foundation_sections(world)

    style = getattr(model, "style", None)
    if style is None or not hasattr(style, "update"):
        return
    if hasattr(style, "_last_update_t"):
        style._last_update_t = -1e9
    try:
        style.update(float(world.t))
    except Exception:
        pass
