#!/usr/bin/env python3
"""Stems-first orchestration helpers for the EDM reverse-DAW branch.

Pipeline contract:
    source mix -> Beat This! beat/downbeat times -> Demucs stems
    -> AWM/object analysis using that shared beat grid -> stem-aligned features

Beat This! supplies the canonical metrical clock.  The AWM transient detector
remains useful for event/object evidence, but its RhythmEngine no longer
creates an independent beat grid when this adapter is installed.
"""
from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class BeatGrid:
    """Canonical beat/downbeat timestamps from the full source mix."""

    beats: np.ndarray
    downbeats: np.ndarray
    confidence: float
    source: str = "beat_this"

    @property
    def beat_period_s(self) -> float:
        if self.beats.size < 2:
            return 0.0
        return float(np.median(np.diff(self.beats)))

    @property
    def tempo_bpm(self) -> float:
        period = self.beat_period_s
        return float(60.0 / period) if period > 0.0 else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "confidence": float(self.confidence),
            "tempo_bpm": float(self.tempo_bpm),
            "beats": self.beats.tolist(),
            "downbeats": self.downbeats.tolist(),
        }


def track_source_beats(audio: np.ndarray, sample_rate: int) -> BeatGrid:
    """Run Beat This! directly on the loaded full-mix source waveform.

    The model performs its own preprocessing/resampling.  We deliberately use
    the minimal postprocessor rather than the optional madmom DBN so the
    branch does not add the old DBN dependency just to obtain beat times.
    """
    try:
        import torch
        from beat_this.inference import Audio2Beats
    except Exception as exc:  # pragma: no cover - exercised in Colab when absent
        raise RuntimeError(
            "Beat This! is unavailable. Install with: !pip install -r requirements-beatthis.txt"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker = Audio2Beats(
        checkpoint_path="final0",
        device=device,
        float16=(device == "cuda"),
        dbn=False,
    )
    beats, downbeats = tracker(np.asarray(audio, dtype=np.float32), int(sample_rate))

    beats = np.asarray(beats, dtype=np.float64).reshape(-1)
    downbeats = np.asarray(downbeats, dtype=np.float64).reshape(-1)
    beats = np.unique(beats[np.isfinite(beats)])
    downbeats = np.unique(downbeats[np.isfinite(downbeats)])

    if beats.size < 2:
        raise RuntimeError("Beat This! returned fewer than two beat positions; no stable beat grid available.")

    # This is a data-sufficiency confidence, not a claimed neural probability.
    confidence = float(np.clip(beats.size / 32.0, 0.0, 1.0))
    return BeatGrid(beats=beats, downbeats=downbeats, confidence=confidence)


def prepare_stems_first(model: Any, audio: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
    """Separate the source once before the expensive AWM run and cache stems.

    The existing StemSeparationEngine remains the authoritative separator;
    this helper only changes orchestration order and prevents a second
    Demucs pass inside AuditoryWorldModel.run().
    """
    engine = getattr(model, "stem_separation", None)
    if engine is None or not getattr(engine, "available", False):
        return {}
    stems = engine.separate(audio, sample_rate)
    if not stems:
        return {}

    normalized = {
        str(name): np.asarray(wav, dtype=np.float32).reshape(-1)
        for name, wav in stems.items()
        if np.asarray(wav).size
    }
    model._stems_first_cache = normalized
    # AuditoryWorldModel.run() asks the engine for stems after mix analysis.
    # Return the cached result there rather than invoking Demucs a second time.
    def cached_separate(_self, _audio, _sample_rate):
        return normalized
    engine.separate = types.MethodType(cached_separate, engine)
    return normalized


def _external_rhythm_update(self: Any, t: float) -> bool:
    grid: BeatGrid = getattr(self, "_external_beat_grid", None)
    if grid is None or grid.beats.size < 2:
        return False

    beats = grid.beats
    index = int(np.searchsorted(beats, float(t), side="right") - 1)
    index = int(np.clip(index, 0, len(beats) - 1))
    previous_beat = float(beats[index])
    if index + 1 < len(beats):
        next_beat = float(beats[index + 1])
    else:
        next_beat = previous_beat + grid.beat_period_s

    phase = 0.0 if next_beat <= previous_beat else float(
        np.clip((float(t) - previous_beat) / (next_beat - previous_beat), 0.0, 1.0)
    )

    # Event-level statistics still come from AWM transient evidence; only the
    # metrical clock itself comes from Beat This!.
    transient_times = np.asarray([
        float(e.time)
        for e in getattr(self.world, "events", [])
        if getattr(e, "event_type", "") == "transient_detected"
    ], dtype=np.float64)
    if transient_times.size:
        lookback = transient_times[transient_times >= float(t) - 8.0]
        if lookback.size:
            period = max(grid.beat_period_s, 1e-6)
            nearest = previous_beat + np.round((lookback - previous_beat) / period) * period
            timing_deviation_ms = float(np.median(np.abs(lookback - nearest)) * 1000.0)
            event_density = float(lookback.size / max(float(t) - float(lookback[0]), 1e-6))
            diffs = np.diff(lookback)
            regularity = float(np.clip(1.0 - np.std(diffs) / max(np.mean(diffs), 1e-9), 0.0, 1.0)) if diffs.size else 0.0
        else:
            timing_deviation_ms = 0.0
            event_density = 0.0
            regularity = 0.0
    else:
        timing_deviation_ms = 0.0
        event_density = 0.0
        regularity = 0.0

    old_groove = self.world.groove
    new_groove = self.world.__class__.__dict__.get("groove", None)
    _ = new_groove  # keeps this function independent of implementation lookup details

    # Import through the instance's module so the adapter works with the
    # repository's dynamically loaded AWM module.
    groove_type = type(old_groove)
    groove = groove_type(
        tempo_bpm=grid.tempo_bpm,
        tempo_confidence=max(float(grid.confidence), 0.5),
        beat_phase=phase,
        next_beat_time=next_beat,
        timing_deviation_ms=timing_deviation_ms,
        swing_ratio=0.5,
        syncopation_index=0.0,
        event_density=event_density,
        regularity=regularity,
        accent_positions=[],
        last_updated=float(t),
    )
    self.world.update_groove(groove, float(t))

    cursor = int(getattr(self, "_external_beat_cursor", -1))
    crossed = index > cursor
    if crossed:
        for beat_index in range(cursor + 1, index + 1):
            beat_t = float(beats[beat_index])
            self.world.emit_event(
                "beat",
                None,
                {"tempo_bpm": grid.tempo_bpm, "phase": 0.0, "source": grid.source},
                float(grid.confidence),
                beat_t,
            )
        self._external_beat_cursor = index

    return crossed


def install_canonical_beat_grid(model: Any, grid: BeatGrid) -> None:
    """Replace the model instance's internal metrical clock with Beat This!."""
    rhythm = model.rhythm
    rhythm._external_beat_grid = grid
    rhythm._external_beat_cursor = -1
    rhythm.observe_onset = types.MethodType(lambda _self, _t, _strength, _bass_weight=0.3: None, rhythm)
    rhythm.update = types.MethodType(_external_rhythm_update, rhythm)
    model._beat_grid = grid


def align_stems_to_beats(
    stems: Dict[str, np.ndarray],
    grid: BeatGrid,
    sample_rate: int,
    half_window_ms: float = 50.0,
) -> Dict[str, Dict[str, Any]]:
    """Compute lightweight beat-aligned per-stem energy features.

    These are non-destructive stem observations. They do not create semantic
    labels or replace the AWM object world. All stems use exactly the same
    source-track beat timestamps.
    """
    result: Dict[str, Dict[str, Any]] = {}
    half = max(1, int(round(float(sample_rate) * half_window_ms / 1000.0)))
    n = int(grid.beats.size)
    for name, waveform in stems.items():
        x = np.asarray(waveform, dtype=np.float32).reshape(-1)
        energies = np.zeros(n, dtype=np.float32)
        peaks = np.zeros(n, dtype=np.float32)
        for i, beat_time in enumerate(grid.beats):
            center = int(round(float(beat_time) * sample_rate))
            lo = max(0, center - half)
            hi = min(len(x), center + half)
            if hi > lo:
                seg = x[lo:hi]
                energies[i] = float(np.sqrt(np.mean(seg * seg) + 1e-12))
                peaks[i] = float(np.max(np.abs(seg)))
        result[name] = {
            "beat_times": grid.beats.tolist(),
            "rms": energies.tolist(),
            "peak": peaks.tolist(),
            "mean_rms": float(np.mean(energies)) if energies.size else 0.0,
            "max_rms": float(np.max(energies)) if energies.size else 0.0,
        }
    return result


def attach_pipeline_state(model: Any, grid: Optional[BeatGrid], stem_features: Dict[str, Dict[str, Any]]) -> None:
    """Keep branch-specific orchestration state available for export/debugging."""
    model._beat_grid = grid
    model._stem_beat_features = stem_features
