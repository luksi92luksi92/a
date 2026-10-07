#!/usr/bin/env python3
"""Role-specific melody and note extraction for the stems-first EDM pipeline.

The detectors are deliberately different by stem role:

- bass   -> CREPE F0 tracking + voiced confidence + semitone stabilization
- other  -> YourMT3+ + MuScriptor multitrack transcription ensemble
- vocals -> pYIN monophonic F0 tracking
- drums  -> melody disabled

This module never uses an amplitude-only "is this loud enough?" gate to decide
whether a pitch is musical. Audio activity is useful elsewhere, but melody
acceptance is based on pitch evidence (voicing / periodicity / note activation).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from edm_musical_layers import analyze_other_music


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def hz_to_midi(hz: np.ndarray | float) -> np.ndarray:
    x = np.asarray(hz, dtype=float)
    return 69.0 + 12.0 * np.log2(np.maximum(x, 1e-9) / 440.0)


def midi_to_hz(midi: np.ndarray | float) -> np.ndarray:
    x = np.asarray(midi, dtype=float)
    return 440.0 * np.power(2.0, (x - 69.0) / 12.0)


def midi_name(midi: int) -> str:
    midi = int(midi)
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def _median_filter(values: np.ndarray, size: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 2 or size <= 1:
        return values.copy()
    try:
        from scipy.ndimage import median_filter
        return median_filter(values, size=int(size), mode="nearest")
    except Exception:
        out = values.copy()
        radius = int(size) // 2
        for i in range(len(values)):
            a = max(0, i - radius)
            b = min(len(values), i + radius + 1)
            out[i] = float(np.median(values[a:b]))
        return out


def _resample_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if int(sample_rate) == 16000:
        return x
    import librosa
    return np.asarray(
        librosa.resample(x, orig_sr=int(sample_rate), target_sr=16000),
        dtype=np.float32,
    )


def _note_events_to_beat_track(
    beat_times: np.ndarray,
    note_events: Sequence[Dict[str, Any]],
    default_zero: float = 0.0,
) -> np.ndarray:
    """Project note events to one primary note per beat for the legacy plot.

    The complete polyphonic note list remains available separately. This
    projection is only a readable single-line summary.
    """
    beats = np.asarray(beat_times, dtype=float)
    out = np.full(len(beats), float(default_zero), dtype=float)
    for i, t in enumerate(beats):
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        for ev in note_events:
            start = float(ev["start_time_s"])
            end = float(ev["end_time_s"])
            if start - 0.02 <= float(t) <= end + 0.02:
                overlap = max(
                    0.0,
                    min(end, float(t) + 0.05) - max(start, float(t) - 0.05),
                )
                confidence = float(ev.get("confidence", ev.get("velocity", 0.0)))
                pitch = float(ev["pitch_midi"])
                # Mild upper-register preference helps select a lead candidate
                # when several simultaneous notes are present, without erasing
                # the complete polyphonic note-event output.
                score = confidence * (0.5 + min(1.0, max(overlap, 0.05) / 0.10)) * (
                    1.0 + 0.03 * np.clip((pitch - 48.0) / 24.0, -1.0, 2.0)
                )
                candidates.append((float(score), ev))
        if candidates:
            _, best = max(candidates, key=lambda x: x[0])
            out[i] = float(best["pitch_midi"])
    return out


def _stabilize_midi_sequence(
    midi: np.ndarray,
    min_note_change_beats: int = 1,
) -> np.ndarray:
    """Quantize to semitones and remove isolated one-sample jumps."""
    x = np.asarray(midi, dtype=float).copy()
    voiced = x > 0.0
    if not np.any(voiced):
        return np.zeros_like(x)
    q = np.zeros_like(x)
    q[voiced] = np.round(x[voiced])

    # Remove isolated single-beat pitch spikes when both neighbors agree.
    for i in range(1, len(q) - 1):
        if q[i] > 0 and q[i - 1] > 0 and q[i + 1] > 0 and q[i - 1] == q[i + 1] and q[i] != q[i - 1]:
            q[i] = q[i - 1]

    return q


def _make_run_events(
    times: np.ndarray,
    midi: np.ndarray,
    confidence: np.ndarray,
    min_duration_s: float = 0.06,
) -> List[Dict[str, Any]]:
    """Turn a quantized framewise F0 into discrete note events."""
    t = np.asarray(times, dtype=float)
    m = np.asarray(midi, dtype=float)
    c = np.asarray(confidence, dtype=float)
    events: List[Dict[str, Any]] = []
    if len(t) == 0:
        return events

    start = None
    current = None
    confs: List[float] = []

    def flush(end_index: int) -> None:
        nonlocal start, current, confs
        if start is None or current is None:
            return
        end_t = float(t[min(end_index, len(t) - 1)])
        start_t = float(t[start])
        if end_t - start_t >= min_duration_s:
            events.append(
                {
                    "start_time_s": start_t,
                    "end_time_s": end_t,
                    "pitch_midi": int(current),
                    "pitch_hz": float(midi_to_hz(current)),
                    "confidence": float(np.median(confs)) if confs else 0.0,
                }
            )
        start = None
        current = None
        confs = []

    for i, value in enumerate(m):
        if value <= 0:
            flush(i)
            continue
        iv = int(round(value))
        if current is None:
            start = i
            current = iv
            confs = [float(c[i])]
        elif iv != current:
            flush(i)
            start = i
            current = iv
            confs = [float(c[i])]
        else:
            confs.append(float(c[i]))
    flush(len(m) - 1)
    return events


def _harmonic_support(
    audio: np.ndarray,
    sample_rate: int,
    start_time_s: float,
    end_time_s: float,
    f0_hz: float,
) -> float:
    """Estimate whether a candidate bass F0 is supported by its harmonics."""
    if f0_hz <= 0:
        return 0.0
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0

    center = int(round(0.5 * (start_time_s + end_time_s) * sample_rate))
    frame = 4096
    half = frame // 2
    a = max(0, center - half)
    b = min(len(x), center + half)
    seg = x[a:b]
    if seg.size < 256:
        return 0.0
    win = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * win))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sample_rate)

    total_band = float(np.sum(spec[(freqs >= 35.0) & (freqs <= 1200.0)]) + 1e-9)
    support = 0.0
    weights = 0.0
    for harmonic in range(1, 9):
        freq = float(f0_hz * harmonic)
        if freq < 35.0 or freq > 1200.0:
            continue
        width = max(2.0, 0.02 * freq)
        idx = np.abs(freqs - freq) <= width
        if not np.any(idx):
            continue
        local = float(np.max(spec[idx]))
        weight = 1.0 / harmonic
        support += weight * local
        weights += weight
    if weights <= 0:
        return 0.0
    # Normalize by total low/mid spectral mass so isolated unrelated energy
    # does not look like a supported bass fundamental.
    return float(np.clip((support / weights) / max(total_band, 1e-9) * 16.0, 0.0, 1.0))


def detect_bass_crepe(
    audio: np.ndarray,
    sample_rate: int,
    beat_times: np.ndarray,
) -> Dict[str, Any]:
    """Detect bass melody using CREPE F0 + periodicity + harmonic validation."""
    import torch
    import torchcrepe

    x16 = _resample_16k(audio, sample_rate)
    tensor = torch.from_numpy(x16).float().unsqueeze(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hop_length = 160  # 10 ms at 16 kHz

    pitch, periodicity = torchcrepe.predict(
        tensor,
        16000,
        hop_length,
        40.0,
        300.0,
        "tiny",
        batch_size=2048,
        device=device,
        return_periodicity=True,
        decoder=torchcrepe.decode.viterbi,
    )
    pitch = np.asarray(pitch.detach().cpu().numpy(), dtype=float).squeeze()
    periodicity = np.asarray(periodicity.detach().cpu().numpy(), dtype=float).squeeze()
    periodicity = _median_filter(periodicity, 5)
    times = np.arange(len(pitch), dtype=float) * 0.010

    valid = (pitch > 40.0) & (pitch < 300.0) & (periodicity >= 0.20)
    frame_midi = np.zeros_like(pitch)
    frame_midi[valid] = hz_to_midi(pitch[valid])
    frame_midi[valid] = np.round(frame_midi[valid])

    # Remove very short pitch flickers at frame level.
    frame_midi = _stabilize_midi_sequence(frame_midi)
    frame_events = _make_run_events(
        times,
        frame_midi,
        periodicity,
        min_duration_s=0.06,
    )

    # Harmonic confirmation at the event level.
    confirmed: List[Dict[str, Any]] = []
    for ev in frame_events:
        f0 = float(ev["pitch_hz"])
        support = _harmonic_support(
            np.asarray(audio, dtype=np.float32),
            int(sample_rate),
            float(ev["start_time_s"]),
            float(ev["end_time_s"]),
            f0,
        )
        ev = dict(ev)
        ev["harmonic_support"] = support
        ev["detector"] = "crepe"
        # CREPE periodicity is already the main pitch-confidence signal.
        # Harmonic support is deliberately a soft score, not an amplitude gate.
        if float(ev["confidence"]) >= 0.20 and support >= 0.05:
            confirmed.append(ev)

    # Project confirmed note events to beat positions.
    beat_midi = _note_events_to_beat_track(np.asarray(beat_times), confirmed)
    beat_midi = _stabilize_midi_sequence(beat_midi)

    return {
        "detector": "crepe_tiny_viterbi",
        "voicing_gate": "periodicity>=0.20; harmonic_support>=0.05",
        "beat_midi": beat_midi,
        "note_events": confirmed,
        "raw_frame_count": int(len(pitch)),
        "raw_voiced_frames": int(np.sum(valid)),
        "device": device,
        "sample_rate": 16000,
        "fmin_hz": 40.0,
        "fmax_hz": 300.0,
    }


def detect_vocal_pyin(
    audio: np.ndarray,
    sample_rate: int,
    beat_times: np.ndarray,
) -> Dict[str, Any]:
    import librosa

    f0, voiced_flag, voiced_prob = librosa.pyin(
        np.asarray(audio, dtype=np.float32),
        fmin=80.0,
        fmax=min(1200.0, sample_rate * 0.45),
        sr=int(sample_rate),
        frame_length=2048,
        hop_length=256,
        fill_na=np.nan,
    )
    frame_times = librosa.times_like(f0, sr=int(sample_rate), hop_length=256)
    valid = np.isfinite(f0) & (voiced_flag.astype(bool)) & (np.asarray(voiced_prob) >= 0.50)
    midi = np.zeros(len(f0), dtype=float)
    midi[valid] = np.round(hz_to_midi(f0[valid]))
    events = _make_run_events(
        frame_times,
        midi,
        np.nan_to_num(np.asarray(voiced_prob), nan=0.0),
        min_duration_s=0.06,
    )
    beat_midi = _stabilize_midi_sequence(
        _note_events_to_beat_track(np.asarray(beat_times), [
            {**ev, "detector": "pyin"} for ev in events
        ])
    )
    return {
        "detector": "librosa_pyin",
        "voicing_gate": "voiced_probability>=0.50",
        "beat_midi": beat_midi,
        "note_events": [{**ev, "detector": "pyin"} for ev in events],
        "raw_frame_count": int(len(f0)),
        "raw_voiced_frames": int(np.sum(valid)),
    }


def detect_other_multitrack(
    audio: np.ndarray,
    sample_rate: int,
    beat_times: np.ndarray,
) -> Dict[str, Any]:
    """Multi-instrument transcription for the residual/other stem.

    This intentionally uses YourMT3+ and MuScriptor rather than Basic Pitch.
    Their independent outputs are fused before EDM musical-layer inference.
    """
    from edm_multitrack_transcription import transcribe_other_multitrack

    result = transcribe_other_multitrack(
        np.asarray(audio, dtype=np.float32),
        int(sample_rate),
        np.asarray(beat_times, dtype=float),
    )

    events = list(result.get("note_events", []))
    try:
        music_layers = analyze_other_music(
            np.asarray(audio, dtype=np.float32),
            int(sample_rate),
            np.asarray(beat_times, dtype=float),
            events,
        )
    except Exception as exc:
        music_layers = {
            "analysis_version": "edm-musical-input-failed-soft",
            "error": repr(exc),
            "all_note_events": list(events),
            "lead_events": [],
            "chord_events": [],
            "counter_melody_events": [],
            "arpeggio_events": [],
            "pad_events": [],
            "stab_events": [],
            "one_shot_events": [],
            "motifs": [],
            "octave_equivalent_motifs": [],
            "call_response_events": [],
        }

    enriched_events = list(music_layers.get("all_note_events", events))
    lead_events = list(music_layers.get("lead_events", []))
    beat_midi = _stabilize_midi_sequence(
        _note_events_to_beat_track(
            np.asarray(beat_times),
            lead_events or enriched_events,
        )
    )

    return {
        **result,
        "detector": "yourmt3_plus_muscriptor_ensemble",
        "voicing_gate": "multi-track model decoding; no Basic Pitch fallback",
        "beat_midi": beat_midi,
        "note_events": enriched_events,
        "all_note_events": enriched_events,
        "raw_frame_count": 0,
        "raw_voiced_frames": int(result.get("raw_voiced_frames", len(enriched_events))),
        "polyphonic": True,
        "music_layers": {
            **music_layers,
            "transcription_backends": result.get("transcription_backends", []),
            "model_agreement_summary": result.get("model_agreement_summary", {}),
            "backend_note_events": result.get("backend_note_events", {}),
        },
    }

def detect_stem_melody(
    stem_name: str,
    audio: np.ndarray,
    sample_rate: int,
    beat_times: np.ndarray,
) -> Dict[str, Any]:
    role = str(stem_name).strip().lower()
    if role == "drums":
        return {
            "detector": "disabled_drums",
            "voicing_gate": "disabled",
            "beat_midi": np.zeros(len(beat_times), dtype=float),
            "note_events": [],
            "raw_frame_count": 0,
            "raw_voiced_frames": 0,
        }
    if role == "bass":
        return detect_bass_crepe(audio, sample_rate, beat_times)
    if role == "other":
        return detect_other_multitrack(audio, sample_rate, beat_times)
    if role == "vocals":
        return detect_vocal_pyin(audio, sample_rate, beat_times)
    # Unknown stems get a conservative monophonic fallback.
    return detect_vocal_pyin(audio, sample_rate, beat_times)


__all__ = [
    "detect_stem_melody",
    "detect_bass_crepe",
    "detect_other_multitrack",
    "detect_vocal_pyin",
]
