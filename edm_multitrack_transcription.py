#!/usr/bin/env python3
"""Multi-track transcription ensemble for the EDM "other" stem.

Basic Pitch is intentionally not used here.  The detector combines:
- YourMT3 (via MT3-Infer): multi-instrument / multi-track transcription
- MuScriptor: multi-instrument transcription with explicit instrument groups

Both outputs are normalized into a common note-event representation, then
deduplicated by pitch/onset evidence.  The downstream EDM musical-layer
analysis works on the resulting polyphonic event cloud.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import os
import tempfile

import numpy as np
import soundfile as sf


_YOURMT3_MODEL = None
_MUSCRIPTOR_MODEL = None


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _write_temp_wav(audio: np.ndarray, sample_rate: int) -> str:
    fd, path = tempfile.mkstemp(prefix="edm_mt_", suffix=".wav")
    os.close(fd)
    sf.write(path, np.asarray(audio, dtype=np.float32), int(sample_rate))
    return path


def _load_yourmt3():
    global _YOURMT3_MODEL
    if _YOURMT3_MODEL is None:
        from mt3_infer import load_model
        _YOURMT3_MODEL = load_model(
            "yourmt3",
            device=_device(),
            cache=True,
            auto_download=True,
        )
    return _YOURMT3_MODEL


def _load_muscriptor():
    global _MUSCRIPTOR_MODEL
    if _MUSCRIPTOR_MODEL is None:
        from muscriptor.transcription_model import TranscriptionModel
        _MUSCRIPTOR_MODEL = TranscriptionModel.load_model(device=_device())
    return _MUSCRIPTOR_MODEL


def _instrument_name(program: int, is_drum: bool = False, name: str = "") -> str:
    if is_drum:
        return "drums"
    name = str(name or "").strip()
    if name:
        return name
    try:
        import pretty_midi
        return str(pretty_midi.program_to_instrument_name(int(program)))
    except Exception:
        return f"program_{int(program)}"


def _parse_yourmt3_midi(midi: Any) -> List[Dict[str, Any]]:
    """Parse MT3-Infer's mido.MidiFile into normalized note events."""
    path = _write_temp_midi(midi)
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(path)
        events: List[Dict[str, Any]] = []
        for track_index, inst in enumerate(pm.instruments):
            instrument = _instrument_name(inst.program, inst.is_drum, inst.name)
            for note in inst.notes:
                start = float(note.start)
                end = float(note.end)
                if end <= start:
                    continue
                events.append(
                    {
                        "start_time_s": start,
                        "end_time_s": end,
                        "pitch_midi": int(note.pitch),
                        "pitch_hz": float(440.0 * 2.0 ** ((int(note.pitch) - 69) / 12.0)),
                        "confidence": float(np.clip(note.velocity / 127.0, 0.0, 1.0)),
                        "velocity": float(np.clip(note.velocity / 127.0, 0.0, 1.0)),
                        "detector": "yourmt3",
                        "source_model": "YourMT3+",
                        "instrument": instrument,
                        "track_id": f"yourmt3_track_{track_index}",
                    }
                )
        return events
    finally:
        try:
            Path(path).unlink()
        except Exception:
            pass


def _write_temp_midi(midi: Any) -> str:
    fd, path = tempfile.mkstemp(prefix="edm_yourmt3_", suffix=".mid")
    os.close(fd)
    midi.save(path)
    return path


def transcribe_yourmt3(audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
    model = _load_yourmt3()
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    midi = model.transcribe(x, sr=int(sample_rate))
    return _parse_yourmt3_midi(midi)


def transcribe_muscriptor(audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
    model = _load_muscriptor()
    path = _write_temp_wav(audio, sample_rate)
    try:
        from muscriptor.events import NoteEndEvent, NoteStartEvent

        active: Dict[int, Dict[str, Any]] = {}
        events: List[Dict[str, Any]] = []
        for ev in model.transcribe(
            path,
            use_sampling=False,
            temperature=1.0,
            cfg_coef=1.0,
            instruments=None,
            batch_size=1,
            no_eos_is_ok=True,
            beam_size=1,
            prelude_forcing=True,
        ):
            if isinstance(ev, NoteStartEvent):
                active[int(ev.index)] = {
                    "start_time_s": float(ev.start_time),
                    "pitch_midi": int(ev.pitch),
                    "instrument": str(ev.instrument),
                    "detector": "muscriptor",
                    "source_model": "MuScriptor",
                    "track_id": f"muscriptor_{ev.instrument}",
                    "confidence": 0.70,
                    "velocity": 0.70,
                }
            elif isinstance(ev, NoteEndEvent):
                idx = int(ev.start_event_index)
                row = active.pop(idx, None)
                if row is None:
                    continue
                end = float(ev.end_time)
                if end <= float(row["start_time_s"]):
                    continue
                row["end_time_s"] = end
                pitch = int(row["pitch_midi"])
                row["pitch_hz"] = float(440.0 * 2.0 ** ((pitch - 69) / 12.0))
                events.append(row)
        return events
    finally:
        try:
            Path(path).unlink()
        except Exception:
            pass


def _cluster_key(event: Dict[str, Any]) -> Tuple[int, int]:
    return int(event["pitch_midi"]), int(round(float(event["start_time_s"]) * 20.0))


def _ensemble_events(
    backend_events: Sequence[Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Fuse the two model outputs without imposing a note-count template."""
    all_events: List[Dict[str, Any]] = []
    for events in backend_events:
        all_events.extend(dict(e) for e in events)

    # Ignore model-generated drum tracks in melodic analysis, but retain them
    # in backend diagnostics.  The "other" stem is being analyzed for pitched
    # musical layers here.
    pitched = [e for e in all_events if str(e.get("instrument", "")).lower() != "drums"]
    pitched.sort(key=lambda e: (float(e["start_time_s"]), int(e["pitch_midi"]), float(e["end_time_s"])))

    clusters: List[List[Dict[str, Any]]] = []
    for ev in pitched:
        best_idx = None
        best_distance = 1e9
        for i, cluster in enumerate(clusters):
            rep = cluster[0]
            if int(rep["pitch_midi"]) != int(ev["pitch_midi"]):
                continue
            onset_distance = abs(float(rep["start_time_s"]) - float(ev["start_time_s"]))
            end_distance = abs(float(rep["end_time_s"]) - float(ev["end_time_s"]))
            overlap = min(float(rep["end_time_s"]), float(ev["end_time_s"])) - max(
                float(rep["start_time_s"]), float(ev["start_time_s"])
            )
            if onset_distance <= 0.075 and (overlap >= -0.04 or end_distance <= 0.12):
                score = onset_distance + 0.20 * end_distance
                if score < best_distance:
                    best_idx = i
                    best_distance = score
        if best_idx is None:
            clusters.append([ev])
        else:
            clusters[best_idx].append(ev)

    fused: List[Dict[str, Any]] = []
    for cluster in clusters:
        start = float(np.average([e["start_time_s"] for e in cluster]))
        end = float(np.average([e["end_time_s"] for e in cluster]))
        source_models = sorted(set(str(e.get("source_model", "")) for e in cluster))
        instruments = sorted(set(str(e.get("instrument", "")) for e in cluster))
        velocities = [float(e.get("velocity", e.get("confidence", 0.7))) for e in cluster]
        pitch = int(cluster[0]["pitch_midi"])
        agreement = len(source_models)
        fused.append(
            {
                "start_time_s": start,
                "end_time_s": max(end, start + 0.01),
                "pitch_midi": pitch,
                "pitch_hz": float(440.0 * 2.0 ** ((pitch - 69) / 12.0)),
                "confidence": float(0.70 + 0.15 * min(agreement - 1, 1)),
                "velocity": float(np.clip(np.mean(velocities), 0.0, 1.0)),
                "detector": "yourmt3_plus_muscriptor_ensemble",
                "source_models": source_models,
                "model_agreement": agreement,
                "instruments": instruments,
                "instrument": instruments[0] if instruments else "unknown",
                "track_ids": sorted(set(str(e.get("track_id", "")) for e in cluster)),
            }
        )
    fused.sort(key=lambda e: (float(e["start_time_s"]), int(e["pitch_midi"]), float(e["end_time_s"])))
    return fused


def _summary(events: Sequence[Dict[str, Any]], backend: str, error: str | None = None) -> Dict[str, Any]:
    instruments = Counter(str(e.get("instrument", "unknown")) for e in events)
    return {
        "backend": backend,
        "success": error is None,
        "note_count": int(len(events)),
        "instrument_counts": dict(sorted(instruments.items())),
        **({"error": error} if error else {}),
    }


def transcribe_other_multitrack(
    audio: np.ndarray,
    sample_rate: int,
    beat_times: np.ndarray,
) -> Dict[str, Any]:
    """Run both multi-track models and return an ensemble note representation.

    No Basic Pitch fallback exists by design.  If either model fails, its error
    is exposed explicitly.  The default is strict: both model runs are needed
    for the ensemble to be considered complete.  Set
    EDM_MT_ALLOW_PARTIAL=1 only for debugging when one backend is unavailable.
    """
    backend_results: List[Tuple[str, List[Dict[str, Any]], str | None]] = []

    for backend_name, fn in (
        ("YourMT3+", transcribe_yourmt3),
        ("MuScriptor", transcribe_muscriptor),
    ):
        try:
            events = fn(np.asarray(audio, dtype=np.float32), int(sample_rate))
            backend_results.append((backend_name, events, None))
        except Exception as exc:
            backend_results.append((backend_name, [], repr(exc)))

    failures = [name for name, _, err in backend_results if err]
    if failures and os.getenv("EDM_MT_ALLOW_PARTIAL", "0") != "1":
        joined = "; ".join(f"{name}: {err}" for name, _, err in backend_results if err)
        raise RuntimeError(
            "Multi-track transcription failed for "
            + ", ".join(failures)
            + ". No Basic Pitch fallback is enabled. "
            + joined
            + ". For MuScriptor, authenticate to Hugging Face and accept its model license."
        )

    events_by_backend = [events for _, events, err in backend_results if err is None]
    if not events_by_backend:
        raise RuntimeError("Neither multi-track transcription backend returned notes.")

    ensemble_events = _ensemble_events(events_by_backend)

    return {
        "detector": "yourmt3_plus_muscriptor_ensemble",
        "voicing_gate": "multi-track model decoding; note evidence from independent transcription backends",
        "polyphonic": True,
        "note_events": ensemble_events,
        "all_note_events": ensemble_events,
        "raw_frame_count": 0,
        "raw_voiced_frames": int(sum(len(events) for _, events, err in backend_results)),
        "transcription_backends": [
            _summary(events, name, err) for name, events, err in backend_results
        ],
        "backend_note_events": {
            name: events for name, events, err in backend_results if err is None
        },
        "model_agreement_summary": {
            "fused_notes": int(len(ensemble_events)),
            "notes_supported_by_both_models": int(
                sum(1 for e in ensemble_events if int(e.get("model_agreement", 0)) >= 2)
            ),
        },
    }


__all__ = ["transcribe_other_multitrack"]
