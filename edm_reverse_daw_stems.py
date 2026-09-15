#!/usr/bin/env python3
"""EDM reverse-DAW stems-first runner.

Architecture:
    source mix -> Beat This! canonical beat/downbeat grid
               -> Demucs stems
               -> independent AWM analysis per stem
               -> stem-local objects/tracking/grouping/structure
               -> stem-local Essentia timbre evidence

The mix is used for beat tracking only. It is not passed through the AWM
physical/object/EDM analysis pipeline. Every Demucs stem gets its own
WorldState, physical analysis, object tracking, masking, relationships,
grouping, rhythm context, timbre, roles, elements, patterns, sections, and
arrangement. All stem worlds share the source-track Beat This! clock.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import edm_reverse_daw_core as _core
from edm_quality_control import clean_foundation_patterns, correct_groove, refresh_style
from edm_stems_first_pipeline import (
    align_stems_to_beats,
    attach_pipeline_state,
    install_canonical_beat_grid,
    prepare_stems_first,
    track_source_beats,
)

try:
    from auditory_world_model_essentia import EssentiaTimbreAdapter, HAVE_ESSENTIA
except Exception:
    EssentiaTimbreAdapter = None
    HAVE_ESSENTIA = False

logging.getLogger("ObjectTracker").setLevel(logging.WARNING)


def _detect_transients_fixed(flux: np.ndarray, hop: int, sr: int):
    """Multi-scale adaptive onset detector used by every stem-local AWM."""
    x = np.log1p(np.maximum(np.asarray(flux, dtype=np.float64), 0.0))
    n = x.size
    onset_strength = np.zeros(n, dtype=np.float32)
    is_transient = np.zeros(n, dtype=bool)
    if n == 0:
        return onset_strength, is_transient

    short_history = max(12, int(round(0.28 * sr / max(hop, 1))))
    long_history = max(short_history + 4, int(round(0.90 * sr / max(hop, 1))))
    refractory = max(1, int(round(0.008 * sr / max(hop, 1))))
    last_trigger = -refractory
    for i in range(n):
        lo_s = max(0, i - short_history)
        lo_l = max(0, i - long_history)
        hs = x[lo_s:i]
        hl = x[lo_l:i]

        def robust_z(hist):
            if hist.size >= 6:
                med = float(np.median(hist))
                mad = float(np.median(np.abs(hist - med)))
                scale = max(1.4826 * mad, 0.003)
                return (float(x[i]) - med) / scale
            if hist.size >= 2:
                return (float(x[i]) - float(np.mean(hist))) / max(float(np.std(hist)), 0.003)
            return 0.0

        z = max(robust_z(hs), robust_z(hl))
        onset_strength[i] = np.float32(z)
        left = max(0, i - 2)
        right = min(n, i + 3)
        local_peak = x[i] >= np.max(x[left:right]) - 1e-12
        previous = float(np.max(x[max(0, i - 3):i])) if i > 0 else float(x[i])
        local_rise = float(x[i]) - previous
        strong_enough = z >= 1.35 or (z >= 1.05 and local_rise >= 0.025)
        if i >= 6 and strong_enough and local_peak and (i - last_trigger) >= refractory:
            is_transient[i] = True
            last_trigger = i
    return onset_strength, is_transient


_core.awm.PhysicalAnalyzer.detect_transients = staticmethod(_detect_transients_fixed)


def _print_structure_state_fixed(model):
    world = model.world
    candidates = list(world.patterns.values())
    patterns = [
        p for p in candidates
        if int(getattr(p, "occurrence_count", 0)) >= 2
        and int(getattr(p, "period_bars", 0)) >= 1
        and float(getattr(p, "last_seen", 0.0)) >= float(getattr(p, "first_seen", 0.0))
    ]
    print("\n" + "=" * 78)
    print("STRUCTURE (stem-local patterns / sections)")
    print("=" * 78)
    print(f"Patterns detected: {len(patterns)}")
    if len(patterns) < len(candidates):
        print(f"One-off structural candidates rejected as patterns: {len(candidates) - len(patterns)}")
    tempo = float(getattr(world.groove, "tempo_bpm", 0.0)) or 120.0
    bar_period = 4.0 * 60.0 / tempo
    for p in sorted(patterns, key=lambda x: (float(x.first_seen), str(x.pattern_id))):
        duration = max(0.0, int(p.period_bars) * bar_period)
        end = max(float(p.last_seen) + duration, float(p.first_seen) + duration)
        print(
            f"  {p.pattern_id} period={p.period_bars} bar(s) status={p.status:<7} "
            f"confidence={p.confidence:.2f} occurrences={p.occurrence_count} "
            f"[{p.first_seen:.2f}s - {end:.2f}s] duration={end - p.first_seen:.2f}s"
        )
    print(f"\nSections: {len(world.sections)}")
    for sec in world.sections:
        end = f"{sec.end_time:.2f}s" if sec.end_time is not None else "(ongoing)"
        print(
            f"  {sec.section_id} [{sec.start_time:.2f}s - {end}] "
            f"novelty={sec.novelty_score:.2f} reason: {sec.boundary_reason}"
        )


def analyze_one_stem(stem_name, stem_audio, sample_rate, beat_grid, args):
    """Run the complete perception stack on exactly one separated stem."""
    params = _core.awm.Parameters()
    model = _core.awm.AuditoryWorldModel(params)
    if beat_grid is not None:
        install_canonical_beat_grid(model, beat_grid)

    x = np.asarray(stem_audio, dtype=np.float32).reshape(-1)
    print(f"\n{'#' * 96}\nSTEM: {stem_name}\n{'#' * 96}")
    print(f"[AWM/{stem_name}] physical + objects + tracking: RUNNING")
    model.run(x, use_stem_separation=False)
    print(f"[AWM/{stem_name}] physical + objects + tracking: DONE")

    quality = clean_foundation_patterns(model.world)
    if beat_grid is None:
        groove = correct_groove(model.world)
        print(
            f"[AWM/{stem_name}] tempo_source=stem transient consensus "
            f"bpm={groove['bpm']:.2f} confidence={groove['confidence']:.2f}"
        )
    else:
        groove = {
            "bpm": beat_grid.tempo_bpm,
            "confidence": beat_grid.confidence,
            "intervals": float(max(len(beat_grid.beats) - 1, 0)),
        }
        print(
            f"[AWM/{stem_name}] tempo_source=beat_this_shared "
            f"bpm={beat_grid.tempo_bpm:.2f} grid_confidence={beat_grid.confidence:.2f}"
        )

    refresh_style(model)
    model.print_structure_state = lambda: _print_structure_state_fixed(model)

    essentia_features = []
    if HAVE_ESSENTIA and EssentiaTimbreAdapter is not None:
        print(f"[AWM/{stem_name}] Essentia timbre: RUNNING")
        essentia = EssentiaTimbreAdapter(sample_rate=sample_rate)
        essentia_features = essentia.analyze(x, hop_size=2048)
        attached = essentia.attach_to_world(model.world, essentia_features)
        print(f"[AWM/{stem_name}] Essentia timbre: DONE frames={len(essentia_features)} objects_enriched={attached}")
    else:
        print(f"[AWM/{stem_name}] Essentia timbre: UNAVAILABLE; native AWM timbre retained")

    stem_grid_features = {}
    if beat_grid is not None:
        stem_grid_features = align_stems_to_beats({stem_name: x}, beat_grid, sample_rate)
        attach_pipeline_state(model, beat_grid, stem_grid_features)

    edm = _core.EDMReverseDAW(model.world)
    snapshot = edm.run()

    stem_json = None
    if args.export_json:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem_name)
        stem_path = Path(args.export_json).with_suffix("")
        per_stem_path = stem_path.parent / f"{stem_path.name}_{safe_name}.json"
        edm.export_json(per_stem_path)
        stem_json = str(per_stem_path)

    if args.plot_dir:
        plot_path = Path(args.plot_dir) / f"auditory_world_model_{stem_name}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        model.plot_activity(save_path=str(plot_path))
        print(f"[AWM/{stem_name}] activity plot={plot_path}")

    print(
        f"[AWM/{stem_name}] objects={len(snapshot['object_metadata'])} "
        f"elements={len(snapshot['elements'])} patterns={len(snapshot['patterns'])} "
        f"sections={len(snapshot['sections'])}"
    )
    if snapshot["arrangement"]:
        print(
            f"[AWM/{stem_name}] arrangement=yes "
            f"confidence={snapshot['arrangement']['confidence']:.3f}"
        )
    else:
        print(f"[AWM/{stem_name}] arrangement=not enough structural evidence")

    return {
        "snapshot": snapshot,
        "beat_aligned_features": stem_grid_features.get(stem_name, {}),
        "stem_export": stem_json,
        "essentia_frames": len(essentia_features),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete EDM reverse-DAW pipeline independently on every Demucs stem."
    )
    parser.add_argument("audio", nargs="?", default=None)
    parser.add_argument("--export-json", default=None)
    parser.add_argument("--plot-dir", default="stem_activity")
    args = parser.parse_args()

    print("\n" + "=" * 96)
    print("AUDITORY WORLD MODEL — FULL STEMS-FIRST PIPELINE")
    print("=" * 96)

    params = _core.awm.Parameters()
    source = _core.awm.AudioSource(params)
    audio, _ = source.load(args.audio)
    duration = len(audio) / float(params.sample_rate)
    print(f"source duration={duration:.3f}s sample_rate={params.sample_rate}Hz")

    print("\n[1/3] Beat This! on FULL SOURCE MIX")
    beat_grid = None
    try:
        beat_grid = track_source_beats(audio, params.sample_rate)
        print(
            f"  beats={len(beat_grid.beats)} downbeats={len(beat_grid.downbeats)} "
            f"tempo={beat_grid.tempo_bpm:.3f} BPM confidence={beat_grid.confidence:.2f}"
        )
    except Exception as exc:
        print(f"  Beat This! failed: {exc}")
        print("  stem-local AWM RhythmEngine will provide fallback metrical state")

    print("\n[2/3] Demucs source separation")
    separator_holder = _core.awm.AuditoryWorldModel(params)
    stems = prepare_stems_first(separator_holder, audio, params.sample_rate)
    if not stems:
        raise RuntimeError("No Demucs stems available. Install with: !pip install demucs")
    print(f"  stems={', '.join(sorted(stems))}")
    print("  source mix will NOT enter the AWM physical/object pipeline")

    print("\n[3/3] COMPLETE PER-STEM PERCEPTION")
    results = {}
    for stem_name in sorted(stems):
        results[stem_name] = analyze_one_stem(
            stem_name, stems[stem_name], params.sample_rate, beat_grid, args
        )

    result = {
        "version": "stems-first-1",
        "architecture": {
            "beat_source": "full_source_mix",
            "beat_tracker": "Beat This!",
            "analysis_source": "Demucs stems",
            "mix_awmpipeline": False,
            "independent_stem_worlds": True,
            "shared_beat_grid": beat_grid.as_dict() if beat_grid is not None else None,
        },
        "source": {"duration_s": duration, "sample_rate": params.sample_rate},
        "stems": {
            name: {
                "snapshot": data["snapshot"],
                "beat_aligned_features": data["beat_aligned_features"],
                "essentia_frames": data["essentia_frames"],
                "stem_export": data["stem_export"],
            }
            for name, data in results.items()
        },
    }

    if args.export_json:
        out = Path(args.export_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\ncombined export={out}")

    print("\n" + "=" * 96)
    print("STEMS-FIRST ANALYSIS COMPLETE")
    print("=" * 96)
    print(f"beat_source={'Beat This! full mix' if beat_grid is not None else 'stem-local fallback'}")
    print(f"stems_analyzed={len(results)}")
    for name, data in results.items():
        snap = data["snapshot"]
        print(
            f"  {name:<8} objects={len(snap['object_metadata']):>4} "
            f"elements={len(snap['elements']):>3} patterns={len(snap['patterns']):>3} "
            f"sections={len(snap['sections']):>3}"
        )

    return results


if __name__ == "__main__":
    main()
