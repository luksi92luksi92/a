#!/usr/bin/env python3
"""EDM reverse-DAW entry point with stems-first orchestration.

This branch uses the full source mix for canonical Beat This! beat/downbeat
tracking, separates the source into Demucs stems before AWM processing, then
uses the same beat grid for the mix and every stem-aligned task. AWM transient
and physical analysis remain event/object evidence; the internal RhythmEngine
is bypassed when Beat This! is available so it cannot invent a competing beat
clock.
"""
from __future__ import annotations

import argparse
import logging
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
    """Multi-scale adaptive onset detector designed to recover weaker EDM hits."""
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


def _print_structure_state_fixed(self) -> None:
    """Report only genuine repeated foundation patterns with real durations."""
    print("\n" + "=" * 78)
    print("STRUCTURE (patterns / sections)")
    print("=" * 78)

    candidates = list(self.world.patterns.values())
    patterns = [
        pat for pat in candidates
        if int(getattr(pat, "occurrence_count", 0)) >= 2
        and float(getattr(pat, "last_seen", 0.0)) >= float(getattr(pat, "first_seen", 0.0))
        and int(getattr(pat, "period_bars", 0)) >= 1
    ]
    print(f"Patterns detected: {len(patterns)}")
    if len(patterns) < len(candidates):
        print(f"One-off structural candidates rejected as patterns: {len(candidates) - len(patterns)}")

    tempo = float(getattr(self.world.groove, "tempo_bpm", 0.0))
    if tempo <= 0.0:
        tempo = 120.0
    beat_period = 60.0 / tempo
    bar_period = 4.0 * beat_period

    for pat in sorted(patterns, key=lambda p: (float(p.first_seen), str(p.pattern_id))):
        first = float(pat.first_seen)
        last_occurrence_start = float(pat.last_seen)
        pattern_duration = max(0.0, int(pat.period_bars) * bar_period)
        end = max(last_occurrence_start + pattern_duration, first + pattern_duration)
        print(
            f"  {pat.pattern_id}  period={pat.period_bars} bar(s)  "
            f"status={pat.status:<7}  confidence={pat.confidence:.2f}  "
            f"occurrences={pat.occurrence_count}  "
            f"[{first:.2f}s - {end:.2f}s]  duration={end - first:.2f}s"
        )

    print(f"\nSections: {len(self.world.sections)}")
    for sec in self.world.sections:
        end = f"{sec.end_time:.2f}s" if sec.end_time is not None else "(ongoing)"
        print(f"  {sec.section_id}  [{sec.start_time:.2f}s - {end}]  novelty={sec.novelty_score:.2f}  "
              f"reason: {sec.boundary_reason}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the stems-first EDM reverse-DAW with Beat This! as the canonical beat clock."
    )
    parser.add_argument("audio", nargs="?", default=None)
    parser.add_argument("--export-json", default=None)
    parser.add_argument("--plot", default="auditory_world_model_activity.png")
    args = parser.parse_args()

    print("\n" + "=" * 96)
    print("AUDITORY WORLD MODEL — EDM REVERSE-DAW / STEMS-FIRST")
    print("=" * 96)
    print("[1/8] Loading source audio...")

    params = _core.awm.Parameters()
    source = _core.awm.AudioSource(params)
    audio, _ = source.load(args.audio)
    audio_duration = len(audio) / float(params.sample_rate)
    print(f"        duration={audio_duration:.3f}s  sample_rate={params.sample_rate}Hz")

    model = _core.awm.AuditoryWorldModel(params)

    print("[2/8] Beat tracking on full source mix: RUNNING")
    beat_grid = None
    try:
        beat_grid = track_source_beats(audio, params.sample_rate)
        print(
            f"        source={beat_grid.source} beats={len(beat_grid.beats)} "
            f"downbeats={len(beat_grid.downbeats)} tempo={beat_grid.tempo_bpm:.2f} BPM"
        )
        print("        canonical grid -> mix + all Demucs stems")
    except Exception as exc:
        print(f"[2/8] Beat This! unavailable/failed: {exc}")
        print("        fallback: AWM RhythmEngine + post-run transient consensus")

    print("[3/8] Demucs source separation: RUNNING")
    stems = prepare_stems_first(model, audio, params.sample_rate)
    if stems:
        print(f"        stems={', '.join(sorted(stems))}")
        print("        separation completed BEFORE AWM object processing; cached for reuse")
    else:
        stem_engine = getattr(model, "stem_separation", None)
        if stem_engine is not None and not stem_engine.available:
            print("[3/8] Demucs source separation: UNAVAILABLE")
            print("        install in Colab with: !pip install demucs")
        else:
            print("[3/8] Demucs source separation: FAILED/EMPTY; continuing on source mix")

    if beat_grid is not None:
        install_canonical_beat_grid(model, beat_grid)

    print("[4/8] Physical analysis + tracking: RUNNING")
    model.run(audio, use_stem_separation=bool(stems))
    print("[4/8] Physical analysis + SoundObject tracking: DONE")

    # Beat-aligned stem observations use exactly the source beat grid; no
    # independent stem beat tracker is allowed to redefine the clock.
    stem_features = {}
    if stems and beat_grid is not None:
        stem_features = align_stems_to_beats(stems, beat_grid, params.sample_rate)
        print(f"        beat-aligned stem tasks: {len(stem_features)} stems x {len(beat_grid.beats)} beats")
        for name in sorted(stem_features):
            stats = stem_features[name]
            print(
                f"        {name:<7} mean_rms={stats['mean_rms']:.5f} "
                f"max_rms={stats['max_rms']:.5f}"
            )
    attach_pipeline_state(model, beat_grid, stem_features)

    quality = clean_foundation_patterns(model.world)
    if beat_grid is None:
        groove = correct_groove(model.world)
        refresh_style(model)
        print(
            f"[5/8] Quality gates: patterns_kept={quality['confirmed']} "
            f"one_off_patterns_rejected={quality['rejected']}"
        )
        print(
            f"        tempo_source=AWM+transient_consensus bpm={groove['bpm']:.1f} "
            f"confidence={groove['confidence']:.2f} intervals={int(groove['intervals'])}"
        )
    else:
        refresh_style(model)
        print(
            f"[5/8] Quality gates: patterns_kept={quality['confirmed']} "
            f"one_off_patterns_rejected={quality['rejected']}"
        )
        print(
            f"        tempo_source={beat_grid.source} bpm={beat_grid.tempo_bpm:.2f} "
            f"grid_confidence={beat_grid.confidence:.2f}"
        )

    model.print_structure_state = _print_structure_state_fixed.__get__(model, type(model))

    print("[6/8] Essentia source timbre analysis:", "RUNNING" if HAVE_ESSENTIA else "UNAVAILABLE (install essentia)")
    if HAVE_ESSENTIA and EssentiaTimbreAdapter is not None:
        essentia = EssentiaTimbreAdapter(sample_rate=params.sample_rate)
        essentia_features = essentia.analyze(audio, hop_size=2048)
        attached = essentia.attach_to_world(model.world, essentia_features)
        print(f"        source frames={len(essentia_features)} objects_enriched={attached}")
        print("        descriptors -> MFCC / spectral peaks / contrast / inharmonicity / spectral shape")
        print("        mode=coarse source timbre track (2048-sample hop; AWM remains full-resolution)")
    else:
        essentia_features = []
        print("        continuing with native AWM timbre features")

    print("[7/8] EDM elements / patterns / sections / arrangement...")
    edm = _core.EDMReverseDAW(model.world)
    snapshot = edm.run()
    print("[7/8] EDM structural layer: DONE")

    print("[8/8] Evaluation / export / activity plot...")

    if args.export_json:
        path = edm.export_json(args.export_json)
        print(f"exported={path}")

    original_subplots = plt.subplots

    def wide_subplots(*plot_args, **plot_kwargs):
        if plot_kwargs.get("figsize") == (14, 9):
            plot_kwargs["figsize"] = (32, 10)
        return original_subplots(*plot_args, **plot_kwargs)

    plt.subplots = wide_subplots
    try:
        model.plot_activity(save_path=args.plot)
    finally:
        plt.subplots = original_subplots

    print(f"Activity plot saved to: {args.plot}")

    print("\n" + "=" * 96)
    print("EDM REVERSE-DAW LAYER")
    print("=" * 96)
    print(f"audio_duration={audio_duration:.3f}s")
    print(f"time={snapshot['time']:.3f}s")
    print(f"beat_source={beat_grid.source if beat_grid is not None else 'AWM_RhythmEngine_fallback'}")
    print(f"beat_times={len(beat_grid.beats) if beat_grid is not None else 0}")
    print(f"stems={len(stems)}")
    print(f"source hypotheses={len(snapshot['source_hypotheses'])}")
    print(f"elements={len(snapshot['elements'])}")
    print(f"patterns={len(snapshot['patterns'])}")
    print(f"sections={len(snapshot['sections'])}")
    arrangement = snapshot["arrangement"]
    print(f"arrangement={'yes' if arrangement else 'not enough structural evidence'}")
    if arrangement:
        print(f"  confidence={arrangement['confidence']:.3f}")
        print(f"  sections={len(arrangement['section_ids'])}")

    for label, method_name in (
        ("Masking / bass recovery", "print_masked_bass_recovery_trace"),
        ("Groove", "print_groove_state"),
        ("Structure", "print_structure_state"),
        ("Roles", "print_role_summary"),
        ("Style", "print_style_summary"),
    ):
        method = getattr(model, method_name, None)
        if method is not None:
            print(f"\n--- {label} ---")
            method()

    print("\nAnalysis complete.")
    return model


if __name__ == "__main__":
    main()
