#!/usr/bin/env python3
"""EDM reverse-DAW entry point with improved transient detection, Demucs, and progression output."""
from __future__ import annotations

import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

import edm_reverse_daw_core as _core

# Keep the analysis readable: suppress the per-object ObjectTracker birth/match
# spam while retaining the explicit stage/progression output below.
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


def main():
    parser = argparse.ArgumentParser(description="Run EDM reverse-DAW analysis with improved transient detection.")
    parser.add_argument("audio", nargs="?", default=None)
    parser.add_argument("--export-json", default=None)
    parser.add_argument("--plot", default="auditory_world_model_activity.png")
    args = parser.parse_args()

    print("\n" + "=" * 96)
    print("AUDITORY WORLD MODEL — EDM REVERSE-DAW")
    print("=" * 96)
    print("[1/7] Loading audio...")

    params = _core.awm.Parameters()
    source = _core.awm.AudioSource(params)
    audio, _ = source.load(args.audio)

    model = _core.awm.AuditoryWorldModel(params)

    # Demucs/HTDemucs runs inside the AWM stem-separation stage.  Make that
    # stage explicit in the console instead of hiding it behind model.run().
    print("[2/7] Demucs coarse source separation: RUNNING")
    print("        stems -> drum / bass / other / vocal cues")
    model.run(audio, use_stem_separation=True)
    print("[2/7] Demucs coarse source separation: DONE")

    print("[3/7] Physical analysis + transient detection: DONE")
    print("[4/7] SoundObject tracking / perceptual state: DONE")

    print("[5/7] EDM elements / patterns / sections / arrangement...")
    edm = _core.EDMReverseDAW(model.world)
    snapshot = edm.run()
    print("[5/7] EDM structural layer: DONE")

    print("[6/7] Evaluation / export state: DONE")
    print("[7/7] Generating activity plot...")

    if args.export_json:
        path = edm.export_json(args.export_json)
        print(f"exported={path}")

    # Make the shared AWM decomposition/spectrogram plot much wider so closely
    # spaced transient markers can be inspected at higher time resolution.
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

    # Restore the progression/state printouts that were present before the
    # object-level print spam was introduced.  Do NOT print object summaries
    # or per-object tracker lines here.
    print("\n" + "=" * 96)
    print("EDM REVERSE-DAW LAYER")
    print("=" * 96)
    print(f"time={snapshot['time']:.3f}s")
    print(f"source hypotheses={len(snapshot['source_hypotheses'])}")
    print(f"elements={len(snapshot['elements'])}")
    print(f"patterns={len(snapshot['patterns'])}")
    print(f"sections={len(snapshot['sections'])}")
    arrangement = snapshot["arrangement"]
    print(f"arrangement={'yes' if arrangement else 'not enough structural evidence'}")
    if arrangement:
        print(f"  confidence={arrangement['confidence']:.3f}")
        print(f"  sections={len(arrangement['section_ids'])}")

    # Keep the useful foundation progression/state reports without the huge
    # object-by-object summary/event dump.
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
