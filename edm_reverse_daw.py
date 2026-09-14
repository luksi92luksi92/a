#!/usr/bin/env python3
"""EDM reverse-DAW entry point with improved transient detection and plot."""
from __future__ import annotations

import argparse
import numpy as np

import edm_reverse_daw_core as _core


def _detect_transients_fixed(flux: np.ndarray, hop: int, sr: int):
    """Multi-scale adaptive onset detector designed to recover weaker EDM hits."""
    x = np.log1p(np.maximum(np.asarray(flux, dtype=np.float64), 0.0))
    n = x.size
    onset_strength = np.zeros(n, dtype=np.float32)
    is_transient = np.zeros(n, dtype=bool)
    if n == 0:
        return onset_strength, is_transient

    # Two causal baselines: a short one catches local changes, while the long
    # one prevents sustained loud passages from becoming the reference level.
    short_history = max(12, int(round(0.28 * sr / max(hop, 1))))
    long_history = max(short_history + 4, int(round(0.90 * sr / max(hop, 1))))
    refractory = max(1, int(round(0.008 * sr / max(hop, 1))))  # ~8 ms
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

        z_short = robust_z(hs)
        z_long = robust_z(hl)
        # Prefer whichever baseline exposes the onset more strongly.
        z = max(z_short, z_long)
        onset_strength[i] = np.float32(z)

        # A hit can occupy a small plateau rather than a strict one-frame peak.
        # Accept the strongest frame within a ±2-frame neighborhood.
        left = max(0, i - 2)
        right = min(n, i + 3)
        local_peak = x[i] >= np.max(x[left:right]) - 1e-12

        # Recover quieter hits while requiring a meaningful local rise.
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

    params = _core.awm.Parameters()
    source = _core.awm.AudioSource(params)
    audio, _ = source.load(args.audio)
    model = _core.awm.AuditoryWorldModel(params)
    model.run(audio, use_stem_separation=False)

    edm = _core.EDMReverseDAW(model.world)
    snapshot = edm.run()

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
    if args.export_json:
        path = edm.export_json(args.export_json)
        print(f"exported={path}")

    model.plot_activity(save_path=args.plot)
    print(f"\nActivity plot saved to: {args.plot}")
    return model


if __name__ == "__main__":
    main()
