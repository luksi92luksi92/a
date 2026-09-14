#!/usr/bin/env python3
"""EDM reverse-DAW entry point with corrected transient detection."""
from __future__ import annotations

import numpy as np

import edm_reverse_daw_core as _core


def _detect_transients_fixed(flux: np.ndarray, hop: int, sr: int):
    """Robust causal transient detector: log flux + median/MAD + local peaks."""
    x = np.log1p(np.maximum(np.asarray(flux, dtype=np.float64), 0.0))
    n = x.size
    onset_strength = np.zeros(n, dtype=np.float32)
    is_transient = np.zeros(n, dtype=bool)
    if n == 0:
        return onset_strength, is_transient

    history_frames = max(16, int(round(0.75 * sr / max(hop, 1))))
    refractory = max(1, int(round(0.018 * sr / max(hop, 1))))
    threshold = 1.75
    last_trigger = -refractory

    for i in range(n):
        lo = max(0, i - history_frames)
        hist = x[lo:i]
        if hist.size >= 8:
            baseline = float(np.median(hist))
            mad = float(np.median(np.abs(hist - baseline)))
            scale = max(1.4826 * mad, 1e-6)
        elif hist.size > 1:
            baseline = float(np.mean(hist))
            scale = max(float(np.std(hist)), 1e-6)
        else:
            baseline = float(x[i])
            scale = 1e-6

        z = (float(x[i]) - baseline) / scale
        onset_strength[i] = np.float32(z)
        local_peak = i == 0 or (x[i] >= x[i - 1] and (i + 1 >= n or x[i] >= x[i + 1]))
        if i >= 8 and z >= threshold and local_peak and (i - last_trigger) >= refractory:
            is_transient[i] = True
            last_trigger = i

    return onset_strength, is_transient


_core.awm.PhysicalAnalyzer.detect_transients = staticmethod(_detect_transients_fixed)

if __name__ == "__main__":
    _core.main()
