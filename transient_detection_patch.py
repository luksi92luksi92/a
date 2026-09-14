from __future__ import annotations

import numpy as np


def detect_transients_robust(flux: np.ndarray, hop: int, sr: int,
                             threshold: float = 2.2,
                             refractory_ms: float = 18.0,
                             baseline_window_s: float = 0.8):
    """Causal robust transient detector for spectral-flux onset candidates.

    The input is ordinary positive spectral flux. We log-compress it, estimate
    a local median/MAD from preceding frames, then trigger on local maxima.
    """
    flux = np.asarray(flux, dtype=np.float32).reshape(-1)
    n = flux.size
    if n == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)

    x = np.log1p(np.maximum(flux, 0.0)).astype(np.float32)
    history_frames = max(8, int(round(baseline_window_s * sr / hop)))
    strength = np.zeros(n, dtype=np.float32)

    for i in range(n):
        hist = x[max(0, i - history_frames):i]
        if hist.size == 0:
            continue
        med = float(np.median(hist))
        mad = float(np.median(np.abs(hist - med)))
        sigma = max(1.4826 * mad, 0.01)
        strength[i] = float((x[i] - med) / sigma)

    refractory = max(1, int(round(refractory_ms * 1e-3 * sr / hop)))
    is_transient = np.zeros(n, dtype=bool)
    last = -refractory

    for i in range(n):
        if strength[i] < threshold:
            continue
        left = strength[i - 1] if i > 0 else strength[i]
        right = strength[i + 1] if i + 1 < n else strength[i]
        if strength[i] < left or strength[i] < right:
            continue
        if i - last < refractory:
            continue
        is_transient[i] = True
        last = i

    return strength, is_transient
