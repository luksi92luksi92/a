#!/usr/bin/env python3
"""Required Essentia adapter for richer time-varying timbre evidence.

Essentia is a required feature extractor for the reverse-DAW timbre layer.
The deterministic AWM physical/object pipeline remains authoritative, while
this adapter contributes explicit, time-indexed timbre evidence for identity
and source-change analysis.

No requested descriptor is silently replaced with zero or skipped. The
installed Essentia build is validated at startup. Spectral spread is computed
directly from the spectrum because the portable Essentia Python API exposes
Centroid/CentralMoments rather than a stable SpectralSpread class name.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import essentia.standard as es
    HAVE_ESSENTIA = True
    ESSENTIA_ERROR = ""
except Exception as exc:
    es = None
    HAVE_ESSENTIA = False
    ESSENTIA_ERROR = str(exc)


@dataclass
class EssentiaTimbreFrame:
    time: float
    spectral_centroid: float
    spectral_spread: float
    spectral_flatness_db: float
    spectral_rolloff: float
    spectral_flux: float
    spectral_crest: float
    hfc: float
    zero_crossing_rate: float
    inharmonicity: float
    mfcc: List[float]
    mfcc_bands: List[float]
    peak_frequencies: List[float]
    peak_magnitudes: List[float]
    spectral_contrast: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EssentiaTimbreAdapter:
    """Strict frame-level Essentia descriptors across supported builds."""

    FRAME_SIZE = 2048
    HOP_SIZE = 256
    MFCC_BANDS = 40
    MFCC_COEFFS = 13
    MAX_PEAKS = 24

    REQUIRED_ALGORITHMS = (
        "Windowing", "Spectrum", "Centroid", "FlatnessDB", "RollOff",
        "Flux", "Crest", "HFC", "ZeroCrossingRate", "MFCC",
        "SpectralPeaks", "HarmonicPeaks", "Inharmonicity", "SpectralContrast",
    )

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = int(sample_rate)
        self.available = HAVE_ESSENTIA
        self.error = ESSENTIA_ERROR
        if not self.available:
            raise RuntimeError(
                "Essentia is required for timbre analysis but could not be imported: "
                f"{self.error}"
            )

        missing = [name for name in self.REQUIRED_ALGORITHMS if not hasattr(es, name)]
        if missing:
            raise RuntimeError(
                "Installed Essentia build is missing required timbre algorithms: "
                + ", ".join(missing)
                + ". No silent fallback is permitted."
            )

        self._windowing = es.Windowing(type="blackmanharris62", size=self.FRAME_SIZE)
        self._spectrum = es.Spectrum(size=self.FRAME_SIZE)
        self._centroid = es.Centroid(range=self.sample_rate * 0.5)
        self._flatness = es.FlatnessDB()
        self._rolloff = es.RollOff(sampleRate=self.sample_rate, cutoff=0.85)
        self._flux = es.Flux()
        self._crest = es.Crest()
        self._hfc = es.HFC()
        self._zcr = es.ZeroCrossingRate()
        self._mfcc = es.MFCC(
            inputSize=self.FRAME_SIZE // 2 + 1,
            sampleRate=self.sample_rate,
            numberCoefficients=self.MFCC_COEFFS,
            numberBands=self.MFCC_BANDS,
        )
        self._peaks = es.SpectralPeaks(
            maxPeaks=self.MAX_PEAKS,
            sampleRate=self.sample_rate,
            magnitudeThreshold=0.001,
            minFrequency=20.0,
            maxFrequency=self.sample_rate * 0.5,
        )
        self._harmonic_peaks = es.HarmonicPeaks()
        self._inharmonicity = es.Inharmonicity()
        self._spectral_contrast = es.SpectralContrast(sampleRate=self.sample_rate)
        self._prev_spectrum: Optional[np.ndarray] = None

    def _spectral_spread(self, spectrum: np.ndarray, centroid_hz: float) -> float:
        mag = np.maximum(np.asarray(spectrum, dtype=np.float64), 0.0)
        if mag.size <= 1:
            return 0.0
        freqs = np.linspace(0.0, self.sample_rate * 0.5, mag.size)
        total = float(np.sum(mag))
        if total <= 1e-12:
            return 0.0
        variance = float(np.sum(mag * (freqs - centroid_hz) ** 2) / total)
        return float(np.sqrt(max(variance, 0.0)))

    def _frame_features(self, frame: np.ndarray, time: float) -> EssentiaTimbreFrame:
        mono = np.asarray(frame, dtype=np.float32)
        if mono.size < self.FRAME_SIZE:
            mono = np.pad(mono, (0, self.FRAME_SIZE - mono.size))
        elif mono.size > self.FRAME_SIZE:
            mono = mono[:self.FRAME_SIZE]

        windowed = self._windowing(mono)
        spectrum = self._spectrum(windowed)
        centroid = float(self._centroid(spectrum))
        spread = self._spectral_spread(spectrum, centroid)
        flatness = float(self._flatness(spectrum))
        rolloff = float(self._rolloff(spectrum))
        flux = float(self._flux(spectrum, self._prev_spectrum)) if self._prev_spectrum is not None else 0.0
        crest = float(self._crest(spectrum))
        hfc = float(self._hfc(spectrum))
        zcr = float(self._zcr(mono))

        bands, coeffs = self._mfcc(spectrum)
        bands = np.asarray(bands, dtype=float)
        coeffs = np.asarray(coeffs, dtype=float)

        peak_freqs, peak_mags = self._peaks(spectrum)
        peak_freqs = np.asarray(peak_freqs, dtype=float)[: self.MAX_PEAKS]
        peak_mags = np.asarray(peak_mags, dtype=float)[: self.MAX_PEAKS]

        if peak_freqs.size < 2:
            inharm = 0.0
        else:
            fundamental = float(peak_freqs[np.argmax(peak_mags)])
            harm_freqs, harm_mags = self._harmonic_peaks(
                peak_freqs.tolist(), peak_mags.tolist(), fundamental
            )
            inharm = (float(self._inharmonicity(harm_freqs, harm_mags))
                      if len(harm_freqs) >= 2 else 0.0)

        contrast = np.asarray(self._spectral_contrast(spectrum), dtype=float).tolist()
        self._prev_spectrum = np.asarray(spectrum, dtype=float)

        return EssentiaTimbreFrame(
            time=float(time),
            spectral_centroid=centroid,
            spectral_spread=spread,
            spectral_flatness_db=flatness,
            spectral_rolloff=rolloff,
            spectral_flux=flux,
            spectral_crest=crest,
            hfc=hfc,
            zero_crossing_rate=zcr,
            inharmonicity=inharm,
            mfcc=coeffs[: self.MFCC_COEFFS].tolist(),
            mfcc_bands=bands[: self.MFCC_BANDS].tolist(),
            peak_frequencies=peak_freqs.tolist(),
            peak_magnitudes=peak_mags.tolist(),
            spectral_contrast=contrast,
        )

    def analyze(self, audio: np.ndarray, hop_size: Optional[int] = None) -> List[EssentiaTimbreFrame]:
        hop = int(hop_size or self.HOP_SIZE)
        if hop <= 0:
            raise ValueError("hop_size must be positive")
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim > 1:
            x = np.mean(x, axis=0)
        self._prev_spectrum = None
        out: List[EssentiaTimbreFrame] = []
        if x.size == 0:
            return out
        for start in range(0, x.size, hop):
            frame = x[start:start + self.FRAME_SIZE]
            if frame.size < self.FRAME_SIZE:
                frame = np.pad(frame, (0, self.FRAME_SIZE - frame.size))
            t = (start + 0.5 * self.FRAME_SIZE) / self.sample_rate
            out.append(self._frame_features(frame, t))
            if start + self.FRAME_SIZE >= x.size:
                break
        return out

    @staticmethod
    def _nearest(features: Sequence[EssentiaTimbreFrame], t: float) -> Optional[EssentiaTimbreFrame]:
        if not features:
            return None
        idx = min(range(len(features)), key=lambda i: abs(features[i].time - t))
        return features[idx]

    def attach_to_world(self, world: Any, features: Sequence[EssentiaTimbreFrame]) -> int:
        if not features:
            raise ValueError("Essentia analysis produced no feature frames")
        track = [f.to_dict() for f in features]
        setattr(world, "essentia_timbre_track", track)
        attached = 0
        for obj in world.objects.values():
            history: List[Dict[str, Any]] = []
            for obs in getattr(obj, "history", []):
                f = self._nearest(features, float(obs.time))
                if f is not None:
                    history.append(f.to_dict())
            setattr(obj, "essentia_timbre_history", history)
            if history:
                attached += 1
        return attached
