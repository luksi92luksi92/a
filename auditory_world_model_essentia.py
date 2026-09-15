#!/usr/bin/env python3
"""Optional Essentia adapter for richer time-varying timbre evidence.

Essentia is used as an auxiliary feature extractor; the deterministic AWM
physical/object pipeline remains authoritative. The adapter computes frame-
level descriptors that are useful for arbitrary-source identity/change:
MFCC, Bark-band energy, spectral peaks, spectral contrast, inharmonicity,
and related spectral descriptors.

The adapter never assigns semantic instrument labels and never replaces
tracking/grouping. It attaches a compact, time-indexed timbre history to the
WorldState and SoundObjects so downstream identity logic can compare how a
tracked sound evolves.
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
    """Frame-level Essentia descriptors with compatibility across builds."""

    FRAME_SIZE = 2048
    HOP_SIZE = 256
    MFCC_BANDS = 40
    MFCC_COEFFS = 13
    MAX_PEAKS = 24

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = int(sample_rate)
        self.available = HAVE_ESSENTIA
        self.error = ESSENTIA_ERROR
        if not self.available:
            return

        self._windowing = es.Windowing(type="blackmanharris62", size=self.FRAME_SIZE)
        self._spectrum = es.Spectrum(size=self.FRAME_SIZE)

        # Essentia dev wheels differ in which convenience aliases are exposed.
        # Use the generic algorithms where needed.
        centroid_cls = getattr(es, "SpectralCentroid", None)
        self._centroid = (centroid_cls(sampleRate=self.sample_rate)
                          if centroid_cls is not None
                          else es.Centroid(range=self.sample_rate * 0.5))

        spread_cls = getattr(es, "SpectralSpread", None)
        self._spread = (spread_cls(sampleRate=self.sample_rate)
                        if spread_cls is not None
                        else es.Spread(range=self.sample_rate * 0.5))

        self._flatness = es.SpectralFlatnessDB()

        rolloff_cls = getattr(es, "SpectralRollOff", None)
        self._rolloff = (rolloff_cls(sampleRate=self.sample_rate, cutoff=0.85)
                         if rolloff_cls is not None
                         else es.RollOff(cutoff=0.85, sampleRate=self.sample_rate))

        self._flux = es.Flux()
        crest_cls = getattr(es, "SpectralCrest", None)
        self._crest = crest_cls() if crest_cls is not None else es.Crest()
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
        self._prev_spectrum: Optional[np.ndarray] = None

    def _frame_features(self, frame: np.ndarray, time: float) -> EssentiaTimbreFrame:
        mono = np.asarray(frame, dtype=np.float32)
        if mono.size < self.FRAME_SIZE:
            mono = np.pad(mono, (0, self.FRAME_SIZE - mono.size))
        elif mono.size > self.FRAME_SIZE:
            mono = mono[:self.FRAME_SIZE]

        windowed = self._windowing(mono)
        spectrum = self._spectrum(windowed)
        centroid = float(self._centroid(spectrum))
        spread = float(self._spread(spectrum))
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

        inharm = 0.0
        if peak_freqs.size:
            try:
                fundamental = float(peak_freqs[np.argmax(peak_mags)])
                harm_freqs, harm_mags = self._harmonic_peaks(
                    peak_freqs.tolist(), peak_mags.tolist(), fundamental
                )
                if len(harm_freqs) >= 2:
                    inharm = float(self._inharmonicity(harm_freqs, harm_mags))
            except Exception:
                inharm = 0.0

        contrast = []
        try:
            contrast_algo = es.SpectralContrast(sampleRate=self.sample_rate)
            contrast = np.asarray(contrast_algo(spectrum), dtype=float).tolist()
        except Exception:
            log_bands = np.log1p(np.maximum(bands, 0.0))
            if log_bands.size:
                contrast = np.diff(log_bands).tolist()

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
        """Return a time-indexed timbre track for the supplied mono audio."""
        if not self.available:
            return []
        hop = int(hop_size or self.HOP_SIZE)
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim > 1:
            x = np.mean(x, axis=0)
        self._prev_spectrum = None
        out: List[EssentiaTimbreFrame] = []
        n = max(0, x.size - self.FRAME_SIZE + hop)
        for start in range(0, n, hop):
            frame = x[start:start + self.FRAME_SIZE]
            if frame.size < self.FRAME_SIZE and start > 0:
                frame = np.pad(frame, (0, self.FRAME_SIZE - frame.size))
            t = (start + 0.5 * self.FRAME_SIZE) / self.sample_rate
            out.append(self._frame_features(frame, t))
        return out

    @staticmethod
    def _nearest(features: Sequence[EssentiaTimbreFrame], t: float) -> Optional[EssentiaTimbreFrame]:
        if not features:
            return None
        idx = min(range(len(features)), key=lambda i: abs(features[i].time - t))
        return features[idx]

    def attach_to_world(self, world: Any, features: Sequence[EssentiaTimbreFrame]) -> int:
        """Attach nearest Essentia descriptors to tracked objects by history time."""
        if not features:
            return 0
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
