#!/usr/bin/env python3
"""Beat/bar/phrase/section structure and functional-segment analysis.

The global Beat This! clock is authoritative.  This module is intentionally
audio/structure-first and does not require sound-object detection.

Per-stem analysis includes:
- stem-specific audible/silence/noise thresholds;
- no plot for an empty or inaudible/noise-only stem;
- normalized FFT display with stem name and RMS volume relative to the
  original mix;
- pause and transition beat/bar functional segments;
- phrase/section spans aligned to the global bar/beat grid;
- 10-dimension functional-segment similarity with a >6/10 match rule;
- shared segment-group colors across every plot row;
- beat/bar similarity rows;
- early/late segment start/end offsets measured in beats from bar boundaries;
- volume per beat, syncopation evidence, and melody/f0 evidence.

The similarity system compares complete non-silent functional segments.  It
resamples beat-level evidence to a common length so phrase/section duration
variation and small beat-pattern differences do not prevent matching.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class NoveltyCandidate:
    level: str
    index: int
    time_s: float
    score: float
    prominence: float
    confidence: float
    selected: bool
    reason: str


@dataclass
class NoveltyStructureResult:
    beat_novelty: List[float]
    bar_novelty: List[float]
    phrase_novelty: List[float]
    section_novelty: List[float]
    beat_times: List[float]
    bar_boundaries: List[float]
    phrase_boundaries: List[float]
    section_boundaries: List[float]
    phrase_spans: List[Dict[str, Any]]
    section_spans: List[Dict[str, Any]]
    functional_segments: List[Dict[str, Any]]
    segment_groups: List[Dict[str, Any]]
    similarity_pairs: List[Dict[str, Any]]
    candidates: List[Dict[str, Any]]
    beat_volume_pct_original: List[float]
    bar_volume_pct_original: List[float]
    beat_syncopation: List[float]
    melody_midi: List[float]
    beat_similarity: List[float]
    bar_similarity: List[float]
    stem_stats: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BeatBarPhraseSectionNovelty:
    """Compute multi-scale novelty and functional structure on a fixed clock."""

    GROUP_COLORS = [
        "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
        "#e6ab02", "#a6761d", "#1f78b4", "#b2df8a", "#fb9a99",
        "#6a3d9a", "#ff7f00", "#b15928", "#33a02c", "#cab2d6",
        "#fdbf6f", "#a6cee3", "#8dd3c7", "#bebada", "#fccde5",
    ]
    PAUSE_COLOR = "#bdbdbd"
    TRANSITION_COLOR = "#f46d43"
    BEAT_COLOR = "#aaaaaa"
    BAR_COLOR = "#444444"
    PHRASE_COLOR = "#377eb8"
    SECTION_COLOR = "#984ea3"

    def __init__(
        self,
        sample_rate: int,
        beats_per_bar: int = 4,
        phrase_lengths: Sequence[int] = (4, 8, 16),
        fft_size: int = 2048,
        hop_size: int = 512,
        context_bars: int = 1,
        min_section_bars: int = 4,
        section_refractory_bars: int = 2,
        section_threshold: float = 0.48,
        silence_dbfs: float = -60.0,
        inaudible_relative_db: float = -48.0,
        noise_flatness: float = 0.94,
        noise_dynamic_db: float = 7.0,
        transition_threshold: float = 0.60,
        similarity_threshold: float = 0.60,
        similarity_dimensions_required: int = 7,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.beats_per_bar = int(beats_per_bar)
        self.phrase_lengths = tuple(int(x) for x in phrase_lengths if int(x) >= 1)
        self.fft_size = int(fft_size)
        self.hop_size = int(hop_size)
        self.context_bars = int(max(context_bars, 1))
        self.min_section_bars = int(max(min_section_bars, 1))
        self.section_refractory_bars = int(max(section_refractory_bars, 1))
        self.section_threshold = float(section_threshold)

        self.silence_dbfs = float(silence_dbfs)
        self.inaudible_relative_db = float(inaudible_relative_db)
        self.noise_flatness = float(noise_flatness)
        self.noise_dynamic_db = float(noise_dynamic_db)
        self.transition_threshold = float(transition_threshold)
        self.similarity_threshold = float(similarity_threshold)
        self.similarity_dimensions_required = int(similarity_dimensions_required)

    @staticmethod
    def _db(x: float) -> float:
        return float(20.0 * np.log10(max(abs(float(x)), 1e-12)))

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return x
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        scale = max(1.4826 * mad, 1e-6)
        return (x - med) / scale

    @staticmethod
    def _row_normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            return x
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norms, 1e-9)

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        n = min(a.size, b.size)
        aa, bb = np.asarray(a[:n], dtype=float), np.asarray(b[:n], dtype=float)
        na = float(np.linalg.norm(aa))
        nb = float(np.linalg.norm(bb))
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        cosine = float(np.clip(np.dot(aa, bb) / (na * nb), -1.0, 1.0))
        return 0.5 * (1.0 - cosine)

    @staticmethod
    def _resample_1d(x: np.ndarray, n: int) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        n = int(max(n, 1))
        if x.size == 0:
            return np.zeros(n, dtype=float)
        if x.size == 1:
            return np.full(n, float(x[0]), dtype=float)
        xp = np.linspace(0.0, 1.0, x.size)
        xn = np.linspace(0.0, 1.0, n)
        return np.interp(xn, xp, x)

    @staticmethod
    def _normalize01(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if not x.size:
            return x
        lo, hi = np.percentile(x, [5, 95])
        if hi - lo < 1e-9:
            return np.zeros_like(x)
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    @staticmethod
    def _peak_candidates(
        values: np.ndarray,
        times: np.ndarray,
        level: str,
        threshold: float,
        refractory: int = 1,
    ) -> List[NoveltyCandidate]:
        x = np.asarray(values, dtype=float)
        if x.size < 3:
            return []
        norm = BeatBarPhraseSectionNovelty._normalize01(x)
        found: List[NoveltyCandidate] = []
        last = -10**9
        for i in range(1, len(x) - 1):
            if norm[i] < threshold:
                continue
            if not (norm[i] >= norm[i - 1] and norm[i] >= norm[i + 1]):
                continue
            if i - last < refractory:
                continue
            local = float(norm[max(0, i - 2):min(len(x), i + 3)].mean())
            prominence = float(max(norm[i] - local, 0.0))
            confidence = float(np.clip(0.60 * norm[i] + 0.40 * prominence, 0.0, 1.0))
            found.append(
                NoveltyCandidate(
                    level, i, float(times[i]), float(norm[i]), prominence,
                    confidence, True, "local novelty peak"
                )
            )
            last = i
        return found

    def _frame_features(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if x.size < self.fft_size:
            x = np.pad(x, (0, self.fft_size - x.size))
        win = np.hanning(self.fft_size).astype(np.float32)
        times: List[float] = []
        feats: List[np.ndarray] = []
        rms_values: List[float] = []
        flatness_values: List[float] = []
        prev_mag = None
        freq = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)
        bands = np.array(
            [[20, 90], [90, 250], [250, 1000], [1000, 4000], [4000, 10000]],
            dtype=float,
        )
        max_start = max(0, x.size - self.fft_size + 1)
        for start in range(0, max_start + 1, self.hop_size):
            frame = x[start:start + self.fft_size]
            if frame.size < self.fft_size:
                frame = np.pad(frame, (0, self.fft_size - frame.size))
            windowed = frame * win
            spec = np.fft.rfft(windowed)
            mag = np.abs(spec).astype(np.float64)
            power = mag * mag + 1e-12
            total = float(np.sum(power))
            centroid = float(np.sum(freq * power) / total) if total > 0 else 0.0
            spread = float(
                np.sqrt(np.sum(((freq - centroid) ** 2) * power) / total)
            ) if total > 0 else 0.0
            flatness = float(
                np.exp(np.mean(np.log(mag + 1e-9))) /
                max(np.mean(mag), 1e-9)
            )
            flux = 0.0
            if prev_mag is not None:
                flux = float(
                    np.linalg.norm(np.maximum(mag - prev_mag, 0.0)) /
                    (np.linalg.norm(prev_mag) + 1e-9)
                )
            prev_mag = mag

            band_energy = []
            for lo, hi in bands:
                m = (freq >= lo) & (freq < hi)
                band_energy.append(float(np.sum(power[m])) / total if total > 0 else 0.0)

            rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            log_rms = float(np.log1p(rms + 1e-9))
            feat = np.asarray(
                [
                    centroid / 10000.0,
                    spread / 10000.0,
                    flatness,
                    flux,
                    log_rms,
                    *band_energy,
                ],
                dtype=float,
            )
            feats.append(feat)
            rms_values.append(rms)
            flatness_values.append(flatness)
            times.append(float(start + self.fft_size * 0.5) / self.sample_rate)

        if not feats:
            return (
                np.zeros((0, 10), dtype=float),
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
            )

        F = np.asarray(feats, dtype=float)
        F[:, :2] = self._zscore(F[:, :2])
        F[:, 3:5] = self._zscore(F[:, 3:5])
        F[:, 5:] = self._row_normalize(F[:, 5:])
        return (
            F,
            np.asarray(times, dtype=float),
            np.asarray(rms_values, dtype=float),
            np.asarray(flatness_values, dtype=float),
        )

    @staticmethod
    def _aggregate_to_times(
        features: np.ndarray,
        frame_times: np.ndarray,
        centers: np.ndarray,
        half_window: float,
    ) -> np.ndarray:
        rows = []
        for t in centers:
            idx = np.where(np.abs(frame_times - float(t)) <= half_window)[0]
            if idx.size == 0:
                j = int(np.argmin(np.abs(frame_times - float(t)))) if frame_times.size else 0
                rows.append(
                    features[j] if frame_times.size
                    else np.zeros(features.shape[1] if features.ndim == 2 else 1)
                )
            else:
                rows.append(np.mean(features[idx], axis=0))
        return np.asarray(rows, dtype=float)

    @staticmethod
    def _aggregate_scalar(values: np.ndarray, frame_times: np.ndarray, centers: np.ndarray, half_window: float) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        frame_times = np.asarray(frame_times, dtype=float).reshape(-1)
        rows = []
        for t in centers:
            idx = np.where(np.abs(frame_times - float(t)) <= half_window)[0]
            if idx.size == 0:
                j = int(np.argmin(np.abs(frame_times - float(t)))) if frame_times.size else 0
                rows.append(float(values[j]) if values.size else 0.0)
            else:
                rows.append(float(np.sqrt(np.mean(values[idx] ** 2))))
        return np.asarray(rows, dtype=float)

    def _local_change(self, X: np.ndarray, radius: int = 1) -> np.ndarray:
        n = len(X)
        out = np.zeros(n, dtype=float)
        if n == 0:
            return out
        r = int(max(radius, 1))
        for i in range(n):
            left = X[max(0, i - r):i]
            right = X[i + 1:min(n, i + 1 + r)]
            if left.size and right.size:
                out[i] = self._distance(np.mean(left, axis=0), np.mean(right, axis=0))
            elif i > 0:
                out[i] = self._distance(X[i - 1], X[i])
        return out

    def _self_similarity_novelty(self, X: np.ndarray, radius: int = 2) -> np.ndarray:
        n = len(X)
        out = np.zeros(n, dtype=float)
        if n < 3:
            return out
        r = int(max(radius, 1))
        Xn = self._row_normalize(X)
        S = np.clip(Xn @ Xn.T, -1.0, 1.0)
        for i in range(n):
            for off in range(-r, r + 1):
                for off2 in range(-r, r + 1):
                    if off * off2 >= 0:
                        continue
                    a = i + off
                    b = i + off2
                    if 0 <= a < n and 0 <= b < n:
                        out[i] += 1.0 - float(S[a, b])
        denom = float(max(2 * r * (r + 1), 1))
        return out / denom

    def _bar_map(self, beat_times: np.ndarray, downbeats: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        if downbeats.size >= 2:
            starts = np.asarray(downbeats, dtype=float)
            beat_indices = [
                int(np.argmin(np.abs(beat_times - t))) for t in starts
            ]
            beat_indices = sorted(
                set(max(0, min(len(beat_times) - 1, i)) for i in beat_indices)
            )
            if len(beat_indices) >= 2:
                return starts, beat_indices
        starts = beat_times[::self.beats_per_bar]
        return starts, list(range(0, len(beat_times), self.beats_per_bar))

    @staticmethod
    def _running_db(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return 20.0 * np.log10(np.maximum(values, 1e-12))

    def _stem_stats(
        self,
        audio: np.ndarray,
        original_audio: np.ndarray | None,
        frame_rms: np.ndarray,
        flatness: np.ndarray,
    ) -> Dict[str, Any]:
        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        orig = np.asarray(original_audio if original_audio is not None else audio, dtype=np.float64).reshape(-1)

        stem_rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        orig_rms = float(np.sqrt(np.mean(orig * orig))) if orig.size else 0.0
        stem_peak = float(np.max(np.abs(x))) if x.size else 0.0
        orig_peak = float(np.max(np.abs(orig))) if orig.size else 0.0
        stem_dbfs = self._db(stem_rms)
        orig_dbfs = self._db(orig_rms)
        relative_db = float(stem_dbfs - orig_dbfs)
        volume_pct = float(100.0 * stem_rms / max(orig_rms, 1e-12))

        frame_db = self._running_db(frame_rms)
        low_db = float(np.percentile(frame_db, 10)) if frame_db.size else -120.0
        high_db = float(np.percentile(frame_db, 95)) if frame_db.size else -120.0
        median_flat = float(np.median(flatness)) if flatness.size else 1.0

        threshold_dbfs = max(
            self.silence_dbfs,
            orig_dbfs + self.inaudible_relative_db,
        )
        if high_db > low_db + self.noise_dynamic_db:
            threshold_dbfs = max(threshold_dbfs, low_db + self.noise_dynamic_db)

        audible = frame_db >= threshold_dbfs
        audible_ratio = float(np.mean(audible)) if audible.size else 0.0
        dynamic_db = float(high_db - low_db)

        noise_only = (
            relative_db < self.inaudible_relative_db
            or (
                median_flat >= self.noise_flatness
                and dynamic_db < self.noise_dynamic_db
                and relative_db < -24.0
            )
        )
        empty = stem_rms <= 1e-10 or stem_peak <= 1e-8
        has_audible_content = bool(not empty and not noise_only and np.any(audible))

        return {
            "stem_rms": stem_rms,
            "stem_rms_dbfs": stem_dbfs,
            "stem_peak": stem_peak,
            "original_rms": orig_rms,
            "original_rms_dbfs": orig_dbfs,
            "original_peak": orig_peak,
            "relative_rms_db": relative_db,
            "volume_pct_of_original_rms": volume_pct,
            "threshold_dbfs": float(threshold_dbfs),
            "noise_floor_dbfs_p10": low_db,
            "p95_dbfs": high_db,
            "noise_dynamic_db": dynamic_db,
            "median_spectral_flatness": median_flat,
            "audible_frame_ratio": audible_ratio,
            "has_audible_content": has_audible_content,
            "empty": bool(empty),
            "noise_only_or_inaudible": bool(noise_only),
        }

    def _melody_track(self, audio: np.ndarray, beat_times: np.ndarray, beat_period: float, beat_volume: np.ndarray, threshold_rms: float) -> np.ndarray:
        try:
            import librosa
            f0 = librosa.yin(
                np.asarray(audio, dtype=np.float32),
                fmin=55.0,
                fmax=min(2200.0, self.sample_rate * 0.45),
                sr=self.sample_rate,
                frame_length=self.fft_size,
                hop_length=self.hop_size,
            )
            times = librosa.times_like(f0, sr=self.sample_rate, hop_length=self.hop_size)
            midi = 69.0 + 12.0 * np.log2(np.maximum(f0, 1e-6) / 440.0)
            out = np.zeros(len(beat_times), dtype=float)
            for i, t in enumerate(beat_times):
                idx = np.where(np.abs(times - t) <= max(beat_period * 0.45, 0.05))[0]
                vals = midi[idx] if idx.size else np.zeros(0)
                valid = vals[np.isfinite(vals)]
                if valid.size and beat_volume[i] >= threshold_rms:
                    out[i] = float(np.median(valid))
            return out
        except Exception:
            return np.zeros(len(beat_times), dtype=float)

    def _syncopation(self, audio: np.ndarray, beat_times: np.ndarray, beat_period: float) -> np.ndarray:
        if len(beat_times) == 0:
            return np.zeros(0, dtype=float)
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        out = np.zeros(len(beat_times), dtype=float)
        for i, center in enumerate(beat_times):
            start = int(max(0, round(float(center) * self.sample_rate)))
            sub = np.linspace(0.0, beat_period, 5)
            vals = []
            for j in range(4):
                a = int(round((float(center) + sub[j] - 0.5 * beat_period) * self.sample_rate))
                b = int(round((float(center) + sub[j + 1] - 0.5 * beat_period) * self.sample_rate))
                a = max(0, a)
                b = min(len(x), max(a + 1, b))
                seg = x[a:b]
                vals.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if seg.size else 0.0)
            v = np.asarray(vals, dtype=float)
            if not np.any(v):
                continue
            strong = v[0] + 0.75 * v[2]
            off = v[1] + v[3]
            total = float(np.sum(v))
            out[i] = float(np.clip(off / max(total, 1e-12) * 1.8 - strong / max(total, 1e-12) * 0.4, 0.0, 1.0))
        return out

    def _frame_activity_mask(self, frame_rms: np.ndarray, threshold_dbfs: float) -> np.ndarray:
        if frame_rms.size == 0:
            return np.zeros(0, dtype=bool)
        return self._running_db(frame_rms) >= threshold_dbfs

    def _segment_bounds_for_bars(
        self,
        bar_index_start: int,
        bar_index_end: int,
        bar_ranges: List[Tuple[int, int]],
        beat_times: np.ndarray,
    ) -> Tuple[float, float, int, int]:
        bs = max(0, min(bar_index_start, len(bar_ranges) - 1))
        be = max(bs + 1, min(bar_index_end, len(bar_ranges)))
        b0 = bar_ranges[bs][0]
        b1 = bar_ranges[be - 1][1]
        start_t = float(beat_times[b0])
        end_beat = max(b0, min(len(beat_times) - 1, b1 - 1))
        end_t = float(beat_times[end_beat])
        if end_beat + 1 < len(beat_times):
            end_t = float(beat_times[end_beat + 1])
        return start_t, end_t, b0, b1

    def _build_functional_segments(
        self,
        beat_times: np.ndarray,
        bar_times: np.ndarray,
        bar_ranges: List[Tuple[int, int]],
        beat_silent: np.ndarray,
        beat_transition: np.ndarray,
        section_boundaries_idx: Sequence[int],
        phrase_candidates: Sequence[NoveltyCandidate],
        frame_times: np.ndarray,
        frame_active: np.ndarray,
        beat_period: float,
    ) -> List[Dict[str, Any]]:
        n_bars = len(bar_ranges)
        segments: List[Dict[str, Any]] = []

        def classify_run(start_bar: int, end_bar: int, kind: str, reason: str) -> None:
            if end_bar <= start_bar or not bar_ranges:
                return
            start_t, end_t, b0, b1 = self._segment_bounds_for_bars(start_bar, end_bar, bar_ranges, beat_times)
            frame_idx = np.where(
                (frame_times >= start_t) & (frame_times <= end_t) & frame_active
            )[0]
            actual_start = float(frame_times[frame_idx[0]]) if frame_idx.size else start_t
            actual_end = float(frame_times[frame_idx[-1]]) if frame_idx.size else end_t
            nominal_start_bar = int(start_bar)
            nominal_end_bar = int(end_bar)
            start_bar_t = float(bar_times[min(start_bar, len(bar_times) - 1)])
            end_bar_t = (
                float(bar_times[min(end_bar, len(bar_times) - 1)])
                if end_bar < len(bar_times)
                else float(end_t)
            )
            start_offset = (actual_start - start_bar_t) / max(beat_period, 1e-9)
            end_offset = (actual_end - end_bar_t) / max(beat_period, 1e-9)
            segments.append({
                "segment_id": f"seg_{len(segments):04d}",
                "kind": kind,
                "reason": reason,
                "start_bar": int(nominal_start_bar),
                "end_bar": int(nominal_end_bar),
                "start_beat": int(b0),
                "end_beat": int(b1),
                "grain": "bar",
                "start_time_s": start_t,
                "end_time_s": end_t,
                "actual_start_time_s": actual_start,
                "actual_end_time_s": actual_end,
                "start_offset_beats_from_bar": float(start_offset),
                "end_offset_beats_from_bar": float(end_offset),
                "group_id": None,
                "group_color": self.PAUSE_COLOR if kind == "pause" else self.TRANSITION_COLOR if kind == "transition" else None,
                "similarity_score": 0.0,
                "similarity_dimensions": 0,
            })

        bar_pause = np.asarray(
            [
                bool(np.all(beat_silent[s:e])) if e > s else True
                for s, e in bar_ranges
            ],
            dtype=bool,
        )
        bar_transition = np.asarray(
            [
                bool(np.any(beat_transition[s:e])) if e > s else False
                for s, e in bar_ranges
            ],
            dtype=bool,
        )

        # Beat-granularity pause/transition spans are kept as functional
        # segments too.  These never enter complete-segment similarity grouping.
        def classify_beat_run(start_beat: int, end_beat: int, kind: str, reason: str) -> None:
            if end_beat <= start_beat:
                return
            start_t = float(beat_times[start_beat])
            end_t = float(beat_times[min(end_beat, len(beat_times) - 1)])
            if end_beat < len(beat_times):
                end_t = float(beat_times[end_beat])
            segments.append({
                "segment_id": f"seg_{len(segments):04d}",
                "kind": kind,
                "grain": "beat",
                "reason": reason,
                "start_bar": int(start_beat // max(self.beats_per_bar, 1)),
                "end_bar": int(math.ceil(end_beat / max(self.beats_per_bar, 1))),
                "start_beat": int(start_beat),
                "end_beat": int(end_beat),
                "start_time_s": start_t,
                "end_time_s": end_t,
                "actual_start_time_s": start_t,
                "actual_end_time_s": end_t,
                "start_offset_beats_from_bar": 0.0,
                "end_offset_beats_from_bar": 0.0,
                "group_id": None,
                "group_color": self.PAUSE_COLOR if kind == "pause" else self.TRANSITION_COLOR,
                "similarity_score": 0.0,
                "similarity_dimensions": 0,
            })

        i = 0
        while i < len(beat_silent):
            if not beat_silent[i]:
                i += 1
                continue
            j = i + 1
            while j < len(beat_silent) and beat_silent[j]:
                j += 1
            classify_beat_run(i, j, "pause", "beat below audible threshold")
            i = j

        i = 0
        active_transition = np.logical_and(beat_transition, np.logical_not(beat_silent))
        while i < len(active_transition):
            if not active_transition[i]:
                i += 1
                continue
            j = i + 1
            while j < len(active_transition) and active_transition[j]:
                j += 1
            classify_beat_run(i, j, "transition", "beat-scale novelty/energy transition")
            i = j

        # First-class pauses: continuous silent bars are their own functional spans.
        i = 0
        while i < n_bars:
            if not bar_pause[i]:
                i += 1
                continue
            j = i + 1
            while j < n_bars and bar_pause[j]:
                j += 1
            classify_run(i, j, "pause", "stem below audible threshold")
            i = j

        # Active material is split by novelty section boundaries and transition bars.
        cuts = {0, n_bars}
        for idx in section_boundaries_idx:
            cuts.add(int(max(0, min(n_bars, idx))))
        for idx in range(n_bars):
            if bar_pause[idx]:
                cuts.add(idx)
                cuts.add(idx + 1)
            if bar_transition[idx]:
                cuts.add(idx)
                cuts.add(idx + 1)
        cuts = sorted(cuts)

        for a, b in zip(cuts[:-1], cuts[1:]):
            if b <= a:
                continue
            if np.all(bar_pause[a:b]):
                continue
            if np.any(bar_transition[a:b]):
                classify_run(a, b, "transition", "novelty/energy change on transition bar")
            else:
                classify_run(a, b, "section", "active functional span")

        # Add phrase-level functional spans, but never span a pause.
        for c in phrase_candidates:
            if not c.selected:
                continue
            p_end = int(c.index)
            for length in self.phrase_lengths:
                p_start = p_end - int(length)
                if p_start < 0 or p_end > n_bars:
                    continue
                if np.any(bar_pause[p_start:p_end]):
                    continue
                classify_run(p_start, p_end, "phrase", f"{length}-bar phrase boundary evidence")
                break

        segments.sort(key=lambda s: (s["start_time_s"], s["end_time_s"], s["kind"]))
        return segments

    def _segment_sequence(
        self,
        seg: Dict[str, Any],
        bar_ranges: List[Tuple[int, int]],
        beat_F: np.ndarray,
        beat_sync: np.ndarray,
        beat_volume: np.ndarray,
        beat_melody: np.ndarray,
        beat_novelty: np.ndarray,
        target_len: int = 16,
    ) -> np.ndarray:
        b0 = int(seg["start_beat"])
        b1 = int(seg["end_beat"])
        b0 = max(0, min(len(beat_F), b0))
        b1 = max(b0 + 1, min(len(beat_F), b1))
        if b1 <= b0:
            return np.zeros((target_len, 10), dtype=float)
        x = np.asarray(beat_F[b0:b1], dtype=float)
        seq = np.column_stack([
            x[:, 0],
            x[:, 1],
            x[:, 3],
            x[:, 4],
            x[:, 5],
            x[:, 6],
            x[:, 8],
            np.asarray(beat_sync[b0:b1], dtype=float),
            np.asarray(beat_melody[b0:b1], dtype=float),
            np.asarray(beat_novelty[b0:b1], dtype=float),
        ])
        out = np.zeros((target_len, seq.shape[1]), dtype=float)
        for j in range(seq.shape[1]):
            vals = seq[:, j]
            if j == 8:
                valid = vals > 0
                fill = float(np.median(vals[valid])) if np.any(valid) else 0.0
                vals = np.where(valid, vals, fill)
                vals = (vals - fill) / 24.0
            out[:, j] = self._resample_1d(vals, target_len)
        return out

    def _segment_pair_similarity(self, a: np.ndarray, b: np.ndarray) -> Tuple[int, float]:
        if a.shape != b.shape or a.size == 0:
            return 0, 0.0
        dim_sim = []
        for j in range(a.shape[1]):
            aa = a[:, j].astype(float)
            bb = b[:, j].astype(float)
            scale = max(
                float(np.percentile(np.abs(np.r_[aa, bb]), 75)),
                0.05,
            )
            if j == 8:
                # Melody contour: correlation is more useful than absolute pitch.
                aa = aa - float(np.mean(aa))
                bb = bb - float(np.mean(bb))
                na = float(np.linalg.norm(aa))
                nb = float(np.linalg.norm(bb))
                sim = 0.5 * (1.0 + float(np.dot(aa, bb) / max(na * nb, 1e-9)))
                sim = float(np.clip(sim, 0.0, 1.0))
            else:
                sim = float(np.exp(-float(np.mean(np.abs(aa - bb))) / scale))
            dim_sim.append(sim)
        dim_sim = np.asarray(dim_sim, dtype=float)
        count = int(np.sum(dim_sim >= self.similarity_threshold))
        return count, float(np.mean(dim_sim))

    def _group_segments(
        self,
        segments: List[Dict[str, Any]],
        bar_ranges: List[Tuple[int, int]],
        beat_F: np.ndarray,
        beat_sync: np.ndarray,
        beat_volume: np.ndarray,
        beat_melody: np.ndarray,
        beat_novelty: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Find every direct complete-segment match above the >6/10 rule.

        No best-pair selection is made. Every direct pair with 7..10 matching
        dimensions is retained. A stable color is assigned to each pair for the
        pair-comparison rows. Segments may participate in multiple pairs.
        """
        eligible = [
            s for s in segments
            if s.get("grain") == "bar"
            and s["kind"] in ("phrase", "section", "transition")
            and s["end_beat"] > s["start_beat"]
        ]
        signatures = [
            self._segment_sequence(
                s, bar_ranges, beat_F, beat_sync, beat_volume, beat_melody, beat_novelty
            )
            for s in eligible
        ]

        pairs: List[Dict[str, Any]] = []
        pair_index = 0
        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                count, score = self._segment_pair_similarity(signatures[i], signatures[j])
                if count <= 6 or count < self.similarity_dimensions_required:
                    continue
                color = self.GROUP_COLORS[pair_index % len(self.GROUP_COLORS)]
                a = eligible[i]
                b = eligible[j]
                pairs.append({
                    "pair_id": f"pair_{pair_index:03d}",
                    "segment_a": a["segment_id"],
                    "segment_b": b["segment_id"],
                    "segment_a_kind": a["kind"],
                    "segment_b_kind": b["kind"],
                    "dimensions_similar": int(count),
                    "similarity_score": float(score),
                    "color": color,
                    "start_a_s": float(a["start_time_s"]),
                    "end_a_s": float(a["end_time_s"]),
                    "start_b_s": float(b["start_time_s"]),
                    "end_b_s": float(b["end_time_s"]),
                    "start_a_beat": int(a["start_beat"]),
                    "end_a_beat": int(a["end_beat"]),
                    "start_b_beat": int(b["start_beat"]),
                    "end_b_beat": int(b["end_beat"]),
                    "start_a_bar": int(a["start_bar"]),
                    "end_a_bar": int(a["end_bar"]),
                    "start_b_bar": int(b["start_bar"]),
                    "end_b_bar": int(b["end_bar"]),
                })
                pair_index += 1

        # Keep direct-match evidence on each segment, but never collapse pairs
        # into a single "winner".
        for seg in eligible:
            matches = [
                p for p in pairs
                if p["segment_a"] == seg["segment_id"] or p["segment_b"] == seg["segment_id"]
            ]
            seg["similarity_match_count"] = len(matches)
            seg["similarity_dimensions"] = max(
                [int(p["dimensions_similar"]) for p in matches] or [0]
            )
            seg["similarity_score"] = max(
                [float(p["similarity_score"]) for p in matches] or [0.0]
            )
            seg["similarity_pair_ids"] = [p["pair_id"] for p in matches]

        # Segment grouping is an indicator only. When a segment participates in
        # multiple direct pairs, use its first pair color for the single timeline
        # segment label; the dedicated pair rows retain each pair's own color.
        for seg in eligible:
            match_ids = seg["similarity_pair_ids"]
            if match_ids:
                first_pair = next(p for p in pairs if p["pair_id"] == match_ids[0])
                seg["group_id"] = int(match_ids[0].split("_")[-1])
                seg["group_color"] = str(first_pair["color"])
            else:
                seg["group_id"] = None
                seg["group_color"] = None

        group_payload: List[Dict[str, Any]] = []
        for p in pairs:
            group_payload.append({
                "group_id": p["pair_id"],
                "color": p["color"],
                "member_segment_ids": [p["segment_a"], p["segment_b"]],
                "member_count": 2,
                "rule": f">{self.similarity_dimensions_required - 1}/10 direct segment similarity",
                "dimensions_similar": p["dimensions_similar"],
                "similarity_score": p["similarity_score"],
            })
        return group_payload, pairs

    def _pairwise_bar_beat_similarity(
        self,
        pairs: List[Dict[str, Any]],
        segments: List[Dict[str, Any]],
        beat_F: np.ndarray,
        bar_F: np.ndarray,
        bar_ranges: List[Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        """Compare bars and beats inside every accepted segment pair.

        Both sides use the same global Beat This! grid. Segment-local positions
        are normalized onto each other, which permits phrase/section duration
        differences without changing the authoritative clock.
        """
        by_id = {s["segment_id"]: s for s in segments}
        outputs: List[Dict[str, Any]] = []

        for p in pairs:
            a = by_id[p["segment_a"]]
            b = by_id[p["segment_b"]]

            a_b0, a_b1 = int(a["start_beat"]), int(a["end_beat"])
            b_b0, b_b1 = int(b["start_beat"]), int(b["end_beat"])

            a_bars = [
                k for k, (s0, e0) in enumerate(bar_ranges)
                if s0 >= a_b0 and e0 <= a_b1
            ]
            b_bars = [
                k for k, (s0, e0) in enumerate(bar_ranges)
                if s0 >= b_b0 and e0 <= b_b1
            ]

            a_bar_scores: List[Dict[str, Any]] = []
            for j, ak in enumerate(a_bars):
                if not b_bars:
                    break
                bj = min(
                    len(b_bars) - 1,
                    int(round((j / max(len(a_bars) - 1, 1)) * (len(b_bars) - 1))),
                )
                bk = b_bars[bj]
                score = float(np.clip(1.0 - self._distance(bar_F[ak], bar_F[bk]), 0.0, 1.0))
                a_bar_scores.append({
                    "a_bar_index": int(ak),
                    "b_bar_index": int(bk),
                    "score": score,
                })

            a_beat_scores: List[Dict[str, Any]] = []
            a_beats = list(range(a_b0, min(a_b1, len(beat_F))))
            b_beats = list(range(b_b0, min(b_b1, len(beat_F))))
            for j, ai in enumerate(a_beats):
                if not b_beats:
                    break
                bj = min(
                    len(b_beats) - 1,
                    int(round((j / max(len(a_beats) - 1, 1)) * (len(b_beats) - 1))),
                )
                bi = b_beats[bj]
                score = float(np.clip(1.0 - self._distance(beat_F[ai], beat_F[bi]), 0.0, 1.0))
                a_beat_scores.append({
                    "a_beat_index": int(ai),
                    "b_beat_index": int(bi),
                    "score": score,
                })

            p2 = dict(p)
            p2["bar_matches"] = a_bar_scores
            p2["beat_matches"] = a_beat_scores
            p2["bar_match_mean"] = float(
                np.mean([x["score"] for x in a_bar_scores]) if a_bar_scores else 0.0
            )
            p2["beat_match_mean"] = float(
                np.mean([x["score"] for x in a_beat_scores]) if a_beat_scores else 0.0
            )
            outputs.append(p2)

        return outputs

    def analyze(
        self,
        audio: np.ndarray,
        beat_grid: Any,
        original_audio: np.ndarray | None = None,
        stem_name: str = "stem",
    ) -> Tuple[NoveltyStructureResult, Dict[str, Any]]:
        beats = np.asarray(beat_grid.beats, dtype=float).reshape(-1)
        downbeats = np.asarray(getattr(beat_grid, "downbeats", []), dtype=float).reshape(-1)
        if beats.size < 2:
            raise ValueError("A stable global beat grid is required for novelty analysis.")

        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        F, frame_times, frame_rms, flatness = self._frame_features(x)
        stats = self._stem_stats(x, original_audio, frame_rms, flatness)

        period = float(np.median(np.diff(beats)))
        bar_times, bar_beat_indices = self._bar_map(beats, downbeats)

        beat_F = self._aggregate_to_times(
            F, frame_times, beats, max(period * 0.45, 0.05)
        )
        beat_volume = self._aggregate_scalar(
            frame_rms, frame_times, beats, max(period * 0.45, 0.05)
        )
        beat_silent = beat_volume < (10.0 ** (stats["threshold_dbfs"] / 20.0))
        beat_nov = self._normalize01(
            0.65 * self._local_change(beat_F, 1)
            + 0.35 * self._self_similarity_novelty(beat_F, 2)
        )

        bars: List[np.ndarray] = []
        bar_ranges: List[Tuple[int, int]] = []
        for j, start_beat in enumerate(bar_beat_indices):
            end_beat = (
                bar_beat_indices[j + 1]
                if j + 1 < len(bar_beat_indices)
                else min(len(beats), start_beat + self.beats_per_bar)
            )
            if end_beat > start_beat:
                bars.append(np.mean(beat_F[start_beat:end_beat], axis=0))
                bar_ranges.append((start_beat, end_beat))
        bar_F = np.asarray(bars, dtype=float)
        bar_nov = self._normalize01(
            0.55 * self._local_change(bar_F, 1)
            + 0.45 * self._self_similarity_novelty(bar_F, 1)
        )
        beat_sync = self._syncopation(x, beats, period)
        beat_melody = self._melody_track(
            x,
            beats,
            period,
            beat_volume,
            10.0 ** (stats["threshold_dbfs"] / 20.0),
        )

        phrase_nov = np.zeros(max(len(bar_F), 1), dtype=float)
        phrase_spans: List[Dict[str, Any]] = []
        for length in self.phrase_lengths:
            if len(bar_F) < 2 * length:
                continue
            for b in range(length, len(bar_F) - length + 1, length):
                left = np.mean(bar_F[b - length:b], axis=0)
                right = np.mean(bar_F[b:b + length], axis=0)
                score = self._distance(left, right)
                phrase_nov[b] = max(phrase_nov[b], score)
                phrase_spans.append({
                    "length_bars": length,
                    "start_bar": b - length,
                    "end_bar": b,
                    "boundary_time_s": float(bar_times[min(b, len(bar_times) - 1)]),
                    "novelty_score": float(score),
                    "repetition_score": float(np.clip(1.0 - score, 0.0, 1.0)),
                })
        phrase_nov = self._normalize01(phrase_nov)

        combined_section = self._normalize01(
            0.55 * bar_nov + 0.45 * phrase_nov[:len(bar_nov)]
        )
        section_times = np.asarray(bar_times[:len(combined_section)], dtype=float)
        raw_sections = self._peak_candidates(
            combined_section,
            section_times,
            "section",
            self.section_threshold,
            refractory=self.section_refractory_bars,
        )
        selected_sections: List[NoveltyCandidate] = []
        last_bar = -10**9
        for c in raw_sections:
            if c.index - last_bar < self.min_section_bars:
                c.selected = False
                c.reason = "minimum section duration constraint"
                continue
            selected_sections.append(c)
            last_bar = c.index

        phrase_candidates = self._peak_candidates(
            phrase_nov,
            np.asarray(bar_times[:len(phrase_nov)], dtype=float),
            "phrase",
            0.34,
            refractory=1,
        )
        bar_candidates = self._peak_candidates(
            bar_nov,
            np.asarray(bar_times[:len(bar_nov)], dtype=float),
            "bar",
            0.30,
            refractory=1,
        )
        beat_candidates = self._peak_candidates(
            beat_nov,
            beats[:len(beat_nov)],
            "beat",
            0.40,
            refractory=1,
        )

        for c in phrase_candidates:
            c.reason = "phrase-scale novelty peak aligned to a bar boundary"
        candidates = [
            asdict(c) for c in beat_candidates + bar_candidates + phrase_candidates + raw_sections
        ]
        for c in raw_sections:
            if not c.selected:
                for row in candidates:
                    if row["level"] == c.level and row["index"] == c.index:
                        row["selected"] = False
                        row["reason"] = c.reason

        section_bounds = [int(c.index) for c in selected_sections]
        functional_segments = self._build_functional_segments(
            beats,
            bar_times,
            bar_ranges,
            beat_silent,
            beat_nov >= self.transition_threshold,
            section_bounds,
            phrase_candidates,
            frame_times,
            self._frame_activity_mask(frame_rms, stats["threshold_dbfs"]),
            period,
        )
        segment_groups, similarity_pairs = self._group_segments(
            functional_segments,
            bar_ranges,
            beat_F,
            beat_sync,
            beat_volume,
            beat_melody,
            beat_nov,
        )
        similarity_pairs = self._pairwise_bar_beat_similarity(
            similarity_pairs,
            functional_segments,
            beat_F,
            bar_F,
            bar_ranges,
        )

        # Backward-compatible aggregate similarity arrays: maximum direct-pair
        # score at each global bar/beat position. The plot itself uses the full
        # pair-specific data and never selects a winner.
        beat_sim = np.zeros(len(beat_F), dtype=float)
        bar_sim = np.zeros(len(bar_F), dtype=float)
        for pair in similarity_pairs:
            for match in pair.get("beat_matches", []):
                idx = int(match["a_beat_index"])
                if 0 <= idx < len(beat_sim):
                    beat_sim[idx] = max(beat_sim[idx], float(match["score"]))
            for match in pair.get("bar_matches", []):
                idx = int(match["a_bar_index"])
                if 0 <= idx < len(bar_sim):
                    bar_sim[idx] = max(bar_sim[idx], float(match["score"]))

        result = NoveltyStructureResult(
            beat_novelty=beat_nov.tolist(),
            bar_novelty=bar_nov.tolist(),
            phrase_novelty=phrase_nov.tolist(),
            section_novelty=combined_section.tolist(),
            beat_times=beats.tolist(),
            bar_boundaries=bar_times.tolist(),
            phrase_boundaries=[c.time_s for c in phrase_candidates if c.selected],
            section_boundaries=[c.time_s for c in selected_sections],
            phrase_spans=phrase_spans,
            section_spans=[
                {
                    "section_index": i,
                    "start_bar": int(c.index),
                    "boundary_time_s": float(c.time_s),
                    "novelty_score": float(c.score),
                }
                for i, c in enumerate(selected_sections)
            ],
            functional_segments=functional_segments,
            segment_groups=segment_groups,
            similarity_pairs=similarity_pairs,
            candidates=candidates,
            beat_volume_pct_original=(
                (100.0 * beat_volume / max(stats["original_rms"], 1e-12)).tolist()
            ),
            bar_volume_pct_original=(
                (
                    100.0 * np.asarray([
                        float(np.sqrt(np.mean(beat_volume[s:e] ** 2))) if e > s else 0.0
                        for s, e in bar_ranges
                    ]) / max(stats["original_rms"], 1e-12)
                ).tolist()
            ),
            beat_syncopation=beat_sync.tolist(),
            melody_midi=beat_melody.tolist(),
            beat_similarity=beat_sim.tolist(),
            bar_similarity=bar_sim.tolist(),
            stem_stats=stats,
        )
        plot_data = {
            "frame_times": frame_times,
            "frame_rms": frame_rms,
            "frame_active": self._frame_activity_mask(frame_rms, stats["threshold_dbfs"]),
            "fft_db": self._fft_db(x, frame_times, normalize_peak=True),
            "beat_novelty": beat_nov,
            "bar_novelty": bar_nov,
            "phrase_novelty": phrase_nov,
            "section_novelty": combined_section,
            "beat_times": beats,
            "bar_times": bar_times,
            "phrase_candidates": phrase_candidates,
            "section_candidates": selected_sections,
            "similarity_pairs": similarity_pairs,
            "beat_volume_pct_original": np.asarray(result.beat_volume_pct_original, dtype=float),
            "bar_volume_pct_original": np.asarray(result.bar_volume_pct_original, dtype=float),
            "beat_syncopation": beat_sync,
            "melody_midi": beat_melody,
            "functional_segments": functional_segments,
            "segment_groups": segment_groups,
            "duration_s": float(len(x) / self.sample_rate),
            "stem_stats": stats,
            "skip_plot": not stats["has_audible_content"],
            "skip_reason": (
                "empty stem"
                if stats["empty"]
                else "inaudible/noise-only stem"
                if not stats["has_audible_content"]
                else ""
            ),
        }
        return result, plot_data

    def _fft_db(self, audio: np.ndarray, frame_times: np.ndarray, normalize_peak: bool = True) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if normalize_peak:
            peak = float(np.max(np.abs(x))) if x.size else 0.0
            if peak > 1e-9:
                x = x * (0.90 / peak)
        if frame_times.size == 0:
            return np.zeros((1, 1), dtype=float)
        cols = []
        win = np.hanning(self.fft_size)
        max_db = -120.0
        for t in frame_times:
            center = int(round(float(t) * self.sample_rate))
            start = max(0, center - self.fft_size // 2)
            frame = x[start:start + self.fft_size]
            if frame.size < self.fft_size:
                frame = np.pad(frame, (0, self.fft_size - frame.size))
            mag = np.abs(np.fft.rfft(frame * win)) + 1e-8
            db = 20.0 * np.log10(mag)
            max_db = max(max_db, float(np.max(db)))
            cols.append(db)
        spec = np.asarray(cols, dtype=float).T
        return np.clip(spec, max_db - 80.0, max_db)

    @classmethod
    def _segment_color(cls, seg: Dict[str, Any]) -> str:
        return str(seg.get("group_color") or cls.PHRASE_COLOR)

    def _plot_boundary_overlay(self, ax: Any, plot_data: Dict[str, Any], include_beat: bool = False) -> None:
        if include_beat:
            for t in np.asarray(plot_data["beat_times"], dtype=float):
                ax.axvline(float(t), color=self.BEAT_COLOR, alpha=0.10, linewidth=0.35)
        for t in np.asarray(plot_data["bar_times"], dtype=float):
            ax.axvline(float(t), color=self.BAR_COLOR, alpha=0.20, linewidth=0.65)

        for seg in plot_data["functional_segments"]:
            if seg["kind"] not in ("phrase", "section"):
                continue
            color = self._segment_color(seg)
            ax.axvline(float(seg["start_time_s"]), color=color, alpha=0.34, linewidth=0.95)
            ax.axvline(float(seg["end_time_s"]), color=color, alpha=0.24, linewidth=0.75)

    def plot(self, plot_data: Dict[str, Any], output_path: str, title: str) -> None:
        if plot_data.get("skip_plot"):
            return

        duration = float(plot_data["duration_s"])
        pair_list = list(plot_data.get("similarity_pairs", []))

        # Two dedicated rows per accepted pair: bars and beats. This avoids
        # overplotting when several different segment pairs are >6/10 similar.
        # All pair boundary lines are also drawn through the full stack.
        n_pair_rows = max(1, 2 * len(pair_list))
        height_pairs = 0.68 * n_pair_rows
        row_heights = [4.4, 1.55] + [0.78] * n_pair_rows + [1.15, 1.05, 1.05, 1.30]
        fig_h = max(10.0, 4.8 + sum(row_heights))
        fig, axes = plt.subplots(
            len(row_heights),
            1,
            figsize=(19, fig_h),
            sharex=True,
            gridspec_kw={"height_ratios": row_heights, "hspace": 0.0},
        )
        axes = np.atleast_1d(axes)
        ax_fft = axes[0]
        ax_nov = axes[1]
        pair_axes = axes[2:2 + n_pair_rows]
        ax_offset = axes[2 + n_pair_rows]
        ax_vol = axes[3 + n_pair_rows]
        ax_sync = axes[4 + n_pair_rows]
        ax_melody = axes[5 + n_pair_rows]

        frame_times = np.asarray(plot_data["frame_times"], dtype=float)
        fft_db = np.asarray(plot_data["fft_db"], dtype=float)
        freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)
        ax_fft.pcolormesh(
            frame_times,
            freqs,
            fft_db,
            shading="auto",
            cmap="magma",
            vmin=float(np.max(fft_db) - 80.0),
            vmax=float(np.max(fft_db)),
        )
        ax_fft.set_yscale("log")
        ax_fft.set_ylim(20, min(self.sample_rate / 2.0, 16000.0))
        ax_fft.set_ylabel("FFT / Hz")

        stats = plot_data["stem_stats"]
        norm_db = self._db(0.90 / max(stats["stem_peak"], 1e-12))
        ax_fft.set_title(
            f"{title} | stem volume {stats['volume_pct_of_original_rms']:.1f}% of original RMS "
            f"| plot normalized (gain {norm_db:+.1f} dB)",
            loc="left",
            fontsize=12,
            fontweight="bold",
        )

        t_bar = np.asarray(plot_data["bar_times"], dtype=float)
        t_beat = np.asarray(plot_data["beat_times"], dtype=float)

        # Novelty stays before all similarity/offset/volume rows.
        beat_nov = np.asarray(plot_data["beat_novelty"], dtype=float)
        bar_nov = np.asarray(plot_data["bar_novelty"], dtype=float)
        phrase_nov = np.asarray(plot_data["phrase_novelty"], dtype=float)
        section_nov = np.asarray(plot_data["section_novelty"], dtype=float)

        if beat_nov.size:
            cmap = plt.get_cmap("RdYlGn_r")
            for i in range(min(len(t_beat) - 1, len(beat_nov))):
                ax_nov.plot(
                    t_beat[i:i + 2],
                    beat_nov[i:i + 2],
                    color=cmap(float(beat_nov[i])),
                    linewidth=2.0,
                    solid_capstyle="round",
                )
        if bar_nov.size:
            ax_nov.plot(t_bar[:len(bar_nov)], bar_nov, color="#666666", linewidth=0.9, alpha=0.55)
        if phrase_nov.size:
            ax_nov.plot(t_bar[:len(phrase_nov)], phrase_nov, color=self.PHRASE_COLOR, linewidth=1.0, alpha=0.60)
        if section_nov.size:
            ax_nov.plot(t_bar[:len(section_nov)], section_nov, color=self.SECTION_COLOR, linewidth=1.2, alpha=0.72)

        # Keep the original structural annotation style on the novelty row.
        for idx, cand in enumerate(plot_data.get("phrase_candidates", [])):
            if not cand.selected:
                continue
            t = float(cand.time_s)
            ax_nov.axvline(t, color=self.PHRASE_COLOR, alpha=0.72, linewidth=1.5)
            ax_nov.text(
                t,
                0.92,
                f"PHRASE {idx + 1}",
                transform=ax_nov.get_xaxis_transform(),
                color=self.PHRASE_COLOR,
                rotation=90,
                va="top",
                ha="right",
                fontsize=7.5,
                fontweight="bold",
            )
        for idx, cand in enumerate(plot_data.get("section_candidates", [])):
            t = float(cand.time_s)
            ax_nov.axvline(t, color=self.SECTION_COLOR, alpha=0.82, linewidth=2.4)
            ax_nov.text(
                t,
                0.98,
                f"SECTION {idx + 1}",
                transform=ax_nov.get_xaxis_transform(),
                color=self.SECTION_COLOR,
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                fontweight="bold",
            )

        ax_nov.set_ylim(-0.02, 1.02)
        ax_nov.set_ylabel("NOVELTY")
        ax_nov.text(
            0.002,
            0.76,
            "green → red = beat novelty",
            transform=ax_nov.transAxes,
            fontsize=7,
            color="#555555",
        )

        # Grid / structural lines. The same global Beat This! grid is used for
        # every stem and every comparison.
        for t in t_beat:
            ax_fft.axvline(float(t), color=self.BEAT_COLOR, alpha=0.09, linewidth=0.3)
            ax_nov.axvline(float(t), color=self.BEAT_COLOR, alpha=0.08, linewidth=0.3)
        for t in t_bar:
            ax_fft.axvline(float(t), color=self.BAR_COLOR, alpha=0.18, linewidth=0.55)
            ax_nov.axvline(float(t), color=self.BAR_COLOR, alpha=0.16, linewidth=0.55)

        # All functional segment boundaries extend through the whole plot.
        # Matched segments use their pair color; unmatched phrase/section/transition
        # spans use the regular structural colors.
        for seg in plot_data["functional_segments"]:
            if seg.get("grain") != "bar":
                continue
            if seg["kind"] == "pause":
                continue
            color = self._segment_color(seg)
            if seg["kind"] == "transition":
                color = self.TRANSITION_COLOR
            for t in (float(seg["start_time_s"]), float(seg["end_time_s"])):
                for ax in axes:
                    ax.axvline(t, color=color, alpha=0.16, linewidth=0.65)

        # Matched pair boundaries are stronger and use the exact same pair color
        # on both member segments.
        for p in pair_list:
            color = str(p["color"])
            times = (
                float(p["start_a_s"]), float(p["end_a_s"]),
                float(p["start_b_s"]), float(p["end_b_s"]),
            )
            for t in times:
                for ax in axes:
                    ax.axvline(t, color=color, alpha=0.28, linewidth=0.9)

        # Similarity rows: one BAR row + one BEAT row for every direct >6/10
        # complete-segment pair. No pair is selected over another.
        if pair_list:
            for pair_index, p in enumerate(pair_list):
                ax_b = pair_axes[2 * pair_index]
                ax_bt = pair_axes[2 * pair_index + 1]
                color = str(p["color"])
                label = (
                    f"{p['pair_id']}  {p['segment_a']} ↔ {p['segment_b']}  "
                    f"{p['dimensions_similar']}/10"
                )

                for ax, row_label in ((ax_b, "BAR"), (ax_bt, "BEAT")):
                    ax.text(
                        0.002,
                        0.80,
                        label,
                        transform=ax.transAxes,
                        color=color,
                        fontsize=7.5,
                        fontweight="bold",
                        va="top",
                    )
                    ax.set_yticks([])

                # Mark the paired segment extents in the shared pair color.
                for ax in (ax_b, ax_bt):
                    ax.axvspan(p["start_a_s"], p["end_a_s"], color=color, alpha=0.08)
                    ax.axvspan(p["start_b_s"], p["end_b_s"], color=color, alpha=0.08)
                    ax.axvline(p["start_a_s"], color=color, linewidth=2.0, alpha=0.85)
                    ax.axvline(p["end_a_s"], color=color, linewidth=1.1, alpha=0.65)
                    ax.axvline(p["start_b_s"], color=color, linewidth=2.0, alpha=0.85)
                    ax.axvline(p["end_b_s"], color=color, linewidth=1.1, alpha=0.65)

                for m in p.get("bar_matches", []):
                    ai = int(m["a_bar_index"])
                    bi = int(m["b_bar_index"])
                    sa, ea = self._bar_times_from_index(ai, t_bar, bar_ranges=None)
                    sb, eb = self._bar_times_from_index(bi, t_bar, bar_ranges=None)
                    # A connector is intentionally not drawn across the full
                    # plot; the paired extents and score bar encode the mapping.
                    ax_b.plot(
                        [sa, sb],
                        [0.72, 0.72],
                        color=color,
                        linewidth=2.5,
                        alpha=float(0.25 + 0.70 * m["score"]),
                    )
                    ax_b.scatter([sa, sb], [0.72, 0.72], color=color, s=10)

                for m in p.get("beat_matches", []):
                    ai = int(m["a_beat_index"])
                    bi = int(m["b_beat_index"])
                    if ai >= len(t_beat) or bi >= len(t_beat):
                        continue
                    sa, sb = float(t_beat[ai]), float(t_beat[bi])
                    ax_bt.plot(
                        [sa, sb],
                        [0.65, 0.65],
                        color=color,
                        linewidth=2.0,
                        alpha=float(0.25 + 0.70 * m["score"]),
                    )
                    ax_bt.scatter([sa, sb], [0.65, 0.65], color=color, s=8)

                ax_b.set_ylim(0.0, 1.0)
                ax_bt.set_ylim(0.0, 1.0)
                ax_b.set_ylabel("BAR")
                ax_bt.set_ylabel("BEAT")
        else:
            pair_axes[0].text(
                0.5,
                0.5,
                "no complete functional segment pairs >6/10",
                transform=pair_axes[0].transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#666666",
            )
            pair_axes[0].set_yticks([])
            pair_axes[0].set_ylabel("SIM")
            for ax in pair_axes:
                ax.set_ylim(0.0, 1.0)
                ax.set_yticks([])

        # Early/late offsets come immediately after similarity rows.
        ax_offset.axhline(0.0, color="#666666", linewidth=0.7)
        for seg in plot_data["functional_segments"]:
            if seg.get("grain") != "bar" or seg["kind"] not in ("phrase", "section"):
                continue
            c = self._segment_color(seg)
            tmid = 0.5 * (seg["start_time_s"] + seg["end_time_s"])
            ax_offset.plot(
                [tmid, tmid],
                [seg["start_offset_beats_from_bar"], seg["end_offset_beats_from_bar"]],
                color=c,
                linewidth=2.0,
            )
            ax_offset.scatter([tmid], [seg["start_offset_beats_from_bar"]], marker="^", color=c, s=20, zorder=4)
            ax_offset.scatter([tmid], [seg["end_offset_beats_from_bar"]], marker="v", color=c, s=20, zorder=4)
        ax_offset.set_ylim(-1.5, 1.5)
        ax_offset.set_yticks([-1.0, 0.0, 1.0])
        ax_offset.set_ylabel("EARLY/LATE")
        ax_offset.set_yticklabels(["-1 beat", "on bar", "+1 beat"])

        beat_vol = np.asarray(plot_data["beat_volume_pct_original"], dtype=float)
        if beat_vol.size:
            ax_vol.plot(t_beat[:len(beat_vol)], beat_vol, color="#222222", linewidth=1.25)
            ax_vol.fill_between(t_beat[:len(beat_vol)], 0.0, beat_vol, color="#aaaaaa", alpha=0.16)
        ax_vol.set_ylim(0.0, max(1.0, float(np.max(beat_vol) * 1.10) if beat_vol.size else 1.0))
        ax_vol.set_ylabel("VOL/BEAT")
        ax_vol.text(0.002, 0.80, "% original RMS", transform=ax_vol.transAxes, fontsize=7, color="#555555")

        sync = np.asarray(plot_data["beat_syncopation"], dtype=float)
        if sync.size:
            ax_sync.plot(t_beat[:len(sync)], sync, color="#7a3e00", linewidth=1.4)
            ax_sync.fill_between(t_beat[:len(sync)], 0.0, sync, color="#d95f02", alpha=0.14)
        ax_sync.axhline(0.60, color="#7a3e00", alpha=0.45, linestyle="--", linewidth=0.75)
        ax_sync.set_ylim(0.0, 1.0)
        ax_sync.set_ylabel("SYNC")

        melody = np.asarray(plot_data["melody_midi"], dtype=float)
        voiced = melody > 0.0
        if melody.size and np.any(voiced):
            ax_melody.plot(
                t_beat[voiced],
                melody[voiced],
                color="#2c7fb8",
                linewidth=1.8,
                marker=".",
                markersize=3,
            )
            ax_melody.set_ylim(
                max(20.0, float(np.min(melody[voiced]) - 3.0)),
                min(120.0, float(np.max(melody[voiced]) + 3.0)),
            )
        else:
            ax_melody.text(
                0.5,
                0.5,
                "no stable pitch evidence",
                transform=ax_melody.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#666666",
            )
            ax_melody.set_ylim(30.0, 90.0)
        ax_melody.set_ylabel("MELODY / MIDI")

        for ax in axes:
            ax.grid(axis="y", alpha=0.12, linewidth=0.5)
            ax.tick_params(axis="x", labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax_melody.set_xlabel("Time (s)")

        fig.subplots_adjust(left=0.055, right=0.995, top=0.965, bottom=0.055, hspace=0.0)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _bar_times_from_index(index: int, bar_times: np.ndarray, bar_ranges: Any = None) -> Tuple[float, float]:
        i = int(index)
        if i < 0 or i >= len(bar_times):
            return 0.0, 0.0
        start = float(bar_times[i])
        if i + 1 < len(bar_times):
            end = float(bar_times[i + 1])
        else:
            end = start
        return start, end

    def run_and_export(
        self,
        audio: np.ndarray,
        beat_grid: Any,
        output_path: str,
        json_path: str,
        stem_name: str,
        original_audio: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        result, plot_data = self.analyze(
            audio,
            beat_grid,
            original_audio=original_audio,
            stem_name=stem_name,
        )
        payload = result.as_dict()
        payload["stem"] = str(stem_name)
        payload["global_beat_grid_source"] = str(getattr(beat_grid, "source", "unknown"))
        payload["global_tempo_bpm"] = float(getattr(beat_grid, "tempo_bpm", 0.0))
        payload["beats_per_bar"] = int(self.beats_per_bar)
        payload["plot_written"] = False
        payload["plot_skipped"] = bool(plot_data.get("skip_plot"))
        payload["plot_skip_reason"] = str(plot_data.get("skip_reason", ""))
        payload["object_detection"] = "not used by novelty analyzer"
        payload["notes"] = [
            "Beat and bar positions are locked to the global Beat This! grid.",
            "Functional similarity compares complete non-silent segments using 10 beat-sequence dimensions.",
            "Every direct pair with 7/10 through 10/10 matching dimensions is retained; no best-pair selection is performed.",
            "Pause/silence segments are excluded from similarity matching.",
            "Bar and beat comparisons use the same global Beat This! grid and normalized within-segment positions.",
            "Each accepted pair has its own color and dedicated BAR/BEAT rows; boundary lines use the same pair color through the full plot stack.",
            "Pair rows are repeated as needed so different accepted pairs do not visually overwrite one another.",
            "Plot FFT is peak-normalized for readability; the title reports stem RMS as a percentage of original-mix RMS.",
        ]
        if not plot_data.get("skip_plot"):
            self.plot(plot_data, output_path, f"{stem_name}")
            payload["plot_written"] = True
            payload["plot_path"] = str(output_path)
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload
