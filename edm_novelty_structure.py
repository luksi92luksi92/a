#!/usr/bin/env python3
"""Beat-synchronous novelty analysis for EDM structure discovery.

This module is deliberately separate from the existing PatternEngine/
StructureEngine. It produces transparent beat/bar/phrase/section evidence and
plots which can be consumed by the existing structure layer later.

Authoritative clock:
    Beat This! global BeatGrid

Feature timeline:
    waveform -> frame features -> global beats -> bars -> phrases -> sections

Plot:
    top    = FFT/spectral magnitude view
    bottom = novelty evidence
    hspace = 0

Novelty colors run green -> red. Beat/bar/phrase/section boundaries are
explicitly annotated; phrase and section labels use the same color as their
boundary lines.
"""
from __future__ import annotations

import json
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
    candidates: List[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BeatBarPhraseSectionNovelty:
    """Compute multi-scale novelty on a fixed global beat grid."""

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
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norms, 1e-9)

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        n = min(a.size, b.size)
        aa, bb = a[:n], b[:n]
        na = float(np.linalg.norm(aa))
        nb = float(np.linalg.norm(bb))
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        cosine = float(np.clip(np.dot(aa, bb) / (na * nb), -1.0, 1.0))
        return 0.5 * (1.0 - cosine)

    def _frame_features(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if x.size < self.fft_size:
            x = np.pad(x, (0, self.fft_size - x.size))
        win = np.hanning(self.fft_size).astype(np.float32)
        times: List[float] = []
        feats: List[np.ndarray] = []
        prev_mag = None
        freq = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)
        bands = np.array([[20, 90], [90, 250], [250, 1000], [1000, 4000], [4000, 10000]], dtype=float)
        for start in range(0, max(1, x.size - self.fft_size + 1), self.hop_size):
            frame = x[start:start + self.fft_size]
            if frame.size < self.fft_size:
                frame = np.pad(frame, (0, self.fft_size - frame.size))
            spec = np.fft.rfft(frame * win)
            mag = np.abs(spec).astype(np.float64)
            power = mag * mag + 1e-12
            total = float(np.sum(power))
            centroid = float(np.sum(freq * power) / total) if total > 0 else 0.0
            spread = float(np.sqrt(np.sum(((freq - centroid) ** 2) * power) / total)) if total > 0 else 0.0
            flatness = float(np.exp(np.mean(np.log(mag + 1e-9))) / max(np.mean(mag), 1e-9))
            flux = 0.0
            if prev_mag is not None:
                flux = float(np.linalg.norm(np.maximum(mag - prev_mag, 0.0)) / (np.linalg.norm(prev_mag) + 1e-9))
            prev_mag = mag
            band_energy = []
            for lo, hi in bands:
                m = (freq >= lo) & (freq < hi)
                band_energy.append(float(np.sum(power[m])) / total if total > 0 else 0.0)
            log_rms = float(np.log1p(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) + 1e-9))
            feat = np.asarray([centroid / 10000.0, spread / 10000.0, flatness, flux, log_rms, *band_energy], dtype=float)
            feats.append(feat)
            times.append(float(start + self.fft_size * 0.5) / self.sample_rate)
        if not feats:
            return np.zeros((0, 10), dtype=float), np.zeros(0, dtype=float)
        F = np.asarray(feats, dtype=float)
        F[:, :2] = self._zscore(F[:, :2])
        F[:, 3:5] = self._zscore(F[:, 3:5])
        F[:, 5:] = self._row_normalize(F[:, 5:])
        return F, np.asarray(times, dtype=float)

    @staticmethod
    def _aggregate_to_times(features: np.ndarray, frame_times: np.ndarray, centers: np.ndarray, half_window: float) -> np.ndarray:
        rows = []
        for t in centers:
            idx = np.where(np.abs(frame_times - float(t)) <= half_window)[0]
            if idx.size == 0:
                j = int(np.argmin(np.abs(frame_times - float(t)))) if frame_times.size else 0
                rows.append(features[j] if frame_times.size else np.zeros(features.shape[1] if features.ndim == 2 else 1))
            else:
                rows.append(np.mean(features[idx], axis=0))
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
    def _peak_candidates(values: np.ndarray, times: np.ndarray, level: str, threshold: float, refractory: int = 1) -> List[NoveltyCandidate]:
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
            found.append(NoveltyCandidate(level, i, float(times[i]), float(norm[i]), prominence, confidence, True, "local novelty peak"))
            last = i
        return found

    def _bar_map(self, beat_times: np.ndarray, downbeats: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        if downbeats.size >= 2:
            starts = np.asarray(downbeats, dtype=float)
            beat_indices = [int(np.argmin(np.abs(beat_times - t))) for t in starts]
            beat_indices = sorted(set(max(0, min(len(beat_times) - 1, i)) for i in beat_indices))
            if len(beat_indices) >= 2:
                return starts, beat_indices
        starts = beat_times[::self.beats_per_bar]
        return starts, list(range(0, len(beat_times), self.beats_per_bar))

    def analyze(self, audio: np.ndarray, beat_grid: Any) -> Tuple[NoveltyStructureResult, Dict[str, Any]]:
        beats = np.asarray(beat_grid.beats, dtype=float).reshape(-1)
        downbeats = np.asarray(getattr(beat_grid, "downbeats", []), dtype=float).reshape(-1)
        if beats.size < 2:
            raise ValueError("A stable global beat grid is required for novelty analysis.")

        F, frame_times = self._frame_features(audio)
        period = float(np.median(np.diff(beats)))
        bar_starts, bar_beat_indices = self._bar_map(beats, downbeats)
        bar_times = bar_starts

        beat_F = self._aggregate_to_times(F, frame_times, beats, max(period * 0.45, 0.05))
        beat_nov = self._normalize01(0.65 * self._local_change(beat_F, 1) + 0.35 * self._self_similarity_novelty(beat_F, 2))

        bars: List[np.ndarray] = []
        valid_bar_ranges: List[Tuple[int, int]] = []
        for j, start_beat in enumerate(bar_beat_indices):
            end_beat = bar_beat_indices[j + 1] if j + 1 < len(bar_beat_indices) else min(len(beats), start_beat + self.beats_per_bar)
            if end_beat > start_beat:
                bars.append(np.mean(beat_F[start_beat:end_beat], axis=0))
                valid_bar_ranges.append((start_beat, end_beat))
        bar_F = np.asarray(bars, dtype=float)
        bar_nov = self._normalize01(0.55 * self._local_change(bar_F, 1) + 0.45 * self._self_similarity_novelty(bar_F, 1))

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

        combined_section = self._normalize01(0.55 * bar_nov + 0.45 * phrase_nov[:len(bar_nov)])
        section_times = np.asarray(bar_times[:len(combined_section)], dtype=float)
        min_sep = self.section_refractory_bars
        raw_sections = self._peak_candidates(
            combined_section,
            section_times,
            "section",
            self.section_threshold,
            refractory=min_sep,
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

        section_spans: List[Dict[str, Any]] = []
        boundaries = [0] + [c.index for c in selected_sections]
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(bar_F)
            if end - start < self.min_section_bars and i > 0:
                continue
            section_spans.append({
                "section_index": i,
                "start_bar": int(start),
                "end_bar": int(end),
                "start_time_s": float(bar_times[min(start, len(bar_times) - 1)]),
                "end_time_s": float(bar_times[min(max(end, start), len(bar_times) - 1)]),
                "novelty_score": float(combined_section[start]) if start < len(combined_section) else 0.0,
                "evidence": {
                    "bar_novelty": float(bar_nov[start]) if start < len(bar_nov) else 0.0,
                    "phrase_novelty": float(phrase_nov[start]) if start < len(phrase_nov) else 0.0,
                },
            })

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
        candidates = [asdict(c) for c in beat_candidates + bar_candidates + phrase_candidates + raw_sections]
        for c in raw_sections:
            if not c.selected:
                for row in candidates:
                    if row["level"] == c.level and row["index"] == c.index and abs(row["time_s"] - c.time_s) < 1e-9:
                        row["selected"] = False
                        row["reason"] = c.reason

        result = NoveltyStructureResult(
            beat_novelty=beat_nov.tolist(),
            bar_novelty=bar_nov.tolist(),
            phrase_novelty=phrase_nov.tolist(),
            section_novelty=combined_section.tolist(),
            beat_times=beats.tolist(),
            bar_boundaries=bar_times.tolist(),
            phrase_boundaries=[c.time_s for c in phrase_candidates],
            section_boundaries=[c.time_s for c in selected_sections],
            phrase_spans=phrase_spans,
            section_spans=section_spans,
            candidates=candidates,
        )
        plot_data = {
            "frame_times": frame_times,
            "fft_db": self._fft_db(audio, frame_times),
            "beat_novelty": beat_nov,
            "bar_novelty": bar_nov,
            "phrase_novelty": phrase_nov,
            "section_novelty": combined_section,
            "beat_times": beats,
            "bar_times": bar_times,
            "phrase_candidates": phrase_candidates,
            "section_candidates": selected_sections,
            "duration_s": float(len(np.asarray(audio).reshape(-1)) / self.sample_rate),
        }
        return result, plot_data

    def _fft_db(self, audio: np.ndarray, frame_times: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
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

    @staticmethod
    def _boundary_color(level: str) -> Any:
        return {
            "beat": "#888888",
            "bar": "#444444",
            "phrase": "#1976d2",
            "section": "#8e24aa",
        }.get(level, "#444444")

    def plot(self, plot_data: Dict[str, Any], output_path: str, title: str) -> None:
        duration = float(plot_data["duration_s"])
        fig, (ax_fft, ax_nov) = plt.subplots(
            2,
            1,
            figsize=(18, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.25], "hspace": 0.0},
        )
        frame_times = np.asarray(plot_data["frame_times"], dtype=float)
        fft_db = np.asarray(plot_data["fft_db"], dtype=float)
        freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)
        ax_fft.pcolormesh(frame_times, freqs, fft_db, shading="auto", cmap="magma", vmin=float(np.max(fft_db) - 80.0), vmax=float(np.max(fft_db)))
        ax_fft.set_yscale("log")
        ax_fft.set_ylim(20, min(self.sample_rate / 2.0, 16000.0))
        ax_fft.set_ylabel("Hz")
        ax_fft.set_title(title)

        t_bar = np.asarray(plot_data["bar_times"], dtype=float)
        t_beat = np.asarray(plot_data["beat_times"], dtype=float)
        beat_nov = np.asarray(plot_data["beat_novelty"], dtype=float)
        bar_nov = np.asarray(plot_data["bar_novelty"], dtype=float)
        phrase_nov = np.asarray(plot_data["phrase_novelty"], dtype=float)
        section_nov = np.asarray(plot_data["section_novelty"], dtype=float)

        # Green -> red novelty signal for the primary combined curve.
        tt = t_beat[:len(beat_nov)]
        cmap = plt.get_cmap("RdYlGn_r")
        if len(tt) >= 2:
            for i in range(len(tt) - 1):
                v = float(beat_nov[i])
                ax_nov.plot(tt[i:i + 2], beat_nov[i:i + 2], color=cmap(v), linewidth=2.0, solid_capstyle="round")
        else:
            ax_nov.plot(tt, beat_nov, color="green", linewidth=2.0)

        if len(t_bar) and len(bar_nov):
            ax_nov.plot(t_bar[:len(bar_nov)], bar_nov, color="#666666", linewidth=1.0, alpha=0.55, label="bar novelty")
        if len(t_bar) and len(phrase_nov):
            ax_nov.plot(t_bar[:len(phrase_nov)], phrase_nov, color="#1976d2", linewidth=1.1, alpha=0.65, label="phrase novelty")
        if len(t_bar) and len(section_nov):
            ax_nov.plot(t_bar[:len(section_nov)], section_nov, color="#8e24aa", linewidth=1.4, alpha=0.75, label="section evidence")

        # Fine beat grid; bar grid; named phrase/section boundaries.
        for i, t in enumerate(t_beat):
            ax_fft.axvline(float(t), color="#aaaaaa", alpha=0.14, linewidth=0.35)
            ax_nov.axvline(float(t), color="#aaaaaa", alpha=0.14, linewidth=0.35)
        for i, t in enumerate(t_bar):
            ax_fft.axvline(float(t), color=self._boundary_color("bar"), alpha=0.35, linewidth=0.8)
            ax_nov.axvline(float(t), color=self._boundary_color("bar"), alpha=0.35, linewidth=0.8)

        for idx, c in enumerate(plot_data["phrase_candidates"]):
            if not c.selected:
                continue
            color = self._boundary_color("phrase")
            t = float(c.time_s)
            for ax in (ax_fft, ax_nov):
                ax.axvline(t, color=color, alpha=0.9, linewidth=2.0)
            ax_nov.text(t, 0.92, f"PHRASE {idx + 1}", transform=ax_nov.get_xaxis_transform(), color=color, rotation=90, va="top", ha="right", fontsize=8, fontweight="bold")

        for idx, c in enumerate(plot_data["section_candidates"]):
            color = self._boundary_color("section")
            t = float(c.time_s)
            for ax in (ax_fft, ax_nov):
                ax.axvline(t, color=color, alpha=0.95, linewidth=3.0)
            ax_nov.text(t, 0.96, f"SECTION {idx + 1}", transform=ax_nov.get_xaxis_transform(), color=color, rotation=90, va="top", ha="right", fontsize=9, fontweight="bold")

        ax_nov.set_ylim(-0.02, 1.02)
        ax_nov.set_ylabel("Novelty")
        ax_nov.set_xlabel("Time (s)")
        ax_nov.set_xlim(0.0, max(duration, 0.1))
        ax_nov.grid(axis="y", alpha=0.15)
        ax_nov.legend(loc="upper right", ncol=3, fontsize=8)
        fig.tight_layout(pad=0.4)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def run_and_export(self, audio: np.ndarray, beat_grid: Any, output_path: str, json_path: str, stem_name: str) -> Dict[str, Any]:
        result, plot_data = self.analyze(audio, beat_grid)
        self.plot(plot_data, output_path, f"Novelty Structure — {stem_name}")
        payload = result.as_dict()
        payload["stem"] = str(stem_name)
        payload["global_beat_grid_source"] = str(getattr(beat_grid, "source", "unknown"))
        payload["global_tempo_bpm"] = float(getattr(beat_grid, "tempo_bpm", 0.0))
        payload["beats_per_bar"] = int(self.beats_per_bar)
        payload["notes"] = [
            "Beat and bar positions are locked to the global Beat This! grid.",
            "Novelty is evidence; it does not by itself declare a semantic section label.",
            "Existing PatternEngine/StructureEngine can consume the candidate evidence without replacing their contracts.",
        ]
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload
