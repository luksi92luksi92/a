#!/usr/bin/env python3
"""Fuse stem-local structural evidence into canonical musical phrases/sections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


class GlobalStructureFusion:
    """Compare aligned stem evidence and emit one global musical structure."""

    PHRASE_COLOR = "#377eb8"
    SECTION_COLOR = "#984ea3"
    EVIDENCE_COLOR = "#555555"
    SUPPORT_COLOR = "#2c7fb8"

    def __init__(
        self,
        phrase_threshold: float = 0.42,
        section_threshold: float = 0.48,
        phrase_min_support: float = 0.25,
        section_min_support: float = 0.25,
        section_refractory_bars: int = 16,
        min_section_bars: int = 16,
        max_section_bars: int = 128,
    ) -> None:
        self.phrase_threshold = float(phrase_threshold)
        self.section_threshold = float(section_threshold)
        self.phrase_min_support = float(phrase_min_support)
        self.section_min_support = float(section_min_support)
        self.section_refractory_bars = int(max(section_refractory_bars, 16))
        self.min_section_bars = int(max(min_section_bars, 16))
        self.max_section_bars = int(max(max_section_bars, self.min_section_bars))

    @staticmethod
    def _pad(values: Sequence[float], n: int) -> np.ndarray:
        out = np.full(n, np.nan, dtype=float)
        x = np.asarray(values, dtype=float).reshape(-1)
        out[:min(n, len(x))] = x[:n]
        return out

    @staticmethod
    def _peak_indices(
        values: np.ndarray,
        threshold: float,
        refractory: int = 1,
    ) -> List[int]:
        x = np.asarray(values, dtype=float)
        if x.size < 3:
            return []
        candidates: List[int] = []
        last = -10**9
        for i in range(1, len(x) - 1):
            if not np.isfinite(x[i]) or x[i] < threshold:
                continue
            if x[i] < x[i - 1] or x[i] < x[i + 1]:
                continue
            if i - last < max(int(refractory), 1):
                continue
            candidates.append(i)
            last = i
        return candidates

    @staticmethod
    def _candidate_support(
        payloads: List[Tuple[str, Dict[str, Any]]],
        level: str,
        n_bars: int,
    ) -> Tuple[np.ndarray, List[List[str]]]:
        counts = np.zeros(n_bars, dtype=float)
        sources: List[List[str]] = [[] for _ in range(n_bars)]
        for stem_name, novelty in payloads:
            seen: set[int] = set()
            for row in novelty.get("candidates", []):
                if row.get("level") != level or not row.get("selected"):
                    continue
                idx = int(row.get("index", -1))
                if idx < 0 or idx >= n_bars or idx in seen:
                    continue
                seen.add(idx)
                counts[idx] += 1.0
                sources[idx].append(stem_name)
        denom = float(max(len(payloads), 1))
        return counts / denom, sources

    @staticmethod
    def _select_boundaries(
        evidence: np.ndarray,
        support: np.ndarray,
        threshold: float,
        min_support: float,
        refractory: int,
    ) -> List[Dict[str, Any]]:
        peaks = GlobalStructureFusion._peak_indices(
            evidence,
            threshold=threshold,
            refractory=refractory,
        )
        selected: List[Dict[str, Any]] = []
        for idx in peaks:
            strong_single_stem = float(evidence[idx]) >= min(0.85, threshold + 0.30)
            if float(support[idx]) < min_support and not strong_single_stem:
                continue
            selected.append(
                {
                    "bar_index": int(idx),
                    "score": float(evidence[idx]),
                    "support": float(support[idx]),
                }
            )
        return selected

    def fuse(
        self,
        stem_results: Dict[str, Dict[str, Any]],
        beat_times: Sequence[float],
        bar_times: Sequence[float],
    ) -> Dict[str, Any]:
        usable: List[Tuple[str, Dict[str, Any]]] = []
        for stem_name, result in stem_results.items():
            novelty = result.get("novelty_structure")
            if not novelty or novelty.get("plot_skipped"):
                continue
            usable.append((str(stem_name), novelty))

        bars = np.asarray(bar_times, dtype=float).reshape(-1)
        beats = np.asarray(beat_times, dtype=float).reshape(-1)
        n_bars = len(bars)
        if not usable or n_bars == 0:
            return {
                "scope": "global",
                "status": "insufficient_stem_evidence",
                "stem_count": len(usable),
                "phrase_boundaries": [],
                "section_boundaries": [],
                "phrases": [],
                "sections": [],
            }

        phrase_stack = np.vstack([
            self._pad(novelty.get("phrase_novelty", []), n_bars)
            for _, novelty in usable
        ])
        section_stack = np.vstack([
            self._pad(novelty.get("section_novelty", []), n_bars)
            for _, novelty in usable
        ])
        bar_stack = np.vstack([
            self._pad(novelty.get("bar_novelty", []), n_bars)
            for _, novelty in usable
        ])

        phrase_mean = np.nanmean(phrase_stack, axis=0)
        section_mean = np.nanmean(section_stack, axis=0)
        bar_mean = np.nanmean(bar_stack, axis=0)
        phrase_mean = np.nan_to_num(phrase_mean, nan=0.0)
        section_mean = np.nan_to_num(section_mean, nan=0.0)
        bar_mean = np.nan_to_num(bar_mean, nan=0.0)

        phrase_support, phrase_sources = self._candidate_support(
            usable, "phrase", n_bars
        )
        section_support, section_sources = self._candidate_support(
            usable, "section", n_bars
        )

        # Mean novelty is the primary signal; cross-stem agreement provides the
        # structural confidence. A strong single-stem event may still survive.
        phrase_evidence = np.clip(
            0.72 * phrase_mean + 0.28 * phrase_support,
            0.0,
            1.0,
        )
        section_evidence = np.clip(
            0.72 * section_mean + 0.28 * section_support,
            0.0,
            1.0,
        )

        phrase_boundaries = self._select_boundaries(
            phrase_evidence,
            phrase_support,
            threshold=self.phrase_threshold,
            min_support=self.phrase_min_support,
            refractory=1,
        )
        section_boundaries = self._select_boundaries(
            section_evidence,
            section_support,
            threshold=self.section_threshold,
            min_support=self.section_min_support,
            refractory=self.section_refractory_bars,
        )

        # Canonical sections must be at least 16 bars apart. Remove boundaries
        # that would create shorter sections, including an early boundary before
        # the minimum initial section length.
        accepted_sections: List[Dict[str, Any]] = []
        last_idx = 0
        for row in section_boundaries:
            idx = int(row["bar_index"])
            if idx - last_idx < self.min_section_bars:
                continue
            accepted_sections.append(row)
            last_idx = idx
        section_boundaries = accepted_sections

        # A section may not exceed 128 bars. Long intervals are split at the
        # maximum length even when no novelty peak exists at that exact point.
        length_capped_boundaries: List[Dict[str, Any]] = []
        last_idx = 0
        for row in section_boundaries:
            idx = int(row["bar_index"])
            while idx - last_idx > self.max_section_bars:
                last_idx += self.max_section_bars
                length_capped_boundaries.append({
                    "bar_index": int(last_idx),
                    "score": 0.0,
                    "support": 0.0,
                    "time_s": float(bars[last_idx]),
                    "source_stems": [],
                    "type": "section_length_cap",
                })
            length_capped_boundaries.append(row)
            last_idx = idx
        while n_bars - last_idx > self.max_section_bars:
            last_idx += self.max_section_bars
            length_capped_boundaries.append({
                "bar_index": int(last_idx),
                "score": 0.0,
                "support": 0.0,
                "time_s": float(bars[last_idx]),
                "source_stems": [],
                "type": "section_length_cap",
            })
        section_boundaries = length_capped_boundaries

        for row in phrase_boundaries:
            idx = row["bar_index"]
            row["time_s"] = float(bars[idx])
            row["source_stems"] = phrase_sources[idx]
            row["type"] = "phrase_boundary"
        for row in section_boundaries:
            idx = row["bar_index"]
            row["time_s"] = float(bars[idx])
            row["source_stems"] = section_sources[idx]
            row["type"] = "section_boundary"

        section_indices = sorted({int(x["bar_index"]) for x in section_boundaries})
        phrase_indices = sorted({int(x["bar_index"]) for x in phrase_boundaries})

        # Sections are authoritative only after cross-stem fusion.
        if n_bars < self.min_section_bars:
            section_boundaries = []
            section_indices = []
        section_cuts = [0] + [i for i in section_indices if 0 < i < n_bars] + [n_bars]
        sections: List[Dict[str, Any]] = []
        for i, (start_bar, end_bar) in enumerate(
            zip(section_cuts[:-1], section_cuts[1:])
        ):
            if end_bar <= start_bar:
                continue
            boundary_idx = start_bar if start_bar in section_indices else None
            score = float(section_evidence[start_bar]) if boundary_idx is not None else 0.0
            support = float(section_support[start_bar]) if boundary_idx is not None else 0.0
            sections.append(
                {
                    "section_index": int(i),
                    "start_bar": int(start_bar),
                    "end_bar": int(end_bar),
                    "start_time_s": float(bars[start_bar]),
                    "end_time_s": float(
                        bars[end_bar] if end_bar < n_bars else (
                            bars[-1] + (
                                float(np.median(np.diff(bars)))
                                if len(bars) > 1
                                else (float(np.median(np.diff(beats))) * 4.0 if len(beats) > 1 else 0.0)
                            )
                        )
                    ),
                    "boundary_score": score,
                    "boundary_support": support,
                }
            )

        # Phrase spans are constrained by both phrase evidence and section
        # boundaries, so a phrase never straddles a canonical section change.
        phrase_cuts = sorted(
            {0, n_bars}
            | {i for i in phrase_indices if 0 < i < n_bars}
            | {i for i in section_indices if 0 < i < n_bars}
        )
        phrases: List[Dict[str, Any]] = []
        for i, (start_bar, end_bar) in enumerate(
            zip(phrase_cuts[:-1], phrase_cuts[1:])
        ):
            if end_bar <= start_bar:
                continue
            evidence_slice = phrase_evidence[start_bar:end_bar]
            support_slice = phrase_support[start_bar:end_bar]
            phrases.append(
                {
                    "phrase_index": int(i),
                    "start_bar": int(start_bar),
                    "end_bar": int(end_bar),
                    "start_time_s": float(bars[start_bar]),
                    "end_time_s": float(
                        bars[end_bar] if end_bar < n_bars else (
                            bars[-1] + (
                                float(np.median(np.diff(bars)))
                                if len(bars) > 1
                                else (float(np.median(np.diff(beats))) * 4.0 if len(beats) > 1 else 0.0)
                            )
                        )
                    ),
                    "boundary_evidence_max": float(np.max(evidence_slice)) if evidence_slice.size else 0.0,
                    "boundary_support_max": float(np.max(support_slice)) if support_slice.size else 0.0,
                    "section_boundary_at_start": bool(start_bar in section_indices),
                    "section_boundary_at_end": bool(end_bar in section_indices),
                }
            )

        return {
            "scope": "global",
            "status": "ok",
            "stem_count": len(usable),
            "stems_used": [name for name, _ in usable],
            "bar_times": bars.tolist(),
            "beat_times": beats.tolist(),
            "bar_novelty_mean": bar_mean.tolist(),
            "phrase_evidence": phrase_evidence.tolist(),
            "phrase_support": phrase_support.tolist(),
            "section_evidence": section_evidence.tolist(),
            "section_support": section_support.tolist(),
            "phrase_boundaries": phrase_boundaries,
            "section_boundaries": section_boundaries,
            "section_constraints": {
                "min_bars": self.min_section_bars,
                "max_bars": self.max_section_bars,
            },
            "phrases": phrases,
            "sections": sections,
            "notes": [
                "Stem-local phrase/section candidates are evidence only.",
                "Global phrase/section boundaries require aligned cross-stem evidence or a strong single-stem event.",
                "A stem dropout remains a transition subtype and does not terminate a global phrase by itself.",
                "All timings are on the shared full-source Beat This! beat/bar grid.",
            ],
        }

    def plot(
        self,
        structure: Dict[str, Any],
        output_path: str | Path,
        title: str = "Global Musical Structure",
    ) -> None:
        if structure.get("status") != "ok":
            return

        bars = np.asarray(structure["bar_times"], dtype=float)
        if bars.size == 0:
            return
        if len(bars) > 1:
            bar_step = float(np.median(np.diff(bars)))
        else:
            beats = np.asarray(structure.get("beat_times", []), dtype=float)
            bar_step = float(np.median(np.diff(beats)) * 4.0) if len(beats) > 1 else 0.0
        x_end = float(bars[-1] + max(bar_step, 1e-6))
        x_bounds = np.r_[bars, x_end]

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(19, 8.5),
            sharex=True,
            gridspec_kw={"height_ratios": [2.1, 1.2], "hspace": 0.04},
        )
        ax_ev, ax_structure = axes

        t = bars
        ax_ev.plot(
            t,
            np.asarray(structure["phrase_evidence"], dtype=float),
            color=self.PHRASE_COLOR,
            linewidth=1.8,
            label="global phrase-boundary evidence",
        )
        ax_ev.plot(
            t,
            np.asarray(structure["section_evidence"], dtype=float),
            color=self.SECTION_COLOR,
            linewidth=1.9,
            label="global section-boundary evidence",
        )
        ax_ev.plot(
            t,
            np.asarray(structure["phrase_support"], dtype=float),
            color=self.SUPPORT_COLOR,
            linewidth=1.0,
            alpha=0.55,
            linestyle="--",
            label="cross-stem phrase support",
        )
        ax_ev.set_ylim(0.0, 1.0)
        ax_ev.set_ylabel("STRUCTURE EVIDENCE")
        ax_ev.legend(loc="upper right", fontsize=8, frameon=False)

        for row in structure["section_boundaries"]:
            x = float(row["time_s"])
            ax_ev.axvline(x, color=self.SECTION_COLOR, linewidth=2.2, alpha=0.85)
            ax_ev.text(
                x,
                0.98,
                f"SECTION {int(row['bar_index'])}",
                transform=ax_ev.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                color=self.SECTION_COLOR,
            )

        for row in structure["phrase_boundaries"]:
            x = float(row["time_s"])
            ax_ev.axvline(x, color=self.PHRASE_COLOR, linewidth=1.35, alpha=0.65)
            ax_ev.text(
                x,
                0.84,
                f"PHRASE {int(row['bar_index'])}",
                transform=ax_ev.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=7,
                color=self.PHRASE_COLOR,
            )

        for seg in structure["sections"]:
            start = float(seg["start_time_s"])
            end = float(seg["end_time_s"])
            ax_structure.broken_barh(
                [(start, max(end - start, 1e-6))],
                (0.54, 0.32),
                facecolors=self.SECTION_COLOR,
                alpha=0.24,
            )
            ax_structure.text(
                0.5 * (start + end),
                0.70,
                f"SECTION {seg['section_index'] + 1}",
                ha="center",
                va="center",
                fontsize=8,
                color=self.SECTION_COLOR,
            )

        for seg in structure["phrases"]:
            start = float(seg["start_time_s"])
            end = float(seg["end_time_s"])
            ax_structure.broken_barh(
                [(start, max(end - start, 1e-6))],
                (0.08, 0.32),
                facecolors=self.PHRASE_COLOR,
                alpha=0.20,
            )
            ax_structure.text(
                0.5 * (start + end),
                0.24,
                f"PHRASE {seg['phrase_index'] + 1}",
                ha="center",
                va="center",
                fontsize=7,
                color=self.PHRASE_COLOR,
            )

        for x in x_bounds:
            ax_ev.axvline(float(x), color="#777777", alpha=0.10, linewidth=0.45)
            ax_structure.axvline(float(x), color="#777777", alpha=0.10, linewidth=0.45)

        ax_structure.set_ylim(0.0, 1.0)
        ax_structure.set_yticks([0.24, 0.70])
        ax_structure.set_yticklabels(["PHRASES", "SECTIONS"])
        ax_structure.set_xlabel("Time (s)")
        ax_structure.set_title(
            f"{title} | {structure['stem_count']} stems fused on shared Beat This! grid",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )

        for ax in axes:
            ax.grid(axis="y", alpha=0.12, linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.subplots_adjust(left=0.055, right=0.995, top=0.95, bottom=0.07)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def fuse_and_export(
        self,
        stem_results: Dict[str, Dict[str, Any]],
        beat_times: Sequence[float],
        bar_times: Sequence[float],
        output_plot: str | Path,
        output_json: str | Path,
    ) -> Dict[str, Any]:
        structure = self.fuse(stem_results, beat_times, bar_times)
        if structure.get("status") == "ok":
            self.plot(structure, output_plot)
        structure["plot_path"] = str(output_plot) if structure.get("status") == "ok" else None
        structure["json_path"] = str(output_json)
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(
            json.dumps(structure, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return structure
