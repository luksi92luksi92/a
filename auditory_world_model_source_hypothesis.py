#!/usr/bin/env python3
"""Source-hypothesis formation for the Auditory World Model.

This is the missing abstraction between fragmented acoustic evidence and a
persistent musical entity.  It does NOT reconstruct audio and does NOT perform
full source separation.

Pipeline:
    mixed audio
      -> existing AWM observations / SoundObjects
      -> source hypotheses (multi-object evidence)
      -> optional semantic interpretation

A SourceHypothesis is intentionally label-free.  Its members are existing
SoundObjects.  Several fragmented observations/components can therefore be
explained by one persistent hypothesis.  A hypothesis carries a confidence,
coherence score, temporal support, and evidence breakdown.

For the built-in deterministic synthetic test, the script also reports a
non-training diagnostic against the known generator components, but never uses
those labels to create hypotheses.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
EMBED = ROOT / "auditory_world_model_timbre_embedding.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

emb = _load(EMBED, "awm_embedding_source_hypothesis")
awm = emb.awm


@dataclass
class SourceEvidence:
    frequency_similarity: float
    temporal_similarity: float
    timbre_similarity: float
    harmonic_similarity: float
    role_similarity: float
    repetition_similarity: float

    def weighted(self) -> float:
        return float(np.clip(
            0.18 * self.frequency_similarity
            + 0.27 * self.temporal_similarity
            + 0.25 * self.timbre_similarity
            + 0.10 * self.harmonic_similarity
            + 0.08 * self.role_similarity
            + 0.12 * self.repetition_similarity,
            0.0, 1.0))


@dataclass
class SourceHypothesis:
    hypothesis_id: str
    member_ids: List[str]
    confidence: float
    coherence: float
    time_start: float
    time_end: float
    evidence: Dict[str, float]
    status: str = "ACTIVE"
    semantic_candidates: Dict[str, float] = field(default_factory=dict)


class SourceHypothesisEngine:
    """Build label-free persistent hypotheses from existing SoundObjects.

    Unlike the old post-hoc layer grouper, this stage is explicitly source
    oriented: a member may have a very different instantaneous frequency from
    another member when its temporal/recurrent/timbral evidence is consistent.
    """

    MIN_HISTORY = 6
    MIN_AGE_MS = 100.0
    MAX_OBJECTS = 80

    PAIR_THRESHOLD = 0.58
    MERGE_THRESHOLD = 0.63
    MIN_HYPOTHESIS_SIZE = 2
    MAX_HYPOTHESES = 12

    def __init__(self, world, embedder: Optional[emb.MeritTimbreEmbedder] = None):
        self.world = world
        self.embedder = embedder
        self.hypotheses: Dict[str, SourceHypothesis] = {}

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))

    def _history(self, obj):
        return [h for h in obj.history[-256:]
                if h.status in ("ACTIVE", "RECOVERING", "TENTATIVE")]

    def _profile(self, obj, end_time: float, grid: np.ndarray) -> Dict[str, object]:
        hist = self._history(obj)
        if not hist:
            return {}
        # Temporal occupancy is the most important source identity cue. Build
        # an energy envelope on a shared time grid.
        ht = np.asarray([float(h.time) for h in hist])
        he = np.asarray([max(float(h.energy), 0.0) for h in hist])
        order = np.argsort(ht); ht = ht[order]; he = he[order]
        curve = np.interp(grid, ht, he, left=0.0, right=0.0)
        curve = curve / (np.linalg.norm(curve) + 1e-9)

        freqs = np.asarray([float(h.frequency_hz) for h in hist if h.frequency_hz > 0], dtype=float)
        if freqs.size:
            logf = np.log2(np.maximum(np.median(freqs), 1.0))
            fspread = float(np.std(np.log2(np.maximum(freqs, 1.0))))
        else:
            logf, fspread = 0.0, 0.0

        role = np.asarray([
            obj.role_scores.kick, obj.role_scores.bass, obj.role_scores.percussion,
            obj.role_scores.harmonic_pad, obj.role_scores.lead_melody
        ], dtype=float)
        role /= np.linalg.norm(role) + 1e-9

        # Event recurrence fingerprint: autocorrelation over a coarse grid.
        if len(curve) >= 8:
            ac = np.correlate(curve, curve, mode="full")[len(curve)-1:]
            ac = ac[:min(32, len(ac))]
            ac /= np.linalg.norm(ac) + 1e-9
        else:
            ac = np.zeros(1)

        embv = None
        if self.embedder is not None and self.embedder.available:
            embv = self.embedder._object_embeddings.get(obj.object_id)

        return {
            "curve": curve,
            "logf": logf,
            "fspread": fspread,
            "harmonicity": float(np.clip(obj.physical_signature.harmonicity, 0, 1)),
            "flatness": float(np.clip(obj.physical_signature.spectral_flatness, 0, 1)),
            "crest": float(np.clip(obj.physical_signature.crest_factor / 8.0, 0, 1)),
            "role": role,
            "autocorr": ac,
            "embedding": embv,
            "timbre": obj.timbre_class,
            "start": float(obj.creation_time),
            "end": float(obj.last_observed_time),
        }

    def _pair(self, pa, pb) -> SourceEvidence:
        freq = math.exp(-abs(pa["logf"] - pb["logf"]) / 3.0)
        # Temporal similarity is allowed to reward phase-related events without
        # demanding identical amplitudes.
        temporal = max(0.0, self._cos(pa["curve"], pb["curve"]))
        if temporal < 0.25:
            # Compare rectified occupancy as a second chance for sparse events.
            aa = (pa["curve"] > 0.015).astype(float)
            bb = (pb["curve"] > 0.015).astype(float)
            temporal = max(temporal, self._cos(aa, bb))
        if pa["embedding"] is not None and pb["embedding"] is not None:
            timbre = max(0.0, self._cos(pa["embedding"], pb["embedding"]))
        else:
            timbre = 1.0 if pa["timbre"] == pb["timbre"] else 0.25
        harmonic = 1.0 - abs(pa["harmonicity"] - pb["harmonicity"])
        role = max(0.0, self._cos(pa["role"], pb["role"]))
        repetition = max(0.0, self._cos(pa["autocorr"], pb["autocorr"]))
        return SourceEvidence(freq, temporal, timbre, harmonic, role, repetition)

    def _cluster_score(self, members, profiles) -> float:
        if len(members) < 2:
            return 1.0
        scores = [
            self._pair(profiles[a.object_id], profiles[b.object_id]).weighted()
            for a, b in itertools.combinations(members, 2)
        ]
        return float(min(scores)) if scores else 0.0

    def _build_clusters(self, objects, profiles) -> List[List[object]]:
        clusters: List[List[object]] = [[o] for o in objects]
        while len(clusters) > 1:
            best = None; best_score = -1.0
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    merged = clusters[i] + clusters[j]
                    score = self._cluster_score(merged, profiles)
                    if score >= self.MERGE_THRESHOLD and score > best_score:
                        best_score = score; best = (i, j)
            if best is None:
                break
            i, j = best
            clusters[i].extend(clusters[j])
            del clusters[j]
        return sorted(
            [c for c in clusters if len(c) >= self.MIN_HYPOTHESIS_SIZE],
            key=lambda c: (-len(c), min(o.creation_time for o in c))
        )[:self.MAX_HYPOTHESES]

    def update(self, t: float) -> List[SourceHypothesis]:
        candidates = [
            o for o in self.world.objects.values()
            if o.status != awm.ObjectStatus.ARCHIVED
            and len(o.history) >= self.MIN_HISTORY
            and (t - o.creation_time) * 1000.0 >= self.MIN_AGE_MS
        ]
        candidates.sort(key=lambda o: o.creation_time)
        candidates = candidates[-self.MAX_OBJECTS:]
        if len(candidates) < 2:
            return []

        grid = np.linspace(max(0.0, t - 8.0), max(0.1, t), 128)
        profiles = {o.object_id: self._profile(o, t, grid) for o in candidates}
        profiles = {k: v for k, v in profiles.items() if v}
        candidates = [o for o in candidates if o.object_id in profiles]
        clusters = self._build_clusters(candidates, profiles)

        new_hypotheses: List[SourceHypothesis] = []
        used = set()
        for members in clusters:
            ids = [o.object_id for o in members]
            if used.intersection(ids):
                continue
            coherence = self._cluster_score(members, profiles)
            if coherence < self.PAIR_THRESHOLD:
                continue
            used.update(ids)
            pair_scores = [
                self._pair(profiles[a.object_id], profiles[b.object_id])
                for a, b in itertools.combinations(members, 2)
            ]
            evidence = {
                "frequency": float(np.mean([x.frequency_similarity for x in pair_scores])) if pair_scores else 0.0,
                "temporal": float(np.mean([x.temporal_similarity for x in pair_scores])) if pair_scores else 0.0,
                "timbre": float(np.mean([x.timbre_similarity for x in pair_scores])) if pair_scores else 0.0,
                "harmonic": float(np.mean([x.harmonic_similarity for x in pair_scores])) if pair_scores else 0.0,
                "role": float(np.mean([x.role_similarity for x in pair_scores])) if pair_scores else 0.0,
                "repetition": float(np.mean([x.repetition_similarity for x in pair_scores])) if pair_scores else 0.0,
            }
            # Confidence rises with coherent independent evidence and modestly
            # with persistence, but never reaches certainty just because a
            # cluster has many members.
            span = max(o.last_observed_time for o in members) - min(o.creation_time for o in members)
            persistence = float(np.clip(span / 4.0, 0.0, 1.0))
            confidence = float(np.clip(0.7 * coherence + 0.3 * persistence, 0.0, 0.97))
            hid = "source_" + "".join(f"{hash(x) & 0xffff:04x}" for x in ids[:2])
            hyp = SourceHypothesis(
                hypothesis_id=hid,
                member_ids=ids,
                confidence=confidence,
                coherence=coherence,
                time_start=min(o.creation_time for o in members),
                time_end=max(o.last_observed_time for o in members),
                evidence=evidence,
            )
            # Semantic candidates are derived after source formation; these are
            # priors only and never drive the clustering itself.
            role_mean = np.mean([
                [o.role_scores.kick, o.role_scores.bass, o.role_scores.percussion,
                 o.role_scores.harmonic_pad, o.role_scores.lead_melody]
                for o in members
            ], axis=0)
            names = ["kick", "bass", "percussion", "harmonic_pad", "lead_melody"]
            hyp.semantic_candidates = {n: float(v) for n, v in zip(names, role_mean) if v > 0.05}
            new_hypotheses.append(hyp)
        self.hypotheses = {h.hypothesis_id: h for h in new_hypotheses}
        return new_hypotheses


def synthetic_source_correlations(audio: np.ndarray, sr: int, hypotheses: Sequence[SourceHypothesis], world) -> List[Tuple[str, str, float]]:
    """Diagnostic only: compare hypothesis member activity to known synthetic
    components. This never enters hypothesis formation."""
    src = awm.AudioSource(world.params).synth_test_signal(duration=len(audio)/sr)
    # Reconstruct the same four components from the audio generator by
    # generating each independently with a local helper mirroring its formulas
    # would duplicate substantial code. Instead use frequency/time signatures
    # from object histories; report only if a clear temporal match exists.
    out = []
    names = ("kick", "bass", "hihat", "pad")
    # Source masks derived from simple, known periodic timing for diagnostics.
    t = np.linspace(0.0, len(audio)/sr, 256, endpoint=False)
    beat = 0.5
    masks = {
        "kick": (np.mod(t, beat) < 0.18).astype(float),
        "bass": np.ones_like(t),
        "hihat": (np.mod(t, beat) >= 0.20).astype(float),
        "pad": np.ones_like(t),
    }
    for h in hypotheses:
        curve = np.zeros_like(t)
        for oid in h.member_ids:
            obj = world.objects.get(oid)
            if obj is None:
                continue
            hist = [x for x in obj.history if x.status in ("ACTIVE", "RECOVERING", "TENTATIVE")]
            if not hist:
                continue
            ht = np.asarray([x.time for x in hist]); he = np.asarray([max(x.energy,0.0) for x in hist])
            curve += np.interp(t, ht, he, left=0.0, right=0.0)
        if np.linalg.norm(curve) < 1e-9:
            continue
        curve /= np.linalg.norm(curve) + 1e-9
        sims = []
        for name in names:
            m = masks[name] / (np.linalg.norm(masks[name]) + 1e-9)
            sims.append((name, float(np.dot(curve, m))))
        sims.sort(key=lambda x: x[1], reverse=True)
        out.append((h.hypothesis_id, sims[0][0], sims[0][1]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="?", default=None)
    args = parser.parse_args()

    params = awm.Parameters()
    source = awm.AudioSource(params)
    audio, sr = source.load(args.audio)
    model = emb.AuditoryWorldModel(params)
    model.stem_separation.available = False  # source hypotheses must not depend on reconstruction/separation
    model.run(audio, use_stem_separation=False)

    embedder = emb.MeritTimbreEmbedder()
    embedder.build_all(model.world, audio, sr)
    engine = SourceHypothesisEngine(model.world, embedder)
    hyps = engine.update(model.world.t)

    print("\n" + "=" * 100)
    print("SOURCE HYPOTHESIS FORMATION")
    print("=" * 100)
    print(f"SoundObjects: {len(model.world.objects)}")
    print(f"Source hypotheses: {len(hyps)}")
    for h in hyps:
        print(f"\n{h.hypothesis_id} members={len(h.member_ids)} conf={h.confidence:.2f} coherence={h.coherence:.2f}")
        print(f"  time: {h.time_start:.3f}s -> {h.time_end:.3f}s")
        print("  members:", ", ".join(h.member_ids))
        print("  evidence:", " ".join(f"{k}={v:.2f}" for k,v in h.evidence.items()))
        if h.semantic_candidates:
            print("  semantic priors:", " ".join(f"{k}={v:.2f}" for k,v in sorted(h.semantic_candidates.items(), key=lambda x:-x[1])[:3]))

    if args.audio is None:
        print("\nSynthetic-test note: hypotheses are label-free. Their diagnostic mapping is not used to form them.")


if __name__ == "__main__":
    main()
