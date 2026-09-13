#!/usr/bin/env python3
"""EDM-specific reverse-DAW perception layer.

This module extends the Auditory World Model without replacing its
WorldState/Observation/SoundObject contracts. It adds the musical-element
layer required by EDM_REVERSE_DAW_ARCHITECTURE_ADDENDUM.md:

    SoundObjects -> Elements -> PatternInstances -> StructuralSpans -> Arrangement

Design rules:
- evidence remains separate from persistent hypotheses;
- object identities are never collapsed merely because they share a family;
- event boundaries are never used as section boundaries;
- learned embeddings and separators are optional adapters;
- all tunable numbers live in EDMParameterRegistry;
- every derived hypothesis has inspectable evidence and provenance;
- export is JSON-safe and deterministic for replay/debugging.

The implementation is deliberately dependency-light: NumPy plus the existing
AWM module and, optionally, source_hypothesis.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
AWM_PATH = ROOT / "auditory_world_model (5).py"
SOURCE_PATH = ROOT / "auditory_world_model_source_hypothesis.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


awm = _load_module(AWM_PATH, "awm_edm_reverse_daw")
if SOURCE_PATH.exists():
    source_hyp = _load_module(SOURCE_PATH, "awm_edm_source_hypothesis")
else:
    source_hyp = None


class ParameterClass(str, Enum):
    DEFAULT = "DEFAULT"
    TUNABLE = "TUNABLE"
    DERIVED = "DERIVED"


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    value: float
    unit: str
    classification: ParameterClass
    owner: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None


@dataclass
class EDMParameterRegistry:
    """All EDM-layer tuning knobs in one inspectable registry."""

    element_similarity_threshold: float = 0.62
    element_merge_hysteresis: float = 0.05
    element_min_members: int = 2
    element_max_members: int = 256
    element_max_families: int = 64

    bar_match_threshold: float = 0.72
    pattern_min_bars: int = 2
    pattern_max_period_bars: int = 16
    pattern_similarity_tolerance: float = 0.12

    section_min_bars: int = 4
    section_novelty_threshold: float = 0.33
    section_refractory_bars: int = 2
    section_silence_threshold: float = 0.08

    arrangement_confidence_floor: float = 0.45
    timing_tolerance_ms: float = 35.0

    def registry(self) -> List[ParameterSpec]:
        T, D = ParameterClass.TUNABLE, ParameterClass.DEFAULT
        return [
            ParameterSpec("element_similarity_threshold", self.element_similarity_threshold, "score", T, "ElementEngine", 0, 1),
            ParameterSpec("element_merge_hysteresis", self.element_merge_hysteresis, "score", T, "ElementEngine", 0, 1),
            ParameterSpec("element_min_members", self.element_min_members, "count", D, "ElementEngine", 1),
            ParameterSpec("element_max_members", self.element_max_members, "count", D, "ElementEngine", 1),
            ParameterSpec("element_max_families", self.element_max_families, "count", D, "ElementEngine", 1),
            ParameterSpec("bar_match_threshold", self.bar_match_threshold, "score", T, "PatternEngine", 0, 1),
            ParameterSpec("pattern_min_bars", self.pattern_min_bars, "count", D, "PatternEngine", 1),
            ParameterSpec("pattern_max_period_bars", self.pattern_max_period_bars, "bars", D, "PatternEngine", 1),
            ParameterSpec("pattern_similarity_tolerance", self.pattern_similarity_tolerance, "distance", T, "PatternEngine", 0, 1),
            ParameterSpec("section_min_bars", self.section_min_bars, "bars", D, "StructureEngine", 1),
            ParameterSpec("section_novelty_threshold", self.section_novelty_threshold, "distance", T, "StructureEngine", 0, 1),
            ParameterSpec("section_refractory_bars", self.section_refractory_bars, "bars", D, "StructureEngine", 0),
            ParameterSpec("section_silence_threshold", self.section_silence_threshold, "relative_energy", T, "StructureEngine", 0, 1),
            ParameterSpec("arrangement_confidence_floor", self.arrangement_confidence_floor, "score", D, "ArrangementEngine", 0, 1),
            ParameterSpec("timing_tolerance_ms", self.timing_tolerance_ms, "ms", T, "EDMReverseDAW", 0),
        ]


@dataclass
class EDMObjectMetadata:
    """Non-destructive EDM annotation attached to a SoundObject by stable id."""

    object_id: str
    beat_index: Optional[int] = None
    bar_index: Optional[int] = None
    beat_position: Optional[float] = None
    timing_residual_ms: float = 0.0
    local_density: float = 0.0
    periodicity: float = 0.0
    tail_ownership: float = 0.0
    overlap_score: float = 0.0
    identity_cluster_ids: List[str] = field(default_factory=list)
    nearest_neighbors: List[Tuple[str, float]] = field(default_factory=list)
    element_memberships: List[str] = field(default_factory=list)
    pattern_memberships: List[str] = field(default_factory=list)
    section_memberships: List[str] = field(default_factory=list)
    stem: str = "unknown"


@dataclass
class ElementFamily:
    element_id: str
    name: str
    member_ids: List[str]
    confidence: float
    formation_time: float
    last_updated: float
    evidence: Dict[str, float]
    status: str = "ACTIVE"


@dataclass
class PatternInstance:
    pattern_id: str
    period_bars: int
    start_bar: int
    end_bar: int
    element_ids: List[str]
    member_object_ids: List[str]
    confidence: float
    variation_tolerance: float
    signature: List[float]
    status: str = "ACTIVE"


@dataclass
class StructuralSpan:
    section_id: str
    start_bar: int
    end_bar: int
    start_time: float
    end_time: float
    label_candidates: Dict[str, float]
    novelty_score: float
    energy_mean: float
    density_mean: float
    evidence: Dict[str, float]
    status: str = "ACTIVE"


@dataclass
class Arrangement:
    arrangement_id: str
    section_ids: List[str]
    confidence: float
    created_time: float
    updated_time: float
    evidence: Dict[str, float]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.size != b.size:
        n = min(a.size, b.size)
        a, b = a[:n], b[:n]
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    return float(np.mean(values)) if values else default


class EDMMeter:
    """Shared metrical context. Never overwrites physical timestamps."""

    def __init__(self, world):
        self.world = world

    @property
    def tempo_bpm(self) -> float:
        g = self.world.groove
        return float(g.tempo_bpm) if g.tempo_bpm > 0 else 120.0

    @property
    def beat_period(self) -> float:
        return 60.0 / max(self.tempo_bpm, 1e-6)

    @property
    def bar_period(self) -> float:
        return 4.0 * self.beat_period

    def beat_index(self, t: float) -> int:
        return int(math.floor(max(0.0, t) / self.beat_period + 1e-9))

    def bar_index(self, t: float) -> int:
        return self.beat_index(t) // 4

    def beat_position(self, t: float) -> float:
        phase = (max(0.0, t) / self.beat_period) % 1.0
        return float(phase)

    def nearest_grid_residual_ms(self, t: float) -> float:
        pos = t / self.beat_period
        residual = abs(pos - round(pos)) * self.beat_period * 1000.0
        return float(residual)


class ElementEngine:
    """Infer reusable musical elements from SoundObjects without collapsing them."""

    ROLE_NAMES = ("kick", "bass", "percussion", "harmonic_pad", "lead_melody")

    def __init__(self, world, registry: EDMParameterRegistry):
        self.world = world
        self.registry = registry
        self.elements: Dict[str, ElementFamily] = {}
        self.metadata: Dict[str, EDMObjectMetadata] = {}

    def _role_vector(self, obj) -> np.ndarray:
        return np.asarray([
            obj.role_scores.kick,
            obj.role_scores.bass,
            obj.role_scores.percussion,
            obj.role_scores.harmonic_pad,
            obj.role_scores.lead_melody,
        ], dtype=float)

    def _object_vector(self, obj) -> np.ndarray:
        sig = obj.physical_signature
        fp = np.asarray(obj.timbre_fingerprint.coefficients, dtype=float)
        if fp.size:
            fp = fp[:12]
            fp = (fp - np.mean(fp)) / (np.std(fp) + 1e-6)
        stem = np.asarray(sig.stem_distribution, dtype=float)
        stem = stem / (np.linalg.norm(stem) + 1e-9) if stem.size else np.zeros(4)
        base = np.asarray([
            math.log2(max(sig.frequency_hz, 20.0)),
            math.log2(max(sig.bandwidth_hz, 20.0)),
            float(sig.harmonicity),
            float(sig.spectral_centroid / 10000.0),
            float(sig.spectral_flatness),
            float(np.clip(sig.crest_factor / 8.0, 0.0, 1.0)),
        ], dtype=float)
        role = self._role_vector(obj)
        return np.concatenate([base, role, stem, fp])

    def _stem_name(self, obj) -> str:
        d = np.asarray(obj.physical_signature.stem_distribution, dtype=float)
        if d.size >= 4 and np.max(d) > 0:
            return ("drums", "bass", "vocals", "other")[int(np.argmax(d[:4]))]
        return "unknown"

    def _pair_score(self, a, b) -> Tuple[float, Dict[str, float]]:
        va, vb = self._object_vector(a), self._object_vector(b)
        detail = {
            "feature": max(0.0, _cosine(va, vb)),
            "role": max(0.0, _cosine(self._role_vector(a), self._role_vector(b))),
            "frequency": math.exp(-abs(math.log2(max(a.physical_signature.frequency_hz, 20.0)) -
                                       math.log2(max(b.physical_signature.frequency_hz, 20.0))) / 2.0),
            "harmonic": 1.0 - abs(a.physical_signature.harmonicity - b.physical_signature.harmonicity),
            "stem": 1.0 if self._stem_name(a) == self._stem_name(b) else 0.25,
        }
        score = (
            0.45 * detail["feature"]
            + 0.20 * detail["role"]
            + 0.15 * detail["frequency"]
            + 0.10 * detail["harmonic"]
            + 0.10 * detail["stem"]
        )
        return float(np.clip(score, 0, 1)), detail

    def _eligible_objects(self, t: float) -> List[Any]:
        objs = [
            o for o in self.world.objects.values()
            if o.status != awm.ObjectStatus.ARCHIVED
            and o.last_observed_time <= t + 1e-9
        ]
        return sorted(objs, key=lambda o: (o.creation_time, o.object_id))[-self.registry.element_max_members:]

    def _propose(self, objects: Sequence[Any], t: float) -> List[List[Any]]:
        clusters: List[List[Any]] = []
        threshold = self.registry.element_similarity_threshold
        for obj in objects:
            best_i, best_s = None, -1.0
            for i, cluster in enumerate(clusters):
                scores = [self._pair_score(obj, member)[0] for member in cluster[-12:]]
                score = min(scores) if scores else 0.0
                if score > best_s:
                    best_i, best_s = i, score
            if best_i is not None and best_s >= threshold:
                clusters[best_i].append(obj)
            else:
                clusters.append([obj])
        return [c for c in clusters if len(c) >= self.registry.element_min_members]

    @staticmethod
    def _family_name(members: Sequence[Any]) -> str:
        role = np.mean([
            [m.role_scores.kick, m.role_scores.bass, m.role_scores.percussion,
             m.role_scores.harmonic_pad, m.role_scores.lead_melody]
            for m in members
        ], axis=0)
        role_name = ElementEngine.ROLE_NAMES[int(np.argmax(role))]
        stem_values = []
        for m in members:
            d = np.asarray(m.physical_signature.stem_distribution, dtype=float)
            if d.size >= 4 and np.max(d) > 0:
                stem_values.append(int(np.argmax(d[:4])))
        if stem_values:
            stem = ("drums", "bass", "vocals", "other")[int(round(np.mean(stem_values)))]
            return f"{stem}_{role_name}"
        return role_name

    def update(self, t: float) -> Dict[str, ElementFamily]:
        objects = self._eligible_objects(t)
        proposals = self._propose(objects, t)
        new_families: Dict[str, ElementFamily] = {}

        for cluster in proposals[:self.registry.element_max_families]:
            member_ids = [o.object_id for o in cluster]
            pair_data = [
                self._pair_score(a, b)
                for i, a in enumerate(cluster)
                for b in cluster[i + 1:]
            ]
            evidence = {
                "feature": _safe_mean([d["feature"] for _, d in pair_data]),
                "role": _safe_mean([d["role"] for _, d in pair_data]),
                "frequency": _safe_mean([d["frequency"] for _, d in pair_data]),
                "harmonic": _safe_mean([d["harmonic"] for _, d in pair_data]),
                "stem": _safe_mean([d["stem"] for _, d in pair_data]),
            }
            confidence = float(np.clip(
                0.70 * _safe_mean([s for s, _ in pair_data])
                + 0.30 * min(1.0, len(cluster) / 6.0),
                0.0, 0.97))
            eid = "element_" + uuid.uuid4().hex[:8]
            family = ElementFamily(
                element_id=eid,
                name=self._family_name(cluster),
                member_ids=member_ids,
                confidence=confidence,
                formation_time=min(o.creation_time for o in cluster),
                last_updated=t,
                evidence=evidence,
            )
            new_families[eid] = family

        # Reconcile with prior families so repeated updates do not manufacture
        # a new identity for the same element family. The overlap threshold is
        # explicit hysteresis, not an implicit "last result wins" rule.
        reconciled: Dict[str, ElementFamily] = {}
        prior_items = list(self.elements.values())
        used_prior: set[str] = set()
        for proposal in new_families.values():
            proposed = set(proposal.member_ids)
            best = None
            best_score = 0.0
            for prior in prior_items:
                if prior.element_id in used_prior:
                    continue
                prior_members = set(prior.member_ids)
                union = len(proposed | prior_members)
                overlap = len(proposed & prior_members) / max(union, 1)
                if overlap > best_score:
                    best_score = overlap
                    best = prior
            if best is not None and best_score >= max(
                self.registry.element_similarity_threshold - self.registry.element_merge_hysteresis,
                0.50,
            ):
                proposal.element_id = best.element_id
                proposal.formation_time = best.formation_time
                proposal.confidence = float(
                    0.70 * best.confidence + 0.30 * proposal.confidence
                )
                used_prior.add(best.element_id)
            reconciled[proposal.element_id] = proposal

        self.elements = reconciled
        self.metadata.clear()
        for obj in objects:
            meta = EDMObjectMetadata(object_id=obj.object_id, stem=self._stem_name(obj))
            self.metadata[obj.object_id] = meta

        # Attach only annotation through a sidecar map; core SoundObjects remain
        # authoritative and unchanged.
        for family in self.elements.values():
            for oid in family.member_ids:
                meta = self.metadata[oid]
                meta.element_memberships.append(family.element_id)

        self._record_provenance(t)
        return self.elements

    def _record_provenance(self, t: float) -> None:
        for family in self.elements.values():
            reason = (
                f"element family '{family.name}' from feature/role/frequency/"
                f"harmonic/stem agreement; members remain separate SoundObjects"
            )
            self.world.record_decision(awm.DecisionRecord(
                time=t, module="ElementEngine", decision_type="element_hypothesis",
                target_id=family.element_id, candidates=[], chosen=family.element_id,
                reason=reason, confidence=family.confidence
            ))
            self.world.emit_event(
                "element_hypothesis",
                family.element_id,
                {"members": family.member_ids, "name": family.name, "evidence": family.evidence},
                family.confidence, t
            )


class PatternEngine:
    """Detect repeated organization over bars with variation tolerance."""

    def __init__(self, world, registry: EDMParameterRegistry, elements: ElementEngine):
        self.world = world
        self.registry = registry
        self.elements = elements
        self.meter = EDMMeter(world)
        self.patterns: Dict[str, PatternInstance] = {}

    def _live_objects(self, t: float) -> List[Any]:
        return [
            o for o in self.world.objects.values()
            if o.status != awm.ObjectStatus.ARCHIVED and o.creation_time <= t
        ]

    def _bar_signature(self, bar_index: int, objects: Sequence[Any]) -> np.ndarray:
        vec = np.zeros(48, dtype=float)
        bar_start = bar_index * self.meter.bar_period
        family_ids = sorted(self.elements.elements)
        for obj in objects:
            t = obj.creation_time
            if not (bar_start <= t < bar_start + self.meter.bar_period):
                continue
            beat = int(np.clip(math.floor((t - bar_start) / self.meter.beat_period), 0, 3))
            family_id = next(
                iter(self.elements.metadata.get(obj.object_id, EDMObjectMetadata(obj.object_id)).element_memberships),
                None,
            )
            family_idx = family_ids.index(family_id) % 8 if family_id in family_ids else 0
            idx = family_idx * 4 + beat
            energy = max(float(obj.physical_signature.energy), 0.0)
            vec[idx] += energy

        # Add broad role/energy summaries to tolerate small timing/sample variation.
        role = np.zeros(5)
        energy = 0.0
        count = 0
        for obj in objects:
            if bar_start <= obj.creation_time < bar_start + self.meter.bar_period:
                role += np.asarray([
                    obj.role_scores.kick, obj.role_scores.bass, obj.role_scores.percussion,
                    obj.role_scores.harmonic_pad, obj.role_scores.lead_melody
                ])
                energy += max(float(obj.physical_signature.energy), 0.0)
                count += 1
        vec[40:45] = role / max(count, 1)
        vec[45] = min(energy, 10.0) / 10.0
        vec[46] = min(count / 16.0, 1.0)
        vec[47] = min(sum(abs(x) for x in vec[:40]) / 10.0, 1.0)
        n = np.linalg.norm(vec)
        return vec / (n + 1e-9)

    def _bars(self, t: float) -> range:
        last_bar = max(0, self.meter.bar_index(t))
        return range(0, last_bar + 1)

    def update(self, t: float) -> Dict[str, PatternInstance]:
        objects = self._live_objects(t)
        signatures = {b: self._bar_signature(b, objects) for b in self._bars(t)}

        found: List[PatternInstance] = []
        max_period = min(self.registry.pattern_max_period_bars, max(1, len(signatures) // 2))
        last_signature_bar = max(signatures.keys(), default=-1)
        for period in range(1, max_period + 1):
            for end_bar in range(
                period * self.registry.pattern_min_bars - 1,
                last_signature_bar + 1,
            ):
                starts = [end_bar - k * period for k in range(self.registry.pattern_min_bars)]
                if min(starts) < 0:
                    continue
                sims = [
                    _cosine(signatures[a], signatures[b])
                    for a, b in zip(starts[:-1], starts[1:])
                ]
                confidence = _safe_mean(sims)
                if confidence < self.registry.bar_match_threshold:
                    continue
                start_bars = set(starts)
                members = [
                    o.object_id for o in objects
                    if self.meter.bar_index(o.creation_time) in start_bars
                ]
                element_ids = sorted({
                    eid
                    for oid in members
                    for eid in self.elements.metadata.get(oid, EDMObjectMetadata(oid)).element_memberships
                })
                pid = "pattern_" + uuid.uuid4().hex[:8]
                found.append(PatternInstance(
                    pattern_id=pid,
                    period_bars=period,
                    start_bar=min(starts),
                    end_bar=end_bar,
                    element_ids=element_ids,
                    member_object_ids=members,
                    confidence=confidence,
                    variation_tolerance=float(1.0 - confidence),
                    signature=signatures[end_bar].tolist(),
                ))
                break

        # Retain only non-overlapping high-confidence hypotheses.
        found.sort(key=lambda p: (-p.confidence, p.period_bars, p.start_bar))
        selected: List[PatternInstance] = []
        occupied: set = set()
        for pattern in found:
            footprint = set(range(pattern.start_bar, pattern.end_bar + 1))
            if footprint.intersection(occupied):
                continue
            occupied.update(footprint)
            selected.append(pattern)

        # Preserve pattern identity across updates when the period and recent
        # footprint still agree. This is deliberately conservative: a changed
        # period becomes a new pattern hypothesis rather than mutating history.
        reconciled: Dict[str, PatternInstance] = {}
        for pattern in selected:
            best = None
            for prior in self.patterns.values():
                if prior.period_bars != pattern.period_bars:
                    continue
                overlap = len(
                    set(range(pattern.start_bar, pattern.end_bar + 1))
                    & set(range(prior.start_bar, prior.end_bar + 1))
                )
                span = max(
                    len(set(range(pattern.start_bar, pattern.end_bar + 1))),
                    len(set(range(prior.start_bar, prior.end_bar + 1))),
                    1,
                )
                if overlap / span >= self.registry.bar_match_threshold:
                    best = prior
                    break
            if best is not None:
                pattern.pattern_id = best.pattern_id
                pattern.confidence = float(
                    0.70 * best.confidence + 0.30 * pattern.confidence
                )
            reconciled[pattern.pattern_id] = pattern

        self.patterns = reconciled
        for pattern in self.patterns.values():
            for oid in pattern.member_object_ids:
                if oid in self.elements.metadata:
                    self.elements.metadata[oid].pattern_memberships.append(pattern.pattern_id)
            self.world.record_decision(awm.DecisionRecord(
                time=t, module="PatternEngine", decision_type="pattern_hypothesis",
                target_id=pattern.pattern_id, candidates=[], chosen=pattern.pattern_id,
                reason=f"{pattern.period_bars}-bar tolerant repetition",
                confidence=pattern.confidence
            ))
            self.world.emit_event(
                "pattern_instance",
                pattern.pattern_id,
                {"period_bars": pattern.period_bars, "start_bar": pattern.start_bar,
                 "end_bar": pattern.end_bar, "elements": pattern.element_ids},
                pattern.confidence, t
            )
        return self.patterns


class StructureEngine:
    """Infer section/phrase-scale boundaries from accumulated bar evidence."""

    LABELS = ("intro", "build", "drop", "break", "breakdown", "post-drop", "outro")

    def __init__(self, world, registry: EDMParameterRegistry, elements: ElementEngine, patterns: PatternEngine):
        self.world = world
        self.registry = registry
        self.elements = elements
        self.patterns = patterns
        self.meter = EDMMeter(world)
        self.sections: List[StructuralSpan] = []
        self._bar_features: Dict[int, np.ndarray] = {}

    def _bar_feature(self, bar: int, objects: Sequence[Any]) -> np.ndarray:
        start, end = bar * self.meter.bar_period, (bar + 1) * self.meter.bar_period
        in_bar = [o for o in objects if start <= o.creation_time < end]
        if not in_bar:
            return np.zeros(11, dtype=float)
        energies = np.asarray([max(float(o.physical_signature.energy), 0.0) for o in in_bar])
        roles = np.mean([
            [o.role_scores.kick, o.role_scores.bass, o.role_scores.percussion,
             o.role_scores.harmonic_pad, o.role_scores.lead_melody]
            for o in in_bar
        ], axis=0)
        low = _safe_mean([float(o.physical_signature.frequency_hz < 140.0) for o in in_bar])
        fx = _safe_mean([float(max(o.physical_signature.spectral_centroid, 0.0) > 3500.0) for o in in_bar])
        unique_elements = len({
            eid for o in in_bar for eid in
            self.elements.metadata.get(o.object_id, EDMObjectMetadata(o.object_id)).element_memberships
        })
        return np.asarray([
            min(len(in_bar) / 16.0, 1.0),
            float(np.clip(np.mean(energies), 0, 1)),
            float(np.clip(np.max(energies), 0, 1)),
            low,
            fx,
            min(unique_elements / 8.0, 1.0),
            *roles,
        ], dtype=float)

    def _novelty(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.clip(1.0 - _cosine(a, b), 0.0, 1.0))

    def _labels(self, feature: np.ndarray, prev: Optional[np.ndarray]) -> Dict[str, float]:
        density, mean_e, peak_e, low, fx, elements = feature[:6]
        delta = 0.0 if prev is None else float(mean_e - prev[1])
        raw = {
            "intro": max(0.0, 0.6 * (1 - density) + 0.4 * (1 - elements)),
            "build": max(0.0, 0.7 * max(delta, 0) + 0.3 * fx),
            "drop": max(0.0, 0.6 * density + 0.3 * low + 0.1 * peak_e),
            "break": max(0.0, 0.8 * (1 - density) + 0.2 * (1 - low)),
            "breakdown": max(0.0, 0.7 * (1 - density) + 0.3 * (1 - low)),
            "post-drop": max(0.0, 0.5 * density + 0.5 * max(delta, 0)),
            "outro": max(0.0, 0.5 * (1 - density) + 0.5 * (1 - mean_e)),
        }
        s = sum(raw.values()) or 1.0
        return {k: float(v / s) for k, v in raw.items()}

    def update(self, t: float) -> List[StructuralSpan]:
        objects = [
            o for o in self.world.objects.values()
            if o.status != awm.ObjectStatus.ARCHIVED and o.creation_time <= t
        ]
        last_bar = self.meter.bar_index(t)
        self._bar_features = {b: self._bar_feature(b, objects) for b in range(last_bar + 1)}
        if last_bar + 1 < self.registry.section_min_bars:
            self.sections = []
            return []

        boundaries = [0]
        last_boundary = 0
        for b in range(1, last_bar + 1):
            prev = self._bar_features[b - 1]
            cur = self._bar_features[b]
            novelty = self._novelty(cur, prev)
            if (b - last_boundary) >= max(self.registry.section_min_bars, self.registry.section_refractory_bars) and novelty >= self.registry.section_novelty_threshold:
                boundaries.append(b)
                last_boundary = b

        sections: List[StructuralSpan] = []
        for i, start_bar in enumerate(boundaries):
            end_bar = (boundaries[i + 1] - 1) if i + 1 < len(boundaries) else last_bar
            fs = np.mean([self._bar_features[b] for b in range(start_bar, end_bar + 1)], axis=0)
            prev = self._bar_features[start_bar - 1] if start_bar > 0 else None
            labels = self._labels(fs, prev)
            novelty = 0.0 if start_bar == 0 else self._novelty(
                self._bar_features[start_bar], self._bar_features[start_bar - 1]
            )
            sid = "section_" + uuid.uuid4().hex[:8]
            section = StructuralSpan(
                section_id=sid,
                start_bar=start_bar,
                end_bar=end_bar,
                start_time=start_bar * self.meter.bar_period,
                end_time=min(t, (end_bar + 1) * self.meter.bar_period),
                label_candidates=labels,
                novelty_score=novelty,
                energy_mean=float(fs[1]),
                density_mean=float(fs[0]),
                evidence={
                    "bar_count": float(end_bar - start_bar + 1),
                    "pattern_count": float(len(self.patterns.patterns)),
                    "element_count": float(len(self.elements.elements)),
                    "low_end": float(fs[3]),
                    "fx_activity": float(fs[4]),
                },
            )
            sections.append(section)

        self.sections = sections
        for section in sections:
            self.world.record_decision(awm.DecisionRecord(
                time=section.start_time, module="StructureEngine",
                decision_type="section_hypothesis", target_id=section.section_id,
                candidates=[], chosen=section.section_id,
                reason="bar-level density/energy/element/pattern/low-end/FX change",
                confidence=max(section.novelty_score, 0.5)
            ))
            self.world.emit_event(
                "structural_span",
                section.section_id,
                {"start_bar": section.start_bar, "end_bar": section.end_bar,
                 "labels": section.label_candidates},
                max(section.novelty_score, 0.5),
                section.start_time
            )
            member_ids = [
                o.object_id for o in objects
                if section.start_time <= o.creation_time <= section.end_time
            ]
            for oid in member_ids:
                if oid in self.elements.metadata:
                    self.elements.metadata[oid].section_memberships.append(section.section_id)
        return sections


class ArrangementEngine:
    """Produce a song-level ordered arrangement hypothesis."""

    def __init__(self, world, registry: EDMParameterRegistry, structure: StructureEngine):
        self.world = world
        self.registry = registry
        self.structure = structure
        self.arrangement: Optional[Arrangement] = None

    def update(self, t: float) -> Optional[Arrangement]:
        sections = [
            s for s in self.structure.sections
            if max(s.label_candidates.values() or [0.0]) >= self.registry.arrangement_confidence_floor
        ]
        if not sections:
            self.arrangement = None
            return None

        label_mass = {label: _safe_mean([s.label_candidates.get(label, 0.0) for s in sections])
                      for label in StructureEngine.LABELS}
        confidence = float(np.clip(
            _safe_mean([max(s.label_candidates.values()) for s in sections]) *
            (0.5 + 0.5 * min(len(sections) / 8.0, 1.0)), 0.0, 0.97
        ))
        if self.arrangement is None:
            aid = "arrangement_" + uuid.uuid4().hex[:8]
            created = sections[0].start_time
        else:
            aid = self.arrangement.arrangement_id
            created = self.arrangement.created_time
        self.arrangement = Arrangement(
            arrangement_id=aid,
            section_ids=[s.section_id for s in sections],
            confidence=confidence,
            created_time=created,
            updated_time=t,
            evidence={
                "section_count": float(len(sections)),
                "intro_mass": label_mass["intro"],
                "build_mass": label_mass["build"],
                "drop_mass": label_mass["drop"],
                "break_mass": label_mass["break"],
                "outro_mass": label_mass["outro"],
            },
        )
        self.world.record_decision(awm.DecisionRecord(
            time=t, module="ArrangementEngine", decision_type="arrangement_hypothesis",
            target_id=aid, candidates=[], chosen=aid,
            reason="ordered structural spans with probabilistic EDM labels",
            confidence=confidence
        ))
        self.world.emit_event(
            "arrangement_hypothesis", aid,
            {"sections": self.arrangement.section_ids, "evidence": self.arrangement.evidence},
            confidence, t
        )
        return self.arrangement


class EDMEvaluation:
    """Annotation-driven evaluation metrics required by the addendum."""

    @staticmethod
    def _match_times(predicted: Sequence[float], truth: Sequence[float], tolerance: float) -> Tuple[int, int, int]:
        used = set()
        tp = 0
        for p in predicted:
            best = None
            best_err = tolerance
            for i, t in enumerate(truth):
                if i in used:
                    continue
                err = abs(float(p) - float(t))
                if err <= best_err:
                    best = i
                    best_err = err
            if best is not None:
                used.add(best)
                tp += 1
        return tp, len(predicted), len(truth)

    @staticmethod
    def _prf(tp: int, pred: int, truth: int) -> Dict[str, float]:
        precision = tp / pred if pred else 0.0
        recall = tp / truth if truth else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    def evaluate_events(self, predicted: Sequence[float], truth: Sequence[float], tolerances: Sequence[float]) -> Dict[str, Any]:
        return {
            f"{tol_ms:.1f}ms": self._prf(
                *self._match_times(predicted, truth, tol_ms / 1000.0)
            )
            for tol_ms in tolerances
        }

    def evaluate_boundaries(self, predicted: Sequence[float], truth: Sequence[float], tolerances: Sequence[float]) -> Dict[str, Any]:
        return self.evaluate_events(predicted, truth, tolerances)

    @staticmethod
    def identity_accuracy(predicted_clusters: Dict[str, str], truth_clusters: Dict[str, str]) -> float:
        pairs = [(k, truth_clusters[k]) for k in truth_clusters if k in predicted_clusters]
        if not pairs:
            return 0.0
        return float(np.mean([predicted_clusters[k] == truth for k, truth in pairs]))

    @staticmethod
    def false_merge_rate(predicted_clusters: Sequence[Sequence[str]], truth_labels: Dict[str, str]) -> float:
        if not predicted_clusters:
            return 0.0
        bad = 0
        for cluster in predicted_clusters:
            labels = {truth_labels[x] for x in cluster if x in truth_labels}
            if len(labels) > 1:
                bad += 1
        return bad / len(predicted_clusters)

    @staticmethod
    def false_split_rate(predicted_clusters: Sequence[Sequence[str]], truth_labels: Dict[str, str]) -> float:
        groups: Dict[str, int] = {}
        for cluster in predicted_clusters:
            labels = {truth_labels[x] for x in cluster if x in truth_labels}
            for label in labels:
                groups[label] = groups.get(label, 0) + 1
        if not groups:
            return 0.0
        return float(np.mean([max(0, count - 1) / count for count in groups.values()]))


class EDMReverseDAW:
    """Top-level additive orchestration and JSON persistence."""

    VERSION = "0.1.0"

    def __init__(self, world, registry: Optional[EDMParameterRegistry] = None):
        self.world = world
        self.registry = registry or EDMParameterRegistry()
        self.elements = ElementEngine(world, self.registry)
        self.patterns = PatternEngine(world, self.registry, self.elements)
        self.structure = StructureEngine(world, self.registry, self.elements, self.patterns)
        self.arrangement_engine = ArrangementEngine(world, self.registry, self.structure)
        self.evaluation = EDMEvaluation()
        self.source_hypotheses: Dict[str, Any] = {}

    def run(self, t: Optional[float] = None) -> Dict[str, Any]:
        analysis_time = self.world.t if t is None else float(t)

        if source_hyp is not None and not self.source_hypotheses:
            try:
                engine = source_hyp.SourceHypothesisEngine(self.world)
                self.source_hypotheses = {
                    h.hypothesis_id: h for h in engine.update(analysis_time)
                }
            except Exception as exc:
                # Source hypotheses are an optional upstream stage. EDM layers
                # remain usable when an adapter/model is unavailable.
                self.world.record_decision(awm.DecisionRecord(
                    time=analysis_time, module="EDMReverseDAW",
                    decision_type="source_hypothesis_unavailable",
                    target_id=None, candidates=[], chosen=None,
                    reason=f"upstream source-hypothesis adapter unavailable: {exc}",
                    confidence=0.0
                ))

        self.elements.update(analysis_time)
        self.patterns.update(analysis_time)
        self.structure.update(analysis_time)
        self.arrangement_engine.update(analysis_time)
        self._attach_rhythm_context(analysis_time)
        return self.snapshot()

    def _attach_rhythm_context(self, t: float) -> None:
        meter = EDMMeter(self.world)
        live = [
            o for o in self.world.objects.values()
            if o.status != awm.ObjectStatus.ARCHIVED and o.creation_time <= t
        ]
        for obj in live:
            meta = self.elements.metadata.get(obj.object_id)
            if meta is None:
                continue
            meta.beat_index = meter.beat_index(obj.creation_time)
            meta.bar_index = meter.bar_index(obj.creation_time)
            meta.beat_position = meter.beat_position(obj.creation_time)
            meta.timing_residual_ms = meter.nearest_grid_residual_ms(obj.creation_time)

            window_start = max(0.0, obj.creation_time - 2.0 * meter.beat_period)
            nearby = [
                other for other in live
                if other.object_id != obj.object_id
                and window_start <= other.creation_time <= obj.creation_time + 2.0 * meter.beat_period
            ]
            meta.local_density = len(nearby) / max(4.0 * meter.beat_period, 1e-6)
            meta.periodicity = self._estimate_periodicity(obj, live, meter)
            meta.tail_ownership = self._estimate_tail_ownership(obj)
            meta.overlap_score = self._estimate_overlap(obj, nearby)

            distances = sorted(
                (
                    other.object_id,
                    abs(other.creation_time - obj.creation_time),
                )
                for other in nearby
            )[:8]
            meta.nearest_neighbors = [
                (oid, float(np.clip(1.0 - d / max(2.0 * meter.beat_period, 1e-6), 0.0, 1.0)))
                for oid, d in distances
            ]

        # Identity clusters are kept as evidence, not object merges. Element
        # ids are useful cluster anchors for later nearest-neighbor retrieval.
        for family in self.elements.elements.values():
            for oid in family.member_ids:
                if oid in self.elements.metadata:
                    self.elements.metadata[oid].identity_cluster_ids.append(family.element_id)

    @staticmethod
    def _estimate_periodicity(obj, live: Sequence[Any], meter: EDMMeter) -> float:
        candidates = sorted([
            abs(other.creation_time - obj.creation_time)
            for other in live if other.object_id != obj.object_id
        ])
        if len(candidates) < 3:
            return 0.0
        target = meter.beat_period
        distances = [abs(d - target) for d in candidates[:12]]
        return float(np.clip(1.0 - _safe_mean(distances) / max(target, 1e-6), 0, 1))

    @staticmethod
    def _estimate_tail_ownership(obj) -> float:
        history = getattr(obj, "history", [])
        if len(history) < 2:
            return 0.0
        e = np.asarray([max(float(h.energy), 0.0) for h in history[-8:]])
        if e[-1] <= 1e-9:
            return 0.0
        decay = np.diff(e)
        return float(np.clip(np.mean(decay < 0), 0, 1))

    @staticmethod
    def _estimate_overlap(obj, nearby: Sequence[Any]) -> float:
        if not nearby:
            return 0.0
        f0 = max(obj.physical_signature.frequency_hz, 20.0)
        overlaps = []
        for other in nearby:
            f1 = max(other.physical_signature.frequency_hz, 20.0)
            overlaps.append(math.exp(-abs(math.log2(f0 / f1)) / 1.5))
        return float(np.clip(np.mean(overlaps), 0, 1))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "time": self.world.t,
            "parameters": [asdict(p) for p in self.registry.registry()],
            "source_hypotheses": {k: asdict(v) for k, v in self.source_hypotheses.items()},
            "object_metadata": {k: asdict(v) for k, v in self.elements.metadata.items()},
            "elements": {k: asdict(v) for k, v in self.elements.elements.items()},
            "patterns": {k: asdict(v) for k, v in self.patterns.patterns.items()},
            "sections": [asdict(v) for v in self.structure.sections],
            "arrangement": asdict(self.arrangement_engine.arrangement)
            if self.arrangement_engine.arrangement else None,
        }

    def export_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.snapshot(), indent=2, sort_keys=True), encoding="utf-8")
        return target


def _run_demo(audio_path: Optional[str], output: Optional[str]) -> None:
    params = awm.Parameters()
    source = awm.AudioSource(params)
    audio, sr = source.load(audio_path)
    model = awm.AuditoryWorldModel(params)
    model.run(audio, use_stem_separation=False)
    edm = EDMReverseDAW(model.world)
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
    if output:
        path = edm.export_json(output)
        print(f"exported={path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the additive EDM reverse-DAW layer.")
    parser.add_argument("audio", nargs="?", default=None)
    parser.add_argument("--export-json", default=None, help="Write a JSON interchange snapshot.")
    args = parser.parse_args()
    _run_demo(args.audio, args.export_json)


if __name__ == "__main__":
    main()
