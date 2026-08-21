#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUDITORY WORLD MODEL — FOUNDATION IMPLEMENTATION
==================================================
Implements, per the CANONICAL SPECIFICATION (v1.1) and the IMPLEMENTATION
SPEC WITH NUMERICAL CONTRACT, the STRICT foundation phase requested:

  1. Audio input + clock + buffering
  2. Multi-resolution physical analysis (short + long STFT, energy,
     spectral features, transient detection)
  3. Observation generation
  4. Full SoundObject tracking: multi-cue matching, hysteresis, prediction,
     lifecycle TENTATIVE -> ACTIVE -> HIDDEN -> RECOVERING -> ARCHIVED
  5. Non-destructive perceptual masking + loudness
  6. Basic relationship graph
  7. Event generation
  8. Full WorldState with provenance and decision log
  9. Replayable history
  10. A working demonstration (synthetic or real audio) with printed
      summary, printed events, a matplotlib plot, and a belief-at-time-T
      inspector.

Explicitly DEFERRED (per spec priority map, Level C/D): full layer
grouping with merge/split, rhythm/beat/groove, semantic roles, style /
genre interpretation, embeddings, multi-hypothesis escalation, and
realtime threading. Interfaces are written so those layers can be added
without breaking this foundation (Part 41 Sec 21, "Replacement Rule").

Colab / T4 usage:
    # !pip install soundfile librosa   # optional, only needed for real
    #                                   # audio files / alternate resampling
    !python auditory_world_model.py my_track.wav
    # or, with no file, a synthetic multi-object test signal is generated.

Only these libraries are used: numpy, scipy, soundfile (optional),
librosa (optional), sounddevice (optional, unused for pure playback here),
torch (optional GPU acceleration), matplotlib, IPython.display (optional).
"""

from __future__ import annotations

import itertools
import logging
import math
import sys
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import get_window, resample_poly, butter, lfilter

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import soundfile as sf
    HAVE_SOUNDFILE = True
except Exception:
    HAVE_SOUNDFILE = False

try:
    import librosa
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False

try:
    import torch
    HAVE_TORCH = True
    HAVE_CUDA = torch.cuda.is_available()
except Exception:
    HAVE_TORCH = False
    HAVE_CUDA = False

try:
    from IPython.display import display, Audio  # noqa: F401
    HAVE_IPYTHON = True
except Exception:
    HAVE_IPYTHON = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("AWM")
LOG.info("Backends: soundfile=%s librosa=%s torch=%s cuda=%s ipython=%s",
         HAVE_SOUNDFILE, HAVE_LIBROSA, HAVE_TORCH, HAVE_CUDA, HAVE_IPYTHON)


# =============================================================================
# 1. PARAMETER REGISTRY  (Numerical Contract, Part 57/64)
# =============================================================================

class ParamClass(Enum):
    LOCKED = "LOCKED"
    DEFAULT = "DEFAULT"
    DERIVED = "DERIVED"
    ADAPTIVE = "ADAPTIVE"
    TUNABLE = "TUNABLE"
    OPEN = "OPEN"


@dataclass(frozen=True)
class ParamSpec:
    name: str
    value: float
    unit: str
    classification: ParamClass
    owner: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None


@dataclass
class Parameters:
    # --- audio engine / physical analysis (LOCKED, from numerical contract) ---
    sample_rate: int = 48000
    fast_hop: int = 256
    short_fft_size: int = 1024
    long_fft_size: int = 4096
    long_hop: int = 1024                       # DERIVED = 4 * fast_hop

    # --- tracking / lifecycle (LOCKED, from numerical contract) ---
    candidate_confirmation_window_ms: float = 50.0
    prediction_horizon_ms: float = 100.0
    recovery_window_ms: float = 250.0
    archive_delay_ms: float = 2000.0
    strong_match: float = 0.80
    candidate_match: float = 0.55
    reject_match: float = 0.40
    confirmation_confidence: float = 0.70       # DEFAULT, Part 59 "Object creation"

    # --- masking (DEFAULT, Part 51/14) ---
    mask_attack_ms: float = 20.0
    mask_release_ms: float = 150.0

    # --- grouping / relationships tolerances (DEFAULT, Part 51/15-16) ---
    temporal_group_tolerance_ms: float = 30.0
    strong_temporal_group_tolerance_ms: float = 10.0
    merge_confirmation: int = 3
    split_confirmation: int = 3
    minimum_split_duration_ms: float = 30.0
    minimum_merge_duration_ms: float = 30.0

    # --- prediction error (DEFAULT, Part 51/17) ---
    prediction_warning_threshold: float = 0.25
    prediction_failure_threshold: float = 0.50

    # --- events (DEFAULT, Part 51/20) ---
    same_event_debounce_ms: float = 20.0

    # --- resource limits (DEFAULT, Part 51/21) ---
    soft_object_limit: int = 256
    hard_object_limit: int = 1024
    working_observation_limit: int = 4096

    # --- matching weights (TUNABLE, versioned, Part 59) ---
    w_frequency: float = 0.25
    w_energy: float = 0.15
    w_envelope: float = 0.15
    w_harmonic: float = 0.20
    w_temporal: float = 0.15
    w_prediction: float = 0.10

    def registry(self) -> List[ParamSpec]:
        L, D, DER, T = (ParamClass.LOCKED, ParamClass.DEFAULT,
                        ParamClass.DERIVED, ParamClass.TUNABLE)
        return [
            ParamSpec("sample_rate", self.sample_rate, "Hz", L, "AudioEngine"),
            ParamSpec("fast_hop", self.fast_hop, "samples", L, "PhysicalAnalyzer"),
            ParamSpec("short_fft_size", self.short_fft_size, "samples", L, "PhysicalAnalyzer"),
            ParamSpec("long_fft_size", self.long_fft_size, "samples", L, "PhysicalAnalyzer"),
            ParamSpec("long_hop", self.long_hop, "samples", DER, "PhysicalAnalyzer"),
            ParamSpec("candidate_confirmation_window_ms", self.candidate_confirmation_window_ms, "ms", L, "ObjectTracker"),
            ParamSpec("prediction_horizon_ms", self.prediction_horizon_ms, "ms", L, "ObjectTracker"),
            ParamSpec("recovery_window_ms", self.recovery_window_ms, "ms", L, "ObjectTracker"),
            ParamSpec("archive_delay_ms", self.archive_delay_ms, "ms", L, "ObjectTracker"),
            ParamSpec("strong_match", self.strong_match, "score", L, "ObjectTracker", 0, 1),
            ParamSpec("candidate_match", self.candidate_match, "score", L, "ObjectTracker", 0, 1),
            ParamSpec("reject_match", self.reject_match, "score", L, "ObjectTracker", 0, 1),
            ParamSpec("confirmation_confidence", self.confirmation_confidence, "score", D, "ObjectTracker", 0, 1),
            ParamSpec("mask_attack_ms", self.mask_attack_ms, "ms", D, "PerceptualModel"),
            ParamSpec("mask_release_ms", self.mask_release_ms, "ms", D, "PerceptualModel"),
            ParamSpec("temporal_group_tolerance_ms", self.temporal_group_tolerance_ms, "ms", D, "RelationshipEngine"),
            ParamSpec("strong_temporal_group_tolerance_ms", self.strong_temporal_group_tolerance_ms, "ms", D, "RelationshipEngine"),
            ParamSpec("merge_confirmation", self.merge_confirmation, "updates", D, "ObjectTracker"),
            ParamSpec("split_confirmation", self.split_confirmation, "updates", D, "ObjectTracker"),
            ParamSpec("minimum_split_duration_ms", self.minimum_split_duration_ms, "ms", D, "ObjectTracker"),
            ParamSpec("minimum_merge_duration_ms", self.minimum_merge_duration_ms, "ms", D, "ObjectTracker"),
            ParamSpec("prediction_warning_threshold", self.prediction_warning_threshold, "norm.err", D, "ObjectTracker", 0, 1),
            ParamSpec("prediction_failure_threshold", self.prediction_failure_threshold, "norm.err", D, "ObjectTracker", 0, 1),
            ParamSpec("same_event_debounce_ms", self.same_event_debounce_ms, "ms", D, "WorldState"),
            ParamSpec("soft_object_limit", self.soft_object_limit, "count", D, "ObjectTracker"),
            ParamSpec("hard_object_limit", self.hard_object_limit, "count", D, "ObjectTracker"),
            ParamSpec("working_observation_limit", self.working_observation_limit, "count", D, "WorldState"),
            ParamSpec("w_frequency", self.w_frequency, "weight", T, "ObjectTracker", 0, 1),
            ParamSpec("w_energy", self.w_energy, "weight", T, "ObjectTracker", 0, 1),
            ParamSpec("w_envelope", self.w_envelope, "weight", T, "ObjectTracker", 0, 1),
            ParamSpec("w_harmonic", self.w_harmonic, "weight", T, "ObjectTracker", 0, 1),
            ParamSpec("w_temporal", self.w_temporal, "weight", T, "ObjectTracker", 0, 1),
            ParamSpec("w_prediction", self.w_prediction, "weight", T, "ObjectTracker", 0, 1),
        ]


def print_parameter_registry(params: Parameters) -> None:
    print("\n" + "=" * 92)
    print("PARAMETER REGISTRY  (numerical contract — every constant has unit/owner/class)")
    print("=" * 92)
    print(f"{'name':<36}{'value':>10}  {'unit':<10}{'class':<10}{'owner':<18}")
    print("-" * 92)
    for spec in params.registry():
        print(f"{spec.name:<36}{str(spec.value):>10}  {spec.unit:<10}{spec.classification.value:<10}{spec.owner:<18}")
    print("=" * 92)


# =============================================================================
# 2. CORE DATA MODEL  (WorldState, Observation, SoundObject, Update/Decision)
# =============================================================================

class ObjectStatus(Enum):
    TENTATIVE = "TENTATIVE"
    ACTIVE = "ACTIVE"
    HIDDEN = "HIDDEN"
    RECOVERING = "RECOVERING"
    ARCHIVED = "ARCHIVED"


class ObservationType(Enum):
    TRANSIENT = "TRANSIENT"
    TONAL = "TONAL"
    NOISE_TEXTURE = "NOISE_TEXTURE"


class RelationshipType(Enum):
    MASKS = "MASKS"
    HARMONIC_RELATION = "HARMONIC_RELATION"
    CO_OCCURS_WITH = "CO_OCCURS_WITH"
    SECTION_MEMBERSHIP = "SECTION_MEMBERSHIP"


@dataclass
class Confidence:
    """Multidimensional confidence (Sec 4.7 / Part 41 Sec 8). Never collapsed
    to a single number except via an explicit, inspectable aggregate()."""
    existence: float = 0.0
    identity: float = 0.0
    perceptual: float = 1.0
    prediction: float = 0.5
    grouping: float = 0.5

    def aggregate(self) -> float:
        w = {"existence": 0.30, "identity": 0.30, "perceptual": 0.20,
             "prediction": 0.10, "grouping": 0.10}
        return float(self.existence * w["existence"] + self.identity * w["identity"]
                     + self.perceptual * w["perceptual"] + self.prediction * w["prediction"]
                     + self.grouping * w["grouping"])


@dataclass
class Observation:
    """Temporary evidence (Sec 4.1 / Invariant I1: no permanent identity)."""
    id: str
    obs_type: ObservationType
    start_time: float
    end_time: float
    frequency_hz: float
    bandwidth_hz: float
    energy: float
    band_energy: float
    bark_band: int
    physical_features: Dict[str, float]
    confidence: float
    provenance: str


@dataclass
class Prediction:
    """Expected future state, never a fact (Part 41 Sec 12)."""
    target_id: str
    made_at: float
    expected_time: float
    expected_frequency_hz: float
    expected_energy: float
    confidence: float
    basis: str


@dataclass
class PhysicalSignature:
    frequency_hz: float = 0.0          # most recent measured/matched frequency (reporting)
    stable_frequency_hz: float = 0.0   # slow-moving anchor used for identity matching (anti-drift)
    bandwidth_hz: float = 0.0
    energy: float = 0.0
    band_energy: float = 0.0
    envelope: float = 0.0
    harmonicity: float = 0.0
    spectral_centroid: float = 0.0
    spectral_flatness: float = 0.5
    crest_factor: float = 1.0


@dataclass
class PerceptualState:
    loudness: float = 0.0
    perceptual_availability: float = 1.0
    masked_by: List[str] = field(default_factory=list)


@dataclass
class RoleScores:
    """Probabilistic, simultaneous role assignment (Sec 15: 'Semantic
    interpretation assigns probabilistic roles to objects... Do not turn
    semantic roles into immutable classifications.'). An object holds a
    score for EVERY role at once -- never collapsed to a single label."""
    kick: float = 0.0
    bass: float = 0.0
    percussion: float = 0.0
    harmonic_pad: float = 0.0
    lead_melody: float = 0.0
    last_updated: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {"kick": self.kick, "bass": self.bass, "percussion": self.percussion,
                "harmonic_pad": self.harmonic_pad, "lead_melody": self.lead_melody}

    def top(self, k: int = 2) -> List[Tuple[str, float]]:
        return sorted(self.as_dict().items(), key=lambda kv: -kv[1])[:k]


@dataclass
class HistoryEntry:
    """One replayable snapshot of a SoundObject at a point in time."""
    time: float
    status: str
    confidence: Dict[str, float]
    frequency_hz: float
    energy: float
    perceptual_availability: float
    timbre_class: str = "UNKNOWN"
    timbre_confidence: float = 0.0


@dataclass
class SoundObject:
    """Persistent perceptual entity (Sec 4.4). Stable id, survives gaps."""
    object_id: str
    status: ObjectStatus
    creation_time: float
    last_observed_time: float
    last_active_time: float
    confidence: Confidence
    physical_signature: PhysicalSignature
    perceptual_state: PerceptualState
    prediction: Optional[Prediction] = None
    history: List[HistoryEntry] = field(default_factory=list)
    relationship_ids: List[str] = field(default_factory=list)
    consecutive_confirmations: int = 0
    consecutive_misses: int = 0
    hidden_since: Optional[float] = None
    recovering_since: Optional[float] = None
    timbre_class: str = "UNKNOWN"
    timbre_confidence: float = 0.0
    role_scores: RoleScores = field(default_factory=RoleScores)

    def age(self, t: float) -> float:
        return t - self.creation_time


@dataclass
class Relationship:
    """Typed, weighted, first-class edge (Sec 4.6 / Part 41 Sec 14)."""
    rel_id: str
    source_id: str
    target_id: str
    rel_type: RelationshipType
    strength: float
    confidence: float
    evidence_count: int
    first_seen: float
    last_updated: float


@dataclass
class Layer:
    """A perceptual grouping of >=2 SoundObjects that consistently co-occur
    (Sec 36 Level B 'layer grouping', e.g. a kick's low-frequency body +
    its broadband click transient). Strictly additive/non-destructive:
    member objects keep their own identity, history and tracking untouched
    -- a Layer only ever references member_ids, it never merges or deletes
    the underlying objects."""
    layer_id: str
    member_ids: List[str] = field(default_factory=list)
    status: str = "CONFIRMED"        # "CONFIRMED" | "DISSOLVED"
    confidence: float = 0.6
    formed_time: float = 0.0
    last_updated: float = 0.0
    dissolved_time: Optional[float] = None
    formation_reason: str = ""


@dataclass
class GrooveVector:
    """Structured, multi-component rhythmic state (Sec 14: 'Groove is not
    BPM. Groove is the relationship between events through time.'; Sec 14.1:
    'Maintain a structured state rather than one label.'). This is
    CONTINUOUS state on WorldState (Sec 20 explicitly lists 'groove' among
    continuous state, not an event stream) -- discrete beat/accent/
    groove_changed events are emitted only for meaningful transitions on
    top of this vector, by RhythmEngine."""
    tempo_bpm: float = 0.0
    tempo_confidence: float = 0.0
    beat_phase: float = 0.0                # 0..1, position within the current beat period
    next_beat_time: float = 0.0
    timing_deviation_ms: float = 0.0       # microtiming: avg |onset - nearest grid position|
    swing_ratio: float = 0.5               # 0.5 = straight; >0.5 = long-short swung feel
    syncopation_index: float = 0.0         # 0 = all onset energy on strong beats, 1 = all off-beat
    event_density: float = 0.0             # onsets/sec, recent window
    regularity: float = 0.0                # 0..1, inverse of inter-onset-interval CV
    accent_positions: List[float] = field(default_factory=list)  # relative energy per beat subdivision
    last_updated: float = 0.0


@dataclass
class Pattern:
    """A repeated rhythmic/temporal pattern (Sec 12.4, 14.2). Identity
    tolerates variation -- 'A repeated musical pattern is not required to
    have sample-identical events' (Sec 14.2) -- so matching is a similarity
    threshold over a per-bar signature, not exact repetition."""
    pattern_id: str
    period_bars: int
    confidence: float
    occurrence_count: int
    first_seen: float
    last_seen: float
    status: str = "ACTIVE"   # "ACTIVE" | "BROKEN"


@dataclass
class Section:
    """A phrase/section span (Sec 14.3: 'Use accumulated object/pattern/
    energy/structural changes to infer phrases and sections.'). This
    foundation implementation treats phrase and section as one mechanism,
    per the spec's own single '14.3 Phrase/section model' heading -- finer
    phrase-vs-section distinction is a documented follow-up, not built here.
    'Fast events should not wait for section analysis' (Sec 14.3) -- this
    runs on the bar-level slow loop and never blocks per-hop processing."""
    section_id: str
    start_time: float
    end_time: Optional[float]
    boundary_reason: str
    novelty_score: float


@dataclass
class Event:
    event_id: str
    time: float
    event_type: str
    target_id: Optional[str]
    payload: Dict[str, Any]
    confidence: float


@dataclass
class UpdateRecord:
    """A proposed/applied mutation with provenance (Part 41 Sec 4)."""
    time: float
    module: str
    target_id: str
    property: str
    old_value: Any
    new_value: Any
    reason: str
    confidence: float


@dataclass
class DecisionRecord:
    """Why: why does an object exist / why was it matched / why did it change."""
    time: float
    module: str
    decision_type: str
    target_id: Optional[str]
    candidates: List[Dict[str, Any]]
    chosen: Optional[str]
    reason: str
    confidence: float


class WorldState:
    """The single, authoritative source of truth (Sec 3 / Part 41 Sec 3).

    Only WorldState performs authoritative mutation (apply_update,
    register_object, upsert_relationship, emit_event). All other modules
    compute proposals and hand them to WorldState -- they never poke a
    SoundObject's attributes directly.
    """

    def __init__(self, params: Parameters):
        self.params = params
        self.t: float = 0.0
        self.objects: Dict[str, SoundObject] = {}
        self.observations: List[Observation] = []
        self.relationships: Dict[Tuple[str, str, str], Relationship] = {}
        self.layers: Dict[str, Layer] = {}
        self.groove: GrooveVector = GrooveVector()
        self.groove_history: List[GrooveVector] = []
        self.patterns: Dict[str, Pattern] = {}
        self.current_pattern: Optional[Pattern] = None
        self.sections: List[Section] = []
        self.events: List[Event] = []
        self.update_log: List[UpdateRecord] = []
        self.decision_log: List[DecisionRecord] = []
        self._last_event_time: Dict[Tuple[str, str], float] = {}
        self.logger = logging.getLogger("WorldState")

    # ---- identity ----
    def new_object_id(self) -> str:
        return f"obj_{uuid.uuid4().hex[:8]}"

    # ---- authoritative mutation ----
    def register_object(self, obj: SoundObject) -> None:
        if len(self.objects) >= self.params.hard_object_limit:
            self.logger.warning("hard_object_limit reached (%d); refusing new object %s",
                                 self.params.hard_object_limit, obj.object_id)
            return
        if len(self.objects) >= self.params.soft_object_limit:
            self.logger.warning("soft_object_limit exceeded (%d active objects)", len(self.objects))
        self.objects[obj.object_id] = obj

    def apply_update(self, upd: UpdateRecord) -> None:
        obj = self.objects.get(upd.target_id)
        if obj is None:
            self.logger.warning("apply_update: unknown target %s ignored", upd.target_id)
            return
        self._set_dotted(obj, upd.property, upd.new_value)
        self.update_log.append(upd)

    @staticmethod
    def _set_dotted(root: Any, dotted_property: str, value: Any) -> None:
        parts = dotted_property.split(".")
        target = root
        for p in parts[:-1]:
            target = getattr(target, p)
        setattr(target, parts[-1], value)

    def record_decision(self, dec: DecisionRecord) -> None:
        self.decision_log.append(dec)

    def upsert_relationship(self, source_id: str, target_id: str, rel_type: RelationshipType,
                             strength: float, confidence: float, t: float) -> Relationship:
        key = (source_id, target_id, rel_type.value)
        rel = self.relationships.get(key)
        if rel is None:
            rel = Relationship(rel_id=f"rel_{uuid.uuid4().hex[:8]}", source_id=source_id,
                                target_id=target_id, rel_type=rel_type, strength=float(strength),
                                confidence=float(confidence), evidence_count=1, first_seen=t, last_updated=t)
            self.relationships[key] = rel
            for oid in (source_id, target_id):
                o = self.objects.get(oid)
                if o is not None and rel.rel_id not in o.relationship_ids:
                    o.relationship_ids.append(rel.rel_id)
            self.emit_event("relationship_formed", source_id,
                             {"target": target_id, "type": rel_type.value, "strength": float(strength)},
                             float(confidence), t)
        else:
            rel.strength = float(0.7 * rel.strength + 0.3 * strength)
            rel.confidence = float(0.7 * rel.confidence + 0.3 * confidence)
            rel.evidence_count += 1
            rel.last_updated = t
        return rel

    # ---- layer grouping (Sec 36 Level B): additive, never mutates member objects -------------
    def form_layer(self, member_ids: List[str], t: float, reason: str) -> Layer:
        lid = f"layer_{uuid.uuid4().hex[:8]}"
        layer = Layer(layer_id=lid, member_ids=list(dict.fromkeys(member_ids)), status="CONFIRMED",
                       confidence=0.6, formed_time=t, last_updated=t, formation_reason=reason)
        self.layers[lid] = layer
        self.record_decision(DecisionRecord(time=t, module="GroupingEngine", decision_type="layer_formed",
                                             target_id=lid, candidates=[], chosen=lid,
                                             reason=reason, confidence=layer.confidence))
        self.emit_event("layer_formed", lid, {"members": list(layer.member_ids)}, layer.confidence, t)
        self.logger.info("[%7.3fs] LAYER FORMED %-12s members=%s (%s)", t, lid, layer.member_ids, reason)
        return layer

    def add_layer_member(self, layer_id: str, object_id: str, t: float, reason: str) -> None:
        layer = self.layers.get(layer_id)
        if layer is None or layer.status != "CONFIRMED" or object_id in layer.member_ids:
            return
        layer.member_ids.append(object_id)
        layer.last_updated = t
        layer.confidence = float(min(0.95, layer.confidence + 0.1))
        self.record_decision(DecisionRecord(time=t, module="GroupingEngine", decision_type="layer_member_added",
                                             target_id=layer_id, candidates=[], chosen=object_id,
                                             reason=reason, confidence=layer.confidence))
        self.emit_event("layer_member_added", layer_id, {"member": object_id}, layer.confidence, t)

    def merge_layers(self, layer_id_a: str, layer_id_b: str, t: float, reason: str) -> None:
        a, b = self.layers.get(layer_id_a), self.layers.get(layer_id_b)
        if a is None or b is None or a.status != "CONFIRMED" or b.status != "CONFIRMED":
            return
        for m in b.member_ids:
            if m not in a.member_ids:
                a.member_ids.append(m)
        a.last_updated = t
        a.confidence = float(min(0.95, max(a.confidence, b.confidence) + 0.05))
        b.status = "DISSOLVED"
        b.dissolved_time = t
        self.record_decision(DecisionRecord(time=t, module="GroupingEngine", decision_type="layer_merged",
                                             target_id=layer_id_a, candidates=[], chosen=layer_id_a,
                                             reason=reason, confidence=a.confidence))
        self.emit_event("layer_merged", layer_id_a, {"absorbed": layer_id_b}, a.confidence, t)

    def remove_layer_member(self, layer_id: str, object_id: str, t: float, reason: str) -> None:
        layer = self.layers.get(layer_id)
        if layer is None or layer.status != "CONFIRMED" or object_id not in layer.member_ids:
            return
        layer.member_ids.remove(object_id)
        layer.last_updated = t
        self.record_decision(DecisionRecord(time=t, module="GroupingEngine", decision_type="layer_member_removed",
                                             target_id=layer_id, candidates=[], chosen=object_id,
                                             reason=reason, confidence=layer.confidence))
        self.emit_event("layer_member_removed", layer_id, {"member": object_id}, layer.confidence, t)
        if len(layer.member_ids) < 2:
            layer.status = "DISSOLVED"
            layer.dissolved_time = t
            self.emit_event("layer_dissolved", layer_id, {"reason": "fewer than 2 members remain"},
                             layer.confidence, t)

    # ---- groove (Sec 14/20): continuous state, updated wholesale by RhythmEngine's proposal ---
    def update_groove(self, new_groove: GrooveVector, t: float) -> None:
        old = self.groove
        self.update_log.append(UpdateRecord(t, "RhythmEngine", "world.groove", "tempo_bpm",
                                             old.tempo_bpm, new_groove.tempo_bpm,
                                             "onset-strength autocorrelation + phase-locked tracking",
                                             new_groove.tempo_confidence))
        self.groove = new_groove
        self.groove_history.append(new_groove)
        if len(self.groove_history) > 2000:  # bounded: this only grows ~5/sec at the slow-loop cadence
            self.groove_history = self.groove_history[-2000:]

    # ---- patterns (Sec 12.4, 14.2): tolerant-match repetition, not exact -----------------------
    def set_pattern(self, period_bars: Optional[int], confidence: float, t: float) -> bool:
        """Reconciles the currently-confirmed pattern with this bar's result.
        Returns True iff the confirmed pattern actually changed this call
        (used by StructureEngine as one signal feeding section-boundary
        novelty)."""
        cur = self.current_pattern
        if period_bars is None:
            changed = cur is not None
            if cur is not None:
                cur.status = "BROKEN"
                self.emit_event("pattern_broken", cur.pattern_id, {"period_bars": cur.period_bars},
                                 cur.confidence, t)
            self.current_pattern = None
            return changed
        if cur is not None and cur.status == "ACTIVE" and cur.period_bars == period_bars:
            cur.confidence = float(0.7 * cur.confidence + 0.3 * confidence)
            cur.occurrence_count += 1
            cur.last_seen = t
            return False
        pid = f"pattern_{uuid.uuid4().hex[:8]}"
        pat = Pattern(pattern_id=pid, period_bars=period_bars, confidence=confidence,
                       occurrence_count=1, first_seen=t, last_seen=t, status="ACTIVE")
        self.patterns[pid] = pat
        self.current_pattern = pat
        self.record_decision(DecisionRecord(time=t, module="StructureEngine", decision_type="pattern_confirmed",
                                             target_id=pid, candidates=[], chosen=pid,
                                             reason=f"{period_bars}-bar repeating pattern "
                                                     f"(similarity {confidence:.2f})", confidence=confidence))
        self.emit_event("pattern_detected", pid, {"period_bars": period_bars}, confidence, t)
        return True

    # ---- sections (Sec 14.3): accumulated structural change, never blocks fast events ----------
    def start_new_section(self, t: float, reason: str, novelty_score: float,
                           active_object_ids: Optional[List[str]] = None) -> Section:
        if self.sections and self.sections[-1].end_time is None:
            self.sections[-1].end_time = t
        sid = f"section_{uuid.uuid4().hex[:8]}"
        sec = Section(section_id=sid, start_time=t, end_time=None,
                       boundary_reason=reason, novelty_score=novelty_score)
        self.sections.append(sec)
        self.record_decision(DecisionRecord(time=t, module="StructureEngine", decision_type="section_boundary",
                                             target_id=sid, candidates=[], chosen=sid,
                                             reason=reason, confidence=novelty_score))
        self.emit_event("section_changed", sid, {"reason": reason, "novelty": novelty_score}, novelty_score, t)
        for oid in (active_object_ids or []):
            self.upsert_relationship(sid, oid, RelationshipType.SECTION_MEMBERSHIP,
                                      strength=1.0, confidence=0.6, t=t)
        return sec

    def emit_event(self, event_type: str, target_id: Optional[str], payload: Dict[str, Any],
                    confidence: float, t: float) -> Optional[Event]:
        key = (event_type, target_id or "")
        last_t = self._last_event_time.get(key, -1e9)
        if (t - last_t) * 1000.0 < self.params.same_event_debounce_ms:
            return None
        self._last_event_time[key] = t
        evt = Event(event_id=f"evt_{uuid.uuid4().hex[:8]}", time=t, event_type=event_type,
                    target_id=target_id, payload=payload, confidence=float(confidence))
        self.events.append(evt)
        return evt

    def snapshot_object_history(self, obj: SoundObject, t: float) -> None:
        obj.history.append(HistoryEntry(
            time=t, status=obj.status.value,
            confidence={"existence": obj.confidence.existence, "identity": obj.confidence.identity,
                        "perceptual": obj.confidence.perceptual, "prediction": obj.confidence.prediction,
                        "grouping": obj.confidence.grouping},
            frequency_hz=obj.physical_signature.frequency_hz, energy=obj.physical_signature.energy,
            perceptual_availability=obj.perceptual_state.perceptual_availability,
            timbre_class=obj.timbre_class, timbre_confidence=obj.timbre_confidence))

    # ---- replay / inspection (Part 41 Sec 20: "what did the system believe at time T, and why?") ----
    def query_at(self, t: float) -> Dict[str, Any]:
        result: Dict[str, Any] = {"time": t, "objects": {}, "events_up_to": [], "nearby_decisions": []}
        for oid, obj in self.objects.items():
            entry = None
            for h in obj.history:
                if h.time <= t:
                    entry = h
                else:
                    break
            if entry is not None:
                result["objects"][oid] = entry
        result["events_up_to"] = [e for e in self.events if e.time <= t]
        result["nearby_decisions"] = [d for d in self.decision_log if abs(d.time - t) <= 0.1]
        return result


# =============================================================================
# 3. AUDIO ENGINE — input, clock, buffering (Sec 5)
# =============================================================================

class AudioClock:
    """Sample-accurate clock. The single time base; never silently mixed
    with another (Part 63, "Clock discontinuity")."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.sample_index: int = 0

    @property
    def t(self) -> float:
        return self.sample_index / self.sample_rate

    def advance(self, n_samples: int) -> None:
        self.sample_index += n_samples


class AudioSource:
    """File or synthetic input, resampled to the locked sample_rate."""

    def __init__(self, params: Parameters):
        self.params = params
        self.logger = logging.getLogger("AudioSource")

    def load(self, path: Optional[str]) -> Tuple[np.ndarray, int]:
        if path is None:
            self.logger.info("No audio file given -> generating synthetic multi-object test signal "
                              "(kick + bass + hi-hat + pad, with a deliberate kick/bass masking scenario).")
            audio = self.synth_test_signal()
            return audio, self.params.sample_rate
        audio, sr = self._load_file(path)
        if sr != self.params.sample_rate:
            self.logger.info("Resampling %d Hz -> %d Hz", sr, self.params.sample_rate)
            audio = self._resample(audio, sr, self.params.sample_rate)
        return audio.astype(np.float32), self.params.sample_rate

    def _load_file(self, path: str) -> Tuple[np.ndarray, int]:
        if HAVE_SOUNDFILE:
            audio, sr = sf.read(path, always_2d=False, dtype="float32")
        else:
            from scipy.io import wavfile
            sr, audio = wavfile.read(path)
            if np.issubdtype(audio.dtype, np.integer):
                audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
            else:
                audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), int(sr)

    def _resample(self, audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
        if HAVE_LIBROSA:
            return librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr_from, target_sr=sr_to)
        g = math.gcd(sr_from, sr_to)
        return resample_poly(audio, sr_to // g, sr_from // g).astype(np.float32)

    def synth_test_signal(self, duration: float = 8.0, bpm: float = 120.0) -> np.ndarray:
        """Kick + bass + hi-hat + pad. The kick's low-frequency sweep
        deliberately overlaps the bass fundamentals so the bass gets
        perceptually masked on every kick hit and must recover afterwards
        (the canonical 'masked bass recovery' behavior test, Sec 32)."""
        sr = self.params.sample_rate
        n = int(duration * sr)
        t = np.arange(n) / sr
        audio = np.zeros(n, dtype=np.float64)
        rng = np.random.default_rng(42)
        beat = 60.0 / bpm

        # --- Pad: sustained harmonic drone (A2 + E3, a fifth), slow tremolo ---
        pad = np.zeros(n)
        for f in (110.0, 165.0):
            pad += 0.10 * np.sin(2 * np.pi * f * t)
        tremolo = 1.0 + 0.15 * np.sin(2 * np.pi * 0.3 * t)
        fade_in = np.clip(t / 1.0, 0.0, 1.0)
        audio += pad * tremolo * fade_in

        # --- Bass: alternating low notes (E1, A1), 2 beats each, low-passed ---
        bass = np.zeros(n)
        bass_notes = (41.2, 55.0)
        note_dur = beat * 2
        n_notes = int(math.ceil(duration / note_dur))
        for i in range(n_notes):
            f = bass_notes[i % 2]
            s = int(i * note_dur * sr)
            e = int(min((i + 1) * note_dur, duration) * sr)
            if e <= s:
                continue
            seg_t = (np.arange(e - s)) / sr
            env = np.clip(seg_t / 0.01, 0.0, 1.0) * np.exp(-seg_t * 0.8)
            bass[s:e] += 0.35 * env * np.sin(2 * np.pi * f * seg_t) + 0.10 * env * np.sin(2 * np.pi * 2 * f * seg_t)
        b_lp, a_lp = butter(4, 300.0 / (sr / 2.0), btype="low")
        bass = lfilter(b_lp, a_lp, bass)
        audio += bass

        # --- Kick: every beat; sub sweep 150Hz->45Hz + transient click ---
        kick = np.zeros(n)
        n_kicks = int(math.ceil(duration / beat))
        for i in range(n_kicks):
            s = int(i * beat * sr)
            e = min(n, s + int(0.25 * sr))
            if e <= s:
                continue
            seg_t = np.arange(e - s) / sr
            f_sweep = 150.0 * np.exp(-seg_t * 35.0) + 45.0
            phase = 2 * np.pi * np.cumsum(f_sweep) / sr
            env = np.exp(-seg_t * 18.0)
            click = 0.25 * np.exp(-seg_t * 400.0) * rng.standard_normal(e - s)
            kick[s:e] += 0.9 * env * np.sin(phase) + click
        audio += kick

        # --- Hi-hat: offbeat 8th notes, bandpassed noise bursts ---
        hihat = np.zeros(n)
        hh_step = beat / 2.0
        n_hh = int(math.ceil(duration / hh_step))
        hh_low = 6000.0 / (sr / 2.0)
        hh_high = min(14000.0, 0.49 * sr) / (sr / 2.0)
        b_hp, a_hp = butter(4, [hh_low, hh_high], btype="band")
        for i in range(n_hh):
            s = int(i * hh_step * sr)
            e = min(n, s + int(0.05 * sr))
            if e <= s:
                continue
            seg_t = np.arange(e - s) / sr
            env = np.exp(-seg_t * 80.0)
            hihat[s:e] += 0.18 * rng.standard_normal(e - s) * env
        hihat = lfilter(b_hp, a_hp, hihat)
        audio += hihat

        audio += rng.standard_normal(n) * 1e-4  # tiny noise floor, avoids exact-zero regions
        peak = float(np.max(np.abs(audio)) + 1e-9)
        audio = (audio / peak * 0.9)
        return audio.astype(np.float32)


# =============================================================================
# 4. PHYSICAL ANALYSIS ENGINE — multi-resolution STFT, spectral features,
#    bark-band energy + spreading matrix (for masking), transients, pitch.
#    (Sec 6). Produces evidence only; never assigns musical meaning.
# =============================================================================

@dataclass
class PhysicalEvidence:
    t: float
    hop_index: int
    energy: float
    crest_factor: float
    spectral_centroid: float
    spectral_spread: float
    spectral_flux: float
    spectral_flatness: float
    onset_strength: float
    is_transient: bool
    bark_energy: np.ndarray
    magnitude: np.ndarray
    pitch_hz: Optional[float]
    pitch_confidence: float


class PhysicalAnalyzer:
    """Cochlea-equivalent / multi-resolution physical analysis (Sec 6).

    Two temporal scales, per Sec 6.2:
      - short window (short_fft_size @ fast_hop): transients, fast spectral
        evidence.
      - long window (long_fft_size @ long_hop): pitch / note-level evidence,
        where frequency resolution matters more than time resolution.

    For efficiency the whole-signal STFT is computed in one vectorized pass
    (optionally on GPU via torch), but every per-hop feature used downstream
    is a causal function of the current and past samples only -- no frame
    ever uses information from a later point in time than its own hop
    boundary, so the streaming/causal semantics required by the architecture
    are preserved even though the pass is batched for speed.
    """

    def __init__(self, params: Parameters):
        self.p = params
        self.logger = logging.getLogger("PhysicalAnalyzer")
        self.short_window = get_window("hann", params.short_fft_size, fftbins=True).astype(np.float32)
        self.long_window = get_window("hann", params.long_fft_size, fftbins=True).astype(np.float32)
        self.short_freqs = np.fft.rfftfreq(params.short_fft_size, d=1.0 / params.sample_rate).astype(np.float32)
        self.long_freqs = np.fft.rfftfreq(params.long_fft_size, d=1.0 / params.sample_rate).astype(np.float32)
        self.bark_filterbank, self.bark_centers_hz = self._build_bark_filterbank(self.short_freqs)
        self.n_bark = self.bark_filterbank.shape[0]
        self.spreading_matrix = self._build_spreading_matrix(self.bark_centers_hz)
        self.logger.info("Bark filterbank: %d bands, 0-%.0f Hz. Spreading matrix %s built (ISO/MPEG-1-style).",
                          self.n_bark, self.bark_centers_hz[-1], self.spreading_matrix.shape)

    # ---- bark / critical-band filterbank (cochlea-equivalent representation, Sec 7.1) ----
    @staticmethod
    def _hz_to_bark(f: np.ndarray) -> np.ndarray:
        f = np.maximum(f, 1e-6)
        return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)

    def _bark_to_hz(self, z: float) -> float:
        lo, hi = 0.0, 24000.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if float(self._hz_to_bark(np.array([mid]))[0]) < z:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _build_bark_filterbank(self, freqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        bark_vals = self._hz_to_bark(freqs)
        bark_max = float(bark_vals[-1])
        n_bands = max(8, int(math.ceil(bark_max)))
        edges = np.linspace(0.0, bark_max, n_bands + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        fb = np.zeros((n_bands, len(freqs)), dtype=np.float32)
        for b in range(n_bands):
            lo, ce, hi = edges[b], centers[b], edges[b + 1]
            left = (bark_vals - lo) / (ce - lo + 1e-9)
            right = (hi - bark_vals) / (hi - ce + 1e-9)
            fb[b] = np.clip(np.minimum(left, right), 0.0, None)
        row_sums = fb.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        fb = fb / row_sums
        centers_hz = np.array([self._bark_to_hz(float(c)) for c in centers], dtype=np.float32)
        return fb, centers_hz

    def _build_spreading_matrix(self, centers_hz: np.ndarray) -> np.ndarray:
        """Simplified ISO/MPEG-1 psychoacoustic-model spreading function
        (Painter & Spanias form): SF(dz) = 15.81 + 7.5(dz+0.474)
        - 17.5*sqrt(1+(dz+0.474)^2)  dB, dz = bark(maskee) - bark(masker).
        This correctly captures the asymmetry of masking (a low-frequency
        masker spreads upward more strongly than a high one spreads
        downward)."""
        z = self._hz_to_bark(centers_hz)
        n = len(z)
        S = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            dz = z[i] - z
            sf_db = 15.81 + 7.5 * (dz + 0.474) - 17.5 * np.sqrt(1.0 + (dz + 0.474) ** 2)
            sf_db = np.clip(sf_db, -100.0, 0.5)
            S[i] = 10.0 ** (sf_db / 10.0)
        return S

    # ---- windowed batch STFT (numpy, optionally torch/GPU) ----
    def _framed_spectrogram(self, audio: np.ndarray, window: np.ndarray, n_fft: int,
                             hop: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (magnitude_spectrogram, windowed_time_frames, raw_time_frames).
        The window is applied only for the FFT; time-domain features (energy,
        crest factor, autocorrelation/pitch) must use the *raw* frames, or a
        Hann taper biases them toward zero lag / distorts amplitude."""
        n = len(audio)
        if n < n_fft:
            audio = np.pad(audio, (0, n_fft - n))
            n = len(audio)
        n_frames = 1 + (n - n_fft) // hop
        idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
        raw_frames = audio[idx].astype(np.float32)
        frames = (raw_frames * window[None, :]).astype(np.float32)
        if HAVE_TORCH and HAVE_CUDA:
            tt = torch.from_numpy(frames).to("cuda")
            spec = torch.fft.rfft(tt, dim=-1)
            mag = spec.abs().detach().cpu().numpy().astype(np.float32)
        else:
            spec = np.fft.rfft(frames, axis=-1)
            mag = np.abs(spec).astype(np.float32)
        return mag, frames, raw_frames

    # ---- transient / onset detection: causal adaptive z-score over spectral flux ----
    def detect_transients(self, flux: np.ndarray, hop: int, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        n = len(flux)
        onset_strength = np.zeros(n, dtype=np.float32)
        is_transient = np.zeros(n, dtype=bool)
        alpha = 0.02                              # DERIVED: reacts over ~a few hundred ms
        refractory = max(1, int(round(0.030 * sr / hop)))  # 30ms refractory, ties to grouping tolerance
        k_threshold = 3.0                          # DEFAULT: 3-sigma adaptive threshold
        warmup = min(n, max(1, int(round(0.05 * sr / hop))))  # 50ms warm-up before triggers are trusted
        ema_mean = float(np.mean(flux[:warmup])) if warmup > 0 else 0.0
        ema_var = float(np.var(flux[:warmup])) + 1e-6 if warmup > 0 else 1e-6
        last_trigger = -refractory
        for i in range(n):
            x = float(flux[i])
            dev = x - ema_mean
            onset_strength[i] = dev / (math.sqrt(ema_var) + 1e-6)
            if i >= warmup and onset_strength[i] > k_threshold and (i - last_trigger) >= refractory:
                is_transient[i] = True
                last_trigger = i
            ema_mean = (1 - alpha) * ema_mean + alpha * x
            ema_var = (1 - alpha) * ema_var + alpha * (dev * dev)
        return onset_strength, is_transient

    # ---- pitch: FFT-based autocorrelation on the long (better freq-res) window ----
    def estimate_pitch_long(self, long_frames_time: np.ndarray, sr: int,
                             fmin: float = 30.0, fmax: float = 600.0) -> Tuple[np.ndarray, np.ndarray]:
        n_frames, n = long_frames_time.shape
        pitches = np.zeros(n_frames, dtype=np.float32)
        confs = np.zeros(n_frames, dtype=np.float32)
        min_lag = max(1, int(sr / fmax))
        max_lag = min(n - 1, int(sr / fmin))
        nfft = 1
        while nfft < 2 * n:
            nfft *= 2
        for i in range(n_frames):
            frame = long_frames_time[i] - long_frames_time[i].mean()
            F = np.fft.rfft(frame, n=nfft)
            ac = np.fft.irfft(F * np.conj(F), n=nfft)[:n]
            if ac[0] <= 1e-9 or max_lag <= min_lag:
                continue
            search = ac[min_lag:max_lag + 1]
            peak_idx = int(np.argmax(search))
            lag = float(min_lag + peak_idx)
            conf = float(search[peak_idx] / ac[0])
            if 0 < peak_idx < len(search) - 1:
                a, bb, c = search[peak_idx - 1], search[peak_idx], search[peak_idx + 1]
                denom = (a - 2 * bb + c)
                if abs(denom) > 1e-9:
                    lag += float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0))
            if peak_idx == 0:
                conf *= 0.3  # boundary-clipped peak: likely not a genuine periodicity
            pitches[i] = float(sr / lag) if lag > 0 else 0.0
            confs[i] = float(np.clip(conf, 0.0, 1.0))
        return pitches, confs

    # ---- full-signal analysis pass ----
    def analyze_full(self, audio: np.ndarray) -> List[PhysicalEvidence]:
        p = self.p
        mag_short, _win_short, raw_short = self._framed_spectrogram(audio, self.short_window, p.short_fft_size, p.fast_hop)
        _, _win_long, raw_long = self._framed_spectrogram(audio, self.long_window, p.long_fft_size, p.long_hop)
        n_short = mag_short.shape[0]

        power_short = mag_short.astype(np.float64) ** 2
        energy = np.sqrt(np.mean(raw_short.astype(np.float64) ** 2, axis=1) + 1e-12)
        peak = np.max(np.abs(raw_short), axis=1)
        crest = peak / (energy + 1e-9)

        mag_sum = np.sum(mag_short, axis=1) + 1e-9
        centroid = np.sum(self.short_freqs[None, :] * mag_short, axis=1) / mag_sum
        spread = np.sqrt(np.sum(mag_short * (self.short_freqs[None, :] - centroid[:, None]) ** 2, axis=1) / mag_sum)
        flatness = np.exp(np.mean(np.log(power_short + 1e-12), axis=1)) / (np.mean(power_short, axis=1) + 1e-12)
        flatness = np.clip(flatness, 0.0, 1.0)

        flux = np.zeros(n_short, dtype=np.float32)
        diff = mag_short[1:] - mag_short[:-1]
        flux[1:] = np.sqrt(np.sum(np.maximum(diff, 0.0) ** 2, axis=1))
        onset_strength, is_transient = self.detect_transients(flux, p.fast_hop, p.sample_rate)

        bark_energy_all = power_short @ self.bark_filterbank.T  # (n_short, n_bark)

        pitches_long, pitch_conf_long = self.estimate_pitch_long(raw_long, p.sample_rate)

        evidence: List[PhysicalEvidence] = []
        for i in range(n_short):
            t = i * p.fast_hop / p.sample_rate
            sample_pos = i * p.fast_hop
            long_idx = int(math.floor((sample_pos - p.long_fft_size) / p.long_hop))
            if 0 <= long_idx < len(pitches_long) and pitch_conf_long[long_idx] > 0.35:
                pitch_hz: Optional[float] = float(pitches_long[long_idx])
                pitch_conf = float(pitch_conf_long[long_idx])
            else:
                pitch_hz, pitch_conf = None, 0.0
            evidence.append(PhysicalEvidence(
                t=t, hop_index=i, energy=float(energy[i]), crest_factor=float(crest[i]),
                spectral_centroid=float(centroid[i]), spectral_spread=float(spread[i]),
                spectral_flux=float(flux[i]), spectral_flatness=float(flatness[i]),
                onset_strength=float(onset_strength[i]), is_transient=bool(is_transient[i]),
                bark_energy=bark_energy_all[i].astype(np.float32), magnitude=mag_short[i],
                pitch_hz=pitch_hz, pitch_confidence=pitch_conf))
        return evidence




# =============================================================================
# 5. OBSERVATION GENERATION — physical evidence -> candidate observations (Sec 8)
# =============================================================================

class ObservationGenerator:
    """Converts one frame of PhysicalEvidence into a small set of primitive
    Observations: a TRANSIENT (if one fired this hop) and up to a handful of
    TONAL / NOISE_TEXTURE observations from local peaks in the bark-band
    energy. These are evidence candidates, not final labels (Sec 8)."""

    MAX_BAND_PEAKS = 6  # bounds working-set growth, spirit of working_observation_limit

    def __init__(self, params: Parameters, bark_centers_hz: np.ndarray):
        self.p = params
        self.bark_centers_hz = bark_centers_hz
        self.n_bark = len(bark_centers_hz)
        self.logger = logging.getLogger("ObservationGenerator")
        # Per-band adaptive noise floor (dB), fast-fall / slow-rise: tracks each band's own
        # typical quiet level so a quiet-but-real sustained band (e.g. a pad) stays detectable
        # regardless of how loud *other*, unrelated bands are in the very same frame.
        self._floor_db: Optional[np.ndarray] = None
        self._floor_fall_alpha = 0.10           # DERIVED: fast tracking down toward quiet
        self._floor_rise_db_per_sec = 0.15      # DERIVED: very slow creep upward
        self._peak_margin_db = 4.0              # DEFAULT: peak must clear its own floor by this much

    def generate(self, ev: PhysicalEvidence) -> List[Observation]:
        obs: List[Observation] = []

        if ev.is_transient:
            band = int(np.argmin(np.abs(self.bark_centers_hz - max(ev.spectral_centroid, 1.0))))
            obs.append(Observation(
                id=f"obs_{uuid.uuid4().hex[:8]}", obs_type=ObservationType.TRANSIENT,
                start_time=ev.t, end_time=ev.t,
                frequency_hz=float(ev.spectral_centroid), bandwidth_hz=float(ev.spectral_spread),
                energy=float(ev.energy), band_energy=float(ev.bark_energy[band]), bark_band=band,
                physical_features={"onset_strength": float(ev.onset_strength),
                                    "crest_factor": float(ev.crest_factor),
                                    "flux": float(ev.spectral_flux),
                                    "harmonic": 0.0,
                                    "pitch_hz": float(ev.pitch_hz or 0.0),
                                    "pitch_confidence": float(ev.pitch_confidence),
                                    "spectral_flatness": float(ev.spectral_flatness)},
                confidence=float(np.clip(ev.onset_strength / 6.0, 0.0, 1.0)),
                provenance="PhysicalAnalyzer.transient(z-score-flux)"))

        energy_db = 10.0 * np.log10(ev.bark_energy + 1e-10)

        if self._floor_db is None:
            self._floor_db = energy_db.copy() - 3.0
        else:
            fdt = self.p.fast_hop / self.p.sample_rate
            falling = energy_db < self._floor_db
            fa = self._floor_fall_alpha
            self._floor_db = np.where(
                falling,
                (1.0 - fa) * self._floor_db + fa * energy_db,
                self._floor_db + self._floor_rise_db_per_sec * fdt,
            )

        peak_bands = []
        n_bands = len(ev.bark_energy)
        for b in range(n_bands):
            if energy_db[b] <= self._floor_db[b] + self._peak_margin_db:
                continue
            ok_left = (b == 0) or (ev.bark_energy[b] >= ev.bark_energy[b - 1])
            ok_right = (b == n_bands - 1) or (ev.bark_energy[b] >= ev.bark_energy[b + 1])
            if ok_left and ok_right:
                peak_bands.append(b)
        peak_bands = sorted(peak_bands, key=lambda b: -ev.bark_energy[b])[: self.MAX_BAND_PEAKS]

        for b in peak_bands:
            is_harmonic = bool(ev.pitch_hz is not None and ev.pitch_confidence > 0.5
                                and self._is_near_harmonic(ev.pitch_hz, float(self.bark_centers_hz[b])))
            otype = (ObservationType.TONAL if (is_harmonic or ev.spectral_flatness < 0.3)
                     else ObservationType.NOISE_TEXTURE)
            conf = float(np.clip((energy_db[b] - self._floor_db[b]) / 24.0, 0.05, 0.95))
            lo_c = float(self.bark_centers_hz[max(b - 1, 0)])
            hi_c = float(self.bark_centers_hz[min(b + 1, len(self.bark_centers_hz) - 1)])
            obs.append(Observation(
                id=f"obs_{uuid.uuid4().hex[:8]}", obs_type=otype,
                start_time=ev.t, end_time=ev.t,
                frequency_hz=float(self.bark_centers_hz[b]), bandwidth_hz=max(hi_c - lo_c, 1.0),
                energy=float(ev.bark_energy[b]), band_energy=float(ev.bark_energy[b]), bark_band=b,
                physical_features={"pitch_hz": float(ev.pitch_hz or 0.0),
                                    "pitch_confidence": float(ev.pitch_confidence),
                                    "spectral_flatness": float(ev.spectral_flatness),
                                    "harmonic": float(is_harmonic),
                                    "onset_strength": float(ev.onset_strength),
                                    "crest_factor": float(ev.crest_factor),
                                    "flux": float(ev.spectral_flux)},
                confidence=conf, provenance="PhysicalAnalyzer.bark_band_peak"))
        return obs

    @staticmethod
    def _is_near_harmonic(f0: float, f: float, tol: float = 0.06) -> bool:
        if f0 <= 1.0:
            return False
        n = max(1, round(f / f0))
        nearest = n * f0
        return abs(f - nearest) / max(nearest, 1.0) < tol


# =============================================================================
# 6. OBJECT TRACKER — multi-cue matching, hysteresis, lifecycle (Sec 10)
# =============================================================================

class ObjectTracker:
    """Momentary evidence -> persistent world state (Sec 10).

    Required order per Part 59 "Tracking":
        new observation -> candidate generation -> multi-cue scoring
        -> hysteresis -> object update -> prediction update
        -> confidence update -> Decision record -> Update record.

    Identity is never a single feature: score = weighted combination of
    frequency, energy, envelope, harmonic, temporal and prediction cues
    (Sec 10.1, Part 59 "Matching"). All object mutation happens through
    WorldState.apply_update / register_object so provenance is real.
    """

    FREQUENCY_VETO_SCORE = 0.15  # DERIVED: below this (~1.15 octaves away) no other cue can rescue a match

    def __init__(self, params: Parameters, world: WorldState):
        self.p = params
        self.world = world
        self.logger = logging.getLogger("ObjectTracker")

    # ---- multi-cue component scores (Sec 10.1 / Part 59) ------------------------------------
    def component_scores(self, obs: Observation, obj: SoundObject, t: float) -> Dict[str, float]:
        sig = obj.physical_signature
        # Identity is anchored to the slow-moving stable frequency, not the last instantaneous
        # match, so a handful of noisy matches (e.g. broadband transient bleed) can't gradually
        # walk an object's identity across the spectrum one small, individually-tolerable step
        # at a time ("track hijacking").
        f_obj = max(sig.stable_frequency_hz or sig.frequency_hz, 1.0)
        f_obs = max(obs.frequency_hz, 1.0)
        freq_dist_oct = abs(math.log2(f_obs / f_obj))
        freq_score = math.exp(-freq_dist_oct / 0.5)  # DERIVED: ~half-octave tolerance

        e_obj = max(sig.band_energy, 1e-8)
        e_obs = max(obs.band_energy, 1e-8)
        energy_ratio_log = abs(math.log10(e_obs / e_obj))
        energy_score = math.exp(-energy_ratio_log / 0.7)

        dt = max(t - obj.last_observed_time, 0.0)
        expected_env = sig.envelope * math.exp(-dt / 0.4)  # DERIVED envelope decay tau = 400ms
        denom = max(expected_env, e_obj, 1e-6)
        envelope_score = max(0.0, 1.0 - min(1.0, abs(obs.band_energy - expected_env) / denom))

        if obs.obs_type == ObservationType.TONAL:
            harmonic_score = float(obs.physical_features.get("harmonic", 0.0)) * 0.7 + 0.3
        elif obs.obs_type == ObservationType.TRANSIENT:
            harmonic_score = 0.4
        else:
            harmonic_score = 0.3

        temporal_score = math.exp(-dt / (self.p.candidate_confirmation_window_ms / 1000.0 * 3.0))

        pred_score = 0.5
        if obj.prediction is not None:
            df = abs(math.log2(max(obj.prediction.expected_frequency_hz, 1.0) / f_obs))
            pred_score = math.exp(-df / 0.5)

        return {"frequency": freq_score, "energy": energy_score, "envelope": envelope_score,
                "harmonic": harmonic_score, "temporal": temporal_score, "prediction": pred_score}

    def weighted_score(self, comp: Dict[str, float]) -> float:
        p = self.p
        return (p.w_frequency * comp["frequency"] + p.w_energy * comp["energy"]
                + p.w_envelope * comp["envelope"] + p.w_harmonic * comp["harmonic"]
                + p.w_temporal * comp["temporal"] + p.w_prediction * comp["prediction"])

    # ---- main per-hop step --------------------------------------------------------------------
    def step(self, observations: List[Observation], t: float) -> None:
        p = self.p
        candidate_objs = [o for o in self.world.objects.values()
                           if o.status in (ObjectStatus.TENTATIVE, ObjectStatus.ACTIVE,
                                           ObjectStatus.HIDDEN, ObjectStatus.RECOVERING)]

        pairs: List[Tuple[float, Observation, SoundObject, Dict[str, float]]] = []
        for obs in observations:
            for obj in candidate_objs:
                comp = self.component_scores(obs, obj, t)
                # Hard frequency-proximity gate (Bregman ASA: frequency proximity is a
                # dominant primitive grouping cue). Without this, a terrible frequency
                # match can still be rescued by the other five cues and silently "hijack"
                # an object's identity into drifting across frequency over time.
                if comp["frequency"] < self.FREQUENCY_VETO_SCORE:
                    continue
                score = self.weighted_score(comp)
                if score >= p.reject_match:
                    pairs.append((score, obs, obj, comp))
        pairs.sort(key=lambda x: -x[0])

        matched_obs_ids: set = set()
        used_obj_ids: set = set()
        for score, obs, obj, comp in pairs:
            if obs.id in matched_obs_ids or obj.object_id in used_obj_ids:
                continue
            matched_obs_ids.add(obs.id)
            used_obj_ids.add(obj.object_id)
            self._apply_match(obj, obs, score, comp, t)

        for obs in observations:
            if obs.id in matched_obs_ids:
                continue
            if obs.confidence >= p.candidate_match:
                self._birth(obs, t)

        for obj in candidate_objs:
            if obj.object_id not in used_obj_ids:
                self._handle_miss(obj, t)

        for obj in list(self.world.objects.values()):
            self.world.snapshot_object_history(obj, t)

    # ---- object birth (Sec 10.3 / Part 59 "Object creation") -------------------------------
    def _birth(self, obs: Observation, t: float) -> SoundObject:
        oid = self.world.new_object_id()
        sig = PhysicalSignature(frequency_hz=obs.frequency_hz, stable_frequency_hz=obs.frequency_hz,
                                 bandwidth_hz=obs.bandwidth_hz,
                                 energy=obs.energy, band_energy=obs.band_energy, envelope=obs.band_energy,
                                 harmonicity=float(obs.physical_features.get("pitch_confidence", 0.0)),
                                 spectral_centroid=obs.frequency_hz,
                                 spectral_flatness=float(obs.physical_features.get("spectral_flatness", 0.5)),
                                 crest_factor=float(obs.physical_features.get("crest_factor", 1.0)))
        conf = Confidence(existence=obs.confidence, identity=obs.confidence * 0.6,
                           perceptual=1.0, prediction=0.3, grouping=0.5)
        obj = SoundObject(object_id=oid, status=ObjectStatus.TENTATIVE, creation_time=t,
                           last_observed_time=t, last_active_time=t, confidence=conf,
                           physical_signature=sig, perceptual_state=PerceptualState(),
                           prediction=None, consecutive_confirmations=1)
        self.world.register_object(obj)
        self._update_prediction(obj, t)
        self.world.record_decision(DecisionRecord(
            time=t, module="ObjectTracker", decision_type="birth", target_id=oid, candidates=[],
            chosen=oid, reason=f"No existing object matched observation {obs.id} "
                                f"(type={obs.obs_type.value}, conf={obs.confidence:.2f}); "
                                f"created TENTATIVE object.", confidence=obs.confidence))
        self.world.emit_event("object_created", oid,
                               {"frequency_hz": obs.frequency_hz, "obs_type": obs.obs_type.value},
                               obs.confidence, t)
        self.logger.info("[%7.3fs] BIRTH        %-12s freq=%7.1fHz type=%-13s conf=%.2f",
                          t, oid, obs.frequency_hz, obs.obs_type.value, obs.confidence)
        return obj

    # ---- matched update (Sec 10.4-10.6) -----------------------------------------------------
    def _apply_match(self, obj: SoundObject, obs: Observation, score: float,
                      comp: Dict[str, float], t: float) -> None:
        p = self.p
        old_status = obj.status
        self.world.record_decision(DecisionRecord(
            time=t, module="ObjectTracker", decision_type="match", target_id=obj.object_id,
            candidates=[{"obs_id": obs.id, "score": score, **comp}], chosen=obj.object_id,
            reason=f"Best multi-cue match score={score:.2f} (>= reject_match {p.reject_match}).",
            confidence=score))

        self.world.apply_update(UpdateRecord(t, "ObjectTracker", obj.object_id,
                                              "physical_signature.frequency_hz",
                                              obj.physical_signature.frequency_hz, obs.frequency_hz,
                                              "matched observation", score))
        anchor = obj.physical_signature.stable_frequency_hz or obs.frequency_hz
        new_anchor = anchor * math.exp(0.05 * math.log(max(obs.frequency_hz, 1.0) / max(anchor, 1.0)))
        self.world.apply_update(UpdateRecord(t, "ObjectTracker", obj.object_id,
                                              "physical_signature.stable_frequency_hz",
                                              obj.physical_signature.stable_frequency_hz, new_anchor,
                                              "slow log-domain EMA anchor (anti-drift)", score))
        self.world.apply_update(UpdateRecord(t, "ObjectTracker", obj.object_id,
                                              "physical_signature.energy",
                                              obj.physical_signature.energy, obs.energy,
                                              "matched observation", score))
        self.world.apply_update(UpdateRecord(t, "ObjectTracker", obj.object_id,
                                              "physical_signature.band_energy",
                                              obj.physical_signature.band_energy, obs.band_energy,
                                              "matched observation", score))
        new_env = 0.3 * obs.band_energy + 0.7 * obj.physical_signature.envelope
        self.world.apply_update(UpdateRecord(t, "ObjectTracker", obj.object_id,
                                              "physical_signature.envelope",
                                              obj.physical_signature.envelope, new_env,
                                              "envelope smoothing (tau~400ms)", score))
        pitch_conf = float(obs.physical_features.get("pitch_confidence", 0.0))
        new_harmonicity = 0.7 * obj.physical_signature.harmonicity + 0.3 * pitch_conf
        self.world.apply_update(UpdateRecord(t, "ObjectTracker", obj.object_id,
                                              "physical_signature.harmonicity",
                                              obj.physical_signature.harmonicity, new_harmonicity,
                                              "EMA of observation pitch_confidence (not a one-shot flag)",
                                              pitch_conf))

        obj.last_observed_time = t
        obj.consecutive_confirmations += 1
        obj.consecutive_misses = 0

        new_identity = 0.6 * obj.confidence.identity + 0.4 * score
        self.world.apply_update(UpdateRecord(t, "ObjectTracker", obj.object_id, "confidence.identity",
                                              obj.confidence.identity, new_identity,
                                              "match-score EMA", score))
        new_existence = 0.5 * obj.confidence.existence + 0.5 * obs.confidence
        self.world.apply_update(UpdateRecord(t, "ObjectTracker", obj.object_id, "confidence.existence",
                                              obj.confidence.existence, new_existence,
                                              "observation-confidence EMA", obs.confidence))

        if obj.prediction is not None:
            pred_err = abs(math.log2(max(obj.prediction.expected_frequency_hz, 1.0)
                                      / max(obs.frequency_hz, 1.0)))
            pred_conf = float(np.clip(1.0 - pred_err, 0.0, 1.0))
            self.world.apply_update(UpdateRecord(t, "ObjectTracker", obj.object_id, "confidence.prediction",
                                                  obj.confidence.prediction, pred_conf,
                                                  "prediction agreement with observation", pred_conf))

        if old_status == ObjectStatus.TENTATIVE:
            elapsed_ms = (t - obj.creation_time) * 1000.0
            if obj.confidence.identity >= p.confirmation_confidence and elapsed_ms >= p.candidate_confirmation_window_ms:
                self._transition(obj, ObjectStatus.ACTIVE, t,
                                  f"Confirmation confidence {obj.confidence.identity:.2f} "
                                  f">= {p.confirmation_confidence} sustained {elapsed_ms:.0f}ms")
        elif old_status == ObjectStatus.HIDDEN:
            self._transition(obj, ObjectStatus.RECOVERING, t,
                              "Evidence reappeared matching historical signature/prediction "
                              f"(score={score:.2f})")
            obj.recovering_since = t
        elif old_status == ObjectStatus.RECOVERING:
            recovering_ms = (t - (obj.recovering_since or t)) * 1000.0
            if recovering_ms >= p.recovery_window_ms:
                self._transition(obj, ObjectStatus.ACTIVE, t,
                                  f"Recovery evidence sustained {recovering_ms:.0f}ms "
                                  f">= recovery_window {p.recovery_window_ms}ms")
                obj.hidden_since = None
                obj.recovering_since = None
        elif old_status == ObjectStatus.ACTIVE:
            obj.last_active_time = t

        self._update_prediction(obj, t)

    # ---- miss handling: temporary absence is NOT death (Sec 10.5) ---------------------------
    def _handle_miss(self, obj: SoundObject, t: float) -> None:
        p = self.p
        obj.consecutive_misses += 1

        if obj.status == ObjectStatus.ACTIVE:
            self._transition(obj, ObjectStatus.HIDDEN, t,
                              "No matching observation this step; evidence temporarily unavailable "
                              "(physical signature and history preserved, not deleted)")
            obj.hidden_since = t

        elif obj.status == ObjectStatus.TENTATIVE:
            elapsed_ms = (t - obj.creation_time) * 1000.0
            if elapsed_ms > p.candidate_confirmation_window_ms and obj.consecutive_confirmations <= 1:
                self._transition(obj, ObjectStatus.ARCHIVED, t,
                                  "TENTATIVE object failed to gather confirming evidence within "
                                  f"candidate_confirmation_window ({p.candidate_confirmation_window_ms}ms)")

        elif obj.status in (ObjectStatus.HIDDEN, ObjectStatus.RECOVERING):
            # Absence is measured from the last REAL evidence match, not from the first
            # HIDDEN transition -- a flappy object that keeps touching partial evidence
            # (e.g. a sustained pad intermittently masked by kick hits) must not have its
            # absence clock treated as continuously running through those partial matches.
            absence_ms = max(0.0, (t - obj.last_observed_time) * 1000.0)
            decay = math.exp(-absence_ms / 1000.0 / 2.0)  # DERIVED: prediction trust halves ~1.4s
            new_pred_conf = obj.confidence.prediction * decay
            self.world.apply_update(UpdateRecord(t, "ObjectTracker", obj.object_id, "confidence.prediction",
                                                  obj.confidence.prediction, new_pred_conf,
                                                  "prediction confidence decays with time since last evidence",
                                                  new_pred_conf))
            if obj.status == ObjectStatus.RECOVERING and absence_ms > self.p.temporal_group_tolerance_ms:
                # Tolerate brief single-hop evidence jitter; only a real gap (longer than the
                # temporal grouping tolerance) counts as recovery actually failing.
                self._transition(obj, ObjectStatus.HIDDEN, t,
                                  f"Recovery evidence gap of {absence_ms:.0f}ms exceeded tolerance "
                                  f"({self.p.temporal_group_tolerance_ms}ms)")
                obj.recovering_since = None
            if absence_ms >= p.archive_delay_ms and obj.confidence.prediction < p.prediction_failure_threshold:
                self._transition(obj, ObjectStatus.ARCHIVED, t,
                                  f"No real evidence for {absence_ms:.0f}ms (>= archive_delay {p.archive_delay_ms}ms) "
                                  f"with weak prediction support ({obj.confidence.prediction:.2f} "
                                  f"< prediction_failure_threshold {p.prediction_failure_threshold})")

        self._update_prediction(obj, t)

    # ---- status transition + events (Sec 4.5) ------------------------------------------------
    def _transition(self, obj: SoundObject, new_status: ObjectStatus, t: float, reason: str) -> None:
        old = obj.status
        if old == new_status:
            return
        self.world.apply_update(UpdateRecord(t, "ObjectTracker", obj.object_id, "status",
                                              old, new_status, reason, obj.confidence.aggregate()))
        self.logger.info("[%7.3fs] TRANSITION   %-12s %-10s -> %-10s  (%s)",
                          t, obj.object_id, old.value, new_status.value, reason)
        etype = {
            ObjectStatus.ACTIVE: "object_confirmed" if old == ObjectStatus.TENTATIVE else "object_reactivated",
            ObjectStatus.HIDDEN: "object_hidden",
            ObjectStatus.RECOVERING: "object_recovering",
            ObjectStatus.ARCHIVED: "object_archived",
        }.get(new_status, "object_status_changed")
        self.world.emit_event(etype, obj.object_id, {"old_status": old.value, "new_status": new_status.value,
                                                       "reason": reason}, obj.confidence.aggregate(), t)

    # ---- prediction (Sec 10.4 / Part 41 Sec 12): never overwrites measured state ------------
    def _update_prediction(self, obj: SoundObject, t: float) -> None:
        horizon = self.p.prediction_horizon_ms / 1000.0
        obj.prediction = Prediction(target_id=obj.object_id, made_at=t, expected_time=t + horizon,
                                     expected_frequency_hz=obj.physical_signature.frequency_hz,
                                     expected_energy=obj.physical_signature.envelope,
                                     confidence=obj.confidence.prediction,
                                     basis="last-state hold (foundation model; replaceable, Part 41 Sec 21)")


# =============================================================================
# 7. PERCEPTUAL MODEL — non-destructive masking + loudness (Sec 7)
# =============================================================================

class PerceptualModel:
    """Frequency + temporal masking and loudness. Strictly non-destructive:
    this module only ever writes to perceptual_state / confidence.perceptual
    on an object. It never touches physical_signature and never deletes an
    object -- masking changes availability, not existence (Sec 7.2)."""

    MASK_THRESHOLD = 0.35  # DERIVED: below this, an object is considered perceptually masked

    def __init__(self, params: Parameters, world: WorldState, bark_centers_hz: np.ndarray,
                 spreading_matrix: np.ndarray):
        self.p = params
        self.world = world
        self.bark_centers_hz = bark_centers_hz
        self.spreading = spreading_matrix
        self.self_gain = np.diag(spreading_matrix).copy()
        self.logger = logging.getLogger("PerceptualModel")
        self._smoothed_avail: Dict[str, float] = {}

    def update(self, ev: PhysicalEvidence, t: float, dt: float) -> None:
        bark_energy = ev.bark_energy.astype(np.float64)
        field_total = self.spreading @ bark_energy  # total masking energy field per band (incl. self)

        active_objs = [o for o in self.world.objects.values()
                        if o.status in (ObjectStatus.ACTIVE, ObjectStatus.HIDDEN,
                                        ObjectStatus.RECOVERING, ObjectStatus.TENTATIVE)]
        for obj in active_objs:
            band = int(np.argmin(np.abs(self.bark_centers_hz - max(obj.physical_signature.frequency_hz, 1.0))))
            own_energy = max(obj.physical_signature.band_energy, 0.0)
            interference = max(field_total[band] - own_energy * self.self_gain[band], 0.0)
            raw_avail = own_energy / (own_energy + interference + 1e-9)

            prev = self._smoothed_avail.get(obj.object_id, raw_avail)
            tau = (self.p.mask_attack_ms if raw_avail < prev else self.p.mask_release_ms) / 1000.0
            alpha = 1.0 - math.exp(-dt / max(tau, 1e-4))
            smoothed = float(prev + alpha * (raw_avail - prev))
            self._smoothed_avail[obj.object_id] = smoothed

            loudness = self._specific_loudness(bark_energy, band)

            old_avail = obj.perceptual_state.perceptual_availability
            self.world.apply_update(UpdateRecord(t, "PerceptualModel", obj.object_id,
                                                  "perceptual_state.perceptual_availability",
                                                  old_avail, smoothed,
                                                  "frequency+temporal masking (ISO/MPEG-1-style spreading, "
                                                  "asymmetric attack/release)", smoothed))
            self.world.apply_update(UpdateRecord(t, "PerceptualModel", obj.object_id,
                                                  "perceptual_state.loudness",
                                                  obj.perceptual_state.loudness, loudness,
                                                  "Zwicker-style specific-loudness summation (E^0.23)", smoothed))
            new_perc_conf = float(np.clip(smoothed, 0.0, 1.0))
            self.world.apply_update(UpdateRecord(t, "PerceptualModel", obj.object_id, "confidence.perceptual",
                                                  obj.confidence.perceptual, new_perc_conf,
                                                  "masking availability", new_perc_conf))

            was_masked = old_avail < self.MASK_THRESHOLD
            now_masked = smoothed < self.MASK_THRESHOLD
            if now_masked and not was_masked:
                self.world.emit_event("masking_onset", obj.object_id,
                                       {"availability": smoothed, "band_hz": float(self.bark_centers_hz[band])},
                                       1.0 - smoothed, t)
            elif was_masked and not now_masked:
                self.world.emit_event("masking_offset", obj.object_id,
                                       {"availability": smoothed, "band_hz": float(self.bark_centers_hz[band])},
                                       smoothed, t)

    def _specific_loudness(self, bark_energy: np.ndarray, band: int) -> float:
        lo, hi = max(0, band - 1), min(len(bark_energy), band + 2)
        local = np.maximum(bark_energy[lo:hi], 0.0)
        specific = np.power(local, 0.23)  # Stevens/Zwicker-style compressive summation
        return float(np.sum(specific))


# =============================================================================
# 8. RELATIONSHIP GRAPH — basic, first-class typed edges (Sec 4.6 / Part 41 Sec 14)
# =============================================================================

class RelationshipEngine:
    """Computes MASKS / HARMONIC_RELATION / CO_OCCURS_WITH edges between
    currently live objects from already-computed physical/perceptual state.
    All mutation goes through world.upsert_relationship (the coordinator)."""

    def __init__(self, params: Parameters, world: WorldState):
        self.p = params
        self.world = world

    def update(self, t: float) -> None:
        objs = [o for o in self.world.objects.values() if o.status != ObjectStatus.ARCHIVED]
        for a, b in itertools.combinations(objs, 2):
            self._check_masking(a, b, t)
            self._check_harmonic(a, b, t)
            self._check_cooccurrence(a, b, t)

    def _check_masking(self, a: SoundObject, b: SoundObject, t: float) -> None:
        avail_a = a.perceptual_state.perceptual_availability
        avail_b = b.perceptual_state.perceptual_availability
        loud_a = a.perceptual_state.loudness
        loud_b = b.perceptual_state.loudness
        if avail_b < PerceptualModel.MASK_THRESHOLD and loud_a > loud_b * 1.5 and a.status == ObjectStatus.ACTIVE:
            self.world.upsert_relationship(a.object_id, b.object_id, RelationshipType.MASKS,
                                            strength=1.0 - avail_b, confidence=avail_a, t=t)
        if avail_a < PerceptualModel.MASK_THRESHOLD and loud_b > loud_a * 1.5 and b.status == ObjectStatus.ACTIVE:
            self.world.upsert_relationship(b.object_id, a.object_id, RelationshipType.MASKS,
                                            strength=1.0 - avail_a, confidence=avail_b, t=t)

    def _check_harmonic(self, a: SoundObject, b: SoundObject, t: float) -> None:
        fa, fb = a.physical_signature.frequency_hz, b.physical_signature.frequency_hz
        if fa <= 1.0 or fb <= 1.0:
            return
        ratio = fa / fb if fa > fb else fb / fa
        nearest = round(ratio)
        if nearest >= 1 and abs(ratio - nearest) < 0.06:
            strength = 1.0 - abs(ratio - nearest) / 0.06
            self.world.upsert_relationship(a.object_id, b.object_id, RelationshipType.HARMONIC_RELATION,
                                            strength=strength,
                                            confidence=min(a.confidence.identity, b.confidence.identity), t=t)

    def _check_cooccurrence(self, a: SoundObject, b: SoundObject, t: float) -> None:
        if a.status == ObjectStatus.ACTIVE and b.status == ObjectStatus.ACTIVE:
            dt = abs(a.last_observed_time - b.last_observed_time)
            if dt * 1000.0 <= self.p.temporal_group_tolerance_ms:
                self.world.upsert_relationship(a.object_id, b.object_id, RelationshipType.CO_OCCURS_WITH,
                                                strength=1.0 - dt, confidence=0.6, t=t)


# =============================================================================
# 9. GROUPING ENGINE — layer grouping (Sec 36 Level B; merge/split, non-destructive)
# =============================================================================

class GroupingEngine:
    """Recognizes that multiple, physically distinct SoundObjects (correctly
    kept separate by the base tracker's frequency-proximity gate, since they
    genuinely occupy different bands) function as ONE perceptual layer
    because they consistently fire together -- the canonical example being a
    kick's low-frequency body and its broadband click transient (Sec 33).

    Strictly additive: a Layer references member object_ids. It never
    merges, deletes, or rewrites the underlying objects' own identities or
    histories -- consistent with 'physical evidence is never destroyed'.

    Uses merge_confirmation / split_confirmation / minimum_merge_duration_ms
    / minimum_split_duration_ms from the numerical contract -- registered
    parameters that had no consumer until this module.

    A raw same-hop co-occurrence COUNT is not, by itself, sufficient evidence
    of shared physical origin: in sequenced/quantized music, two entirely
    different instruments (e.g. bass and hi-hat) can legitimately fire on the
    exact same hop repeatedly simply because they share a beat grid, not
    because they are one source. A true single-source pair (e.g. a kick's
    body + click, both excited by the same strike) should co-fire on nearly
    EVERY hop that either one fires at all. So confirmation additionally
    requires a conditional co-occurrence RATE in both directions -- count
    relative to each object's own independent firing rate -- not just an
    absolute count.
    """

    STALE_CANDIDATE_GAP_MS = 100.0     # DERIVED: a gap this long resets a merge-candidate's streak
    MIN_COOCCURRENCE_RATE = 0.5        # DERIVED: must co-fire on >=50% of EACH member's own firings

    def __init__(self, params: Parameters, world: WorldState):
        self.p = params
        self.world = world
        self.logger = logging.getLogger("GroupingEngine")
        self._candidate_evidence: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._member_miss_streak: Dict[Tuple[str, str], int] = {}
        self._fire_counts: Dict[str, int] = {}

    def update(self, t: float) -> None:
        just_matched = [o for o in self.world.objects.values()
                         if o.status != ObjectStatus.ARCHIVED and abs(o.last_observed_time - t) < 1e-6]
        just_matched_ids = {o.object_id for o in just_matched}
        for oid in just_matched_ids:
            self._fire_counts[oid] = self._fire_counts.get(oid, 0) + 1

        # --- merge candidates: objects that received real evidence on the exact same hop ---
        for a, b in itertools.combinations(just_matched, 2):
            key = tuple(sorted((a.object_id, b.object_id)))
            self._accumulate_candidate(key, t)
        for key, ev in list(self._candidate_evidence.items()):
            a_id, b_id = key
            rate_a = ev["count"] / max(self._fire_counts.get(a_id, 1), 1)
            rate_b = ev["count"] / max(self._fire_counts.get(b_id, 1), 1)
            if (ev["count"] >= self.p.merge_confirmation
                    and (t - ev["first_seen"]) * 1000.0 >= self.p.minimum_merge_duration_ms
                    and rate_a >= self.MIN_COOCCURRENCE_RATE and rate_b >= self.MIN_COOCCURRENCE_RATE):
                self._confirm_merge(key, t, rate_a, rate_b)

        # --- split checking: only when the layer actually had an opportunity to co-fire ---
        for layer in list(self.world.layers.values()):
            if layer.status != "CONFIRMED":
                continue
            fired = [m for m in layer.member_ids if m in just_matched_ids]
            if not fired:
                continue
            for m in list(layer.member_ids):
                skey = (layer.layer_id, m)
                if m in fired:
                    self._member_miss_streak[skey] = 0
                else:
                    self._member_miss_streak[skey] = self._member_miss_streak.get(skey, 0) + 1
                if self._member_miss_streak[skey] >= self.p.split_confirmation:
                    self.world.remove_layer_member(layer.layer_id, m, t,
                        reason=f"missed {self._member_miss_streak[skey]} consecutive group "
                                "activations (>= split_confirmation)")
                    self._member_miss_streak[skey] = 0

    def _accumulate_candidate(self, key: Tuple[str, str], t: float) -> None:
        ev = self._candidate_evidence.setdefault(key, {"count": 0, "first_seen": t, "last_seen": t})
        if (t - ev["last_seen"]) * 1000.0 > self.STALE_CANDIDATE_GAP_MS:
            ev["count"] = 0
            ev["first_seen"] = t
        ev["count"] += 1
        ev["last_seen"] = t

    def _confirm_merge(self, key: Tuple[str, str], t: float, rate_a: float, rate_b: float) -> None:
        a_id, b_id = key
        layer_a, layer_b = self._layer_of(a_id), self._layer_of(b_id)
        reason = (f"{a_id} and {b_id} co-fired {self.p.merge_confirmation}+ consecutive times "
                  f"within a {self.p.minimum_merge_duration_ms:.0f}ms+ window "
                  f"(co-occurrence rate {rate_a:.0%}/{rate_b:.0%} of each object's own firings)")
        if layer_a and layer_b and layer_a.layer_id != layer_b.layer_id:
            self.world.merge_layers(layer_a.layer_id, layer_b.layer_id, t,
                                     reason=f"{a_id}/{b_id} bridge previously separate layers")
        elif layer_a:
            self.world.add_layer_member(layer_a.layer_id, b_id, t, reason=reason)
        elif layer_b:
            self.world.add_layer_member(layer_b.layer_id, a_id, t, reason=reason)
        else:
            self.world.form_layer([a_id, b_id], t, reason=reason)
        del self._candidate_evidence[key]

    def _layer_of(self, object_id: str) -> Optional[Layer]:
        for layer in self.world.layers.values():
            if layer.status == "CONFIRMED" and object_id in layer.member_ids:
                return layer
        return None


# =============================================================================
# 10. EVENT ENGINE — evidence-level events not tied to a persistent object
# =============================================================================

class EventEngine:
    """Object-lifecycle and masking events are emitted directly by
    ObjectTracker/PerceptualModel (they own that decision). This module
    covers the remaining evidence-level event: a raw transient firing,
    independent of whether it gets matched to a persistent object."""

    def __init__(self, world: WorldState):
        self.world = world

    def process_observations(self, observations: List[Observation], t: float) -> None:
        for obs in observations:
            if obs.obs_type == ObservationType.TRANSIENT:
                self.world.emit_event("transient_detected", None,
                                       {"frequency_hz": obs.frequency_hz,
                                        "onset_strength": obs.physical_features.get("onset_strength", 0.0)},
                                       obs.confidence, t)


# =============================================================================
# 11. TIMBRE CLASSIFIER — rule-based attack/sustain classification (new increment)
# =============================================================================

class TimbreLabel(Enum):
    PERCUSSIVE = "PERCUSSIVE"    # fast attack, little sustain (kick, hihat, snare)
    SUSTAINED = "SUSTAINED"      # slow/moderate attack, long sustain (pad, drone, bowed)
    DECAYING = "DECAYING"        # struck-but-resonant, in between (pluck, mallet, bell)
    UNKNOWN = "UNKNOWN"          # not enough history yet to tell


class TimbreClassifier:
    """Classifies an object's timbre class from its OWN energy-over-time
    history -- not from a library of instrument templates, and not from an
    embedding. This distinguishes 'pad-like' from 'drum-like' by attack time
    and sustain ratio (Sec: deferred layer, minimal rule-based version).

    It deliberately does NOT attempt to say 'these two objects are the same
    instrument' or 'the same sample' -- that requires a pitch-normalized
    timbre fingerprint compared *across* objects (a SAME_SOURCE relationship
    built on top of RelationshipEngine) or a learned embedding, both out of
    scope for this increment. This module only classifies each object's own
    timbre shape in isolation.
    """

    MIN_LIFETIME_MS = 150.0          # DERIVED: need onset + partial tail before judging
    REEVALUATE_EVERY_N_HOPS = 20     # DERIVED: classification is slow-changing, not per-hop
    ATTACK_WINDOW_MS = 15.0          # DERIVED: attack faster than this -> percussive-like
    ONSET_WINDOW_MS = 50.0           # DERIVED: window used to measure onset-region energy
    TAIL_WINDOW_MS = (200.0, 400.0)  # DERIVED: window used to measure sustained-region energy

    def __init__(self, params: Parameters, world: WorldState):
        self.p = params
        self.world = world
        self.logger = logging.getLogger("TimbreClassifier")
        self._hops_since_eval: Dict[str, int] = {}

    def update(self, t: float) -> None:
        for obj in self.world.objects.values():
            if obj.status == ObjectStatus.ARCHIVED:
                continue
            lifetime_ms = (t - obj.creation_time) * 1000.0
            if lifetime_ms < self.MIN_LIFETIME_MS:
                continue
            n = self._hops_since_eval.get(obj.object_id, 0) + 1
            self._hops_since_eval[obj.object_id] = n
            if n % self.REEVALUATE_EVERY_N_HOPS != 0:
                continue
            label, conf = self._classify(obj)
            if label != obj.timbre_class or abs(conf - obj.timbre_confidence) > 0.05:
                self.world.record_decision(DecisionRecord(
                    time=t, module="TimbreClassifier", decision_type="timbre_classification",
                    target_id=obj.object_id, candidates=[], chosen=label,
                    reason=f"Attack/sustain rule-based classification -> {label}", confidence=conf))
                self.world.apply_update(UpdateRecord(t, "TimbreClassifier", obj.object_id, "timbre_class",
                                                      obj.timbre_class, label,
                                                      "attack/sustain shape of own energy history", conf))
                self.world.apply_update(UpdateRecord(t, "TimbreClassifier", obj.object_id, "timbre_confidence",
                                                      obj.timbre_confidence, conf,
                                                      "classification confidence", conf))

    def _classify(self, obj: SoundObject) -> Tuple[str, float]:
        hist = obj.history
        if len(hist) < 5:
            return TimbreLabel.UNKNOWN.value, 0.0
        t0 = obj.creation_time

        onset_vals = [h.energy for h in hist if (h.time - t0) * 1000.0 <= self.ONSET_WINDOW_MS]
        early_vals = [(h.time, h.energy) for h in hist if (h.time - t0) * 1000.0 <= 150.0]
        if not onset_vals or not early_vals:
            return TimbreLabel.UNKNOWN.value, 0.0
        peak_time, peak_val = max(early_vals, key=lambda te: te[1])
        attack_ms = (peak_time - t0) * 1000.0
        onset_energy = float(np.mean(onset_vals))

        lo_s, hi_s = self.TAIL_WINDOW_MS[0] / 1000.0, self.TAIL_WINDOW_MS[1] / 1000.0
        tail_vals = [h.energy for h in hist if lo_s <= (h.time - t0) <= hi_s]

        if not tail_vals:
            if attack_ms < self.ATTACK_WINDOW_MS:
                return TimbreLabel.PERCUSSIVE.value, 0.5
            return TimbreLabel.UNKNOWN.value, 0.3

        tail_energy = float(np.mean(tail_vals))
        sustain_ratio = tail_energy / (onset_energy + 1e-9)

        if attack_ms < self.ATTACK_WINDOW_MS and sustain_ratio < 0.25:
            return TimbreLabel.PERCUSSIVE.value, float(np.clip(1.0 - sustain_ratio, 0.5, 0.95))
        elif sustain_ratio > 0.6:
            return TimbreLabel.SUSTAINED.value, float(np.clip(sustain_ratio, 0.5, 0.95))
        else:
            return TimbreLabel.DECAYING.value, 0.55


# =============================================================================
# 12. RHYTHM ENGINE — beat / tempo / groove (Sec 14; 'slow loop', Sec 18/19)
# =============================================================================

class RhythmEngine:
    """Beat/tempo/groove tracking. 'Groove is not BPM. Groove is the
    relationship between events through time' (Sec 14) -- so this maintains
    a structured GrooveVector (Sec 14.1), not a single number, and updates
    it on the 'slow loop' cadence the spec assigns to groove stabilization
    (Sec 18: 'groove stabilization' is a slow-loop concern; Sec 19: rhythmic
    state should update on a tens-hundreds-of-ms timescale, not every hop).

    Tempo is estimated causally: onset-strength evidence is binned into a
    rolling multi-second buffer and autocorrelated to find the dominant
    periodicity (never looks ahead). Beat phase is tracked with a gentle
    phase-lock correction toward recent strong onsets rather than being
    rigidly recomputed from scratch each update, so a single spurious onset
    can't yank the whole beat grid.
    """

    UPDATE_INTERVAL_MS = 200.0            # DERIVED: 'tens-hundreds of ms' per Sec 19
    HISTORY_WINDOW_S = 8.0                # DERIVED: tempo autocorrelation lookback
    TEMPO_MIN_BPM = 60.0
    TEMPO_MAX_BPM = 200.0
    ACCENT_MULTIPLE = 1.5                 # DERIVED: onset > 1.5x recent median strength -> accent
    GROOVE_CHANGE_TEMPO_DELTA_BPM = 4.0   # DERIVED: tempo shift big enough to count as groove_changed
    PHASE_LOCK_GAIN = 0.15                # DERIVED: how strongly a strong onset pulls the beat grid
    N_SUBDIVISIONS = 16                   # 16th-note grid for accent_positions

    def __init__(self, params: Parameters, world: WorldState):
        self.p = params
        self.world = world
        self.logger = logging.getLogger("RhythmEngine")
        self._onset_times: List[float] = []
        self._onset_strengths: List[float] = []
        self._last_update_t = -1e9
        self._phase_origin: Optional[float] = None

    def observe_onset(self, t: float, strength: float) -> None:
        self._onset_times.append(t)
        self._onset_strengths.append(max(float(strength), 0.0))
        cutoff = t - self.HISTORY_WINDOW_S
        while self._onset_times and self._onset_times[0] < cutoff:
            self._onset_times.pop(0)
            self._onset_strengths.pop(0)

    def update(self, t: float) -> bool:
        """Returns True iff a beat boundary was just crossed this call."""
        if (t - self._last_update_t) * 1000.0 < self.UPDATE_INTERVAL_MS:
            return False
        self._last_update_t = t
        if len(self._onset_times) < 4:
            return False

        tempo_bpm, tempo_conf = self._estimate_tempo(t)
        if tempo_bpm <= 0:
            return False
        period_s = 60.0 / tempo_bpm

        if self._phase_origin is None:
            self._phase_origin = self._onset_times[-1]
        else:
            median_strength = float(np.median(self._onset_strengths))
            recent_strong = [ot for ot, os in zip(self._onset_times, self._onset_strengths)
                              if os > median_strength and (t - ot) < period_s * 2]
            if recent_strong:
                latest = recent_strong[-1]
                nearest_grid = self._phase_origin + period_s * round((latest - self._phase_origin) / period_s)
                self._phase_origin += self.PHASE_LOCK_GAIN * (latest - nearest_grid)

        old_groove = self.world.groove
        phase = ((t - self._phase_origin) / period_s) % 1.0
        next_beat_time = t + (1.0 - phase) * period_s

        new_groove = GrooveVector(
            tempo_bpm=tempo_bpm, tempo_confidence=tempo_conf, beat_phase=phase,
            next_beat_time=next_beat_time, timing_deviation_ms=self._timing_deviation(period_s),
            swing_ratio=self._swing_ratio(period_s), syncopation_index=self._syncopation_index(period_s),
            event_density=self._event_density(t), regularity=self._regularity(),
            accent_positions=self._accent_positions(period_s), last_updated=t)
        self.world.update_groove(new_groove, t)

        beat_fired = bool(old_groove.next_beat_time and old_groove.next_beat_time <= t and tempo_conf > 0.3)
        if beat_fired:
            self.world.emit_event("beat", None, {"tempo_bpm": tempo_bpm, "phase": phase}, tempo_conf, t)

        if self._onset_strengths:
            median_strength = float(np.median(self._onset_strengths))
            if self._onset_strengths[-1] > self.ACCENT_MULTIPLE * (median_strength + 1e-6):
                self.world.emit_event("accent", None, {"strength": self._onset_strengths[-1]}, tempo_conf, t)

        if (abs(tempo_bpm - old_groove.tempo_bpm) > self.GROOVE_CHANGE_TEMPO_DELTA_BPM
                or (old_groove.tempo_confidence < 0.3 <= tempo_conf)):
            self.world.emit_event("groove_changed", None,
                                   {"old_tempo": old_groove.tempo_bpm, "new_tempo": tempo_bpm}, tempo_conf, t)
        return beat_fired

    # ---- tempo estimation: causal onset-strength autocorrelation --------------------------
    def _estimate_tempo(self, t: float) -> Tuple[float, float]:
        times = np.array(self._onset_times)
        strengths = np.array(self._onset_strengths)
        if len(times) < 4:
            return 0.0, 0.0
        bin_s = 0.01
        n_bins = int(self.HISTORY_WINDOW_S / bin_s)
        sig = np.zeros(n_bins)
        t0 = t - self.HISTORY_WINDOW_S
        idx = np.clip(((times - t0) / bin_s).astype(int), 0, n_bins - 1)
        np.add.at(sig, idx, strengths)
        if sig.sum() <= 0:
            return 0.0, 0.0
        sig = sig - sig.mean()
        ac = np.correlate(sig, sig, mode="full")[n_bins - 1:]
        ac[0] = 0.0
        min_lag = max(1, int(60.0 / self.TEMPO_MAX_BPM / bin_s))
        max_lag = min(len(ac) - 1, int(60.0 / self.TEMPO_MIN_BPM / bin_s))
        if max_lag <= min_lag:
            return 0.0, 0.0
        search = ac[min_lag:max_lag + 1]
        peak_idx = int(np.argmax(search))
        peak_lag = min_lag + peak_idx
        peak_value = search[peak_idx]

        # Octave-ambiguity resolution: a real periodic pulse at HALF the winning lag
        # means the winning lag was actually the 2nd harmonic of a faster true beat
        # (this is exactly how a 0.5s kick + 0.25s hi-hat alias constructively at their
        # shared 1.0s lag). Test the hypothesis against the signal's own structure
        # rather than assuming any particular genre's typical tempo.
        half_lag = peak_lag // 2
        if half_lag >= min_lag and ac[half_lag] > 0.6 * peak_value:
            peak_lag = half_lag
            peak_value = ac[half_lag]

        conf = float(np.clip(peak_value / (np.max(np.abs(ac)) + 1e-9), 0.0, 1.0))
        return float(60.0 / (peak_lag * bin_s)), conf

    # ---- groove vector components (Sec 14.1) -----------------------------------------------
    def _timing_deviation(self, period_s: float) -> float:
        if self._phase_origin is None or not self._onset_times:
            return 0.0
        devs = []
        for ot in self._onset_times[-16:]:
            rel = (ot - self._phase_origin) % period_s
            devs.append(min(rel, period_s - rel) * 1000.0)
        return float(np.mean(devs)) if devs else 0.0

    def _swing_ratio(self, period_s: float) -> float:
        if len(self._onset_times) < 4:
            return 0.5
        half = period_s / 2.0
        long_gaps, short_gaps = [], []
        for i in range(1, len(self._onset_times)):
            gap = self._onset_times[i] - self._onset_times[i - 1]
            if 0 < gap <= period_s * 1.5:
                (long_gaps if gap >= half else short_gaps).append(gap)
        if not long_gaps or not short_gaps:
            return 0.5
        ratio = float(np.mean(long_gaps)) / (float(np.mean(long_gaps)) + float(np.mean(short_gaps)))
        return float(np.clip(ratio, 0.0, 1.0))

    def _syncopation_index(self, period_s: float) -> float:
        if self._phase_origin is None or not self._onset_times:
            return 0.0
        strong, weak = 0.0, 0.0
        for ot, os in zip(self._onset_times[-32:], self._onset_strengths[-32:]):
            rel_phase = ((ot - self._phase_origin) / period_s) % 1.0
            on_strong = min(rel_phase, 1.0 - rel_phase) < 0.06  # near phase 0 (the downbeat)
            if on_strong:
                strong += os
            else:
                weak += os
        total = strong + weak
        return float(weak / total) if total > 0 else 0.0

    def _event_density(self, t: float) -> float:
        recent = [ot for ot in self._onset_times if t - ot <= 2.0]
        return float(len(recent) / 2.0)

    def _regularity(self) -> float:
        if len(self._onset_times) < 4:
            return 0.0
        gaps = np.diff(self._onset_times[-16:])
        gaps = gaps[gaps > 0]
        if len(gaps) < 2 or gaps.mean() <= 0:
            return 0.0
        cv = float(gaps.std() / gaps.mean())
        return float(np.clip(1.0 - cv, 0.0, 1.0))

    def _accent_positions(self, period_s: float) -> List[float]:
        if self._phase_origin is None or not self._onset_times:
            return [0.0] * self.N_SUBDIVISIONS
        bins = np.zeros(self.N_SUBDIVISIONS)
        counts = np.zeros(self.N_SUBDIVISIONS)
        for ot, os in zip(self._onset_times[-64:], self._onset_strengths[-64:]):
            rel_phase = ((ot - self._phase_origin) / period_s) % 1.0
            b = int(rel_phase * self.N_SUBDIVISIONS) % self.N_SUBDIVISIONS
            bins[b] += os
            counts[b] += 1
        avg = np.divide(bins, np.maximum(counts, 1))
        m = float(avg.max())
        return (avg / m).tolist() if m > 0 else avg.tolist()


# =============================================================================
# 13. STRUCTURE ENGINE — patterns / phrases / sections (Sec 12.4, 14.2, 14.3)
# =============================================================================

class StructureEngine:
    """Bar-level pattern repetition and phrase/section boundary detection.

    Runs on the BAR cadence (driven by RhythmEngine's beat events, grouped
    4 beats per bar) -- coarser even than RhythmEngine's 200ms slow loop,
    since patterns/phrases are inherently multi-bar phenomena (Sec 18/19).
    'Fast events should not wait for section analysis' (Sec 14.3) -- this
    module only ever reads already-computed groove/object state; it never
    sits in the per-hop critical path.

    Pattern matching uses cosine similarity between per-bar signatures with
    a tolerance threshold, honoring Sec 14.2: 'A repeated musical pattern
    is not required to have sample-identical events.'
    """

    BEATS_PER_BAR = 4
    N_SUBDIVISIONS = 16                    # 16th-note resolution across one full bar
    PATTERN_SIMILARITY_THRESHOLD = 0.75    # DERIVED: tolerant match, not exact (Sec 14.2)
    PATTERN_MAX_PERIOD_BARS = 4            # DERIVED: search up to a 4-bar repeating cycle
    MIN_SECTION_BARS = 4                   # DERIVED: minimum section length between boundaries
    NOVELTY_THRESHOLD = 0.55               # DERIVED: structural-change score that triggers a boundary
    SIGNATURE_HISTORY_BARS = 64            # DERIVED: bounded lookback for pattern comparison

    def __init__(self, params: Parameters, world: WorldState):
        self.p = params
        self.world = world
        self.logger = logging.getLogger("StructureEngine")
        self._bar_signatures: List[np.ndarray] = []
        self._beat_count = 0
        self._bars_since_boundary = 0
        self._baseline_object_count = 0.0
        self._baseline_energy = 0.0
        self._baseline_initialized = False
        self._onset_times: List[float] = []       # this engine's OWN onset feed, scoped per-bar
        self._onset_strengths: List[float] = []
        self._current_bar_start: Optional[float] = None

    def observe_onset(self, t: float, strength: float) -> None:
        self._onset_times.append(t)
        self._onset_strengths.append(max(float(strength), 0.0))

    def observe_beat(self, t: float, active_object_ids: List[str], mean_energy: float,
                      period_s: float) -> None:
        if self._current_bar_start is None:
            self._current_bar_start = t - period_s
        self._beat_count += 1
        if self._beat_count % self.BEATS_PER_BAR != 0:
            return  # not a bar boundary yet

        bar_start, bar_end = self._current_bar_start, t
        sig = self._bin_bar_signature(bar_start, bar_end)
        self._current_bar_start = t
        # drop onsets we've now consumed; keep anything spilling into the next bar
        keep = [i for i, ot in enumerate(self._onset_times) if ot >= bar_end]
        self._onset_times = [self._onset_times[i] for i in keep]
        self._onset_strengths = [self._onset_strengths[i] for i in keep]

        self._bar_signatures.append(sig)
        if len(self._bar_signatures) > self.SIGNATURE_HISTORY_BARS:
            self._bar_signatures = self._bar_signatures[-self.SIGNATURE_HISTORY_BARS:]

        pattern_changed = self._detect_pattern(t)
        self._detect_structural_boundary(t, active_object_ids, mean_energy, pattern_changed)

    def _bin_bar_signature(self, bar_start: float, bar_end: float) -> np.ndarray:
        bar_period = max(bar_end - bar_start, 1e-6)
        bins = np.zeros(self.N_SUBDIVISIONS)
        for ot, os in zip(self._onset_times, self._onset_strengths):
            if ot < bar_start or ot >= bar_end:
                continue
            rel = (ot - bar_start) / bar_period
            b = int(rel * self.N_SUBDIVISIONS) % self.N_SUBDIVISIONS
            bins[b] += os
        m = float(bins.max())
        return bins / m if m > 0 else bins

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _detect_pattern(self, t: float) -> bool:
        n = len(self._bar_signatures)
        if n < 3:
            return self.world.set_pattern(None, 0.0, t)
        current = self._bar_signatures[-1]
        best_period, best_sim = None, 0.0
        for period in range(1, min(self.PATTERN_MAX_PERIOD_BARS, n - 1) + 1):
            sim = self._cosine_sim(current, self._bar_signatures[-1 - period])
            if sim > best_sim:
                best_sim, best_period = sim, period
        if best_period is not None and best_sim >= self.PATTERN_SIMILARITY_THRESHOLD:
            confirmed = True
            if n > 2 * best_period:
                sim2 = self._cosine_sim(current, self._bar_signatures[-1 - 2 * best_period])
                confirmed = sim2 >= self.PATTERN_SIMILARITY_THRESHOLD * 0.85
            if confirmed:
                return self.world.set_pattern(best_period, best_sim, t)
        return self.world.set_pattern(None, 0.0, t)

    def _detect_structural_boundary(self, t: float, active_object_ids: List[str],
                                     mean_energy: float, pattern_changed: bool) -> None:
        self._bars_since_boundary += 1
        active_count = len(active_object_ids)
        if not self._baseline_initialized:
            self._baseline_object_count = float(active_count)
            self._baseline_energy = mean_energy
            self._baseline_initialized = True
            return

        obj_change = abs(active_count - self._baseline_object_count) / max(self._baseline_object_count, 1.0)
        energy_change = abs(mean_energy - self._baseline_energy) / (self._baseline_energy + 1e-6)
        novelty = float(np.clip(0.4 * min(obj_change, 1.0) + 0.4 * min(energy_change, 1.0)
                                 + 0.2 * (1.0 if pattern_changed else 0.0), 0.0, 1.0))

        if novelty >= self.NOVELTY_THRESHOLD and self._bars_since_boundary >= self.MIN_SECTION_BARS:
            reason = (f"novelty={novelty:.2f} (object_count_change={obj_change:.2f}, "
                      f"energy_change={energy_change:.2f}, pattern_changed={pattern_changed})")
            self.world.start_new_section(t, reason=reason, novelty_score=novelty,
                                          active_object_ids=active_object_ids)
            self._bars_since_boundary = 0
            self._baseline_object_count = float(active_count)
            self._baseline_energy = mean_energy
        else:
            # slow adaptation so the baseline tracks gradual drift, not just a fixed t=0 reference
            self._baseline_object_count = 0.9 * self._baseline_object_count + 0.1 * active_count
            self._baseline_energy = 0.9 * self._baseline_energy + 0.1 * mean_energy


# =============================================================================
# 14. SEMANTIC ROLE ENGINE — probabilistic, contextual roles (Sec 15)
# =============================================================================

class SemanticRoleEngine:
    """Assigns probabilistic, contextual, revisable role scores.

    Explicitly NOT a classifier (Sec 15: 'Do not turn semantic roles into
    immutable classifications.'): every live object gets a score for EVERY
    role simultaneously, using only signals already computed elsewhere --
    register (frequency, relative to what else is currently active),
    timbre shape (from TimbreClassifier), harmonicity/pitch, perceptual
    prominence (loudness relative to the loudest current object), and
    rhythmic alignment (how close the object's evidence lands to the
    current beat grid, from RhythmEngine's public groove state).

    Sec 15 also lists 'genre/style hypothesis' as a contextual input --
    genre modeling (Sec 16) is a later, unbuilt step, so that input is
    honestly left out rather than faked with a placeholder.

    Scores are recomputed periodically per object (slow, not per-hop) so
    they can genuinely change as context shifts -- never locked in once
    assigned.
    """

    REEVALUATE_EVERY_N_HOPS = 30    # DERIVED: roles are contextual/slow-changing, not per-hop
    MIN_LIFETIME_MS = 100.0         # DERIVED: need at least a little signature before scoring
    ROLE_CHANGE_EVENT_THRESHOLD = 0.5  # DERIVED: only announce a role change once it's a real lead

    def __init__(self, params: Parameters, world: WorldState):
        self.p = params
        self.world = world
        self.logger = logging.getLogger("SemanticRoleEngine")
        self._hops_since_eval: Dict[str, int] = {}
        self._prev_top_role: Dict[str, str] = {}

    def update(self, t: float) -> None:
        live = [o for o in self.world.objects.values() if o.status != ObjectStatus.ARCHIVED]
        if not live:
            return
        freqs = [o.physical_signature.frequency_hz for o in live if o.physical_signature.frequency_hz > 0]
        loud_vals = [o.perceptual_state.loudness for o in live]
        register_lo = min(freqs) if freqs else 50.0
        register_hi = max(freqs) if freqs else 5000.0
        loud_max = max(loud_vals) if loud_vals else 1.0

        for obj in live:
            lifetime_ms = (t - obj.creation_time) * 1000.0
            if lifetime_ms < self.MIN_LIFETIME_MS:
                continue
            n = self._hops_since_eval.get(obj.object_id, 0) + 1
            self._hops_since_eval[obj.object_id] = n
            if n % self.REEVALUATE_EVERY_N_HOPS != 0:
                continue
            scores = self._score(obj, t, register_lo, register_hi, loud_max)
            self._apply(obj, scores, t)

    def _rhythm_alignment(self, obj: SoundObject, t: float) -> float:
        """0 = last real evidence landed exactly on a beat, 1 = exactly
        off-beat. Falls back to a neutral 0.5 when no rhythmic context is
        available yet, rather than silently asserting misalignment."""
        g = self.world.groove
        if g.tempo_confidence <= 0.2 or g.tempo_bpm <= 0 or not g.next_beat_time:
            return 0.5
        period_s = 60.0 / g.tempo_bpm
        k = math.ceil((g.next_beat_time - obj.last_observed_time) / period_s)
        grid_origin = g.next_beat_time - period_s * k
        phase = ((obj.last_observed_time - grid_origin) / period_s) % 1.0
        dist = min(phase, 1.0 - phase)
        return float(np.clip(1.0 - dist * 2.0, 0.0, 1.0))

    @staticmethod
    def _range_membership(f: float, lo: float, hi: float) -> float:
        """Smooth 0..1 membership for f falling within an absolute [lo, hi]
        band, centered/scaled in log-frequency (octave) space so it softens
        gracefully at the edges instead of a hard cutoff."""
        if f <= 0:
            return 0.0
        center = math.sqrt(lo * hi)
        half_width_oct = math.log2(hi / lo) / 2.0
        dist_oct = abs(math.log2(f / center))
        return float(np.clip(1.0 - dist_oct / (half_width_oct + 1e-6), 0.0, 1.0))

    def _score(self, obj: SoundObject, t: float, register_lo: float, register_hi: float,
               loud_max: float) -> Dict[str, float]:
        f = obj.physical_signature.frequency_hz
        span = max(register_hi - register_lo, 1.0)
        register_pos = float(np.clip((f - register_lo) / span, 0.0, 1.0)) if f > 0 else 0.5
        timbre = obj.timbre_class
        is_percussive = 1.0 if timbre == TimbreLabel.PERCUSSIVE.value else (
            0.3 if timbre == TimbreLabel.DECAYING.value else 0.0)
        is_sustained = 1.0 if timbre == TimbreLabel.SUSTAINED.value else (
            0.3 if timbre == TimbreLabel.DECAYING.value else 0.0)
        harmonicity = float(np.clip(obj.physical_signature.harmonicity, 0.0, 1.0))
        prominence = float(np.clip(obj.perceptual_state.loudness / (loud_max + 1e-9), 0.0, 1.0))
        on_beat = 1.0 - self._rhythm_alignment(obj, t)  # rhythm_alignment returns off-beat distance

        # Absolute physical ranges -- a kick lives around 35-180Hz whether or not
        # anything lower happens to be active right now; a bassline covers a wider
        # absolute span; percussion (hihat/cymbal-like) rises above roughly 600Hz.
        kick_range = self._range_membership(f, 35.0, 180.0)
        bass_range = self._range_membership(f, 35.0, 350.0)
        high_register = float(np.clip((f - 600.0) / 4000.0, 0.0, 1.0))

        kick = 0.30 * kick_range + 0.30 * is_percussive + 0.25 * (1.0 - harmonicity) + 0.15 * on_beat
        bass = 0.35 * bass_range + 0.30 * (1.0 - is_percussive) + 0.35 * harmonicity
        percussion = 0.40 * is_percussive + 0.35 * high_register + 0.25 * (1.0 - on_beat)
        harmonic_pad = 0.45 * is_sustained + 0.30 * harmonicity + 0.25 * (1.0 - prominence)
        lead_melody = 0.35 * register_pos + 0.30 * harmonicity + 0.35 * prominence

        return {"kick": float(np.clip(kick, 0, 1)), "bass": float(np.clip(bass, 0, 1)),
                "percussion": float(np.clip(percussion, 0, 1)),
                "harmonic_pad": float(np.clip(harmonic_pad, 0, 1)),
                "lead_melody": float(np.clip(lead_melody, 0, 1))}

    def _apply(self, obj: SoundObject, scores: Dict[str, float], t: float) -> None:
        old = obj.role_scores
        for role_name, value in scores.items():
            self.world.apply_update(UpdateRecord(t, "SemanticRoleEngine", obj.object_id,
                                                  f"role_scores.{role_name}", getattr(old, role_name), value,
                                                  "register + timbre + rhythm-alignment + prominence "
                                                  "(contextual, revisable)", max(scores.values())))
        self.world.apply_update(UpdateRecord(t, "SemanticRoleEngine", obj.object_id, "role_scores.last_updated",
                                              old.last_updated, t, "role re-evaluation", max(scores.values())))

        top_role, top_score = max(scores.items(), key=lambda kv: kv[1])
        prev = self._prev_top_role.get(obj.object_id)
        if top_score >= self.ROLE_CHANGE_EVENT_THRESHOLD and top_role != prev:
            self._prev_top_role[obj.object_id] = top_role
            self.world.emit_event("role_changed", obj.object_id,
                                   {"role": top_role, "score": top_score, "all_scores": scores},
                                   top_score, t)


# =============================================================================
# 15. AUDITORY WORLD MODEL — top-level coordinator (Sec 31 canonical loop)
# =============================================================================

class AuditoryWorldModel:
    """Ties every module to one WorldState and runs the canonical per-hop
    pipeline (Sec 31): physical evidence -> observations -> tracking
    -> perceptual weighting -> relationships -> (events emitted inline)
    -> replay snapshot."""

    def __init__(self, params: Optional[Parameters] = None):
        self.p = params or Parameters()
        self.world = WorldState(self.p)
        self.physical = PhysicalAnalyzer(self.p)
        self.obs_gen = ObservationGenerator(self.p, self.physical.bark_centers_hz)
        self.tracker = ObjectTracker(self.p, self.world)
        self.perceptual = PerceptualModel(self.p, self.world, self.physical.bark_centers_hz,
                                           self.physical.spreading_matrix)
        self.relationships = RelationshipEngine(self.p, self.world)
        self.grouping = GroupingEngine(self.p, self.world)
        self.rhythm = RhythmEngine(self.p, self.world)
        self.structure = StructureEngine(self.p, self.world)
        self.event_engine = EventEngine(self.world)
        self.timbre = TimbreClassifier(self.p, self.world)
        self.roles = SemanticRoleEngine(self.p, self.world)
        self.logger = logging.getLogger("AuditoryWorldModel")
        self.evidence_log: List[PhysicalEvidence] = []  # physical-layer replay support

    def run(self, audio: np.ndarray) -> None:
        evidence_list = self.physical.analyze_full(audio)
        n = len(evidence_list)
        self.logger.info("Processing %d fast-loop frames (%.2fs audio @ hop=%.2fms, %.1f frames/sec)",
                          n, n * self.p.fast_hop / self.p.sample_rate,
                          1000 * self.p.fast_hop / self.p.sample_rate, self.p.sample_rate / self.p.fast_hop)
        prev_t = 0.0
        for ev in evidence_list:
            t = ev.t
            dt = max(t - prev_t, 1e-6)
            self.world.t = t
            self.evidence_log.append(ev)

            observations = self.obs_gen.generate(ev)
            self.world.observations.extend(observations)
            if len(self.world.observations) > self.p.working_observation_limit:
                self.world.observations = self.world.observations[-self.p.working_observation_limit:]

            self.event_engine.process_observations(observations, t)
            for obs in observations:
                if obs.obs_type == ObservationType.TRANSIENT:
                    strength = obs.physical_features.get("onset_strength", 1.0)
                    self.rhythm.observe_onset(t, strength)
                    self.structure.observe_onset(t, strength)
            self.tracker.step(observations, t)          # candidate gen -> scoring -> hysteresis -> update
            self.perceptual.update(ev, t, dt)            # non-destructive masking + loudness
            self.relationships.update(t)                 # basic relationship graph
            self.grouping.update(t)                       # layer grouping (merge/split)
            self.timbre.update(t)                         # attack/sustain timbre classification
            beat_fired = self.rhythm.update(t)            # beat/tempo/groove (self-throttled, slow loop)
            if beat_fired and self.world.groove.tempo_bpm > 0:
                recent_ids = [o.object_id for o in self.world.objects.values()
                              if o.status != ObjectStatus.ARCHIVED and (t - o.last_observed_time) <= 1.0]
                recent_energy = float(np.mean([e.energy for e in self.evidence_log[-40:]])) if self.evidence_log else 0.0
                period_s = 60.0 / self.world.groove.tempo_bpm
                self.structure.observe_beat(t, recent_ids, recent_energy, period_s)
            self.roles.update(t)                          # probabilistic contextual roles (self-throttled)

            prev_t = t

        self.logger.info("Processing complete. objects=%d  events=%d  updates=%d  decisions=%d  relationships=%d",
                          len(self.world.objects), len(self.world.events), len(self.world.update_log),
                          len(self.world.decision_log), len(self.world.relationships))

    # ---- outputs -----------------------------------------------------------------------------
    def print_object_summary(self) -> None:
        print("\n" + "=" * 110)
        print("SOUND OBJECT SUMMARY")
        print("=" * 110)
        print(f"{'ID':<14}{'Status':<12}{'Created':>9}{'LastSeen':>10}{'Lifetime':>10}"
              f"{'Freq(Hz)':>10}{'AggConf':>9}{'PercAvail':>11}{'Timbre':>13}")
        print("-" * 110)
        for obj in sorted(self.world.objects.values(), key=lambda o: o.creation_time):
            lifetime = obj.last_observed_time - obj.creation_time
            timbre = f"{obj.timbre_class}({obj.timbre_confidence:.2f})"
            print(f"{obj.object_id:<14}{obj.status.value:<12}{obj.creation_time:>9.2f}{obj.last_observed_time:>10.2f}"
                  f"{lifetime:>10.2f}{obj.physical_signature.frequency_hz:>10.1f}"
                  f"{obj.confidence.aggregate():>9.2f}{obj.perceptual_state.perceptual_availability:>11.2f}"
                  f"{timbre:>13}")
        print("=" * 110)
        n_by_status: Dict[str, int] = {}
        for obj in self.world.objects.values():
            n_by_status[obj.status.value] = n_by_status.get(obj.status.value, 0) + 1
        print("By status:", ", ".join(f"{k}={v}" for k, v in sorted(n_by_status.items())))
        print(f"Relationships tracked: {len(self.world.relationships)}")
        for rel in list(self.world.relationships.values())[:10]:
            print(f"  {rel.source_id} --{rel.rel_type.value}--> {rel.target_id}  "
                  f"strength={rel.strength:.2f} conf={rel.confidence:.2f} evidence_count={rel.evidence_count}")
        confirmed_layers = [l for l in self.world.layers.values() if l.status == "CONFIRMED"]
        print(f"\nLayers formed (grouping engine): {len(self.world.layers)} total, "
              f"{len(confirmed_layers)} currently CONFIRMED")
        for layer in confirmed_layers:
            print(f"  {layer.layer_id}  members={layer.member_ids}  conf={layer.confidence:.2f}  "
                  f"formed={layer.formed_time:.2f}s  reason: {layer.formation_reason}")

    def print_events(self, limit: int = 60) -> None:
        print(f"\nEVENT LOG (showing last {min(limit, len(self.world.events))} of {len(self.world.events)}):")
        for e in self.world.events[-limit:]:
            tgt = e.target_id or "-"
            print(f"  [{e.time:7.3f}s] {e.event_type:<20} target={tgt:<14} conf={e.confidence:.2f}")

    def print_groove_state(self) -> None:
        """Prints the current GrooveVector (Sec 14.1: structured state, not
        a single BPM label) and how it evolved -- direct evidence this is
        continuous, replayable world state, not a one-shot label."""
        g = self.world.groove
        print("\n" + "=" * 78)
        print("GROOVE STATE (final)")
        print("=" * 78)
        if g.tempo_confidence <= 0.0:
            print("  (Not enough onset evidence accumulated to lock a tempo estimate.)")
            return
        print(f"  tempo            {g.tempo_bpm:6.1f} BPM   (confidence {g.tempo_confidence:.2f})")
        print(f"  beat_phase       {g.beat_phase:.2f}        next_beat_time={g.next_beat_time:.3f}s")
        print(f"  timing_deviation {g.timing_deviation_ms:6.1f} ms   (microtiming, lower = tighter)")
        print(f"  swing_ratio      {g.swing_ratio:.2f}        (0.5 = straight, higher = swung)")
        print(f"  syncopation      {g.syncopation_index:.2f}        (0 = all on-beat, 1 = all off-beat)")
        print(f"  event_density    {g.event_density:.2f} onsets/sec")
        print(f"  regularity       {g.regularity:.2f}        (1 = perfectly even inter-onset spacing)")
        if g.accent_positions:
            bar = " ".join(f"{v:.1f}" for v in g.accent_positions)
            print(f"  accent_positions [{bar}]  (relative energy across {len(g.accent_positions)} beat subdivisions)")
        if len(self.world.groove_history) >= 2:
            early, late = self.world.groove_history[0], self.world.groove_history[-1]
            print(f"\n  Tempo estimate over time: {early.tempo_bpm:.1f} BPM at t={early.last_updated:.2f}s "
                  f"-> {late.tempo_bpm:.1f} BPM at t={late.last_updated:.2f}s "
                  f"({len(self.world.groove_history)} groove updates recorded, replayable via groove_history)")

    def print_structure_state(self) -> None:
        """Prints detected patterns (Sec 12.4/14.2) and phrase/section
        boundaries (Sec 14.3)."""
        print("\n" + "=" * 78)
        print("STRUCTURE (patterns / sections)")
        print("=" * 78)
        print(f"Patterns detected: {len(self.world.patterns)}")
        for pat in self.world.patterns.values():
            print(f"  {pat.pattern_id}  period={pat.period_bars} bar(s)  status={pat.status:<7}  "
                  f"confidence={pat.confidence:.2f}  occurrences={pat.occurrence_count}  "
                  f"[{pat.first_seen:.2f}s - {pat.last_seen:.2f}s]")
        print(f"\nSections: {len(self.world.sections)}")
        for sec in self.world.sections:
            end = f"{sec.end_time:.2f}s" if sec.end_time is not None else "(ongoing)"
            print(f"  {sec.section_id}  [{sec.start_time:.2f}s - {end}]  novelty={sec.novelty_score:.2f}  "
                  f"reason: {sec.boundary_reason}")

    def print_role_summary(self) -> None:
        """Prints every object's full role-score vector (Sec 15: probabilistic,
        simultaneous roles, never a single classification)."""
        print("\n" + "=" * 96)
        print("SEMANTIC ROLES (probabilistic, contextual, revisable -- Sec 15)")
        print("=" * 96)
        print(f"{'ID':<14}{'kick':>7}{'bass':>7}{'percussion':>12}{'harmonic_pad':>14}{'lead_melody':>13}   top role(s)")
        print("-" * 96)
        for obj in sorted(self.world.objects.values(), key=lambda o: o.creation_time):
            if obj.status == ObjectStatus.ARCHIVED or obj.role_scores.last_updated == 0.0:
                continue
            r = obj.role_scores
            top = ", ".join(f"{name}={score:.2f}" for name, score in r.top(2))
            print(f"{obj.object_id:<14}{r.kick:>7.2f}{r.bass:>7.2f}{r.percussion:>12.2f}"
                  f"{r.harmonic_pad:>14.2f}{r.lead_melody:>13.2f}   {top}")

    def print_masked_bass_recovery_trace(self) -> None:
        """Prints the status timeline of whichever object spends the most
        cumulative time in HIDDEN/RECOVERING while ACTIVE elsewhere -- i.e.
        a direct check of the canonical behavior test in Sec 32."""
        best_obj, best_cycles = None, -1
        for obj in self.world.objects.values():
            statuses = [h.status for h in obj.history]
            cycles = sum(1 for i in range(1, len(statuses)) if statuses[i] == "HIDDEN" and statuses[i - 1] == "ACTIVE")
            if cycles > best_cycles:
                best_cycles, best_obj = cycles, obj
        if best_obj is None or best_cycles <= 0:
            print("\n(No object showed an ACTIVE->HIDDEN->RECOVERING->ACTIVE cycle in this run.)")
            return
        print(f"\nMASKING/RECOVERY TRACE for {best_obj.object_id} "
              f"({best_cycles} ACTIVE->HIDDEN transition(s) -- canonical test, Sec 32):")
        prev_status = None
        for h in best_obj.history:
            if h.status != prev_status:
                print(f"  [{h.time:7.3f}s] -> {h.status:<11} freq={h.frequency_hz:7.1f}Hz "
                      f"perc_avail={h.perceptual_availability:.2f} identity_conf={h.confidence['identity']:.2f}")
                prev_status = h.status

    def query_at(self, t: float) -> None:
        snap = self.world.query_at(t)
        print(f"\nWHAT THE SYSTEM BELIEVED AT t={t:.3f}s")
        print("-" * 78)
        if not snap["objects"]:
            print("  (no objects existed yet)")
        for oid, h in sorted(snap["objects"].items()):
            print(f"  {oid:<14} status={h.status:<11} freq={h.frequency_hz:7.1f}Hz energy={h.energy:.5f} "
                  f"perc_avail={h.perceptual_availability:.2f} identity_conf={h.confidence['identity']:.2f} "
                  f"existence_conf={h.confidence['existence']:.2f} timbre={h.timbre_class}({h.timbre_confidence:.2f})")
        layers_at_t = [l for l in self.world.layers.values()
                       if l.formed_time <= t and (l.dissolved_time is None or l.dissolved_time > t)]
        if layers_at_t:
            print(f"  Layers active at this time ({len(layers_at_t)}):")
            for layer in layers_at_t:
                print(f"    {layer.layer_id} members={layer.member_ids} conf={layer.confidence:.2f}")
        groove_at_t = None
        for g in self.world.groove_history:
            if g.last_updated <= t:
                groove_at_t = g
            else:
                break
        if groove_at_t and groove_at_t.tempo_confidence > 0:
            print(f"  Groove: {groove_at_t.tempo_bpm:.1f} BPM (conf {groove_at_t.tempo_confidence:.2f}), "
                  f"swing={groove_at_t.swing_ratio:.2f} syncopation={groove_at_t.syncopation_index:.2f}")
        recent = [e for e in snap["events_up_to"] if t - 0.5 <= e.time <= t]
        print(f"  Events in the preceding 0.5s ({len(recent)}):")
        for e in recent[-10:]:
            print(f"    [{e.time:.3f}s] {e.event_type} ({e.target_id or '-'})")
        if snap["nearby_decisions"]:
            print(f"  Decisions recorded within 100ms of t ({len(snap['nearby_decisions'])}):")
            for d in snap["nearby_decisions"][:5]:
                print(f"    [{d.time:.3f}s] {d.module}.{d.decision_type} -> {d.chosen}  reason: {d.reason}")

    def plot_activity(self, save_path: Optional[str] = None) -> None:
        objs = sorted(self.world.objects.values(), key=lambda o: o.creation_time)
        if not objs:
            print("No objects to plot.")
            return
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})
        status_colors = {"TENTATIVE": "#f0ad4e", "ACTIVE": "#5cb85c", "HIDDEN": "#d9534f",
                          "RECOVERING": "#5bc0de", "ARCHIVED": "#777777"}
        for idx, obj in enumerate(objs):
            hist = obj.history
            if not hist:
                continue
            for i in range(len(hist) - 1):
                color = status_colors.get(hist[i].status, "black")
                ax1.plot([hist[i].time, hist[i + 1].time], [idx, idx], color=color,
                         linewidth=6, solid_capstyle="butt")
            label = f"{obj.object_id} ({obj.physical_signature.frequency_hz:.0f}Hz)"
            ax1.text(hist[0].time - 0.03, idx, label, va="center", ha="right", fontsize=7.5)
        ax1.set_yticks([])
        ax1.set_ylabel("Sound Objects")
        ax1.set_title("Auditory World Model — Object Activity Over Time")
        ax1.set_xlim(left=-1.0)
        handles = [mpatches.Patch(color=c, label=s) for s, c in status_colors.items()]
        ax1.legend(handles=handles, loc="upper right", ncol=5, fontsize=8)

        transient_times = [e.time for e in self.world.events if e.event_type == "transient_detected"]
        if transient_times:
            ax2.eventplot(transient_times, colors="black", lineoffsets=0.5, linelengths=0.8)
        ax2.set_yticks([])
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Transients")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=130)
            self.logger.info("Saved activity plot to %s", save_path)
        plt.show()


# =============================================================================
# 16. DEMONSTRATION / MAIN — Colab-ready entry point
# =============================================================================

def main(audio_path: Optional[str] = None, duration: float = 20.0,
          plot_save_path: Optional[str] = "auditory_world_model_activity.png") -> AuditoryWorldModel:
    print_parameter_registry(Parameters())
    params = Parameters()
    model = AuditoryWorldModel(params)
    source = AudioSource(params)

    if audio_path is None:
        audio = source.synth_test_signal(duration=duration)
    else:
        audio, _sr = source.load(audio_path)

    model.run(audio)
    model.print_object_summary()
    model.print_events(limit=80)
    model.print_masked_bass_recovery_trace()
    model.print_groove_state()
    model.print_structure_state()
    model.print_role_summary()
    model.plot_activity(save_path=plot_save_path)
    if plot_save_path:
        print(f"\nActivity plot saved to: {plot_save_path}")
        if HAVE_IPYTHON:
            try:
                from IPython.display import Image as _IPyImage
                display(_IPyImage(filename=plot_save_path))
            except Exception:
                pass
    return model


if __name__ == "__main__":
    _path = sys.argv[1] if len(sys.argv) > 1 else None
    wm = main(_path)

    print("\n" + "=" * 78)
    print("INSPECTOR — 'what did the system believe at time T?'")
    print("In a notebook, call:  wm.query_at(<seconds>)   e.g. wm.query_at(2.35)")
    print("=" * 78)
    try:
        while True:
            s = input("\nQuery belief at time T (seconds), or 'q' to quit: ").strip()
            if s.lower() in ("q", "quit", ""):
                break
            try:
                wm.query_at(float(s))
            except ValueError:
                print("  Please enter a number of seconds, or 'q'.")
    except (EOFError, KeyboardInterrupt):
        pass
