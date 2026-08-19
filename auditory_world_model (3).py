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
                                 harmonicity=float(obs.physical_features.get("harmonic", 0.0)),
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
# 12. AUDITORY WORLD MODEL — top-level coordinator (Sec 31 canonical loop)
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
        self.event_engine = EventEngine(self.world)
        self.timbre = TimbreClassifier(self.p, self.world)
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
            self.tracker.step(observations, t)          # candidate gen -> scoring -> hysteresis -> update
            self.perceptual.update(ev, t, dt)            # non-destructive masking + loudness
            self.relationships.update(t)                 # basic relationship graph
            self.grouping.update(t)                       # layer grouping (merge/split)
            self.timbre.update(t)                         # attack/sustain timbre classification

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
# 13. DEMONSTRATION / MAIN — Colab-ready entry point
# =============================================================================

def main(audio_path: Optional[str] = None, duration: float = 8.0,
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
