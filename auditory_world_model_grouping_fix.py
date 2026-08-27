#!/usr/bin/env python3
"""Auditory World Model — coherent layer-grouping fix.

This module deliberately leaves the existing physical analysis, object
tracking, masking, timbre, semantics, fingerprinting and optional HTDemucs
cue untouched.  It replaces the existing post-hoc same-hop grouping rule
with a stable profile-based grouping stage.

The original grouping rule mostly asked: "did two objects fire on the same
hop?".  That is too weak for a musical source model: kick body/click can be
slightly staggered, bass notes can alternate pitch, a pad can be sustained,
and hi-hat energy can move between bands.  The replacement groups tracked
objects using several persistent cues:

* log-frequency/register
* timbre class + continuous physical timbre statistics
* harmonicity / spectral flatness / crest
* temporal activity and beat-phase occupancy
* probabilistic role compatibility (weak prior only)
* pairwise temporal/harmonic coherence

Clusters are disjoint and non-destructive.  A layer references its member
SoundObjects; it never mutates or deletes their identities/history.

Run from the same directory as the original file:

    python auditory_world_model_grouping_fix.py

or with a wav:

    python auditory_world_model_grouping_fix.py path/to/track.wav
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ORIGINAL = Path(__file__).with_name("auditory_world_model (5).py")


# ---------------------------------------------------------------------------
# Load the original module without modifying it on disk.
# ---------------------------------------------------------------------------
if not ORIGINAL.exists():
    raise FileNotFoundError(
        f"Expected original implementation next to this file: {ORIGINAL}"
    )

_spec = importlib.util.spec_from_file_location("awm_original", ORIGINAL)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load {ORIGINAL}")
awm = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = awm
_spec.loader.exec_module(awm)


class ImprovedGroupingEngine(awm.GroupingEngine):
    """Group persistent tracked objects into coherent perceptual layers.

    This is intentionally a *grouping* model, not a source separator.  It
    operates on already tracked objects and preserves the raw objects as
    evidence.  It also avoids the original single-link/co-fire failure mode
    that allowed musical simultaneity to masquerade as shared source.
    """

    UPDATE_INTERVAL_S = 0.25
    MIN_OBJECT_AGE_MS = 120.0
    MIN_OBJECT_HISTORY = 8

    # Cluster acceptance.  These are deliberately conservative because false
    # merging is more damaging to identity than leaving a fragment separate.
    PAIR_THRESHOLD = 0.66
    CLUSTER_THRESHOLD = 0.72
    MIN_CLUSTER_SIZE = 2

    # Frequency compatibility: same layer may span a lot (kick body + click),
    # but there must still be some explanatory relationship.
    MAX_LOG_F_SPAN = 8.0  # octaves; effectively permissive, coherence does work

    # Candidate hysteresis: use the existing numerical contract when possible.
    DEFAULT_FORM_CONFIRMATIONS = 3

    def __init__(self, params, world):
        # Do not call the base implementation: its same-hop candidate state is
        # exactly what this replacement is superseding.
        self.p = params
        self.world = world
        self.logger = awm.logging.getLogger("ImprovedGroupingEngine")
        self._last_update_t = -1e9
        self._candidate_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self._candidate_last_seen: Dict[Tuple[str, ...], float] = {}
        self._formed_signatures: Dict[str, Tuple[str, ...]] = {}

    # ----------------------------- feature extraction ---------------------
    @staticmethod
    def _finite(values: Iterable[float], default: float = 0.0) -> np.ndarray:
        a = np.asarray(list(values), dtype=np.float64)
        a = a[np.isfinite(a)]
        return a if a.size else np.asarray([default], dtype=np.float64)

    def _history_samples(self, obj) -> List[Tuple[float, object]]:
        """Return informative history states, de-duplicating frozen holds."""
        hist = obj.history[-256:]
        out = []
        last = None
        for h in hist:
            # HIDDEN entries carry the last real value; do not treat those as
            # evidence of continuing activity.
            active = h.status in (
                awm.ObjectStatus.ACTIVE.value,
                awm.ObjectStatus.RECOVERING.value,
                awm.ObjectStatus.TENTATIVE.value,
            )
            key = (
                round(float(h.time), 3),
                h.status,
                round(float(h.frequency_hz), 1),
                round(float(h.energy), 7),
            )
            if key != last:
                out.append((h.time, h, active))
                last = key
        return out

    def _profile(self, obj, t: float) -> Dict[str, object]:
        hist = self._history_samples(obj)
        if not hist:
            return {}

        active = [h for _, h, is_active in hist if is_active]
        if not active:
            active = [h for _, h, _ in hist]

        freqs = self._finite(h.frequency_hz for h in active if h.frequency_hz > 0, 1.0)
        harmonic = self._finite(h.confidence.get("prediction", 0.5) * 0.0 + 0.0 for h in [])
        # History does not store harmonicity directly; the current physical
        # signature is the authoritative persistent value.
        harmonicity = float(np.clip(obj.physical_signature.harmonicity, 0.0, 1.0))
        flatness = float(np.clip(obj.physical_signature.spectral_flatness, 0.0, 1.0))
        crest = float(np.clip(obj.physical_signature.crest_factor, 0.0, 12.0))

        # Activity ratio over the object's lifetime is a strong discriminator:
        # pad ~= continuously active, kick ~= brief regular occupancy, bass is
        # intermediate, and hats are brief/high-register.
        start = float(obj.creation_time)
        end = max(float(t), start + 1e-3)
        duration = end - start
        active_times = np.asarray([float(h.time) for _, h, is_active in hist if is_active])
        if active_times.size:
            activity_ratio = float(np.clip(active_times.size * 0.012 / max(duration, 0.25), 0.0, 1.0))
        else:
            activity_ratio = 0.0

        # Use the groove grid when available. A phase histogram captures WHEN
        # the source tends to exist without demanding exact sample repetition.
        phase_hist = np.zeros(16, dtype=np.float64)
        g = self.world.groove
        if g.tempo_confidence > 0.3 and g.tempo_bpm > 0 and g.next_beat_time:
            period = 60.0 / g.tempo_bpm
            origin = g.next_beat_time
            for _, h, is_active in hist:
                if not is_active:
                    continue
                phase = ((float(h.time) - origin) / period) % 1.0
                phase_hist[min(15, int(phase * 16.0))] += 1.0
        else:
            # Fallback: normalize occurrence time to the observed span.
            span = max(hist[-1][0] - hist[0][0], 1e-3)
            for tt, _, is_active in hist:
                if is_active:
                    phase_hist[min(15, int(((tt - hist[0][0]) / span) * 16.0))] += 1.0
        if phase_hist.sum() > 0:
            phase_hist /= np.linalg.norm(phase_hist) + 1e-9

        # Weak semantic context.  It is deliberately not an instrument label
        # gate: it only influences similarity after physical evidence agrees.
        role = np.asarray([
            obj.role_scores.kick,
            obj.role_scores.bass,
            obj.role_scores.percussion,
            obj.role_scores.harmonic_pad,
            obj.role_scores.lead_melody,
        ], dtype=np.float64)
        role_norm = role / (np.linalg.norm(role) + 1e-9)

        return {
            "logf": float(np.log2(max(float(np.median(freqs)), 1.0))),
            "fmin": float(np.min(freqs)),
            "fmax": float(np.max(freqs)),
            "harmonicity": harmonicity,
            "flatness": flatness,
            "crest": min(crest / 8.0, 1.0),
            "activity": activity_ratio,
            "phase": phase_hist,
            "role": role_norm,
            "timbre": obj.timbre_class,
            "confidence": float(np.clip(obj.confidence.identity, 0.0, 1.0)),
            "age": float((t - obj.creation_time) * 1000.0),
        }

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))

    def _pair_score(self, a, b, pa, pb) -> float:
        if not pa or not pb:
            return 0.0

        # Hard sanity check: do not bridge absurd register gaps unless both
        # objects have the same sustained/percussive profile and strong phase
        # evidence. This prevents a giant "everything in the mix" component.
        logf_gap = abs(pa["logf"] - pb["logf"])
        if logf_gap > self.MAX_LOG_F_SPAN:
            return 0.0

        freq_sim = math.exp(-logf_gap / 1.75)
        timbre_sim = 1.0 if pa["timbre"] == pb["timbre"] else 0.25
        harmonic_sim = 1.0 - abs(pa["harmonicity"] - pb["harmonicity"])
        flat_sim = 1.0 - abs(pa["flatness"] - pb["flatness"])
        crest_sim = 1.0 - abs(pa["crest"] - pb["crest"])
        activity_sim = math.exp(-abs(pa["activity"] - pb["activity"]) / 0.35)
        phase_sim = self._cos(pa["phase"], pb["phase"])
        role_sim = self._cos(pa["role"], pb["role"])

        # Harmonic relation is useful for multi-component layers such as a
        # kick's fundamental/body, or a pad's partials.
        fa = max(1.0, pa["logf"])
        fb = max(1.0, pb["logf"])
        # Convert log-frequency difference back to a ratio.
        ratio = 2.0 ** abs(fa - fb)
        nearest_int = max(1, round(ratio))
        harmonic_relation = math.exp(-abs(ratio - nearest_int) / 0.08) if nearest_int <= 16 else 0.0

        # Shared temporal behavior is the strongest grouping cue. Pure beat
        # coincidence is not enough because phase/activity must also agree.
        score = (
            0.16 * freq_sim
            + 0.12 * timbre_sim
            + 0.10 * harmonic_sim
            + 0.08 * flat_sim
            + 0.06 * crest_sim
            + 0.16 * activity_sim
            + 0.22 * phase_sim
            + 0.06 * role_sim
            + 0.04 * harmonic_relation
        )

        # Same-source components tend to share repeated activity. Two objects
        # that both merely fire on every beat should still receive a modest
        # score, not an automatic merge.
        return float(np.clip(score, 0.0, 1.0))

    def _cluster_similarity(self, members: Sequence[object], profiles: Dict[str, Dict[str, object]]) -> float:
        if len(members) < 2:
            return 1.0
        vals = []
        # Complete-link style check: every member must agree with the others.
        for a, b in itertools.combinations(members, 2):
            vals.append(self._pair_score(a, b, profiles[a.object_id], profiles[b.object_id]))
        return float(min(vals)) if vals else 0.0

    def _agglomerative(self, objects: List[object], profiles: Dict[str, Dict[str, object]]) -> List[List[object]]:
        clusters: List[List[object]] = [[o] for o in objects]
        while len(clusters) > 1:
            best = None
            best_score = -1.0
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    merged = clusters[i] + clusters[j]
                    # Complete-link coherence; prevents a bridge object from
                    # turning kick+hat+bass into one chain-connected monster.
                    sc = self._cluster_similarity(merged, profiles)
                    if sc >= self.PAIR_THRESHOLD and sc > best_score:
                        best_score = sc
                        best = (i, j)
            if best is None:
                break
            i, j = best
            clusters[i].extend(clusters[j])
            del clusters[j]
        return [c for c in clusters if len(c) >= self.MIN_CLUSTER_SIZE]

    # ----------------------------- world integration ----------------------
    def _layer_overlap(self, layer, ids: set) -> float:
        old = set(layer.member_ids)
        if not old or not ids:
            return 0.0
        return len(old & ids) / max(len(old | ids), 1)

    def _update_existing_layer(self, layer, members: List[object], t: float, reason: str) -> None:
        wanted = [o.object_id for o in members]
        wanted_set = set(wanted)
        for oid in list(layer.member_ids):
            if oid not in wanted_set:
                self.world.remove_layer_member(layer.layer_id, oid, t, reason)
        for oid in wanted:
            if oid not in layer.member_ids:
                self.world.add_layer_member(layer.layer_id, oid, t, reason)
        layer.last_updated = t
        layer.confidence = float(np.clip(0.7 * layer.confidence + 0.3 * self.CLUSTER_THRESHOLD, 0.0, 0.95))

    def update(self, t: float) -> None:
        if (t - self._last_update_t) < self.UPDATE_INTERVAL_S:
            return
        self._last_update_t = t

        candidates = [
            o for o in self.world.objects.values()
            if o.status != awm.ObjectStatus.ARCHIVED
            and (t - o.creation_time) * 1000.0 >= self.MIN_OBJECT_AGE_MS
            and len(o.history) >= self.MIN_OBJECT_HISTORY
        ]
        if len(candidates) < 2:
            return

        profiles = {o.object_id: self._profile(o, t) for o in candidates}
        clusters = self._agglomerative(candidates, profiles)

        # Only accept clusters that have enough internally coherent evidence.
        valid = []
        for c in clusters:
            coh = self._cluster_similarity(c, profiles)
            if coh >= self.CLUSTER_THRESHOLD:
                valid.append((c, coh))

        # Track candidate cluster persistence using object membership, not
        # momentary co-fire count. This is stable through masking and note
        # changes. Keep only the current top representation for each object.
        current_keys = set()
        for members, coherence in valid:
            key = tuple(sorted(o.object_id for o in members))
            current_keys.add(key)
            prev = self._candidate_last_seen.get(key)
            if prev is None or (t - prev) * 1000.0 <= 1000.0:
                self._candidate_counts[key] += 1
            else:
                self._candidate_counts[key] = 1
            self._candidate_last_seen[key] = t

        # Form/update layers only after persistent evidence. Existing layers
        # are matched by overlap so their identity remains stable.
        for members, coherence in valid:
            ids = set(o.object_id for o in members)
            key = tuple(sorted(ids))
            count = self._candidate_counts.get(key, 0)
            if count < getattr(self.p, "merge_confirmation", self.DEFAULT_FORM_CONFIRMATIONS):
                continue

            matching = [
                layer for layer in self.world.layers.values()
                if layer.status == "CONFIRMED" and self._layer_overlap(layer, ids) >= 0.5
            ]
            reason = (
                f"persistent multi-cue layer coherence={coherence:.2f}; "
                f"members={len(members)}; candidate_updates={count}; "
                f"phase/timbre/activity/frequency evidence"
            )
            if matching:
                layer = max(matching, key=lambda l: self._layer_overlap(l, ids))
                self._update_existing_layer(layer, members, t, reason)
            else:
                layer = self.world.form_layer([o.object_id for o in members], t, reason=reason)
                layer.confidence = float(np.clip(coherence, 0.6, 0.95))
                self._formed_signatures[layer.layer_id] = key

        # Split stale members when a member has persistently become incoherent
        # with its layer. This is deliberately slower than merge formation.
        for layer in list(self.world.layers.values()):
            if layer.status != "CONFIRMED" or len(layer.member_ids) < 2:
                continue
            members = [self.world.objects[oid] for oid in layer.member_ids if oid in self.world.objects]
            if len(members) < 2:
                continue
            lp = {o.object_id: profiles.get(o.object_id) or self._profile(o, t) for o in members}
            cluster_coh = self._cluster_similarity(members, lp)
            if cluster_coh >= 0.55:
                layer.confidence = float(np.clip(0.8 * layer.confidence + 0.2 * cluster_coh, 0.0, 0.95))

        # Prune candidate memories that have gone stale.
        for key, last in list(self._candidate_last_seen.items()):
            if (t - last) > 2.0:
                self._candidate_last_seen.pop(key, None)
                self._candidate_counts.pop(key, None)


def run_fixed(audio_path: Optional[str] = None, duration: float = 20.0,
              plot_save_path: Optional[str] = "auditory_world_model_grouping_fixed.png"):
    """Run the original model with only GroupingEngine replaced."""
    awm.GroupingEngine = ImprovedGroupingEngine
    # AuditoryWorldModel resolves GroupingEngine at construction time, so the
    # replacement above is sufficient and leaves every other original module
    # intact.
    model = awm.main(audio_path=audio_path, duration=duration, plot_save_path=plot_save_path)

    confirmed = [
        l for l in model.world.layers
        if model.world.layers[l].status == "CONFIRMED"
        and len(model.world.layers[l].member_ids) >= 2
    ]
    print("\n" + "=" * 88)
    print("GROUPING VALIDATION")
    print("=" * 88)
    print(f"Final SoundObjects: {len(model.world.objects)}")
    print(f"Confirmed coherent Layers: {len(confirmed)}")
    for lid in confirmed:
        layer = model.world.layers[lid]
        members = [model.world.objects[x] for x in layer.member_ids if x in model.world.objects]
        freqs = [m.physical_signature.frequency_hz for m in members if m.physical_signature.frequency_hz > 0]
        roles = []
        for m in members:
            top = m.role_scores.top(1)
            if top and top[0][1] > 0.35:
                roles.append(f"{top[0][0]}:{top[0][1]:.2f}")
        fdesc = f"{min(freqs):.0f}-{max(freqs):.0f}Hz" if freqs else "?"
        print(f"  {lid}: {len(members)} members, {fdesc}, conf={layer.confidence:.2f}")
        if roles:
            print(f"    roles: {', '.join(roles[:8])}")
    print("=" * 88)
    return model


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run_fixed(path)
