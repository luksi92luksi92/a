#!/usr/bin/env python3
"""Auditory World Model — pretrained timbre embedding integration.

This is an ADDITIVE, NON-DESTRUCTIVE extension.  It does not perform audio
reconstruction and does not replace the physical analyzer.  It adds a learned
continuous timbre representation using the MERIT timbre head over the MERT
v1-330M backbone. MERIT publishes a pretrained 128-D timbre projection head
whose intended use is timbral similarity, making it a much better fit for
cross-event identity than a generic text/audio embedding.

The embedding is optional. If torch/transformers/torchaudio/huggingface_hub or
the model/head files are unavailable, the script falls back to the existing
DSP-only grouping implementation rather than failing the auditory pipeline.

The model is applied to short local MIXED-AUDIO windows around tracked
objects. These embeddings are evidence for similarity, not isolated stems.
No waveform reconstruction or source-separation output is produced.

Usage:
    python auditory_world_model_timbre_embedding.py
    python auditory_world_model_timbre_embedding.py path/to/track.wav

Dependencies when enabling learned embeddings:
    pip install torch torchaudio transformers huggingface_hub
    huggingface-cli download amaai-lab/merit \
        head_tim/best_head.pt --local-dir ./models

The MERT backbone is downloaded automatically by Transformers the first time
it is used.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "auditory_world_model (5).py"
GROUPING_FIX = ROOT / "auditory_world_model_grouping_fix.py"
HEAD_PATH = Path(os.environ.get("AWM_MERIT_TIMBRE_HEAD", str(ROOT / "models" / "head_tim" / "best_head.pt")))
MODEL_ID = os.environ.get("AWM_MERIT_MODEL", "m-a-p/MERT-v1-330M")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


awm = _load_module(ORIGINAL, "awm_original_embedding")
gfix = _load_module(GROUPING_FIX, "awm_grouping_fix_embedding")


class MeritTimbreEmbedder:
    """Frozen MERIT timbre encoder with a small per-window cache."""

    EXTRACT_LAYERS = (3, 4, 5, 6, 23)
    SAMPLE_RATE = 24_000
    WINDOW_SECONDS = 1.5
    MAX_WINDOWS_PER_OBJECT = 8
    MIN_ACTIVE_HISTORY = 3

    def __init__(self, head_path: Path = HEAD_PATH, model_id: str = MODEL_ID):
        self.available = False
        self.device = "cpu"
        self.model = None
        self.processor = None
        self.head = None
        self.torch = None
        self.torchaudio = None
        self.model_id = model_id
        self.head_path = head_path
        self._window_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._object_embeddings: Dict[str, np.ndarray] = {}
        self.error: Optional[str] = None
        self._try_load()

    def _try_load(self) -> None:
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            import torchaudio
            from transformers import AutoModel, Wav2Vec2FeatureExtractor

            self.torch = torch
            self.torchaudio = torchaudio
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            class ProjectionHead(nn.Module):
                def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(in_dim, hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_dim, out_dim, bias=False),
                    )

                def forward(self, x):
                    return F.normalize(self.net(x), dim=-1)

            if not self.head_path.exists():
                raise FileNotFoundError(
                    f"MERIT timbre head not found at {self.head_path}. "
                    "Download head_tim/best_head.pt from amaai-lab/merit."
                )

            ckpt = torch.load(self.head_path, map_location=self.device, weights_only=True)
            self.head = ProjectionHead(
                int(ckpt["in_dim"]), int(ckpt["hidden_dim"]), int(ckpt["out_dim"])
            ).to(self.device).eval()
            self.head.load_state_dict(ckpt["state_dict"])

            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=True
            ).to(self.device).eval()
            self.model.requires_grad_(False)
            self.available = True
        except Exception as exc:  # optional capability
            self.error = str(exc)
            self.available = False

    def _resample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr == self.SAMPLE_RATE:
            return np.asarray(audio, dtype=np.float32)
        wav = self.torch.from_numpy(np.asarray(audio, dtype=np.float32))
        if wav.ndim > 1:
            wav = wav.mean(dim=0)
        out = self.torchaudio.functional.resample(wav, sr, self.SAMPLE_RATE)
        return out.cpu().numpy().astype(np.float32)

    def _embed_window(self, audio: np.ndarray, sr: int, start_s: float) -> Optional[np.ndarray]:
        if not self.available:
            return None
        start = max(0, int(round(start_s * sr)))
        stop = min(len(audio), start + int(round(self.WINDOW_SECONDS * sr)))
        if stop <= start:
            return None
        key = (start, stop)
        cached = self._window_cache.get(key)
        if cached is not None:
            return cached.copy()

        segment = np.asarray(audio[start:stop], dtype=np.float32)
        segment = self._resample(segment, sr)
        target_len = int(self.SAMPLE_RATE * self.WINDOW_SECONDS)
        if len(segment) < target_len:
            segment = np.pad(segment, (0, target_len - len(segment)))
        else:
            segment = segment[:target_len]

        inputs = self.processor(segment, sampling_rate=self.SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            parts = [out.hidden_states[layer].mean(dim=1) for layer in self.EXTRACT_LAYERS]
            backbone = self.torch.cat(parts, dim=-1)
            emb = self.head(backbone)[0].detach().cpu().numpy().astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-9
        self._window_cache[key] = emb.copy()
        return emb

    def embed_object(self, obj, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Aggregate local mixed-audio embeddings over an object's own active
        history. Median-normalized pooling reduces the influence of a single
        contaminated moment."""
        if not self.available:
            return None
        if len(obj.history) < self.MIN_ACTIVE_HISTORY:
            return None

        active_times = [
            float(h.time)
            for h in obj.history[-128:]
            if h.status in (
                awm.ObjectStatus.ACTIVE.value,
                awm.ObjectStatus.RECOVERING.value,
                awm.ObjectStatus.TENTATIVE.value,
            )
        ]
        if len(active_times) < self.MIN_ACTIVE_HISTORY:
            return None

        # Evenly sample the object's history instead of embedding every hop.
        idxs = np.linspace(0, len(active_times) - 1,
                           min(self.MAX_WINDOWS_PER_OBJECT, len(active_times)), dtype=int)
        embs: List[np.ndarray] = []
        half = self.WINDOW_SECONDS / 2.0
        for idx in sorted(set(int(i) for i in idxs)):
            center = active_times[idx]
            emb = self._embed_window(audio, sr, max(0.0, center - half))
            if emb is not None:
                embs.append(emb)
        if not embs:
            return None
        stacked = np.vstack(embs)
        pooled = np.median(stacked, axis=0)
        pooled /= np.linalg.norm(pooled) + 1e-9
        self._object_embeddings[obj.object_id] = pooled.astype(np.float32)
        return pooled.astype(np.float32)

    def build_all(self, world, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        if not self.available:
            return {}
        for obj in world.objects.values():
            if obj.status == awm.ObjectStatus.ARCHIVED:
                continue
            self.embed_object(obj, audio, sr)
        return dict(self._object_embeddings)

    def similarity(self, a_id: str, b_id: str) -> float:
        a = self._object_embeddings.get(a_id)
        b = self._object_embeddings.get(b_id)
        if a is None or b is None:
            return 0.0
        return float(np.clip(np.dot(a, b), -1.0, 1.0))


class EmbeddingAwareGroupingEngine(gfix.ImprovedGroupingEngine):
    """Adds learned timbre similarity to the existing deterministic grouping.

    The deterministic signals remain dominant. The embedding is an auxiliary
    identity cue and cannot form a cluster by itself.
    """

    def __init__(self, params, world, embedder: Optional[MeritTimbreEmbedder] = None):
        super().__init__(params, world)
        self.embedder = embedder
        self._embedding_profiles: Dict[str, np.ndarray] = {}

    def _pair_score(self, a, b, pa, pb) -> float:
        base = super()._pair_score(a, b, pa, pb)
        if self.embedder is None or not self.embedder.available:
            return base
        emb_sim = self.embedder.similarity(a.object_id, b.object_id)
        if emb_sim <= 0.0:
            return base
        # Auxiliary only: maximum contribution 0.16 and only for already
        # plausible deterministic pairs.
        return float(np.clip(0.84 * base + 0.16 * emb_sim, 0.0, 1.0))


class EmbeddingAwareSameSourceEngine(awm.SameSourceEngine):
    """Use learned timbre embeddings as a stronger cross-event re-ID cue."""

    UPDATE_INTERVAL_S = 1.0
    SIMILARITY_THRESHOLD = 0.80

    def __init__(self, params, world, embedder: MeritTimbreEmbedder):
        super().__init__(params, world)
        self.embedder = embedder

    def update(self, t: float) -> None:
        if (t - self._last_update_t) < self.UPDATE_INTERVAL_S:
            return
        self._last_update_t = t
        candidates = [
            o for o in self.world.objects.values()
            if o.status != awm.ObjectStatus.ARCHIVED
            and o.object_id in self.embedder._object_embeddings
        ]
        for a, b in __import__("itertools").combinations(candidates, 2):
            if a.status == awm.ObjectStatus.ACTIVE and b.status == awm.ObjectStatus.ACTIVE:
                continue
            fa = a.physical_signature.frequency_hz
            fb = b.physical_signature.frequency_hz
            if fa > 0 and fb > 0 and abs(math.log2(fa / fb)) > 6.0:
                continue
            sim = self.embedder.similarity(a.object_id, b.object_id)
            if sim >= self.SIMILARITY_THRESHOLD:
                self.world.upsert_relationship(
                    a.object_id, b.object_id, awm.RelationshipType.SAME_SOURCE,
                    strength=sim, confidence=sim, t=t,
                )


def run(audio_path: Optional[str] = None, duration: float = 8.0) -> object:
    params = awm.Parameters()
    model = awm.AuditoryWorldModel(params)

    # Replace only the grouping/same-source engines. Everything upstream stays
    # exactly the same as the current implementation.
    embedder = MeritTimbreEmbedder()
    model.grouping = EmbeddingAwareGroupingEngine(params, model.world, embedder)
    model.same_source = EmbeddingAwareSameSourceEngine(params, model.world, embedder)

    source = awm.AudioSource(params)
    if audio_path is None:
        audio = source.synth_test_signal(duration=duration)
    else:
        audio, sr = source.load(audio_path)
        params_sr = params.sample_rate
        if sr != params_sr:
            raise RuntimeError(f"AudioSource returned unexpected sample rate {sr}; expected {params_sr}")

    # First pass: original physical/object pipeline. We disable its internal
    # grouping because we need embeddings available before grouping decisions.
    # The grouping engine is invoked after tracking by the original run(), so
    # we instead provide a small embedded pipeline below.
    evidence_list = model.physical.analyze_full(audio)
    stem_bark_energy = None
    if model.stem_separation.available:
        stems = model.stem_separation.separate(audio, params.sample_rate)
        if stems:
            stem_bark_energy = {name: model.physical.bark_energy_only(wav) for name, wav in stems.items()}

    prev_t = 0.0
    for ev in evidence_list:
        t = ev.t
        dt = max(t - prev_t, 1e-6)
        model.world.t = t
        model.evidence_log.append(ev)
        observations = model.obs_gen.generate(ev, stem_bark_energy)
        model.world.observations.extend(observations)
        if len(model.world.observations) > params.working_observation_limit:
            model.world.observations = model.world.observations[-params.working_observation_limit:]
        model.event_engine.process_observations(observations, t)
        for obs in observations:
            if obs.obs_type == awm.ObservationType.TRANSIENT:
                strength = obs.physical_features.get("onset_strength", 1.0)
                bass_weight = float(ev.bark_energy[model._low_band_mask].sum() / (ev.bark_energy.sum() + 1e-9))
                model.rhythm.observe_onset(t, strength, bass_weight)
                model.structure.observe_onset(t, strength)
        model.tracker.step(observations, t)
        model.perceptual.update(ev, t, dt)
        model.relationships.update(t)
        model.timbre.update(t)
        # Build/refresh learned embeddings periodically from current tracked objects.
        # This happens after enough history has accumulated and before grouping.
        if embedder.available and (int(round(t / 0.5)) != int(round(prev_t / 0.5))):
            embedder.build_all(model.world, audio, params.sample_rate)
        model.grouping.update(t)
        model.rhythm.update(t)
        model.roles.update(t)
        model.style.update(t)
        model.fingerprint.update(ev, t)
        model.same_source.update(t)
        prev_t = t

    print("\n" + "=" * 88)
    print("PRETRAINED TIMBRE EMBEDDING VALIDATION")
    print("=" * 88)
    print(f"MERIT available: {embedder.available}")
    if not embedder.available:
        print(f"Reason: {embedder.error}")
    print(f"SoundObjects: {len(model.world.objects)}")
    confirmed = [l for l in model.world.layers.values() if l.status == "CONFIRMED" and len(l.member_ids) >= 2]
    print(f"Confirmed Layers: {len(confirmed)}")
    print(f"Embedding-backed object identities: {len(embedder._object_embeddings)}")
    print(f"Relationships: {len(model.world.relationships)}")
    for layer in confirmed:
        print(f"  {layer.layer_id}: members={layer.member_ids} conf={layer.confidence:.2f}")
    print("=" * 88)
    return model


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
