#!/usr/bin/env python3
"""EDM reverse-DAW learned embedding layer.

Additive extension of the existing AWM/EDM layers. Embeddings are first-class
evidence, rhythm/timbre/melody channels remain separate, model-specific output
is normalized into stable contracts, and similarity is retained without
collapsing SoundObjects. The retrieval interface can later be replaced by
FAISS without changing callers.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EmbeddingRecord:
    object_id: str
    model: str
    version: str
    vector: List[float]
    dimension: int
    normalized: bool = True
    source: str = "mixed_audio"
    channel: str = "timbre"
    window_start: Optional[float] = None
    window_end: Optional[float] = None
    confidence: float = 1.0
    provenance: Dict[str, object] = field(default_factory=dict)


@dataclass
class SimilarityEvidence:
    evidence_id: str
    object_a: str
    object_b: str
    channel: str
    model: str
    similarity: float
    confidence: float
    reason: str


class EmbeddingStore:
    """Stable in-memory store with JSON export and an ANN-ready contract."""

    def __init__(self) -> None:
        self.records: Dict[Tuple[str, str, str], EmbeddingRecord] = {}
        self.similarities: Dict[str, SimilarityEvidence] = {}

    def upsert(self, record: EmbeddingRecord) -> None:
        self.records[(record.object_id, record.channel, record.model)] = record

    def get(self, object_id: str, channel: str, model: str) -> Optional[EmbeddingRecord]:
        return self.records.get((object_id, channel, model))

    def add_similarity(self, evidence: SimilarityEvidence) -> None:
        a, b = sorted((evidence.object_a, evidence.object_b))
        key = f"{a}|{b}|{evidence.channel}|{evidence.model}"
        self.similarities[key] = evidence

    def objects(self, channel: str, model: Optional[str] = None) -> List[str]:
        return sorted({r.object_id for r in self.records.values()
                       if r.channel == channel and (model is None or r.model == model)})

    def export_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "embeddings": [asdict(r) for r in self.records.values()],
            "similarity_evidence": [asdict(e) for e in self.similarities.values()],
        }, indent=2, sort_keys=True), encoding="utf-8")
        return path


class EmbeddingAdapter:
    """Stable adapter boundary for model-specific representations."""

    MODEL = "unknown"
    VERSION = "0"

    def encode(self, audio: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        return x / (np.linalg.norm(x) + 1e-9)


class MeritTimbreAdapter(EmbeddingAdapter):
    """Adapter for the repository's existing MERIT/MERT timbre implementation."""

    MODEL = "MERIT/MERT-v1-330M"
    VERSION = "merit-timbre-head"

    def __init__(self, embedder) -> None:
        self.embedder = embedder

    def encode_object(self, obj, audio: np.ndarray, sample_rate: int) -> Optional[EmbeddingRecord]:
        if not getattr(self.embedder, "available", False):
            return None
        vector = self.embedder.embed_object(obj, audio, sample_rate)
        if vector is None:
            return None
        vector = self.normalize(vector)
        return EmbeddingRecord(
            object_id=obj.object_id,
            model=self.MODEL,
            version=self.VERSION,
            vector=vector.tolist(),
            dimension=int(vector.size),
            source="mixed_audio",
            channel="timbre",
            confidence=1.0,
            provenance={"adapter": self.__class__.__name__},
        )


class SimpleRhythmAdapter(EmbeddingAdapter):
    """Derived rhythm descriptor; clearly not presented as a learned embedding."""

    MODEL = "derived-rhythm-v1"
    VERSION = "1"

    def encode_object(self, obj, audio: np.ndarray, sample_rate: int) -> EmbeddingRecord:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        env = np.abs(x)
        if env.size > 8192:
            env = env[np.linspace(0, env.size - 1, 8192).astype(int)]
        chunks = np.array_split(env, 16)
        vector = self.normalize(np.asarray([float(np.mean(c)) if len(c) else 0.0 for c in chunks]))
        return EmbeddingRecord(
            object_id=obj.object_id,
            model=self.MODEL,
            version=self.VERSION,
            vector=vector.tolist(),
            dimension=int(vector.size),
            source="mixed_audio",
            channel="derived_rhythm",
            confidence=0.5,
            provenance={"adapter": self.__class__.__name__, "learned": False},
        )


class EmbeddingEngine:
    """Attach embeddings and similarity evidence without mutating object identity."""

    def __init__(self, store: Optional[EmbeddingStore] = None) -> None:
        self.store = store or EmbeddingStore()
        self.adapters: List[object] = []

    def register(self, adapter: object) -> None:
        self.adapters.append(adapter)

    def build_for_object(self, obj, audio: np.ndarray, sample_rate: int) -> List[EmbeddingRecord]:
        records: List[EmbeddingRecord] = []
        for adapter in self.adapters:
            if isinstance(adapter, MeritTimbreAdapter):
                record = adapter.encode_object(obj, audio, sample_rate)
            else:
                record = adapter.encode_object(obj, audio, sample_rate)
            if record is not None:
                self.store.upsert(record)
                records.append(record)
        return records

    @staticmethod
    def cosine(a: EmbeddingRecord, b: EmbeddingRecord) -> float:
        va, vb = np.asarray(a.vector, dtype=np.float32), np.asarray(b.vector, dtype=np.float32)
        if va.size != vb.size or va.size == 0:
            return 0.0
        return float(np.clip(np.dot(va, vb), -1.0, 1.0))

    def compare(self, a_id: str, b_id: str, channel: str, model: str,
                threshold: float = 0.78) -> Optional[SimilarityEvidence]:
        a, b = self.store.get(a_id, channel, model), self.store.get(b_id, channel, model)
        if a is None or b is None:
            return None
        sim = self.cosine(a, b)
        confidence = float(np.clip((sim - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0))
        evidence = SimilarityEvidence(
            evidence_id="sim_" + uuid.uuid4().hex[:10],
            object_a=a_id, object_b=b_id, channel=channel, model=model,
            similarity=sim, confidence=confidence,
            reason=f"{channel} embedding cosine similarity",
        )
        self.store.add_similarity(evidence)
        return evidence

    def retrieve_candidates(self, query: EmbeddingRecord, top_k: int = 8) -> List[Tuple[str, float]]:
        """Brute-force implementation of the ANN contract; FAISS can replace it later."""
        candidates = []
        for record in self.store.records.values():
            if record.channel != query.channel or record.model != query.model or record.object_id == query.object_id:
                continue
            candidates.append((record.object_id, self.cosine(query, record)))
        candidates.sort(key=lambda x: (-x[1], x[0]))
        return candidates[:max(0, int(top_k))]


def attach_to_edm(edm, audio: np.ndarray, sample_rate: int, merit_embedder=None) -> EmbeddingEngine:
    """Populate the EDM embedding layer; SoundObjects remain authoritative."""
    engine = EmbeddingEngine()
    if merit_embedder is not None and getattr(merit_embedder, "available", False):
        engine.register(MeritTimbreAdapter(merit_embedder))
    engine.register(SimpleRhythmAdapter())

    for obj in edm.world.objects.values():
        if getattr(obj, "status", None) == "ARCHIVED":
            continue
        engine.build_for_object(obj, audio, sample_rate)

    for channel in ("timbre", "derived_rhythm"):
        models = sorted({r.model for r in engine.store.records.values() if r.channel == channel})
        for model in models:
            ids = engine.store.objects(channel, model)
            for i, a_id in enumerate(ids):
                for b_id in ids[i + 1:]:
                    engine.compare(a_id, b_id, channel, model,
                                   threshold=0.80 if channel == "timbre" else 0.88)
    return engine
''