import numpy as np
from edm_embedding_layer import EmbeddingEngine, EmbeddingRecord, EmbeddingStore, SimpleRhythmAdapter


def test_cosine_and_retrieval():
    store = EmbeddingStore()
    engine = EmbeddingEngine(store)
    a = EmbeddingRecord('a', 'test', '1', [1, 0, 0], 3)
    b = EmbeddingRecord('b', 'test', '1', [0.99, 0.1, 0], 3)
    c = EmbeddingRecord('c', 'test', '1', [0, 1, 0], 3)
    for r in (a, b, c):
        store.upsert(r)
    assert engine.cosine(a, b) > engine.cosine(a, c)
    assert engine.retrieve_candidates(a, 2)[0][0] == 'b'
    evidence = engine.compare('a', 'b', 'timbre', 'test', threshold=0.5)
    assert evidence is not None
    assert evidence.similarity > 0.9


def test_rhythm_adapter_is_explicitly_derived():
    adapter = SimpleRhythmAdapter()
    class O:
        object_id = 'x'
    record = adapter.encode_object(O(), np.ones(4096, dtype=np.float32), 24000)
    assert record.channel == 'derived_rhythm'
    assert record.provenance['learned'] is False
    assert record.dimension == 16
