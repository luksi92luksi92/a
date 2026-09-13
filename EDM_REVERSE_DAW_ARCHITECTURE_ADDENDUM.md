# EDM Reverse-DAW Architecture Addendum

## Purpose

This addendum extends the Auditory World Model foundation into an EDM-specific reverse-DAW perception and reconstruction system without replacing the core WorldState, observation/object distinction, lifecycle, confidence, provenance, or replay architecture.

## Architectural hierarchy

The system preserves the distinction between temporary evidence and persistent identity and adds an explicit musical-element layer:

```text
Audio
  -> source separation
  -> multi-resolution physical analysis
  -> observations
  -> persistent SoundObjects
  -> Elements
  -> Patterns
  -> Phrases / Sections
  -> Arrangement
  -> reverse-DAW representation
```

`Observation` is evidence. `SoundObject` is persistent identity. `Element` is a higher-level reusable musical component. `Pattern` describes temporally repeated organization and tolerates variation. `Phrase/Section` describes larger structural spans. `Arrangement` represents song-level organization.

## Stem-specialized perception

The front-end detector is specialized by separated stem while publishing the same object contract to WorldState:

- **drums**: transient/onset emphasis, kick/snare/hat/clap/tom/cymbal/impact candidates, very short boundary scales, overlap handling, decay-tail ownership, rhythmic anchoring.
- **bass**: pitch trajectory, note onset/offset, sustain, slide/glide, harmonicity, sub-band energy, repeated-note/motif context.
- **vocals**: voiced/unvoiced evidence, phrase and syllabic activity, breath/silence evidence, pitch trajectory, harmonicity, chop candidates, longer context.
- **other**: synth stabs, pads, keys/guitars, FX, risers, downlifters and impacts using spectral novelty, texture evolution, harmonicity, transient evidence and longer temporal scales.

Stem-specific detectors may use different thresholds, features and models, but they must emit the shared object representation so downstream modules remain stem-agnostic where appropriate.

## Object detector design

The detector is not a silence-trimming segmenter. The full stem remains available to the detector. Silence/activity is represented as evidence and prioritization rather than destructive deletion.

Detection proceeds through:

1. adaptive activity estimation;
2. multi-scale candidate generation;
3. stem-specific event proposals;
4. multi-pass recovery of weaker events;
5. boundary refinement from waveform envelope, spectral change and event-specific cues;
6. merge/split and overlap resolution;
7. artifact/noise rejection;
8. multidimensional confidence and quality scoring;
9. classical feature extraction;
10. compact signatures and learned embeddings;
11. object publication to WorldState.

Objects should represent audio events, not phrase windows. Phrase windows are a later structural-analysis mechanism.

## Non-destructive perceptual processing

Masking, denoising, loudness normalization, and salience weighting must not destroy physical evidence. A quiet or masked event may retain a high physical-existence state and a lower perceptual-availability state.

## Object identity and similarity

Object identity is represented separately from pattern identity. An object may be a member of a sample/event family while participating in many patterns.

Similarity is a cascade:

```text
object audio
  -> cheap signature
  -> candidate retrieval
  -> detailed DSP comparison
  -> learned embedding comparison
  -> identity decision
```

Nearest-neighbor search should use an ANN index such as FAISS when object counts justify it. Similarity evidence is retained rather than immediately collapsing objects into one record.

## Learned representations

Learned music representations are first-class evidence, behind model adapters. Candidate models include MERT, MERIT, CLAP-style audio encoders, and task-specific pretrained models. They are not architectural dependencies of WorldState.

Where a disentangled representation is available, preserve separate dimensions for rhythm, timbre and melody rather than forcing all information into one scalar similarity.

Example object representation:

```text
features:
  temporal: ...
  spectral: ...
  harmonic: ...
  envelope: ...

embeddings:
  mert: ...
  merit_melody: ...
  merit_rhythm: ...
  merit_timbre: ...

signature: ...
```

Model-specific tensor layouts must be converted into the project's own stable contracts.

## Object metadata

A persistent SoundObject should retain, when available:

- onset/start, peak and offset/end;
- duration;
- boundary confidence;
- existence, identity, perceptual, prediction, grouping and semantic confidence;
- physical, spectral, harmonic and envelope features;
- learned embedding references;
- previous/next object links and gap measurements;
- beat/bar position and timing residual;
- local density and periodicity;
- tail ownership and overlap evidence;
- identity-cluster evidence and nearest-neighbor evidence;
- pattern/phrase/section membership;
- provenance and decision evidence.

## Boundary policy

Boundary refinement is a first-class operation because reverse-DAW reconstruction requires cut points suitable for sample/event extraction. Boundary estimates must remain distinct from phrase or section boundaries.

The system should preserve both:

- **event boundaries** — start/end of one audio event;
- **structural boundaries** — transitions between patterns, phrases, sections or arrangement states.

## Rhythm and meter

Tempo/beat/groove context is a shared cross-stem layer. Beat and bar estimates should be attached to objects as contextual evidence and can feed back into identity/grouping decisions.

Beat alignment must not overwrite physical timestamps. Store both measured event time and metrical interpretation.

## Element layer

Elements sit between object identity and pattern/structure:

```text
SoundObjects
  -> Element families
      -> Pattern instances
          -> Phrases / Sections
```

Examples include kick family, snare/clap family, hat family, bass motif, synth stab family, pad layer, vocal chop, riser and impact. Elements are probabilistic hypotheses, not immutable labels.

## EDM structure

EDM-specific structural interpretation should combine:

- object/element density;
- beat/bar regularity;
- pattern changes;
- new element appearance/disappearance;
- energy and spectral change;
- low-end state changes;
- FX buildup evidence;
- silence/dropout evidence;
- repeated-section resets.

Likely `intro`, `build`, `drop`, `break`, `breakdown`, `post-drop`, `outro`, etc. are semantic hypotheses over structural evidence, not hard-coded prerequisites of the object detector.

## Layered grouping

Grouping must support:

- many observations -> one SoundObject;
- one observation -> multiple hypotheses;
- object families -> Elements;
- Elements -> Patterns;
- Patterns/energy/novelty -> Phrases/Sections.

Merge/split decisions require hysteresis, evidence and provenance.

## Variation handling

Do not collapse all instances of a sample family into one object. Preserve variation such as velocity, processing, layering and context. Identity families and occurrence instances are separate concepts.

## Evaluation and tuning

The system requires a ground-truth annotation/evaluation layer for tuning. At minimum measure:

- event precision/recall;
- onset and offset error;
- boundary precision/recall at multiple tolerances;
- object identity accuracy;
- element classification accuracy;
- pattern detection precision/recall;
- section-boundary error;
- false merge rate and false split rate.

Thresholds must be versioned. Tunable parameters belong in the numerical parameter registry rather than scattered literals.

## Implementation stack principles

The stack is capability-driven rather than tied to one library:

- audio I/O: SoundFile, torchaudio;
- DSP: NumPy, SciPy, librosa, torchaudio;
- source separation: Demucs and compatible separators;
- inference: PyTorch and Transformers/model-specific runtimes;
- learned music representations: MERT, MERIT, CLAP-style and task-specific encoders as evaluated;
- approximate nearest-neighbor retrieval: FAISS or equivalent;
- clustering: graph/density/hierarchical methods such as HDBSCAN/DBSCAN where appropriate;
- storage: JSON for interchange, with SQLite/Parquet available for large object graphs;
- visualization/debugging: matplotlib/Plotly or a dedicated local UI.

No downstream module should depend on a library's internal tensor/object representation. Each external model/library is accessed through an adapter that publishes the project's stable data contracts.

## Persistence and provenance

Intermediate evidence must be retained whenever practical. The system should be able to explain why an object, identity, element, pattern or structural hypothesis exists and what evidence changed its confidence.

Recommended persisted layers:

```text
observations
objects
object_features
object_signatures
object_embeddings
object_similarity
identity_clusters
elements
patterns
phrases_sections
arrangement
```

The architecture is additive: existing WorldState and foundation modules remain authoritative; these layers extend perception and musical interpretation rather than replacing the core model.
