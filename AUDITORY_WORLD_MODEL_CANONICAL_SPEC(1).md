# AUDITORY WORLD MODEL — CANONICAL IMPLEMENTATION SPECIFICATION
## Consolidated from SPEC(1).md
## Version 1.1 — Audited and reconciled architecture

> This document is the authoritative implementation specification for the project.
> It is intended to survive conversation limits, model changes, and transfer to a new code-generation conversation.
> Previous conversational wording is non-authoritative when it conflicts with this document.

---

# 0. PROJECT CONTRACT

## 0.1 Goal

Build a real-time auditory perception engine for interactive music visualization. The system converts audio into a persistent internal model of interacting sound objects, their relationships, temporal behavior, musical context, semantic roles, and high-level events. Applications such as VJ systems consume that model but do not modify perception.

The project is **not** primarily:

- a genre classifier;
- a waveform analyzer;
- a stem/source-separation system;
- a collection of independent instrument classifiers;
- an end-to-end black-box neural network.

The core abstraction is an **auditory world model**.

## 0.2 Governing question

The system should continuously answer:

1. What evidence exists?
2. What perceptual objects could that evidence represent?
3. Which observations belong to the same object?
4. Which objects persist over time?
5. Which objects belong together as layers/families?
6. What relationships exist between objects?
7. What musical patterns and structures are emerging?
8. What is changing?
9. How confident is the system?
10. What meaningful events/state should applications receive?

## 0.3 Non-negotiable principles

1. **Do not destroy information early.** Raw/physical evidence must remain available even when perceptually masked.
2. **Masking changes perceptual availability/confidence, not physical existence.** A masked object can become hidden rather than deleted.
3. **Observations are evidence; objects are persistent knowledge.**
4. **Object identity is not equivalent to raw sound shape.** Identity combines temporal continuity, physical features, relationships, prediction, context, and optionally embeddings.
5. **Grouping is probabilistic and multi-cue.** No single feature owns object construction.
6. **Perception, semantics, and application output are separate layers.**
7. **Applications never directly mutate the world model.**
8. **Genre/subgenre knowledge interprets the universal world model; it does not redefine the physical/perceptual engine.**
9. **AI is an optional component inside the architecture, not the architecture itself.**
10. **Derived/adaptive parameters are preferred to large sets of manually tuned constants.**
11. **Every important state transition must be replayable and inspectable.**
12. **Latency requirements differ by information type.** Fast transient/beat information and slow genre/structure information must not share one update rate.
13. **The auditory detector is application-independent. Internal observers are evidence-producing modules; the detector/world model integrates their evidence into persistent objects and musical state. Applications decide what to do visually.**
14. **Do not optimize before the world model and validation are working.**

---

# 1. SYSTEM PIPELINE

Canonical logical pipeline:

```text
Audio input
    ↓
Audio clock / buffering / resampling
    ↓
Cochlea-equivalent / multi-resolution physical analysis
    ↓
Physical observations + low-level auditory evidence fields
    ↓
Candidate-region / primitive-observation generation
    ↓
Layer grouping hypotheses
    ↓
Persistent sound-object construction
    ↓
Object identity matching + tracking
    ↓
Object memory + prediction
    ↓
Object relationships / hierarchy / families
    ↓
Object-level perceptual model
    │   ├── loudness
    │   ├── frequency masking
    │   ├── temporal masking
    │   └── attention / salience
    ↓
Rhythm / beat / groove / pattern / phrase / section context
    ↓
Semantic roles and musical interpretation
    ↓
Genre / subgenre / style interpretation
    ↓
State + semantic event generation
    ↓
Application trigger stream
    ↓
VJ / lighting / visualization / DAW / analysis consumers
```

The ordering deliberately separates two different uses of auditory perception. Low-level auditory evidence fields may be computed from the physical analysis so candidate generation can use hearing-relevant structure, but they must never delete or suppress physical evidence. Object-level masking, loudness, attention, and salience are applied after persistent object hypotheses exist. Masking therefore changes perceptual availability/confidence, not physical existence, and must never prevent later object recovery.

---

# 2. ARCHITECTURAL LAYERS

## 2.1 Physical world

Describes measurable audio evidence without assigning musical meaning.

Examples:

- energy;
- frequency distribution;
- pitch candidates;
- harmonic structure;
- transient evidence;
- envelope;
- modulation;
- noise/texture;
- dynamics;
- distortion;
- spatial cues;
- embeddings when available.

## 2.2 Perceptual world

Represents which physical information is likely available/prominent to a listener.

Includes:

- loudness;
- critical-band relationships;
- frequency masking;
- temporal masking;
- attention;
- salience;
- foreground/background state;
- perceptual stability;
- expectation;
- surprise;
- uncertainty.

The perceptual world references the physical world. It does not overwrite it.

## 2.3 Semantic world

Interprets persistent objects in musical context.

Examples:

- likely kick;
- likely bass foundation;
- percussion role;
- melodic role;
- texture;
- groove foundation;
- tension source;
- pattern membership;
- phrase membership;
- section membership.

Semantic labels are probabilities/roles, not absolute truth.

## 2.4 Application world

Maps semantic/structural events to external actions.

Example:

```text
Kick event
    ↓
application rule
    ↓
visual flash / movement / parameter change
```

Never encode application-specific visual behavior into perception itself.

---

# 3. CENTRAL WORLD MODEL

The world model is the authoritative shared state. Modules do not maintain competing versions of reality.

Conceptual root:

```text
WorldState
├── clock
├── physical graph
├── perceptual graph
├── semantic graph
├── sound objects
├── relationships
├── patterns
├── phrases
├── sections
├── global musical state
├── event history
└── decision/replay metadata
```

## 3.1 Property ownership

A module may own the **calculation** of a property but not the existence of the world model itself.

Examples:

- physical analyzer calculates physical features;
- perception engine calculates perceptual state;
- tracker updates identity/lifecycle;
- grouping engine updates membership hypotheses;
- musical context engine updates rhythm/pattern/structure;
- semantic engine updates role probabilities;
- applications read the resulting state and generate their own actions.

## 3.2 Read/write rule

Modules may:

- read authorized world properties;
- propose updates;
- submit observations;
- submit relationship hypotheses;
- submit events.

They must not silently mutate unrelated properties.

All important updates should be timestamped and versioned.

---

# 4. CORE DATA MODEL

The minimum canonical entities are:

```text
WorldState
Observation
Region
Layer
SoundObject
FeatureSet
Embedding
Relationship
Prediction
Pattern
Phrase
Section
SemanticRole
Event
Confidence
Update
DecisionLog
```

## 4.1 Observation

A temporary piece of evidence tied to a time interval.

Required conceptual fields:

```text
id
start_time
end_time
source_frame
physical_features
perceptual_features (if available)
region/reference coordinates
observation_type
confidence
provenance
```

An observation may disappear without deleting the object it supported.

## 4.2 Region

A candidate localized region in a time-frequency or feature representation. Regions are evidence candidates, not yet persistent musical objects.

## 4.3 Layer

A grouping hypothesis over primitive regions/observations that appear to form one coherent sound component.

A layer may be tentative and may later merge, split, or be reassigned.

## 4.4 SoundObject

Persistent perceptual entity.

Conceptual fields:

```text
object_id
status
creation_time
last_observed_time
age
confidence
physical_signature
perceptual_state
semantic_roles
history
prediction
relationships
family_id / hierarchy links
current importance/salience
```

## 4.5 Object status

Canonical states:

```text
TENTATIVE
ACTIVE
HIDDEN
RECOVERING
ARCHIVED
```

Do not immediately delete objects when evidence disappears.

## 4.6 Relationship

A typed, weighted edge between entities.

Examples:

```text
KICK → BASS
BASS → GROOVE
PERCUSSION → MOVEMENT
LEAD → MELODY
OBJECT_A → OBJECT_B : harmonic_relation
OBJECT_A → OBJECT_B : rhythmic_relation
```

Edges contain confidence and temporal validity.

## 4.7 Confidence

Confidence is not one universal scalar. Maintain confidence by property where useful:

```text
existence_confidence
identity_confidence
grouping_confidence
role_confidence
prediction_confidence
perceptual_confidence
structural_confidence
```

Aggregate confidence may be derived for consumers.

---

# 5. AUDIO ENGINE

Responsibilities:

- input file/stream handling;
- channel handling;
- sample-rate handling;
- buffering;
- audio clock;
- frame scheduling;
- realtime safety.

The audio engine must not block on slow semantic/AI operations.

Canonical execution streams:

```text
Audio/realtime thread
Analysis workers
World-state update scheduler
Semantic/slow workers
Application/output thread
UI/debug/replay thread
```

The exact implementation may use Python initially, with critical components later replaceable by C++/Rust/native/GPU implementations.

---

# 6. PHYSICAL ANALYSIS ENGINE

The physical engine generates evidence only.

## 6.1 Required analyses

At minimum:

- multi-resolution STFT;
- magnitude/power spectrum;
- filterbank representation;
- RMS/energy;
- spectral centroid;
- spectral spread/shape;
- spectral flux;
- transient/onset evidence;
- harmonic evidence;
- pitch candidates;
- envelope;
- modulation;
- noise/tonality indicators;
- dynamics;
- distortion indicators;
- spatial/channel features when input permits.

## 6.2 Multi-scale analysis

Use different temporal resolutions for:

- fast transients;
- note/envelope behavior;
- rhythmic patterns;
- phrases/sections.

Do not force every analysis to run at the fastest audio rate.

## 6.3 Physical graph

Physical nodes represent evidence/features/regions. They should preserve provenance so later modules can determine where a belief came from.

## 6.4 Lossless vs lossy input

Lossless source material generally preserves more recoverable physical information. Lossy compression can alter high-frequency detail, transients, stereo information, and low-level components.

The architecture must not assume that lossy artifacts are meaningful musical objects. Confidence/provenance can record degraded evidence.

---

# 7. PERCEPTUAL MODEL

The perceptual layer models useful properties of human hearing without attempting biological simulation.

## 7.1 Cochlea-equivalent representation

Transform physical spectrum into a perceptually useful frequency representation, such as critical-band/bark-like or comparable auditory bands.

The exact filterbank can be replaced without changing the world-model API.

## 7.2 Masking

Masking has two primary forms:

- frequency masking;
- temporal masking.

Masking affects **perceptual confidence/availability**.

It must not:

- delete a physical object;
- erase object history;
- force an object to cease existing;
- prevent recovery when the masker disappears.

Example:

```text
bass exists physically
    ↓
kick masks bass perceptually
    ↓
bass becomes HIDDEN / low perceptual confidence
    ↓
kick decays
    ↓
bass evidence becomes available
    ↓
bass object recovers using memory/prediction
```

## 7.3 Loudness

Estimate perceptual energy rather than treating raw amplitude as equivalent to perceived importance.

## 7.4 Attention

Attention/salience can prioritize computation and output but must not alter the physical graph.

Factors may include:

- perceptual prominence;
- novelty;
- prediction error;
- musical importance;
- role;
- application relevance;
- temporal stability.

## 7.5 Dynamic computation

Highly important/uncertain objects may receive more analysis detail. Stable low-importance objects can receive less computation.

---

# 8. PRIMITIVE OBSERVATIONS AND CANDIDATE EVENTS

Primitive observers convert physical and low-level auditory evidence into hypotheses such as:

- transient;
- tonal component;
- harmonic group;
- sustained energy region;
- rhythmic onset;
- modulation event;
- spectral texture change;
- spatial movement.

These are **observations**, not final semantic labels.

A primitive observer should expose:

```text
observation type
features
interval
confidence
provenance
```

Do not build hundreds of mutually exclusive classifiers. Prefer reusable observers whose evidence can contribute to multiple hypotheses.

---

# 9. LAYER GROUPING

Layer grouping is one of the hardest parts of the system.

Goal:

```text
many primitive observations
        ↓
coherent layer hypotheses
        ↓
persistent sound objects
```

## 9.1 Grouping cues

Use multiple cues:

- temporal coherence;
- onset proximity;
- envelope relationship;
- modulation relationship;
- harmonic relationship;
- spectral relationship;
- spatial relationship;
- embedding similarity when available;
- common fate / correlated evolution;
- prediction compatibility;
- perceptual context.

## 9.2 Multi-cue fusion

Do not use a single hard threshold such as frequency equality. Maintain component scores and combine them into a grouping belief.

Conceptual:

```text
GroupingScore(A,B) = f(
    temporal,
    envelope,
    modulation,
    harmonic,
    spectral,
    spatial,
    embedding,
    prediction,
    perceptual_context
)
```

The implementation should preserve component scores for debugging.

## 9.3 Merge/split

Merging requires sustained evidence that two hypotheses represent one coherent object.

Splitting requires sustained evidence that one object contains independently evolving components.

Avoid rapid merge/split oscillation using temporal hysteresis and object history.

## 9.4 Multi-scale grouping

Grouping should operate at multiple temporal scales because a short event can belong to a larger sustained object or repeating pattern.

---

# 10. OBJECT IDENTITY AND TRACKING

Tracking is the transition from momentary evidence to persistent world state.

## 10.1 Identity principle

Object identity is not identical to shape, frequency, amplitude, or one embedding.

Identity combines:

```text
physical similarity
perceptual similarity
temporal continuity
envelope behavior
harmonic relationships
embedding similarity
prediction agreement
context
relationships
history
```

## 10.2 Matching

For each observation/layer hypothesis:

1. query plausible active/hidden objects;
2. calculate component match scores;
3. incorporate prediction;
4. rank candidate matches;
5. choose match/create/uncertain outcome;
6. update the selected object;
7. record decision provenance.

Conceptual:

```text
MatchScore = weighted evidence vector
```

Do not collapse diagnostic evidence prematurely into an opaque number.

## 10.3 Object birth

Create a new object when no existing object has sufficient support and the observation is sufficiently coherent.

Initial state:

```text
unique id
creation time
initial signature
initial confidence
status = TENTATIVE
```

## 10.4 Prediction

Objects maintain predicted future behavior. Prediction may include:

- expected time;
- frequency trajectory;
- energy trajectory;
- rhythmic recurrence;
- envelope continuation;
- relationship continuation.

Observation agreeing with prediction increases identity confidence.

## 10.5 Hidden state

Temporary absence must not imply object death.

A hidden object retains:

- identity;
- history;
- last state;
- prediction;
- relationships;
- confidence trajectory.

## 10.6 Recovery

When evidence reappears, use historical signature + prediction + context + current evidence to reconnect the object.

## 10.7 Death/archive

Archive only after prolonged absence, low confidence, weak prediction support, and no contextual evidence requiring persistence.

---

# 11. OBJECT HIERARCHY AND FAMILIES

Objects can exist at multiple levels.

Example:

```text
Kick family
├── attack/transient component
├── body component
├── click component
└── distorted/room component
```

or:

```text
Synth family
├── layer A
├── layer B
└── modulation/FX component
```

The hierarchy represents perceptual/musical organization, not necessarily physical source separation.

Important distinction:

- **same object** = identity continuity;
- **same family** = related objects/components;
- **same pattern** = recurring temporal organization;
- **same role** = semantic function.

Do not collapse these concepts.

---

# 12. MEMORY SYSTEM

Memory is hierarchical.

## 12.1 Short-term memory

Recent observations and transient context.

## 12.2 Object memory

Persistent object signature/history.

## 12.3 Track memory

Trajectory, prediction errors, birth/death/recovery history.

## 12.4 Pattern memory

Repeated rhythmic/harmonic/temporal patterns.

## 12.5 Long-term memory

Optional learned/reference knowledge, including style profiles and embedding databases.

## 12.6 Replay

Every important world-state update should be replayable. Debugging must support questions such as:

> What did the system believe at time T, and why?

Store sufficient provenance and decision logs to reconstruct the state.

---

# 13. RELATIONSHIP GRAPH

Relationships are first-class data.

Possible relationship types:

- temporal;
- harmonic;
- rhythmic;
- modulation;
- grouping;
- hierarchy;
- common-fate;
- call/response;
- foreground/background;
- support/foundation;
- contrast;
- pattern membership;
- phrase membership;
- section membership.

Relationships have:

```text
source
relation_type
target
strength
confidence
start_time
end_time
provenance
```

The graph allows higher-level meaning without requiring a classifier for every possible musical event.

---

# 14. RHYTHM AND GROOVE

Groove is not BPM.

Groove is the relationship between events through time.

Represent:

- beat;
- tempo;
- subdivisions;
- timing grid;
- microtiming;
- swing;
- syncopation;
- event density;
- repetition;
- accent structure;
- object interaction;
- call/response;
- rhythmic stability.

## 14.1 Groove vector

Maintain a structured state rather than one label. Components may include timing deviation, swing, syncopation, density, regularity, accent distribution, and interaction between rhythmic objects.

## 14.2 Pattern identity

Patterns should tolerate variation. A repeated musical pattern is not required to have sample-identical events.

## 14.3 Phrase/section model

Use accumulated object/pattern/energy/structural changes to infer phrases and sections.

Fast events should not wait for section analysis.

---

# 15. MUSICAL SEMANTICS

Semantic interpretation assigns probabilistic roles to objects.

Example:

```text
Object #42
kick_role = 0.91
bass_role = 0.12
percussion_role = 0.33
```

Roles are contextual.

A sound's meaning may change depending on:

- rhythm;
- register;
- surrounding objects;
- arrangement;
- current section;
- genre/style hypothesis.

Do not turn semantic roles into immutable classifications.

---

# 16. GENRE / SUBGENRE / STYLE MODEL

Genre knowledge is an interpretation layer.

Do not build:

```text
one classifier per genre
```

Instead represent style as a combination of observable behavior:

```text
object behavior
rhythm
microtiming
groove
arrangement
sound-design tendencies
production characteristics
structure
semantic role distributions
```

Genre is probabilistic and hierarchical.

Example:

```text
Electronic
└── Techno
    ├── Hard Techno
    ├── Hardgroove
    ├── Industrial Techno
    └── Schranz
```

Subgenres are regions/clusters in a multidimensional style space, not independent universes.

A new/unrecognized subgenre should still be interpretable through its behavior vector.

Genre knowledge may influence semantic interpretation and expected patterns, but must not rewrite universal physical/perceptual facts.

---

# 17. AI / EMBEDDINGS

AI is used selectively where fixed algorithms become weak.

Good uses:

- object embeddings;
- similarity;
- complex pattern discovery;
- prediction;
- style-space modeling;
- optional learned adapters.

Poor default uses:

- replacing the entire world model;
- deleting explainability;
- one end-to-end model for everything;
- one model per genre;
- unnecessary source separation.

## 17.1 Embedding rule

An embedding is evidence for identity/similarity. It is not the identity itself.

A future learned model may map physically/perceptually related observations to nearby representations even when exact frequency, amplitude, or waveform shape changes.

## 17.2 Training

The core deterministic architecture does not require training.

Optional learning can later be trained using:

- self-supervised audio segments;
- positive/negative temporal object pairs;
- augmentation invariance;
- human/object annotations where available;
- style/reference datasets.

Training is a replaceable implementation detail behind stable interfaces.

---

# 18. COMPUTATION MODEL

The system is multi-speed.

## Fast loop

- audio buffering;
- STFT/physical updates;
- transient observations;
- critical timing events.

## Medium loop

- object tracking;
- grouping;
- perceptual state;
- object relationships.

## Slow loop

- groove stabilization;
- patterns;
- phrases;
- sections;
- genre/style interpretation.

Prediction should reduce computation by allowing stable objects to be extrapolated instead of fully reanalyzed every frame.

## CPU

Preferred for:

- streaming;
- scheduling;
- state management;
- graph/world updates;
- lightweight DSP.

## GPU

Preferred when beneficial for:

- embeddings;
- neural models;
- large batch analysis.

## Adaptive compute

Allocate more computation to objects that are:

- important;
- uncertain;
- rapidly changing;
- newly created;
- involved in a conflict/merge/split;
- relevant to an active application.

Stable low-value objects can be updated less frequently.

---

# 19. REALTIME AND LATENCY

Different outputs have different timing requirements.

Approximate target classes:

```text
transient / beat reaction     milliseconds–tens of ms
object tracking               tens of ms
rhythmic state                tens–hundreds of ms
phrase/section                hundreds of ms–seconds
genre/style                   seconds / long-term
```

Do not require slow interpretation to block fast events.

The application receives timestamps and should compensate for known analysis latency when appropriate.

Audio processing must remain non-blocking.

---

# 20. EVENT SYSTEM

Not everything is an event.

Continuous state belongs in the world model:

```text
energy
confidence
groove
tension
current objects
object importance
```

Events represent meaningful transitions:

```text
object_created
object_hidden
object_recovered
object_split
object_merged
beat
accent
pattern_started
pattern_changed
groove_changed
build_started
drop_detected
section_changed
role_changed
style_changed
```

## Event priority

Priority can combine:

- musical importance;
- confidence;
- application relevance;
- novelty;
- perceptual salience.

Do not flood applications with thousands of low-level changes when one semantic transition is sufficient.

---

# 21. APPLICATION / VJ API

The perception engine does not create visuals.

It exposes:

### State API

Current world/object/semantic state.

### Event stream

Timestamped semantic events.

### Raw/debug API

Optional physical/perceptual graphs for advanced users.

Possible transports:

- WebSocket;
- OSC;
- MIDI;
- custom API;
- JSON/file export;
- graph snapshots;
- analysis timeline.

One perception engine can feed multiple applications without changing perception:

```text
World Model
├── VJ
├── lighting
├── visualization
├── analysis
└── DAW tools
```

Applications may define mappings such as:

```text
if drop_confidence > 0.8:
    trigger_scene_change
```

The universal detector/world model remains unchanged.

---

# 22. CONFIGURATION POLICY

The project should not expose thousands of arbitrary knobs.

Every parameter must belong to one category:

```text
FIXED_CONSTANT
DERIVED_PARAMETER
RUNTIME_ADAPTIVE
ARCHITECTURAL_POLICY
PROJECT_PREFERENCE
OPTIONAL_LEARNED_PARAMETER
```

Prefer:

```text
threshold = estimated_background + k * variability
```

over:

```text
magic_threshold = 0.427
```

Parameters should be centralized, versioned, and validated.

Most engineering constants should receive robust defaults automatically. Only genuine architectural or artistic policy decisions should require user input.

---

# 23. FAILURE HANDLING

Important failure categories:

- false observation;
- missed observation;
- false object creation;
- object split;
- object merge;
- identity switch;
- object disappearance;
- failed masking recovery;
- grouping instability;
- semantic instability;
- timing error;
- compute overload;
- model failure.

Every failure should be logged with:

```text
time
module
inputs/provenance
state before
decision
state after
confidence
```

The system should degrade gracefully rather than crash or erase world state.

---

# 24. VALIDATION

Do not evaluate only classification accuracy.

Measure:

## Observation

- event timing;
- feature accuracy;
- false/missed observations.

## Object

- object continuity;
- identity switches;
- false object creation;
- object death/recovery.

## Grouping

- over-merging;
- over-separation;
- layer stability.

## Perception

- masking behavior;
- perceptual priority;
- hidden-object recovery.

## Musical context

- beat timing;
- tempo stability;
- groove stability;
- pattern continuity;
- phrase/section accuracy.

## Application

- event latency;
- trigger stability;
- event flooding;
- semantic usefulness.

## Test pyramid

Progress from:

```text
simple synthetic sounds
→ single instruments/sounds
→ layered sounds
→ dense mixes
→ varied production styles
→ difficult subgenres
→ adversarial/hard cases
→ realtime streams
```

Hard cases should explicitly include dense layered material, similar embeddings, masking, distortion, rapidly changing layers, and ambiguous family boundaries.

---

# 25. DEVELOPMENT ORDER

Build in this order:

```text
0. Project foundation
1. WorldState + data schemas
2. Audio engine / clock / buffering
3. Physical analysis
4. Primitive observations
5. Debug/replay infrastructure
6. Object tracking
7. Object memory/prediction
8. Layer grouping
9. Perceptual model/masking
10. Relationships/hierarchy
11. Rhythm/tempo/groove
12. Patterns/phrases/sections
13. Semantic roles
14. Genre/subgenre/style model
15. Event system
16. VJ/application API
17. Optional embeddings/AI
18. Performance optimization
```

Do not start with genre detection, complex AI, or VJ output.

The first meaningful milestone is:

```text
Audio
→ observations
→ persistent objects
→ replayable world state
```

---

# 26. MVP

The MVP proves only the foundation:

```text
Audio input
↓
RMS/spectral features
↓
primitive transient observations
↓
SoundObject creation
↓
basic tracking
↓
object history
↓
structured event output
```

The MVP must be intentionally incomplete.

It does not need:

- full source separation;
- genre recognition;
- sophisticated embeddings;
- complete psychoacoustic modeling;
- full VJ mapping.

Its purpose is to prove that evidence can become persistent world state.

---

# 27. REPOSITORY ARCHITECTURE

Recommended initial structure:

```text
auditory_world_model/
├── README.md
├── PROJECT_SPEC.md
├── requirements.txt
├── config.yaml
├── main.py
│
├── core/
│   ├── world.py
│   ├── observations.py
│   ├── objects.py
│   ├── features.py
│   ├── relationships.py
│   ├── predictions.py
│   ├── confidence.py
│   ├── memory.py
│   ├── events.py
│   └── decisions.py
│
├── audio/
│   ├── input.py
│   ├── buffer.py
│   ├── clock.py
│   └── streaming.py
│
├── physical/
│   ├── stft.py
│   ├── filterbank.py
│   ├── spectrum.py
│   ├── energy.py
│   ├── transient.py
│   ├── harmonic.py
│   ├── pitch.py
│   ├── envelope.py
│   ├── modulation.py
│   ├── spatial.py
│   └── embeddings.py
│
├── perception/
│   ├── auditory_bands.py
│   ├── loudness.py
│   ├── masking.py
│   ├── attention.py
│   └── salience.py
│
├── grouping/
│   ├── regions.py
│   ├── cues.py
│   ├── scoring.py
│   ├── merge.py
│   └── split.py
│
├── tracking/
│   ├── tracker.py
│   ├── matching.py
│   ├── prediction.py
│   └── lifecycle.py
│
├── semantics/
│   ├── roles.py
│   ├── rhythm.py
│   ├── groove.py
│   ├── patterns.py
│   ├── phrases.py
│   ├── structure.py
│   └── style.py
│
├── output/
│   ├── events.py
│   ├── websocket.py
│   ├── osc.py
│   ├── midi.py
│   └── visualization.py
│
├── models/
│   ├── embeddings/
│   ├── classifiers/
│   └── learned_parameters/
│
├── replay/
├── debug/
└── tests/
```

Modules must have stable interfaces so implementation language can later change for critical realtime components.

---

# 28. IMPLEMENTATION STACK

Initial implementation:

- Python;
- NumPy;
- SciPy;
- soundfile;
- sounddevice;
- librosa or equivalent DSP utilities;
- asyncio/websockets for application transport where appropriate;
- python-osc for OSC;
- PyTorch/ONNX only when learned modules are introduced.

The architecture must not become dependent on one library's internal representation.

---

# 29. DEBUGGING / REPLAY

Every major module should expose or record:

```text
input timestamp
input identifiers
output
confidence
latency
decision/provenance
```

The debugger must support inspection at arbitrary timestamps and visualization of:

- physical graph;
- perceptual graph;
- object graph;
- semantic graph;
- confidence changes;
- merge/split decisions;
- masking/recovery;
- event generation.

This is a core feature, not optional polish.

---

# 30. CODE GENERATION RULES

A future code-generating LLM must obey these rules:

1. Treat this file as canonical.
2. Do not redesign locked architecture unless explicitly requested.
3. Do not replace the world model with a monolithic classifier.
4. Do not delete physical evidence because it is masked.
5. Do not equate observations with objects.
6. Do not equate embeddings with identity.
7. Do not hardcode genre-specific assumptions into universal perception modules.
8. Preserve timestamps and provenance throughout the pipeline.
9. Keep physical, perceptual, semantic, and application layers separable.
10. Keep interfaces replaceable.
11. Implement replay/debugging alongside core modules.
12. Add tests before optimizing.
13. Prefer derived/adaptive parameters over unexplained magic constants.
14. Never introduce a configuration knob without documenting why it exists.
15. Never make an application directly mutate the world model.
16. Preserve uncertainty rather than forcing a false label.
17. Prefer stable object continuity over frame-by-frame classification.
18. Keep fast and slow computations asynchronous.
19. Optimize only after validation identifies a bottleneck.
20. When uncertain, preserve information and defer irreversible decisions.

---

# 31. CANONICAL OBJECT UPDATE ALGORITHM

Conceptual pseudocode:

```text
for each audio time step:

    ingest audio
    update clock

    physical_evidence = physical_engine.analyze(audio)

    observations = observers.generate(physical_evidence)

    perceptual_state = perception.evaluate(
        physical_evidence,
        current_world
    )

    candidates = grouping.generate_candidates(
        observations,
        perceptual_state,
        current_world
    )

    grouping_hypotheses = grouping.score(candidates)

    object_updates = tracker.match(
        grouping_hypotheses,
        current_objects,
        predictions
    )

    memory.update(object_updates)
    predictions.update(memory)

    relationships.update(
        objects,
        memory,
        perceptual_state
    )

    musical_context.update(
        objects,
        relationships,
        memory
    )

    semantics.update(
        objects,
        relationships,
        musical_context,
        style_model
    )

    events = event_engine.generate(
        world_state_changes,
        semantic_state,
        confidence
    )

    publish(events, state)

    record_replay_data()
```

The actual implementation may schedule these asynchronously, but the logical dependency order remains stable.

---

# 32. EXAMPLE: MASKED BASS RECOVERY

A canonical behavior test:

```text
Bass evidence appears
→ Bass object created
→ Kick becomes dominant
→ masking increases
→ Bass physical evidence remains
→ Bass perceptual confidence decreases
→ Bass status becomes HIDDEN
→ Kick decays
→ masking decreases
→ Bass evidence becomes available
→ prediction/history match Bass object
→ Bass returns to ACTIVE
```

The system must not create a brand-new bass object merely because the bass was temporarily inaudible.

---

# 33. EXAMPLE: KICK CONSTRUCTION

A kick may produce multiple primitive components:

```text
sub
body
click
harmonic residue
distortion
room/reverb
```

Grouping evaluates their temporal, envelope, harmonic, modulation, spatial, embedding, and predictive relationships.

If evidence supports a common perceptual identity, these components form one kick object/family rather than independent final objects.

---

# 34. EXAMPLE: DENSE MIX

In a dense section:

- multiple objects may overlap in frequency;
- masking may hide weaker objects;
- embeddings may be similar;
- object boundaries may be ambiguous;
- grouping may have several plausible solutions.

The correct behavior is not forced certainty. Maintain competing hypotheses/confidence, preserve evidence, use temporal continuity and prediction, and resolve ambiguity as later evidence arrives.

---

# 35. WHAT NOT TO BUILD

Avoid:

- 1000 independent genre/instrument classifiers;
- one classifier per subgenre;
- destructive masking;
- early source separation as a prerequisite;
- end-to-end neural replacement of the world model;
- arbitrary hardcoded visual mappings inside perception;
- premature GPU optimization;
- excessive configuration knobs;
- frame-by-frame identity without memory;
- treating all output as events;
- assuming one correct semantic interpretation at every instant.

Full source separation remains optional research capability, not a prerequisite for human-like musical-world understanding.

---

# 36. PRIORITY MAP

## Level A — essential

- WorldState;
- audio pipeline;
- physical features;
- observations;
- object tracking;
- history/replay.

## Level B — major capability

- layer grouping;
- perceptual model;
- rhythm/groove;
- event system.

## Level C — advanced

- embeddings;
- prediction models;
- musical structure;
- style/subgenre model.

## Level D — research

- full source separation;
- end-to-end foundation models;
- realtime learning.

Highest-value architectural improvements are generally:

```text
memory
+ relationships
+ grouping
+ perceptual weighting
+ prediction
```

not simply more labels.

---

# 37. VALIDATION SUCCESS CRITERIA

The project is successful when it can maintain a useful, stable world model through realistic audio rather than merely naming sounds.

Minimum meaningful milestones:

1. Physical observations are temporally correct.
2. Sound objects persist across frames.
3. Identity survives short gaps and masking.
4. Layers can be grouped without excessive merge/split oscillation.
5. Relationships remain stable enough to support musical context.
6. Beat/tempo/groove state is useful and temporally stable.
7. Semantic roles remain probabilistic and contextual.
8. Genre/subgenre interpretation emerges from object/rhythm/structure behavior.
9. Applications receive compact, meaningful events instead of raw analysis floods.
10. The entire state can be replayed and inspected.

---

# 38. FINAL ARCHITECTURAL STATEMENT

The project is a **real-time auditory object perception system for interactive music visualization**.

Its central abstraction is not the waveform, stem, instrument label, or genre label.

It is the persistent **sound object and its relationships inside an evolving auditory world**.

The canonical transformation is:

```text
Audio
→ physical evidence
→ perceptual evidence
→ primitive observations
→ grouped sound objects
→ persistent identity
→ memory/prediction
→ relationships/hierarchy
→ rhythm/groove/structure
→ semantic meaning
→ style interpretation
→ meaningful event/state stream
→ application-specific visual behavior
```

The detector/observer answers:

> **What is happening musically?**

The application answers:

> **What should happen visually?**

The architecture must preserve that separation.

---

# 39. LOCKED DECISIONS

Unless explicitly changed in a future specification revision:

- The world model is the central source of truth.
- Physical evidence is never destroyed by masking.
- Perceptual masking changes availability/confidence, not existence.
- Observations are temporary evidence.
- Sound objects are persistent entities.
- Identity is multi-cue and temporal.
- Grouping is multi-cue and probabilistic.
- Memory and prediction are core, not optional decorations.
- Relationships are first-class state.
- Genre/subgenre is an interpretation layer.
- AI is modular and optional.
- Applications cannot mutate perception directly.
- VJ output is downstream of semantic/world state.
- Realtime and slow interpretation are separated.
- Replay/debugging is part of the architecture.
- Full source separation is not required for the core objective.

---

# 40. REVISION POLICY

Future changes must be expressed as explicit architectural changes.

For each proposed change record:

```text
CHANGE_ID
DATE
OLD_RULE
NEW_RULE
REASON
AFFECTED_MODULES
MIGRATION_REQUIREMENTS
```

Do not silently reinterpret earlier locked decisions.

If a future model cannot determine whether a statement is authoritative, it must preserve the existing architecture rather than invent a new interpretation.

---

# END OF CANONICAL SPECIFICATION
