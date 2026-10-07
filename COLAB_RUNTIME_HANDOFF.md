# Colab Runtime Handoff — Stems-First

Repository: `luksi92luksi92/a`
Branch: `edm-stems-first`

## 2026-10-07 Colab issues

### Runtime dependency incident
The previous requirements file let pip upgrade Colab-managed packages. This caused:
- `google-colab` / `IPython` incompatibility (`IPython 7.34.0` was replaced by `9.17.1`)
- `moviepy` / `decorator` incompatibility
- replacement of the existing PyTorch/CUDA stack

Fix applied:
- `requirements.txt` now pins the known-good dependency versions
- `IPython` removed from project dependencies
- `librosa` pinned to `0.11.0`
- MT3-Infer pinned to the exact Git commit used during the successful build

A damaged runtime must be deleted and recreated before using the corrected requirements.

### Transcription backend incident
The pipeline then reached the `other`-stem transcription stage and failed for two separate reasons:

1. **YourMT3+**: the model downloader uses Git LFS, but the Colab runtime did not have the `git-lfs` command.
2. **MuScriptor**: its model weights are gated on Hugging Face and require account authorization.

Fix applied:
- YourMT3+ is the primary polyphonic backend.
- MuScriptor is now optional and opt-in with `EDM_MT_USE_MUSCRIPTOR=1`.
- A single unavailable optional backend no longer aborts the run when another backend succeeds.
- `EDM_MT_REQUIRE_ALL=1` restores strict all-backend behavior.
- Basic Pitch remains intentionally unused.

## Fresh Colab setup requirement

Before running the pipeline, install Git LFS in the fresh runtime:

    !apt-get -qq update
    !apt-get -qq install -y git-lfs
    !git lfs install
    !git lfs version

Then install the pinned requirements from the `edm-stems-first` branch.

The default pipeline uses YourMT3+ and does not require MuScriptor access.

## Current state

The repository-side fixes are complete for the two failures above. The next execution should be performed from a fresh Colab runtime with Git LFS installed.

The previous failure path was:

`edm_reverse_daw.py`
-> `edm_reverse_daw_stems.py`
-> `edm_novelty_structure.py`
-> `edm_melody.py`
-> `edm_multitrack_transcription.py`

The failure occurred during model checkpoint acquisition, before the actual stems-first analysis completed.

This file is the continuation handoff for future conversations.


## Melody plot export fix — 2026-10-07

The melody/note detector was producing data, but the plotting code only embedded the melody row inside each `novelty_structure_<stem>.png`. It did not write a separate melody image.

Fix applied:
- `edm_novelty_structure.py` now writes a dedicated `melody_<stem>.png` next to the novelty plot.
- The combined novelty JSON records `melody_plot_path` and `melody_plot_written` for each usable stem.
- For the `other` stem, the dedicated plot includes raw YourMT3+/MuScriptor note evidence plus lead/counter/arpeggio/chord layers.
- For monophonic roles such as bass/vocals, it plots the detected MIDI contour.

After pulling the latest `edm-stems-first` branch, rerun the pipeline. Existing plots are not retroactively regenerated; the new files are created on the next successful run.
