# Stems-First EDM Reverse-DAW

The active development branch for the stems-first pipeline is `edm-stems-first`.

## Colab setup

Use a fresh Colab runtime.

First install Git LFS because the public YourMT3+ checkpoint is downloaded through a Git LFS repository:

```python
!apt-get -qq update
!apt-get -qq install -y git-lfs
!git lfs install
!git lfs version
```

Then install the pinned project dependencies:

```python
%cd /content/a
!pip uninstall -y basic-pitch
!pip install -U -r requirements.txt
```

The pipeline uses YourMT3+ as the primary polyphonic detector. MuScriptor is an optional secondary backend because its Hugging Face model is gated. The default pipeline does not require a Hugging Face account.

To enable MuScriptor later, accept the model license and authenticate, then run:

```python
!pip install -U huggingface_hub
!hf auth login
import os
os.environ["EDM_MT_USE_MUSCRIPTOR"] = "1"
```

Do not add `IPython` to the project requirements. Colab manages its Jupyter/IPython runtime separately.

See [COLAB_RUNTIME_HANDOFF.md](COLAB_RUNTIME_HANDOFF.md) for the 2026-10-07 runtime incident, dependency fix, transcription backend failure, and continuation state.
