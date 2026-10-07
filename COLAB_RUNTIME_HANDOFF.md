# Colab Runtime Handoff — Stems-First

## Incident: 2026-10-07

A Colab runtime was broken by running:

    %cd /content/a
    !pip uninstall -y basic-pitch
    !pip install -U -r requirements.txt

The `basic-pitch` uninstall was harmless because it was not installed.

The failure came from the previous `requirements.txt`, which contained unpinned core runtime packages including `IPython`, `torch`, `numpy`, `scipy`, `matplotlib`, `librosa`, and `scikit-learn`. The `-U` install upgraded the Colab-managed environment.

The observed conflicting versions were:

- Colab required `IPython==7.34.0`, but pip installed `IPython==9.17.1`.
- Colab's `moviepy` required `decorator<5.0`, while `librosa==1.0.0` pulled `decorator==5.3.1`.
- PyTorch/CUDA packages were also replaced, including torch 2.11.0 -> 2.14.1 and multiple CUDA 13 packages.

The runtime then required a restart because core packages had already been imported, and the restart was unsuccessful.

## Repository fix

Branch: `edm-stems-first`

`requirements.txt` is now pinned to the known-good runtime versions from immediately before the failed upgrade.

Important changes:

1. `IPython` was removed entirely. The project does not import it directly.
2. `librosa` is pinned to `0.11.0` instead of allowing `1.0.0`, avoiding the Colab/moviepy decorator conflict.
3. Runtime-sensitive packages are pinned, including NumPy, SciPy, Matplotlib, PyTorch, TorchAudio, TorchVision, Transformers, scikit-learn, and the installed audio/ML stack.
4. `mt3-infer` is pinned to the exact repository commit that was resolved during the successful build.
5. The file no longer permits an ordinary `pip install -U -r requirements.txt` to silently replace the known-good runtime versions.

## Correct Colab recovery procedure

Because the damaged runtime may already have incompatible packages installed, start with:

1. Runtime -> Disconnect and delete runtime
2. Start a fresh Colab runtime.
3. Clone/checkout branch `edm-stems-first`.
4. Run:

    %cd /content/a
    !pip uninstall -y basic-pitch
    !pip install -U -r requirements.txt

5. Because the pins match the known-good baseline, the install should leave Colab's IPython/Jupyter runtime at its expected version instead of upgrading it.

Do not manually upgrade `IPython`, Jupyter, `decorator`, or the CUDA/PyTorch stack after this install unless the dependency set is deliberately revalidated.

## Current continuation state

The project being worked on is the stems-first EDM reverse-DAW pipeline in `edm-stems-first`.

The repository is:

`luksi92luksi92/a`

The branch name is **`edm-stems-first`**, not `stemsfirst`.

The installation failure was an environment/dependency issue, not evidence that the stems-first pipeline itself is broken.

This note is the handoff record for future conversations.
