# Stems-First EDM Reverse-DAW

The active development branch for the stems-first pipeline is `edm-stems-first`.

## Colab setup

Use a fresh Colab runtime. The dependency file is pinned specifically to the known-good runtime baseline:

```python
%cd /content/a
!pip uninstall -y basic-pitch
!pip install -U -r requirements.txt
```

Do not add `IPython` to the project requirements. Colab manages its Jupyter/IPython runtime separately.

See [COLAB_RUNTIME_HANDOFF.md](COLAB_RUNTIME_HANDOFF.md) for the 2026-10-07 runtime incident, recovery procedure, and continuation state.
