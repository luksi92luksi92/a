#!/usr/bin/env python3
"""Apply the small fixes requested for the AWM grouping/diagnostic plots.

Run from the repository directory:
    python apply_diagnostic_patches.py

The patch is idempotent and refuses to silently overwrite a mismatched target.
It makes two changes:

1. Fixes the generator-expression parentheses in
   auditory_world_model_grouping_fix.py.
2. Makes the mel spectrogram a subdued background (alpha=0.50) and gives
   object/layer evidence distinct high-contrast plotting colors in
   plot_awmm_layers.py.
"""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
GROUPING = ROOT / "auditory_world_model_grouping_fix.py"
PLOTS = ROOT / "plot_awmm_layers.py"


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Expected target for {label!r} was not found; refusing to patch.")
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one target for {label!r}; found {text.count(old)}.")
    return text.replace(old, new)


def patch_grouping() -> None:
    text = GROUPING.read_text(encoding="utf-8")
    old = 'freqs = self._finite(h.frequency_hz for h in active if h.frequency_hz > 0, 1.0)'
    new = 'freqs = self._finite((h.frequency_hz for h in active if h.frequency_hz > 0), 1.0)'
    patched = replace_exact(text, old, new, "grouping generator expression")
    if patched != text:
        GROUPING.write_text(patched, encoding="utf-8")
        print(f"Patched: {GROUPING.name}")
    else:
        print(f"Already fixed: {GROUPING.name}")


def patch_plots() -> None:
    text = PLOTS.read_text(encoding="utf-8")

    # Add a reusable object/layer palette without changing the mel colormap.
    anchor = 'import numpy as np\nfrom scipy.signal import butter, lfilter, resample_poly\n'
    palette_block = '''import numpy as np\nfrom scipy.signal import butter, lfilter, resample_poly\n\n# Diagnostic overlay palette: intentionally distinct from the mel background.\nOBJECT_COLORS = [\n    "#00E5FF", "#FFEA00", "#FF4D6D", "#7CFF6B", "#B388FF",\n    "#FF9F1C", "#00F5D4", "#FFFFFF", "#F15BB5", "#4CC9F0",\n]\n\n'''
    if 'OBJECT_COLORS = [' not in text:
        text = replace_exact(text, anchor, palette_block, "object palette insertion")

    # Make both mel-background displays subdued. We only change the two calls
    # that actually render the mel spectrogram itself, not the evidence overlay.
    text = text.replace(
        'ax1.imshow(mel_db, origin="lower", aspect="auto", extent=extent)\n',
        'ax1.imshow(mel_db, origin="lower", aspect="auto", extent=extent, alpha=0.50)\n',
    )
    text = text.replace(
        'axes[1].imshow(mel_db, origin="lower", aspect="auto", extent=extent)\n',
        'axes[1].imshow(mel_db, origin="lower", aspect="auto", extent=extent, alpha=0.50)\n',
    )

    # Main combined layer curves: cycle through high-contrast colors and keep
    # the mel spectrogram subdued underneath.
    old = '        ax1.plot(times, y + act * (3.0 / max(1, len(ordered))), linewidth=2.0,\n                 label=f"L{row+1} {lid[:12]} ({len(members)} objs)")\n'
    new = '        ax1.plot(times, y + act * (3.0 / max(1, len(ordered))), linewidth=2.4,\n                 color=OBJECT_COLORS[row % len(OBJECT_COLORS)],\n                 label=f"L{row+1} {lid[:12]} ({len(members)} objs)")\n'
    text = replace_exact(text, old, new, "combined layer color")

    # Layer/object timeline below the spectrogram: layer lines are vivid/thick;
    # object lines are thinner but still use a distinct palette rather than the
    # mel colormap.
    old = '        ax2.plot(times, y + 0.35 * la, linewidth=2.2)\n'
    new = '        ax2.plot(times, y + 0.35 * la, linewidth=2.8,\n                 color=OBJECT_COLORS[i % len(OBJECT_COLORS)])\n'
    text = replace_exact(text, old, new, "timeline layer color")

    # Object traces inside each layer. Use deterministic per-object palette index.
    old = '            ax2.plot(times, y + 0.25 * oa, linewidth=0.8, alpha=0.75)\n'
    new = '            ax2.plot(times, y + 0.25 * oa, linewidth=1.15, alpha=0.95,\n                     color=OBJECT_COLORS[(i + len(member := members[:members.index(obj)])) % len(OBJECT_COLORS)] if obj in members else OBJECT_COLORS[i % len(OBJECT_COLORS)])\n'
    # Avoid the awkward expression above: replace it immediately with a stable
    # local index implementation if the source contains the original line.
    if old in text:
        replacement = '            member_idx = members.index(obj)\n            ax2.plot(times, y + 0.25 * oa, linewidth=1.15, alpha=0.95,\n                     color=OBJECT_COLORS[(i + member_idx + 1) % len(OBJECT_COLORS)])\n'
        text = text.replace(old, replacement)

    old = '        axr.plot(times, 3 + j + oa / max(np.max(oa), 1e-9), linewidth=0.9, alpha=0.75)\n'
    new = '        axr.plot(times, 3 + j + oa / max(np.max(oa), 1e-9), linewidth=1.15, alpha=0.95,\n                 color=OBJECT_COLORS[j % len(OBJECT_COLORS)])\n'
    text = replace_exact(text, old, new, "remaining-object color")

    PLOTS.write_text(text, encoding="utf-8")
    print(f"Patched: {PLOTS.name}")


def main() -> None:
    if not GROUPING.exists():
        raise FileNotFoundError(GROUPING)
    if not PLOTS.exists():
        raise FileNotFoundError(PLOTS)
    patch_grouping()
    patch_plots()
    print("\nDiagnostic patches applied successfully.")
    print("Run: python plot_awmm_layers.py")


if __name__ == "__main__":
    main()
