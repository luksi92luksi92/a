#!/usr/bin/env python3
"""Apply the AWM grouping syntax and ALL diagnostic-plot visibility fixes.

Run from the repository directory:
    python apply_diagnostic_patches.py

The patch is idempotent and refuses to silently overwrite a mismatched target.
It makes these changes:

1. Fixes the generator-expression parentheses in
   auditory_world_model_grouping_fix.py.
2. Makes every mel-spectrogram background panel 50% opacity.
3. Gives every layer-evidence overlay a distinct high-contrast colormap and
   stronger opacity so it remains visibly separate from the mel background.
4. Uses a distinct high-contrast palette for layer/object time-series traces.
"""

from __future__ import annotations

from pathlib import Path

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

    # Add a reusable overlay palette without changing the mel background map.
    anchor = 'import numpy as np\nfrom scipy.signal import butter, lfilter, resample_poly\n'
    palette_block = '''import numpy as np\nfrom scipy.signal import butter, lfilter, resample_poly\n\n# Diagnostic overlay palette: deliberately distinct from the mel background.\nOBJECT_COLORS = [\n    "#00E5FF", "#FFEA00", "#FF4D6D", "#7CFF6B", "#B388FF",\n    "#FF9F1C", "#00F5D4", "#FFFFFF", "#F15BB5", "#4CC9F0",\n]\n\n'''
    if 'OBJECT_COLORS = [' not in text:
        text = replace_exact(text, anchor, palette_block, "object palette insertion")

    # Every mel background occurrence, including per-layer panels, is subdued.
    # Apply to any remaining imshow(mel_db...) call regardless of variable name.
    lines = text.splitlines(keepends=True)
    changed = 0
    out = []
    for line in lines:
        if 'imshow(mel_db,' in line and 'alpha=0.50' not in line:
            line = line.rstrip('\n') + ', alpha=0.50)\n'
            # The above would duplicate ')' only when the original line already
            # ended with ')'. Fix it robustly below.
            line = line.replace('extent=extent), alpha=0.50)', 'extent=extent, alpha=0.50)')
            changed += 1
        out.append(line)
    text = ''.join(out)

    # The layer-specific overlay is the model evidence. Use a separate magma-like
    # map with strong alpha. It is intentionally not the same map as the mel base.
    old = '        ax.imshow(overlay, origin="lower", aspect="auto", extent=extent, alpha=0.65)\n'
    new = '        ax.imshow(overlay, origin="lower", aspect="auto", extent=extent,\n                   cmap="turbo", alpha=0.88, vmin=0.0, vmax=1.0)\n'
    if old in text:
        text = text.replace(old, new)
    elif new not in text:
        raise RuntimeError("Layer evidence overlay call not found; refusing to patch.")

    # Combined overview layer curves.
    old = '        ax1.plot(times, y + act * (3.0 / max(1, len(ordered))), linewidth=2.0,\n                 label=f"L{row+1} {lid[:12]} ({len(members)} objs)")\n'
    new = '        ax1.plot(times, y + act * (3.0 / max(1, len(ordered))), linewidth=2.4,\n                 color=OBJECT_COLORS[row % len(OBJECT_COLORS)],\n                 label=f"L{row+1} {lid[:12]} ({len(members)} objs)")\n'
    if old in text:
        text = text.replace(old, new)
    elif new not in text:
        raise RuntimeError("Combined layer curve target not found; refusing to patch.")

    # Layer/object timeline: vivid layer lines.
    old = '        ax2.plot(times, y + 0.35 * la, linewidth=2.2)\n'
    new = '        ax2.plot(times, y + 0.35 * la, linewidth=2.8,\n                 color=OBJECT_COLORS[i % len(OBJECT_COLORS)])\n'
    if old in text:
        text = text.replace(old, new)
    elif new not in text:
        raise RuntimeError("Timeline layer curve target not found; refusing to patch.")

    # Member-object traces: one distinct color per object within its layer.
    old = '            ax2.plot(times, y + 0.25 * oa, linewidth=0.8, alpha=0.75)\n'
    if old in text:
        replacement = '            member_idx = members.index(obj)\n            ax2.plot(times, y + 0.25 * oa, linewidth=1.15, alpha=0.95,\n                     color=OBJECT_COLORS[(i + member_idx + 1) % len(OBJECT_COLORS)])\n'
        text = text.replace(old, replacement)

    # Remaining-object traces.
    old = '        axr.plot(times, 3 + j + oa / max(np.max(oa), 1e-9), linewidth=0.9, alpha=0.75)\n'
    new = '        axr.plot(times, 3 + j + oa / max(np.max(oa), 1e-9), linewidth=1.15, alpha=0.95,\n                 color=OBJECT_COLORS[j % len(OBJECT_COLORS)])\n'
    if old in text:
        text = text.replace(old, new)
    elif new not in text:
        raise RuntimeError("Remaining-object curve target not found; refusing to patch.")

    PLOTS.write_text(text, encoding="utf-8")
    print(f"Patched: {PLOTS.name} ({changed} mel background panel(s) subdued)")


def main() -> None:
    if not GROUPING.exists():
        raise FileNotFoundError(GROUPING)
    if not PLOTS.exists():
        raise FileNotFoundError(PLOTS)
    patch_grouping()
    patch_plots()
    print("\nAll diagnostic patches applied successfully.")
    print("Run: python plot_awmm_layers.py")


if __name__ == "__main__":
    main()
