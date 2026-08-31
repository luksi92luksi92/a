#!/usr/bin/env python3
"""Comprehensive non-destructive diagnostic plots for the Auditory World Model.

This script runs the current pretrained-timbre pipeline and produces TWO figures:

1) ``awm_diagnostics_combined.png`` — combined mixture waveform + log-frequency
   spectrogram with layer/object evidence overlaid on the SAME time axis.
2) ``awm_diagnostics_stack.png`` — separate full-size panels for the mixture,
   every confirmed layer, remaining/unassigned evidence, and object timelines.

There is deliberately NO source waveform reconstruction here. A layer panel is
an evidence visualization: it projects tracked object histories onto the input
spectrogram so we can see what acoustic regions the model assigned to that layer.

For the no-file synthetic test, the four known generated source signals are
recreated exactly and each inferred layer is correlated with them in time. A
mapping is only printed when correlation + temporal overlap are sufficiently
strong; ambiguous mappings remain ``UNRESOLVED`` rather than being forced.

Usage:
    python plot_awmm_layers.py
    python plot_awmm_layers.py path/to/track.wav

Outputs are written in the current repository directory.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, resample_poly

import auditory_world_model_timbre_embedding as emb


OUT_DIR = Path(__file__).resolve().parent


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 96,
                   fmin: float = 30.0, fmax: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Create a triangular mel filterbank without requiring librosa."""
    if fmax is None:
        fmax = sr / 2.0
    fmax = min(fmax, sr / 2.0)

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

    m = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(m)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c > l:
            fb[i, l:c] = (np.arange(l, c) - l) / float(c - l)
        if r > c:
            fb[i, c:r] = (r - np.arange(c, r)) / float(r - c)
    return fb, freqs


def stft_mag(audio: np.ndarray, sr: int, n_fft: int = 2048,
             hop: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + max(0, (len(audio) - n_fft) // hop)
    frames = np.stack([
        audio[i * hop:i * hop + n_fft] * win
        for i in range(n_frames)
    ], axis=0)
    spec = np.fft.rfft(frames, axis=1)
    mag = np.abs(spec).T
    times = (np.arange(n_frames) * hop + n_fft / 2.0) / sr
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    return mag, times, freqs


def smooth_1d(x: np.ndarray, width: int = 11) -> np.ndarray:
    if len(x) < 3 or width <= 1:
        return x.astype(float)
    width = min(width, len(x) if len(x) % 2 else len(x) - 1)
    width = max(3, width)
    if width % 2 == 0:
        width -= 1
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(x, kernel, mode="same")


def source_test_components(duration: float = 8.0, bpm: float = 120.0,
                           sr: int = 48000) -> Dict[str, np.ndarray]:
    """Recreate the four deterministic synthetic test components exactly enough
    for correlation/temporal validation. Same RNG seed and synthesis structure
    as AudioSource.synth_test_signal()."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(42)
    beat = 60.0 / bpm
    comps = {k: np.zeros(n, dtype=np.float64) for k in ("kick", "bass", "hihat", "pad")}

    pad = np.zeros(n)
    for f in (110.0, 165.0):
        pad += 0.10 * np.sin(2 * np.pi * f * t)
    tremolo = 1.0 + 0.15 * np.sin(2 * np.pi * 0.3 * t)
    fade_in = np.clip(t / 1.0, 0.0, 1.0)
    comps["pad"] = pad * tremolo * fade_in

    bass = np.zeros(n)
    notes = (41.2, 55.0)
    note_dur = beat * 2
    for i in range(int(math.ceil(duration / note_dur))):
        f = notes[i % 2]
        s = int(i * note_dur * sr)
        e = int(min((i + 1) * note_dur, duration) * sr)
        if e <= s:
            continue
        seg_t = np.arange(e - s) / sr
        env = np.clip(seg_t / 0.01, 0.0, 1.0) * np.exp(-seg_t * 0.8)
        bass[s:e] += 0.35 * env * np.sin(2 * np.pi * f * seg_t)
        bass[s:e] += 0.10 * env * np.sin(2 * np.pi * 2 * f * seg_t)
    b_lp, a_lp = butter(4, 300.0 / (sr / 2.0), btype="low")
    comps["bass"] = lfilter(b_lp, a_lp, bass)

    kick = np.zeros(n)
    for i in range(int(math.ceil(duration / beat))):
        s = int(i * beat * sr)
        e = min(n, s + int(0.25 * sr))
        if e <= s:
            continue
        seg_t = np.arange(e - s) / sr
        f_sweep = 150.0 * np.exp(-seg_t * 35.0) + 45.0
        phase = 2 * np.pi * np.cumsum(f_sweep) / sr
        env = np.exp(-seg_t * 18.0)
        click = 0.25 * np.exp(-seg_t * 400.0) * rng.standard_normal(e - s)
        kick[s:e] += 0.9 * env * np.sin(phase) + click
    comps["kick"] = kick

    hihat = np.zeros(n)
    hh_step = beat / 2.0
    b_hp, a_hp = butter(4, [6000.0 / (sr / 2.0), min(14000.0, 0.49 * sr) / (sr / 2.0)], btype="band")
    for i in range(1, int(math.ceil(duration / hh_step)), 2):
        s = int(i * hh_step * sr)
        e = min(n, s + int(0.05 * sr))
        if e <= s:
            continue
        seg_t = np.arange(e - s) / sr
        env = np.exp(-seg_t * 80.0)
        hihat[s:e] += 0.18 * rng.standard_normal(e - s) * env
    comps["hihat"] = lfilter(b_hp, a_hp, hihat)

    # The original normalizes the sum by one common peak. Do the same here.
    mix = sum(comps.values())
    peak = float(np.max(np.abs(mix)) + 1e-9)
    scale = 0.9 / peak
    for k in comps:
        comps[k] = (comps[k] * scale).astype(np.float32)
    return comps


def object_activity(obj, times: np.ndarray) -> np.ndarray:
    """Turn a history into a soft activity curve on a common time axis."""
    if not obj.history:
        return np.zeros_like(times, dtype=float)
    ht = np.asarray([float(h.time) for h in obj.history])
    he = np.asarray([float(max(h.energy, 0.0)) for h in obj.history])
    good = np.asarray([
        h.status in ("ACTIVE", "RECOVERING", "TENTATIVE") for h in obj.history
    ])
    if not np.any(good):
        good[:] = True
    ht = ht[good]
    he = he[good]
    if len(ht) == 0:
        return np.zeros_like(times, dtype=float)
    order = np.argsort(ht)
    ht, he = ht[order], he[order]
    y = np.interp(times, ht, he, left=0.0, right=0.0)
    return np.maximum(y, 0.0)


def layer_members(model) -> Dict[str, List[object]]:
    world = model.world
    result: Dict[str, List[object]] = {}
    for layer in world.layers.values():
        if layer.status != "CONFIRMED" or not layer.member_ids:
            continue
        members = [world.objects[oid] for oid in layer.member_ids if oid in world.objects]
        if members:
            result[layer.layer_id] = members
    return result


def layer_activity(layer_membership: Sequence[object], times: np.ndarray) -> np.ndarray:
    curves = [object_activity(o, times) for o in layer_membership]
    if not curves:
        return np.zeros_like(times, dtype=float)
    stack = np.vstack(curves)
    x = np.sqrt(np.sum(stack * stack, axis=0))
    if np.max(x) > 0:
        x = x / np.max(x)
    return x


def layer_evidence_map(mag: np.ndarray, spec_times: np.ndarray, freqs: np.ndarray,
                       members: Sequence[object], n_mels: int = 96) -> np.ndarray:
    """Project tracked object evidence onto the input mel spectrogram grid.

    This is a visualization mask only. It never reconstructs or claims to have
    separated the waveform.
    """
    fb, _ = mel_filterbank(48000, (len(freqs) - 1) * 2, n_mels=n_mels)
    mel = fb @ (mag ** 2)
    evidence = np.zeros_like(mel, dtype=float)
    mel_centers_hz = np.array([
        np.sum(fb[i] * freqs) / (np.sum(fb[i]) + 1e-12)
        for i in range(n_mels)
    ])
    for obj in members:
        for h in obj.history:
            if h.status not in ("ACTIVE", "RECOVERING", "TENTATIVE"):
                continue
            f = float(h.frequency_hz)
            if not (20.0 <= f <= 22000.0):
                continue
            ti = int(np.argmin(np.abs(spec_times - float(h.time))))
            mi = int(np.argmin(np.abs(mel_centers_hz - f)))
            w = float(max(h.energy, 1e-8))
            # Paint a small frequency neighborhood; narrower at high frequencies
            # in log-frequency space.
            lo = max(0, mi - 2)
            hi = min(n_mels, mi + 3)
            evidence[lo:hi, max(0, ti - 1):min(evidence.shape[1], ti + 2)] += w
    if evidence.max() > 0:
        evidence /= evidence.max()
    return evidence


def correlate_layers_with_sources(model, layer_map: Dict[str, List[object]],
                                  times: np.ndarray, sources: Dict[str, np.ndarray],
                                  sr: int = 48000) -> List[Tuple[str, str, float, float]]:
    # Make source activity curves on the same time grid.
    out = []
    source_curves: Dict[str, np.ndarray] = {}
    for name, x in sources.items():
        env = smooth_1d(np.abs(x.astype(float)), width=max(5, int(sr / 400)))
        source_curves[name] = np.interp(times, np.arange(len(env)) / sr, env)
    for lid, members in layer_map.items():
        la = layer_activity(members, times)
        if np.std(la) < 1e-9:
            continue
        # Normalize layer activity before correlation.
        la = (la - np.mean(la)) / (np.std(la) + 1e-12)
        candidates = []
        for name, sc in source_curves.items():
            sc = (sc - np.mean(sc)) / (np.std(sc) + 1e-12)
            corr = float(np.mean(la * sc))
            overlap = float(np.sum((la > 0.15) & (sc > np.percentile(sc, 60))) /
                            max(1, np.sum(la > 0.15)))
            score = 0.75 * ((corr + 1.0) / 2.0) + 0.25 * overlap
            candidates.append((name, corr, score))
        candidates.sort(key=lambda z: z[2], reverse=True)
        best = candidates[0]
        second = candidates[1][2] if len(candidates) > 1 else -1.0
        if best[2] >= 0.62 and (best[2] - second) >= 0.06 and best[1] >= 0.20:
            label = best[0]
        else:
            label = "UNRESOLVED"
        out.append((lid, label, best[1], best[2]))
    return out


def make_plots(model, audio: np.ndarray, sr: int, synthetic: bool,
               out_prefix: str = "awm_diagnostics") -> None:
    mag, st, freqs = stft_mag(audio, sr)
    fb, _ = mel_filterbank(sr, (len(freqs) - 1) * 2, n_mels=96)
    mel_power = fb @ (mag ** 2)
    mel_db = 10.0 * np.log10(mel_power + 1e-10)
    mel_db = np.maximum(mel_db, mel_db.max() - 80.0)

    times = np.linspace(0.0, len(audio) / sr, max(2, len(audio) // 512), endpoint=False)
    layer_map = layer_members(model)
    correlations = []
    sources = {}
    if synthetic:
        sources = source_test_components(duration=len(audio) / sr, sr=sr)
        correlations = correlate_layers_with_sources(model, layer_map, times, sources, sr)

    # Stable plotting order: strongest/earliest layers first.
    ordered = sorted(layer_map.items(), key=lambda kv: min(o.creation_time for o in kv[1]))

    # ---------- Figure 1: combined overview ----------
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 2.2, max(2.0, 0.55 * max(1, len(ordered) + len(model.world.objects) // 3))])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)

    waveform_t = np.arange(len(audio)) / sr
    ax0.plot(waveform_t, audio, linewidth=0.6)
    ax0.set_ylabel("Amplitude")
    ax0.set_title("Auditory World Model — ORIGINAL MIXED SIGNAL")
    ax0.grid(alpha=0.2)

    extent = [st[0], st[-1], 0, 96]
    ax1.imshow(mel_db, origin="lower", aspect="auto", extent=extent)
    ax1.set_ylabel("Mel band")
    ax1.set_title("Mixed-input mel spectrogram with layer evidence")
    for row, (lid, members) in enumerate(ordered):
        act = layer_activity(members, times)
        # Put activity as a thick transparent trace scaled to mel rows.
        y = 2 + row * (94.0 / max(1, len(ordered)))
        ax1.plot(times, y + act * (3.0 / max(1, len(ordered))), linewidth=2.0,
                 label=f"L{row+1} {lid[:12]} ({len(members)} objs)")
    if ordered:
        ax1.legend(loc="upper right", fontsize=7, ncol=2)

    # Layer rows + member objects below on common seconds axis.
    ylabels: List[str] = []
    yvals: List[float] = []
    y = 0
    for i, (lid, members) in enumerate(ordered):
        ax2.axhspan(y - 0.45, y + 0.45, alpha=0.10)
        la = layer_activity(members, times)
        ax2.plot(times, y + 0.35 * la, linewidth=2.2)
        ylabels.append(f"L{i+1} {lid[:10]} | {len(members)} objs")
        yvals.append(y)
        y -= 1
        for obj in members:
            oa = object_activity(obj, times)
            if np.max(oa) > 0:
                oa = oa / np.max(oa)
            ax2.plot(times, y + 0.25 * oa, linewidth=0.8, alpha=0.75)
            ylabels.append(f"  └ {obj.object_id[:10]} | {obj.physical_signature.frequency_hz:.1f} Hz | {obj.timbre_class}")
            yvals.append(y)
            y -= 1
    # Remaining objects not assigned to a confirmed layer.
    grouped = {o.object_id for ms in layer_map.values() for o in ms}
    remaining = [o for o in model.world.objects.values() if o.object_id not in grouped and o.history]
    for obj in sorted(remaining, key=lambda o: o.creation_time):
        oa = object_activity(obj, times)
        if np.max(oa) > 0:
            oa = oa / np.max(oa)
        ax2.plot(times, y + 0.22 * oa, linewidth=0.7, alpha=0.55)
        ylabels.append(f"  └ REMAIN {obj.object_id[:10]} | {obj.physical_signature.frequency_hz:.1f} Hz")
        yvals.append(y)
        y -= 1
    ax2.set_yticks(yvals)
    ax2.set_yticklabels(ylabels, fontsize=6)
    ax2.set_ylabel("Layers → member objects")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Layer and object evidence — SAME TIME AXIS as the mixture")
    ax2.grid(axis="x", alpha=0.2)

    if synthetic and correlations:
        txt = ["Synthetic temporal correlation / assignment:"]
        for lid, label, corr, score in correlations:
            txt.append(f"{lid[:12]} → {label}  corr={corr:+.2f} score={score:.2f}")
        ax0.text(0.995, 0.95, "\n".join(txt), transform=ax0.transAxes, va="top", ha="right", fontsize=8,
                 bbox=dict(boxstyle="round", alpha=0.8))
    fig.suptitle("Auditory World Model — Combined Diagnostic View", fontsize=16)
    combined_path = OUT_DIR / f"{out_prefix}_combined.png"
    fig.savefig(combined_path, dpi=160)
    plt.show()

    # ---------- Figure 2: large separate panels ----------
    n_layers = max(1, len(ordered))
    nrows = 2 + n_layers + 1
    fig2 = plt.figure(figsize=(20, max(14, 3.5 * nrows)), constrained_layout=True)
    axes = fig2.subplots(nrows, 1, sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    axes[0].plot(waveform_t, audio, linewidth=0.7)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("COMBINED MIXTURE — waveform")
    axes[0].grid(alpha=0.2)

    axes[1].imshow(mel_db, origin="lower", aspect="auto", extent=extent)
    axes[1].set_ylabel("Mel")
    axes[1].set_title("COMBINED MIXTURE — mel spectrogram")

    cor_by_layer = {x[0]: x for x in correlations}
    for i, (lid, members) in enumerate(ordered):
        ax = axes[2 + i]
        evidence = layer_evidence_map(mag, st, freqs, members, n_mels=96)
        # Masked evidence: brighten only where that layer has evidence, while
        # retaining the actual mixture beneath it.
        ax.imshow(mel_db, origin="lower", aspect="auto", extent=extent)
        overlay = np.ma.masked_less(evidence, 0.08)
        ax.imshow(overlay, origin="lower", aspect="auto", extent=extent, alpha=0.65)
        ax2b = ax.twinx()
        la = layer_activity(members, times)
        ax2b.plot(times, la, linewidth=1.7)
        ax2b.set_ylim(0, 1.05)
        ax2b.set_ylabel("layer activity")
        corr_text = ""
        if lid in cor_by_layer:
            _, label, corr, score = cor_by_layer[lid]
            corr_text = f" | synthetic→{label}, corr={corr:+.2f}, score={score:.2f}"
        ax.set_ylabel("Mel")
        ax.set_title(f"LAYER {i+1}: {lid} | {len(members)} objects{corr_text}")

    # Last panel: all remaining/unassigned objects, plus object labels.
    axr = axes[-1]
    axr.imshow(mel_db, origin="lower", aspect="auto", extent=extent)
    grouped = {o.object_id for ms in layer_map.values() for o in ms}
    remaining = [o for o in model.world.objects.values() if o.object_id not in grouped and o.history]
    for j, obj in enumerate(sorted(remaining, key=lambda o: o.creation_time)):
        oa = object_activity(obj, times)
        if np.max(oa) <= 0:
            continue
        axr.plot(times, 3 + j + oa / max(np.max(oa), 1e-9), linewidth=0.9, alpha=0.75)
        axr.text(st[0] + 0.01, 3 + j + 0.5, f"{obj.object_id} {obj.physical_signature.frequency_hz:.1f}Hz",
                 fontsize=6, va="center")
    axr.set_title(f"REMAINING / UNASSIGNED OBJECT EVIDENCE ({len(remaining)} objects) — not reconstructed")
    axr.set_ylabel("Mel + object traces")
    axr.set_xlabel("Time (s)")

    fig2.suptitle("Auditory World Model — Full Layer-by-Layer Evidence Inspection", fontsize=17)
    stack_path = OUT_DIR / f"{out_prefix}_stack.png"
    fig2.savefig(stack_path, dpi=160)
    plt.show()

    print("\nDIAGNOSTIC OUTPUTS")
    print(f"  Combined: {combined_path}")
    print(f"  Stack:    {stack_path}")
    print(f"  Confirmed layers: {len(ordered)}")
    print(f"  SoundObjects: {len(model.world.objects)}")
    print(f"  Remaining/unassigned objects: {len(remaining)}")
    if synthetic:
        print("\nSYNTHETIC SOURCE CORRELATION")
        for lid, label, corr, score in correlations:
            print(f"  {lid}: {label:10s} corr={corr:+.3f} score={score:.3f}")


def load_audio_for_plot(path: Optional[str], model_params) -> Tuple[np.ndarray, int, bool]:
    src = emb.awm.AudioSource(model_params)
    if path is None:
        audio = src.synth_test_signal(duration=8.0)
        return audio, model_params.sample_rate, True
    audio, sr = src.load(path)
    return audio, sr, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="?", default=None)
    parser.add_argument("--prefix", default="awm_diagnostics")
    args = parser.parse_args()

    params = emb.awm.Parameters()
    audio, sr, synthetic = load_audio_for_plot(args.audio, params)
    model = emb.run(args.audio, duration=8.0)
    make_plots(model, audio, sr, synthetic, args.prefix)


if __name__ == "__main__":
    main()
