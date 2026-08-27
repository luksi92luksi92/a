#!/usr/bin/env python3
"""Run the coherent layer-grouping fix.

The first grouping-fix module is kept as the implementation. This wrapper
loads it after applying a Python syntax normalization for a generator
expression so it can be run directly in environments that require
parenthesized generator expressions when another positional argument
follows.
"""
from pathlib import Path
import sys

p = Path(__file__).with_name("auditory_world_model_grouping_fix.py")
src = p.read_text(encoding="utf-8")
src = src.replace(
    "self._finite(h.frequency_hz for h in active if h.frequency_hz > 0, 1.0)",
    "self._finite((h.frequency_hz for h in active if h.frequency_hz > 0), 1.0)",
)
src = src.replace(
    "        harmonic = self._finite(h.confidence.get(\"prediction\", 0.5) * 0.0 + 0.0 for h in [])\n",
    "",
)
ns = {"__file__": str(p), "__name__": "awm_grouping_fix"}
exec(compile(src, str(p), "exec"), ns)
path = sys.argv[1] if len(sys.argv) > 1 else None
ns["run_fixed"](path)
