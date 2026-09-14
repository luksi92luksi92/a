"""Colab-friendly runner with the robust transient detector enabled."""
from __future__ import annotations

import edm_reverse_daw
from transient_detection_patch import detect_transients_robust

# edm_reverse_daw loads the foundation AWM module into this attribute.
edm_reverse_daw.awm.PhysicalAnalyzer.detect_transients = staticmethod(detect_transients_robust)

if __name__ == "__main__":
    edm_reverse_daw.main()
