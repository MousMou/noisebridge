"""
NoiseBridge -- Quantum noise mitigation toolkit
===============================================
Three post-processing strategies for quantum measurement noise:

  1. rem_correct()     -- Readout Error Mitigation via confusion matrix inversion.
     Validated on IQM Garnet (real QPU): avg +0.030 fidelity, 5/5 circuits.

  2. snn_correct()     -- Centering-matrix SNN-NR v2.
     Validated on IQM Garnet: avg +0.057 fidelity, 6/6 circuits.
     Architecture: W[i,j] = (delta(i,j) - 1/N) * W_scale

  3. rem_snn_correct() -- Combined pipeline: REM -> SNN (recommended for production).

Quick start -- combined pipeline (recommended)
---------------------------------------------
>>> from noisebridge import rem_snn_correct
>>> corrected = rem_snn_correct(
...     {"00": 122, "01": 3, "10": 3, "11": 128},
...     n=2, device="iqm_garnet"
... )

Quick start -- REM only
-----------------------
>>> from noisebridge import rem_correct
>>> p_clean = rem_correct({"00": 122, "01": 3, "10": 3, "11": 128},
...                        n=2, device="iqm_garnet")

Quick start -- SNN only
-----------------------
>>> from noisebridge import correct, load_params
>>> params = load_params("iqm_garnet")
>>> p_clean = correct(raw_counts, params, n=3)

Supported devices
-----------------
>>> from noisebridge import list_devices
>>> list_devices()
>>> list_devices(recommended_only=True)

References
----------
FractKit project -- SNN-NR v2 centering architecture (2026)
IQM Garnet real QPU validation: 2026-05-12
  SNN-NR v2: avg +0.057 fidelity (6/6 positive)
  REM:       avg +0.030 fidelity (5/5 positive)
"""

from .correct import snn_correct as correct, counts_to_probs, sparse_snn_correct
from .registry import load_params, list_devices, DEVICE_REGISTRY
from .mitigation import (
    rem_correct,
    rem_snn_correct,
    load_readout_params,
    build_confusion_matrix,
)

__version__ = "0.3.0"
__author__  = "FractKit Project"
__all__     = [
    # Combined pipeline -- recommended
    "rem_snn_correct",
    # REM -- validated on real QPU
    "rem_correct", "load_readout_params", "build_confusion_matrix",
    # SNN-NR v2 -- centering architecture
    "correct", "sparse_snn_correct", "counts_to_probs", "load_params",
    "list_devices", "DEVICE_REGISTRY",
]
