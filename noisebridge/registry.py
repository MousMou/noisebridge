"""
noisebridge.registry
=====================
Device registry for SNN-NR -- centering matrix architecture (v2).

SNN parameters are stored per qubit count n (2-5) under the key
"n_params" in each device entry.  load_params(device_id) returns a
dict with a top-level "n_params" key understood by snn_correct().

Architecture note
-----------------
All parameter sets use the centering-matrix weight:

    W[i,j] = (delta(i,j) - 1/N) * W_scale

which produces  (W@x)[i] = W_scale * (x[i] - mean(x)).

Parameters are derived analytically from each device noise profile:
  - threshold   : 2.5x readout noise floor, capped at 30% of signal excess
  - W_scale     : amplifies centering; increases with total noise level
  - snn_factor  : mixing coefficient; higher for noisier devices
  - beta        : universal 0.5 (LIF decay)
  - T           : 50 time steps (universal)

Validated on IQM Garnet real QPU (2026-05-12, 256 shots):
  bell_2q    RAW->0.9764  SNN-v2->0.9895  delta=+0.0131
  ghz_3q     RAW->0.9336  SNN-v2->0.9737  delta=+0.0401
  ghz_4q     RAW->0.8789  SNN-v2->0.9576  delta=+0.0787
  ghz_5q     RAW->0.8477  SNN-v2->0.9532  delta=+0.1055
  w_state    RAW->0.9413  SNN-v2->0.9822  delta=+0.0409
  qft_3q     RAW->0.9034  SNN-v2->0.9621  delta=+0.0587
  Average improvement: +0.057  (6/6 circuits positive)

Usage
-----
>>> from noisebridge import load_params
>>> params = load_params("iqm_garnet")
>>> print(list_devices())
"""

from typing import Dict, List

DEVICE_REGISTRY: Dict[str, Dict] = {

    # -- IBM Quantum --------------------------------------------------
    "ibm_fakemarrakesh": {
        "name":        "IBM FakeMarrakesh",
        "family":      "IBM Quantum",
        "qubits":      156,
        "2q_error":    0.0049,
        "regime":      "transition",
        "recommended": True,
        "n_params": {
            2: {"beta": 0.5, "threshold": 0.0325, "W_scale": 1.398, "snn_factor": 0.359, "T": 50},
            3: {"beta": 0.5, "threshold": 0.0287, "W_scale": 1.500, "snn_factor": 0.408, "T": 50},
            4: {"beta": 0.5, "threshold": 0.0269, "W_scale": 1.500, "snn_factor": 0.456, "T": 50},
            5: {"beta": 0.5, "threshold": 0.0259, "W_scale": 1.500, "snn_factor": 0.505, "T": 50},
        },
        "benchmark": {
            "win_rate":    0.56,
            "p_value":     0.011,
            "cohens_d":    1.288,
            "source":      "FractKit intensive benchmark (25 tests) -- v1 SNN",
            "note":        "v2 centering architecture -- re-benchmarking pending",
        },
    },

    "ibm_fakesherbrooke": {
        "name":        "IBM FakeSherbrooke",
        "family":      "IBM Quantum",
        "qubits":      127,
        "2q_error":    0.0060,
        "regime":      "transition",
        "recommended": True,
        "n_params": {
            2: {"beta": 0.5, "threshold": 0.0387, "W_scale": 1.475, "snn_factor": 0.390, "T": 50},
            3: {"beta": 0.5, "threshold": 0.0344, "W_scale": 1.500, "snn_factor": 0.450, "T": 50},
            4: {"beta": 0.5, "threshold": 0.0322, "W_scale": 1.500, "snn_factor": 0.509, "T": 50},
            5: {"beta": 0.5, "threshold": 0.0311, "W_scale": 1.500, "snn_factor": 0.568, "T": 50},
        },
        "benchmark": {
            "source": "FractKit noise model training (2026-05-12) -- v2 params",
        },
    },

    "ibm_faketorino": {
        "name":        "IBM FakeTorino (Heron R2)",
        "family":      "IBM Quantum",
        "qubits":      133,
        "2q_error":    0.0030,
        "regime":      "clean",
        "recommended": False,
        "n_params": {
            2: {"beta": 0.5, "threshold": 0.0263, "W_scale": 1.300, "snn_factor": 0.320, "T": 50},
            3: {"beta": 0.5, "threshold": 0.0231, "W_scale": 1.375, "snn_factor": 0.350, "T": 50},
            4: {"beta": 0.5, "threshold": 0.0216, "W_scale": 1.449, "snn_factor": 0.380, "T": 50},
            5: {"beta": 0.5, "threshold": 0.0208, "W_scale": 1.500, "snn_factor": 0.409, "T": 50},
        },
        "benchmark": {
            "source": "FractKit noise model training (2026-05-12) -- v2 params",
            "note":   "Low noise regime -- REM alone may suffice for n <= 3",
        },
    },

    "ibm_miami": {
        "name":        "IBM Miami",
        "family":      "IBM Quantum",
        "qubits":      120,
        "2q_error":    0.0035,
        "regime":      "transition",
        "recommended": True,
        "device_id":   "ibm:ibm:qpu:miami",
        "provider":    "qBraid",
        "n_params": {
            2: {"beta": 0.5, "threshold": 0.0294, "W_scale": 1.337, "snn_factor": 0.335, "T": 50},
            3: {"beta": 0.5, "threshold": 0.0259, "W_scale": 1.425, "snn_factor": 0.370, "T": 50},
            4: {"beta": 0.5, "threshold": 0.0242, "W_scale": 1.500, "snn_factor": 0.405, "T": 50},
            5: {"beta": 0.5, "threshold": 0.0234, "W_scale": 1.500, "snn_factor": 0.439, "T": 50},
        },
        "benchmark": {
            "source":        "FractKit noise model training (2026-05-12) -- v2 params",
            "note":          "Heron R1 heavy-hex. Pending real QPU validation.",
            "real_hardware": False,
        },
    },

    # -- IQM ---------------------------------------------------------
    "iqm_garnet": {
        "name":        "IQM Garnet",
        "family":      "IQM",
        "qubits":      20,
        "2q_error":    0.0049,
        "regime":      "clean",
        "recommended": True,
        "n_params": {
            2: {"beta": 0.5, "threshold": 0.0394, "W_scale": 1.460, "snn_factor": 0.384, "T": 50},
            3: {"beta": 0.5, "threshold": 0.0347, "W_scale": 1.500, "snn_factor": 0.433, "T": 50},
            4: {"beta": 0.5, "threshold": 0.0323, "W_scale": 1.500, "snn_factor": 0.481, "T": 50},
            5: {"beta": 0.5, "threshold": 0.0312, "W_scale": 1.500, "snn_factor": 0.530, "T": 50},
        },
        "benchmark": {
            "win_rate":      1.00,
            "avg_delta":     0.057,
            "source":        "IQM Garnet QPU REAL -- 2026-05-12 (6 circuits, 256 shots) -- v2 centering SNN",
            "note":          "6/6 circuits positive. bell+0.013 ghz3+0.040 ghz4+0.079 ghz5+0.106 w_state+0.041 qft+0.059",
            "real_hardware": True,
            "validated":     True,
        },
    },

    "iqm_emerald": {
        "name":        "IQM Emerald",
        "family":      "IQM",
        "qubits":      20,
        "2q_error":    0.0060,
        "regime":      "transition",
        "recommended": True,
        "n_params": {
            2: {"beta": 0.5, "threshold": 0.0488, "W_scale": 1.500, "snn_factor": 0.425, "T": 50},
            3: {"beta": 0.5, "threshold": 0.0431, "W_scale": 1.500, "snn_factor": 0.485, "T": 50},
            4: {"beta": 0.5, "threshold": 0.0403, "W_scale": 1.500, "snn_factor": 0.544, "T": 50},
            5: {"beta": 0.5, "threshold": 0.0389, "W_scale": 1.500, "snn_factor": 0.600, "T": 50},
        },
        "benchmark": {
            "source": "FractKit noise model training (2026-05-12) -- v2 params",
        },
    },

    # -- Rigetti -----------------------------------------------------
    "rigetti_aspen_m3": {
        "name":        "Rigetti Aspen-M3",
        "family":      "Rigetti",
        "qubits":      79,
        "2q_error":    0.0150,
        "regime":      "noisy",
        "recommended": True,
        "n_params": {
            2: {"beta": 0.5, "threshold": 0.0739, "W_scale": 1.500, "snn_factor": 0.600, "T": 50},
            3: {"beta": 0.5, "threshold": 0.0859, "W_scale": 1.500, "snn_factor": 0.600, "T": 50},
            4: {"beta": 0.5, "threshold": 0.0805, "W_scale": 1.500, "snn_factor": 0.600, "T": 50},
            5: {"beta": 0.5, "threshold": 0.0777, "W_scale": 1.500, "snn_factor": 0.600, "T": 50},
        },
        "benchmark": {
            "source": "FractKit noise model training (2026-05-12) -- v2 params",
            "note":   "Highest noise regime -- maximum SNN correction applied",
        },
    },

    # -- IonQ --------------------------------------------------------
    "ionq_aria": {
        "name":        "IonQ Aria",
        "family":      "IonQ",
        "qubits":      25,
        "2q_error":    0.0030,
        "regime":      "clean",
        "recommended": False,
        "n_params": {
            2: {"beta": 0.5, "threshold": 0.0156, "W_scale": 1.200, "snn_factor": 0.280, "T": 50},
            3: {"beta": 0.5, "threshold": 0.0141, "W_scale": 1.275, "snn_factor": 0.310, "T": 50},
            4: {"beta": 0.5, "threshold": 0.0133, "W_scale": 1.349, "snn_factor": 0.340, "T": 50},
            5: {"beta": 0.5, "threshold": 0.0129, "W_scale": 1.424, "snn_factor": 0.369, "T": 50},
        },
        "benchmark": {
            "source": "FractKit noise model training (2026-05-12) -- v2 params",
            "note":   "All-to-all topology. REM recommended as primary, SNN as secondary.",
        },
    },

    # -- Quantinuum --------------------------------------------------
    "quantinuum_h2": {
        "name":        "Quantinuum H2",
        "family":      "Quantinuum",
        "qubits":      56,
        "2q_error":    0.0010,
        "regime":      "clean",
        "recommended": False,
        "n_params": {
            2: {"beta": 0.5, "threshold": 0.0100, "W_scale": 1.075, "snn_factor": 0.230, "T": 50},
            3: {"beta": 0.5, "threshold": 0.0100, "W_scale": 1.100, "snn_factor": 0.240, "T": 50},
            4: {"beta": 0.5, "threshold": 0.0100, "W_scale": 1.125, "snn_factor": 0.250, "T": 50},
            5: {"beta": 0.5, "threshold": 0.0100, "W_scale": 1.150, "snn_factor": 0.260, "T": 50},
        },
        "benchmark": {
            "source": "FractKit noise model training (2026-05-12) -- v2 params",
            "note":   "Ultra-clean hardware -- minimal correction needed",
        },
    },

    # -- qBraid ------------------------------------------------------
    "qbraid_qir_sim": {
        "name":        "qBraid QIR Simulator",
        "family":      "qBraid",
        "qubits":      30,
        "2q_error":    0.0,
        "regime":      "ideal",
        "recommended": False,
        "device_id":   "qbraid:qbraid:sim:qir-sv",
        "provider":    "qBraid",
        "n_params": {
            2: {"beta": 0.5, "threshold": 0.0100, "W_scale": 1.000, "snn_factor": 0.200, "T": 50},
            3: {"beta": 0.5, "threshold": 0.0100, "W_scale": 1.000, "snn_factor": 0.200, "T": 50},
            4: {"beta": 0.5, "threshold": 0.0100, "W_scale": 1.000, "snn_factor": 0.200, "T": 50},
            5: {"beta": 0.5, "threshold": 0.0100, "W_scale": 1.000, "snn_factor": 0.200, "T": 50},
        },
        "benchmark": {
            "source":        "Ideal simulator -- baseline and algorithm validation",
            "note":          "No hardware noise. SNN near-identity at these params.",
            "real_hardware": False,
        },
    },
}

# qBraid device ID alias for IQM Garnet
DEVICE_REGISTRY["aws:iqm:qpu:garnet"] = dict(DEVICE_REGISTRY["iqm_garnet"])
DEVICE_REGISTRY["aws:iqm:qpu:garnet"]["name"]      = "IQM Garnet (qBraid alias)"
DEVICE_REGISTRY["aws:iqm:qpu:garnet"]["device_id"] = "aws:iqm:qpu:garnet"
DEVICE_REGISTRY["aws:iqm:qpu:garnet"]["provider"]  = "qBraid"


def load_params(device_id: str) -> Dict:
    """
    Load SNN-NR hyperparameters for a specific quantum device.

    Returns a dict with key "n_params" -- a mapping from qubit count
    (int 2-5) to parameter dicts understood by snn_correct().

    Parameters
    ----------
    device_id : str
        Device identifier. Use list_devices() to see available options.

    Returns
    -------
    dict with key: n_params -> {2: {...}, 3: {...}, 4: {...}, 5: {...}}
    Each inner dict has: beta, threshold, W_scale, snn_factor, T

    Raises
    ------
    KeyError : if device_id is not in registry

    Examples
    --------
    >>> params = load_params("iqm_garnet")
    >>> from noisebridge import correct
    >>> corrected = correct(raw_counts, params, n=2)
    """
    if device_id not in DEVICE_REGISTRY:
        available = list(DEVICE_REGISTRY.keys())
        raise KeyError(
            f"Device '{device_id}' not found. "
            f"Available: {available}"
        )
    entry = DEVICE_REGISTRY[device_id]
    return {"n_params": {int(k): v for k, v in entry["n_params"].items()}}


def list_devices(recommended_only: bool = False) -> List[str]:
    """
    List available device IDs in the registry.

    Parameters
    ----------
    recommended_only : bool
        If True, only return devices where SNN-NR is recommended.

    Returns
    -------
    list of str
    """
    # Exclude alias entries from the default listing
    base = [d for d in DEVICE_REGISTRY if d != "aws:iqm:qpu:garnet"]
    if recommended_only:
        return [d for d in base if DEVICE_REGISTRY[d].get("recommended", False)]
    return base


def device_info(device_id: str) -> Dict:
    """Return full device profile including benchmark results."""
    if device_id not in DEVICE_REGISTRY:
        raise KeyError(f"Device '{device_id}' not found.")
    return DEVICE_REGISTRY[device_id]
