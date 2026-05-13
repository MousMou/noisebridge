# NoiseBridge

[![PyPI version](https://badge.fury.io/py/noisebridge.svg)](https://pypi.org/project/noisebridge/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![IBM Quantum](https://img.shields.io/badge/IBM%20Quantum-validated-blue)](https://quantum.ibm.com/)
[![IQM](https://img.shields.io/badge/IQM-validated-orange)](https://www.meetiqm.com/)

**Post-processing quantum noise mitigation via a centering-matrix Syndrome Error Corrector (SEC).**  
Zero decoder overhead demonstrated on IBM Kingston real hardware — Surface Code d=3.

---

## Key Result

> **LER_SNN = LER_raw = 0.0422** on IBM `ibm_kingston` (Surface Code d=3, p=0.25, 4096 shots)  
> First zero-overhead decoding result on real quantum hardware. SNN outperforms standard lookup decoder by **1.72×** at p=0.10.

| Version | Qubits | Depth | CZ gates | LER_raw | LER_SNN | SNN vs std |
|---------|--------|-------|----------|---------|---------|------------|
| v1 linear | 17 | 244 | 153 | 0.1785 | 0.2585 | 0.88× |
| v2 2D 17Q | 17 | 242 | 153 | 0.2449 | 0.2944 | 1.01× |
| **v3 Z-only p=0.10** | **13** | **26** | **25** | **0.0388** | **0.0457** | **1.72×** |
| **v3 Z-only p=0.25** | **13** | **26** | **25** | **0.0422** | **0.0422** | **1.27× (zero overhead)** |

---

## Install

```bash
pip install noisebridge
```

With Qiskit integration:

```bash
pip install "noisebridge[qiskit]"
```

---

## Quickstart

### Surface Code syndrome decoding (IBM Kingston result)

```python
from noisebridge import correct, load_params

# Load calibrated params for IBM Kingston
params = load_params("ibm_kingston")

# Raw Z-syndrome counts from hardware (4096 shots, p=0.25, 2 injected errors)
raw_counts = {
    "0111": 3382, "0011": 213, "0110": 184,
    "0101": 131,  "1111": 41,  "0100": 35,
}

# SNN Syndrome Error Corrector — soft-decodes Hamming-1 neighbours
corrected = correct(raw_counts, params, n=4)
# Dominant syndrome 0111 amplified; noise neighbours suppressed
```

### Combined REM + SNN pipeline (recommended for general use)

```python
from noisebridge import rem_snn_correct

corrected = rem_snn_correct(
    {"00": 122, "01": 3, "10": 3, "11": 128},
    n=2,
    device="iqm_garnet"
)
```

### REM only

```python
from noisebridge import rem_correct

p_clean = rem_correct(
    {"0": 480, "1": 520},
    n=1,
    device="ibm_kingston"
)
```

### List supported devices

```python
from noisebridge import list_devices

list_devices()
list_devices(recommended_only=True)
```

---

## How It Works

NoiseBridge applies three composable post-processing strategies to raw measurement
counts after a quantum circuit executes on hardware. No additional QPU shots required.

**1. REM — Readout Error Mitigation**  
Inverts the readout confusion matrix to correct SPAM errors.  
Validated on IQM Garnet real QPU: avg +0.030 fidelity (5/5 circuits positive).

**2. SNN-SEC — Centering-Matrix Syndrome Error Corrector**  
The weight matrix uses a centering architecture:

```
W[i,j] = (δ(i,j) − 1/N) × W_scale
```

This subtracts the global mean from each syndrome probability, amplifying the
dominant (true) syndrome and suppressing Hamming-1 measurement noise neighbours.  
Validated on IQM Garnet real QPU: avg +0.057 fidelity (6/6 circuits positive).

**3. REM → SNN pipeline (recommended)**  
Sequential application achieves best performance on noisy hardware.

### Why Z-only for Surface Code d=3?

The state |0⟩^⊗9 is **not** an eigenstate of X-stabilizers — measuring them
produces random 50/50 noise with zero information content. Removing X-stabilizer
measurement:

- Reduces transpiled depth 242 → **26** (89% reduction)
- Reduces 2-qubit gate count 153 → **25** CZ (84% reduction)
- Eliminates all SWAP gates via native heavy-hex layout

This is the root cause of the zero-overhead result at p=0.25.

---

## Hardware Support

| Device | Provider | Validated | Notes |
|--------|----------|-----------|-------|
| `ibm_kingston` | IBM Quantum | ✅ | Surface code d=3 benchmark (2026-05-13) |
| `iqm_garnet` | IQM | ✅ | 20Q, avg +0.057 fidelity |
| `iqm_emerald` | IQM | ⚠️ | Params present, not yet validated on real QPU |
| `ibm_fakemarrakesh` | IBM (simulator) | ✅ | 56% win rate vs ZNE, p=0.011 |
| `rigetti_aspen_m3` | Rigetti | ⚠️ | Noise-model calibrated |

---

## IBM Kingston Layout (Surface Code d=3 v3)

```
Physical qubits:
  D0-D8  = [77, 56, 62, 66, 64, 85, 68, 57, 83]
  AZ0-AZ3 = [65, 63, 67, 84]

Z-stabilizer native edges: 10/12
  AZ0: 2/2  AZ1: 3/4  AZ2: 3/4  AZ3: 2/2

Transpiled circuit:  depth=26  CZ=25  SWAP=0
```

See [`examples/ibm_kingston_surface_d3.py`](examples/ibm_kingston_surface_d3.py) for
the full runnable benchmark.

---

## Citation

If you use NoiseBridge in your research, please cite:

```bibtex
@software{noisebridge2026,
  author  = {{FractKit Project}},
  title   = {NoiseBridge: Zero-Overhead Syndrome Decoding via Centering-Matrix SNN},
  year    = {2026},
  url     = {https://github.com/MousMou/noisebridge},
  version = {0.3.0}
}
```

Preprint: *Zero-Overhead Syndrome Decoding on Real Quantum Hardware via a
Centering-Matrix SNN Syndrome Error Corrector* — FractKit Project, 2026.
Available on Zenodo (10.5281/zenodo.20157839).

---

## License

MIT © 2026 FractKit Project
