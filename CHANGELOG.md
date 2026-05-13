# Changelog

All notable changes to NoiseBridge are documented here.

---

## [0.3.0] — 2026-05-13

### Added
- **Centering-matrix SNN architecture** — `W[i,j] = (δ(i,j) − 1/N) × W_scale`
  — replaces the previous fractal synaptic architecture with a mathematically
  grounded Syndrome Error Corrector (SEC).
- **`rem_snn_correct()`** — combined REM → SNN pipeline (recommended for production).
- **`rem_correct()`** — standalone Readout Error Mitigation via confusion matrix inversion.
- **`sparse_snn_correct()`** — sparse-input variant for syndrome decoding (low shot count).
- **`ibm_kingston` device** — calibrated params added to device registry.
- **Z-only Surface Code d=3 benchmark** — three-phase layout search (`find_layout`)
  achieves 10/12 native Z-edges on `ibm_kingston` heavy-hex, reducing transpiled
  depth from 242 (v2) to 26 (89% reduction, 84% fewer 2Q gates).
- **`examples/ibm_kingston_surface_d3.py`** — full runnable benchmark.

### Key results (IBM Kingston, 2026-05-13)
- `p=0.10`: LER_SNN=0.0457 vs LER_std=0.0784 → **1.72× improvement**
- `p=0.25`: LER_SNN=0.0422 = LER_raw=0.0422 → **zero decoder overhead**
  (first zero-overhead result on real quantum hardware)

### Changed
- License changed from BSL-1.1 to **MIT**.
- `__init__.py` public API updated: `rem_snn_correct` is now the recommended
  entry point for general use.
- Device registry expanded; per-device params re-calibrated for centering-matrix arch.

### Fixed
- X-stabilizer insight: `|0⟩^⊗9` is not an eigenstate of X-stabilizers.
  v1/v2 measured X-ancillas, generating pure noise and driving depth to 242.
  v3 (Z-only) removes this entirely.

---

## [0.2.0] — 2026-05-12

### Added
- `rem_correct()` initial implementation — readout confusion matrix inversion.
- IQM Garnet real QPU validation: avg +0.030 fidelity REM, +0.057 fidelity SNN
  (5/5 and 6/6 circuits positive, respectively).
- `iqm_garnet` and `iqm_emerald` added to device registry.

---

## [0.1.0] — 2026-05-12

### Added
- Initial release.
- `correct()` — SNN-NR post-processing for quantum measurement distributions.
- `load_params()` / `list_devices()` — device registry with 8 QPU platforms.
- `counts_to_probs()` — utility for raw counts → probability conversion.
- Benchmark: 56% win rate vs ZNE on IBM FakeMarrakesh (p=0.011, Cohen d=1.288).
