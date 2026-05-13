"""
Tests for noisebridge package.
Run: pytest tests/
"""
import pytest
import numpy as np
from noisebridge import correct, counts_to_probs, load_params, list_devices, DEVICE_REGISTRY


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def ghz5_counts():
    return {
        "00000": 220, "11111": 195, "00001": 8,
        "10000": 6, "01000": 5, "11110": 4, "00010": 3,
    }

@pytest.fixture
def ibm_params():
    return load_params("ibm_fakemarrakesh")


# ── registry tests ─────────────────────────────────────────────────────────────

def test_list_devices_returns_all():
    devices = list_devices()
    assert len(devices) == 10
    assert "ibm_fakemarrakesh" in devices
    assert "iqm_garnet" in devices
    assert "ibm_miami" in devices
    assert "qbraid_qir_sim" in devices

def test_list_devices_recommended_only():
    rec = list_devices(recommended_only=True)
    assert "ibm_fakemarrakesh" in rec
    assert "ibm_fakesherbrooke" in rec
    assert "rigetti_aspen_m3" in rec
    assert len(rec) >= 1

def test_load_params_returns_n_params():
    # load_params returns {"n_params": {n: {beta, threshold, W_scale, snn_factor, T}}}
    params = load_params("ibm_fakemarrakesh")
    assert "n_params" in params
    required_keys = {"beta", "threshold", "W_scale", "snn_factor", "T"}
    for n, p in params["n_params"].items():
        assert required_keys.issubset(p.keys()), f"Missing keys for n={n}: {p.keys()}"

def test_load_params_unknown_device_raises():
    with pytest.raises(KeyError, match="not found"):
        load_params("nonexistent_device_xyz")

def test_all_devices_have_required_fields():
    required = {"name", "family", "qubits", "2q_error", "regime", "recommended", "n_params"}
    for device_id, info in DEVICE_REGISTRY.items():
        assert required.issubset(info.keys()), f"{device_id} missing fields"


# ── counts_to_probs tests ──────────────────────────────────────────────────────

def test_counts_to_probs_sums_to_one(ghz5_counts):
    probs = counts_to_probs(ghz5_counts, n=5)
    assert abs(sum(probs.values()) - 1.0) < 1e-10

def test_counts_to_probs_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        counts_to_probs({}, n=2)


# ── snn_correct tests ──────────────────────────────────────────────────────────

def test_correct_from_counts_sums_to_one(ghz5_counts, ibm_params):
    result = correct(ghz5_counts, ibm_params, n=5)
    assert abs(sum(result.values()) - 1.0) < 1e-9

def test_correct_returns_all_bitstrings(ibm_params):
    counts = {"00": 128, "11": 120}
    result = correct(counts, ibm_params, n=2)
    assert set(result.keys()) == {"00", "01", "10", "11"}

def test_correct_no_negative_probs(ghz5_counts, ibm_params):
    result = correct(ghz5_counts, ibm_params, n=5)
    assert all(v >= 0.0 for v in result.values())

def test_correct_works_with_probabilities(ibm_params):
    probs = {"00000": 0.48, "11111": 0.44, "00001": 0.04, "10000": 0.04}
    result = correct(probs, ibm_params, n=5)
    assert abs(sum(result.values()) - 1.0) < 1e-9

def test_correct_all_devices():
    counts = {"000": 130, "111": 120, "001": 3, "110": 3}
    for device_id in list_devices():
        params = load_params(device_id)
        result = correct(counts, params, n=3)
        assert abs(sum(result.values()) - 1.0) < 1e-9, f"SNN failed for {device_id}"

def test_correct_ghz_dominant_states_preserved(ghz5_counts, ibm_params):
    result = correct(ghz5_counts, ibm_params, n=5)
    top2 = sorted(result, key=result.get, reverse=True)[:2]
    assert "00000" in top2
    assert "11111" in top2
