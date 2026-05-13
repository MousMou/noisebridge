"""
IBM Kingston — Surface Code d=3 Z-only benchmark (noisebridge v0.3.0)
======================================================================
Reproduces the zero-overhead decoding result from 2026-05-13:

    p=0.10:  LER_SNN=0.0457  vs  LER_std=0.0784  →  1.72× improvement
    p=0.25:  LER_SNN=0.0422  =   LER_raw=0.0422  →  ZERO OVERHEAD

Key insight: |0⟩^⊗9 is NOT an eigenstate of X-stabilizers.
Measuring X-ancillas yields random 50/50 noise with no information content.
Z-only circuit reduces depth 242 → 26 and CZ count 153 → 25.

Requirements:
    pip install noisebridge qiskit qiskit-ibm-runtime

Usage:
    python ibm_kingston_surface_d3.py --token YOUR_IBM_TOKEN
    python ibm_kingston_surface_d3.py --token YOUR_IBM_TOKEN --p_error 0.25 --shots 4096
    python ibm_kingston_surface_d3.py --token YOUR_IBM_TOKEN --find-layout-only
"""

import argparse
import numpy as np
from collections import defaultdict
from itertools import combinations

# ── Circuit definition ───────────────────────────────────────────────────────

# 13 qubits: D0-D8 (data, virtual 0-8) + AZ0-AZ3 (Z-ancilla, virtual 9-12)
N_QUBITS = 13
DATA     = list(range(9))
Z_ANC    = list(range(9, 13))

# Z-stabilizers: each entry lists data qubit indices measured by that ancilla
Z_STABS = [
    [0, 3],            # AZ0 — weight-2 boundary
    [1, 2, 4, 5],      # AZ1 — weight-4 bulk
    [3, 4, 6, 7],      # AZ2 — weight-4 bulk
    [5, 8],            # AZ3 — weight-2 boundary
]

# Logical operators
LOGICAL_Z_QUBITS = [0, 1, 2]   # Z_L = Z{D0,D1,D2}  (top row)
LOGICAL_X_QUBITS = [2, 5, 8]   # X_L = X{D2,D5,D8}  (right column)

# 12 Z-edges in virtual qubit space (D0-D8 = 0-8, AZ0-AZ3 = 9-12)
SURF_Z_EDGES = [
    (0, 9),  (3, 9),                           # AZ0
    (1, 10), (2, 10), (4, 10), (5, 10),        # AZ1
    (3, 11), (4, 11), (6, 11), (7, 11),        # AZ2
    (5, 12), (8, 12),                           # AZ3
]

# Minimum native edges required per Z-stabilizer to remain functional
Z_ANC_MIN = {
    9:  ([(0, 9),  (3, 9)],                    1),
    10: ([(1, 10), (2, 10), (4, 10), (5, 10)], 2),
    11: ([(3, 11), (4, 11), (6, 11), (7, 11)], 2),
    12: ([(5, 12), (8, 12)],                    1),
}

# Best known layout on ibm_kingston (found 2026-05-13, 10/12 native Z-edges)
BEST_LAYOUT_KINGSTON = [77, 56, 62, 66, 64, 85, 68, 57, 83,  # D0-D8
                         65, 63, 67, 84]                        # AZ0-AZ3


# ── Layout search ────────────────────────────────────────────────────────────

def find_layout(backend, time_limit=45.0):
    """
    Three-phase subgraph search for the best 13-qubit Z-only surface code
    layout on a heavy-hex backend.

    Returns:
        layout  : list[int] of length 13 — physical qubit for each virtual qubit
        native  : int — number of native Z-edges (max 12)
    """
    import time as _t

    cmap = backend.coupling_map
    adj  = defaultdict(set)
    for a, b in cmap:
        adj[a].add(b)
        adj[b].add(a)
    num_q  = backend.num_qubits
    hw_deg = {q: len(adj[q]) for q in range(num_q)}

    surf_deg = defaultdict(int)
    for v1, v2 in SURF_Z_EDGES:
        surf_deg[v1] += 1
        surf_deg[v2] += 1
    virt_by_deg = sorted(range(13), key=lambda v: (-surf_deg[v], v))

    def score_map(v_to_p):
        # Reject if any Z-stabilizer falls below minimum connectivity
        for anc_v, (edges, min_nat) in Z_ANC_MIN.items():
            if sum(1 for v1, v2 in edges if v_to_p[v2] in adj[v_to_p[v1]]) < min_nat:
                return -1
        return sum(1 for v1, v2 in SURF_Z_EDGES if v_to_p[v2] in adj[v_to_p[v1]])

    def local_search(mapping, max_no_improve=300):
        best, best_s = dict(mapping), score_map(mapping)
        no_imp = 0
        while no_imp < max_no_improve:
            improved = False
            for v1 in range(13):
                for v2 in range(v1 + 1, 13):
                    m = dict(best)
                    m[v1], m[v2] = m[v2], m[v1]
                    ns = score_map(m)
                    if ns > best_s:
                        best, best_s, improved, no_imp = m, ns, True, 0
            if not improved:
                break
        return best_s, best

    def greedy_assignment(hw_set, rng):
        hw_list   = sorted(hw_set)
        local_deg = {q: len(adj[q] & hw_set) for q in hw_list}
        hw_sorted = sorted(hw_list, key=lambda q: (-local_deg[q], q))
        best_s, best_m = -1, None
        for trial in range(20):
            hw_order = hw_sorted[:]
            if trial > 0:
                rng.shuffle(hw_order)
            m = {virt_by_deg[i]: hw_order[i] for i in range(13)}
            s = score_map(m)
            if s > best_s:
                best_s, best_m = s, dict(m)
        return best_s, best_m

    t0            = _t.time()
    deadline      = t0 + time_limit
    best_s, best_m = -1, None
    tried         = set()
    rng           = np.random.default_rng(42)
    seeds         = list(range(num_q))
    rng.shuffle(seeds)

    # Phase 1: BFS diversification
    for seed in seeds:
        if _t.time() > t0 + time_limit * 0.25:
            break
        for variant in range(4):
            nodes, visited = [seed], {seed}
            rng_v = np.random.default_rng(seed * 97 + variant)
            while len(nodes) < 13:
                frontier = {}
                for n in nodes:
                    for nb in adj[n]:
                        if nb not in visited:
                            frontier[nb] = max(frontier.get(nb, 0), len(adj[nb] & visited))
                if not frontier:
                    break
                cands = list(frontier.items())
                next_n = max(cands, key=lambda x: (x[1], hw_deg[x[0]]))[0] \
                    if variant == 0 else cands[rng_v.choice(len(cands))][0]
                nodes.append(next_n); visited.add(next_n)
            if len(nodes) < 13:
                continue
            key = frozenset(nodes)
            if key in tried:
                continue
            tried.add(key)
            s, m = greedy_assignment(set(nodes), rng_v)
            if s > best_s:
                best_s, best_m = s, m
                print(f"  [Phase 1] {s}/12 native Z-edges (seed=q{seed}, t={_t.time()-t0:.1f}s)")

    # Phase 2: local search + perturbation
    if best_m:
        s2, m2 = local_search(best_m, max_no_improve=1000)
        if s2 > best_s:
            best_s, best_m = s2, m2

    it = 0
    while _t.time() < t0 + time_limit * 0.90 and best_m:
        it += 1
        rng_i    = np.random.default_rng(it * 7919)
        pert     = dict(best_m)
        virts    = rng_i.choice(13, size=6, replace=False)
        for k in range(3):
            v1, v2 = virts[2*k], virts[2*k+1]
            pert[v1], pert[v2] = pert[v2], pert[v1]
        sp, mp = local_search(pert, max_no_improve=200)
        if sp > best_s:
            best_s, best_m = sp, mp
            print(f"  [Phase 2] {sp}/12 native Z-edges (iter={it}, t={_t.time()-t0:.1f}s)")
            if best_s >= 11:
                break

    # Phase 3: boundary qubit refinement
    if best_m:
        hw_set = frozenset(best_m[v] for v in range(13))
        for new_q in range(num_q):
            if _t.time() > deadline:
                break
            if new_q in hw_set or not adj[new_q] & hw_set:
                continue
            for old_v in range(13):
                trial = dict(best_m); trial[old_v] = new_q
                st, mt = local_search(trial, max_no_improve=100)
                if st > best_s:
                    best_s, best_m = st, mt
                    hw_set = frozenset(best_m[v] for v in range(13))
                    print(f"  [Phase 3] {st}/12 native Z-edges (t={_t.time()-t0:.1f}s)")

    if best_m is None:
        print("  Layout search failed — using default linear layout")
        return list(range(13)), 0

    layout = [best_m[v] for v in range(13)]
    print(f"\n  Layout found: D0-D8={layout[:9]}  AZ0-AZ3={layout[9:]}")
    print(f"  Native Z-edges: {best_s}/12")
    return layout, best_s


# ── Circuit builder ──────────────────────────────────────────────────────────

def build_circuit(p_error=0.0, logical_state=0, seed=42):
    """
    Build the Z-only Surface Code d=3 circuit.

    Args:
        p_error      : bit-flip error probability per data qubit (injected classically)
        logical_state: 0 for |0_L⟩, 1 for |1_L⟩
        seed         : RNG seed for error injection

    Returns:
        qc       : QuantumCircuit (13 qubits, 13 classical bits)
        injected : list of data qubit indices where errors were injected
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(N_QUBITS, N_QUBITS, name="surface_d3_zonly_v3")

    if logical_state == 1:
        for q in LOGICAL_X_QUBITS:
            qc.x(q)

    injected = []
    if p_error > 0:
        rng = np.random.default_rng(seed)
        for i in range(9):
            if rng.random() < p_error:
                qc.x(i)
                injected.append(i)

    qc.barrier()

    # Z-stabilizer measurement schedule
    # Round 1: AZ0 || AZ3 (disjoint data qubits, fully parallel)
    qc.cx(0, 9);  qc.cx(5, 12)
    qc.cx(3, 9);  qc.cx(8, 12)
    # Round 2-3: AZ1 || AZ2 (serialized on shared D3, D4)
    qc.cx(1, 10); qc.cx(6, 11)
    qc.cx(2, 10); qc.cx(7, 11)
    qc.cx(4, 10); qc.cx(3, 11)
    qc.cx(5, 10); qc.cx(4, 11)

    qc.barrier()

    qc.measure(Z_ANC, range(4))     # bits 0-3: AZ0-AZ3 syndrome
    qc.measure(DATA,  range(4, 13)) # bits 4-12: D0-D8 data

    return qc, injected


# ── Decoder ──────────────────────────────────────────────────────────────────

def build_lookup_decoder():
    """Build Z-syndrome → correction lookup table (all weight-1 and weight-2 errors)."""
    table = {"0" * 4: []}
    for q in range(9):
        key = "".join("1" if q in s else "0" for s in Z_STABS)
        if key not in table:
            table[key] = [q]
    for q1, q2 in combinations(range(9), 2):
        key = "".join(str((1 if q1 in s else 0) ^ (1 if q2 in s else 0)) for s in Z_STABS)
        if key not in table:
            table[key] = [q1, q2]
    return table


def extract_syndrome(bitstring):
    """Extract (data_str, z_syndrome) from a 13-bit measurement string."""
    s     = bitstring.replace(" ", "").zfill(13)
    z_syn = s[-4:][::-1]     # bits 0-3 → AZ0-AZ3
    data  = s[-13:-4][::-1]  # bits 4-12 → D0-D8
    return data, z_syn


def logical_z(data_str, x_corrections):
    """Compute logical Z value (parity of D0,D1,D2 after X corrections)."""
    bits = [int(c) for c in data_str]
    for pos in x_corrections:
        bits[pos] ^= 1
    return (bits[0] + bits[1] + bits[2]) % 2


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Surface Code d=3 Z-only benchmark on IBM Kingston"
    )
    parser.add_argument("--token",            required=True,  help="IBM Quantum API token")
    parser.add_argument("--backend",          default="ibm_kingston")
    parser.add_argument("--shots",            type=int,   default=4096)
    parser.add_argument("--p_error",          type=float, default=0.10,
                        help="Bit-flip error probability (0.0–1.0)")
    parser.add_argument("--state",            type=int,   default=0, choices=[0, 1],
                        help="Logical state: 0=|0_L⟩, 1=|1_L⟩")
    parser.add_argument("--layout-time",      type=float, default=45.0,
                        help="Seconds to spend on layout search")
    parser.add_argument("--use-best-layout",  action="store_true",
                        help="Skip layout search, use the pre-found Kingston layout")
    parser.add_argument("--find-layout-only", action="store_true",
                        help="Run layout search only, do not submit to QPU")
    args = parser.parse_args()

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from noisebridge.correct import sparse_snn_correct

    print(f"\n{'='*70}")
    print(f"  NoiseBridge v0.3.0 — Surface Code d=3 Z-only benchmark")
    print(f"  Backend: {args.backend}  |  p_error={args.p_error:.0%}  |  shots={args.shots}")
    print(f"{'='*70}\n")

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=args.token)
    backend = service.backend(args.backend)
    print(f"  Connected: {backend.name}  ({backend.num_qubits} qubits)\n")

    # Layout
    if args.use_best_layout:
        layout = BEST_LAYOUT_KINGSTON
        native = 10
        print(f"  Using pre-found layout: D0-D8={layout[:9]}  AZ0-AZ3={layout[9:]}")
        print(f"  Native Z-edges: {native}/12\n")
    else:
        print(f"  Searching for optimal 13-qubit Z-only layout ({args.layout_time:.0f}s)...")
        layout, native = find_layout(backend, time_limit=args.layout_time)

    if args.find_layout_only:
        print("\n  [--find-layout-only] Done.")
        return

    # Build and transpile circuit
    qc, injected = build_circuit(p_error=args.p_error, logical_state=args.state)
    expected_syn = "".join(
        str(sum(1 for q in injected if q in stab) % 2) for stab in Z_STABS
    )
    print(f"  Injected errors: {injected}  →  expected syndrome: {expected_syn}")

    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        pm = generate_preset_pass_manager(
            optimization_level=2, backend=backend, initial_layout=layout
        )
        tc = pm.run(qc)
    except Exception:
        from qiskit import transpile
        tc = transpile(qc, backend=backend, optimization_level=2, initial_layout=layout)

    ops  = tc.count_ops()
    n_cz = ops.get("cz", 0) + ops.get("cx", 0) + ops.get("ecr", 0)
    n_sw = ops.get("swap", 0)
    print(f"  Transpiled: depth={tc.depth()}  2Q={n_cz}  SWAP={n_sw}\n")

    # Submit
    sampler = Sampler(backend)
    print(f"  Submitting {args.shots} shots...")
    job    = sampler.run([tc], shots=args.shots)
    print(f"  Job ID: {job.job_id()}")
    result = job.result()

    # Process results
    pub   = result[0]
    creg  = list(pub.data.keys())[0]
    counts = pub.data[creg].get_counts()
    total = sum(counts.values()) or 1

    # Aggregate Z-syndromes
    z_syn_counts = defaultdict(int)
    for s, cnt in counts.items():
        _, z_syn = extract_syndrome(s)
        z_syn_counts[z_syn] += cnt

    # SNN soft-decode
    snn_params = {"n_params": {4: {
        "beta": 0.5, "threshold": 0.050, "W_scale": 1.20, "snn_factor": 0.35, "T": 50
    }}}
    corrected_z = sparse_snn_correct(dict(z_syn_counts), snn_params, 4)

    top_z = sorted(z_syn_counts.items(), key=lambda x: -x[1])[:5]
    dom_syn  = top_z[0][0] if top_z else "N/A"
    dom_prob = top_z[0][1] / total if top_z else 0.0
    syn_ok   = dom_syn == expected_syn

    print(f"  Z-syndromes: " + "  ".join(f"{s}={c/total:.3f}" for s, c in top_z))
    print(f"  Expected: {expected_syn}  Dominant: {dom_syn}  "
          f"Match: {'✓' if syn_ok else '✗'}  ({dom_prob:.1%})\n")

    # Decode and compute LER
    decoder = build_lookup_decoder()
    err_raw = err_std = err_snn = 0

    for s, cnt in counts.items():
        data, z_syn = extract_syndrome(s)

        # Raw: no correction
        err_raw += cnt * logical_z(data, [])

        # Standard lookup decoder
        err_std += cnt * logical_z(data, decoder.get(z_syn, []))

        # SNN soft-decode: pick best Hamming-1 neighbour from corrected distribution
        best_z, best_p = z_syn, corrected_z.get(z_syn, 0.0)
        for cand, prob in corrected_z.items():
            if sum(a != b for a, b in zip(z_syn, cand)) <= 1 and prob > best_p:
                best_z, best_p = cand, prob
        err_snn += cnt * logical_z(data, decoder.get(best_z, []))

    ler_raw = err_raw / total
    ler_std = err_std / total
    ler_snn = err_snn / total
    snn_win = ler_snn < ler_std

    print(f"{'─'*60}")
    print(f"  {'Metric':<26} {'LER':>8}  {'vs RAW':>8}")
    print(f"{'─'*60}")
    print(f"  {'LER_raw  (no correction)':<26} {ler_raw:>8.4f}")
    print(f"  {'LER_std  (lookup decoder)':<26} {ler_std:>8.4f}  {ler_std-ler_raw:>+7.4f}")
    print(f"  {'LER_snn  (SNN+lookup)':<26} {ler_snn:>8.4f}  {ler_snn-ler_raw:>+7.4f}")
    print(f"{'─'*60}")

    if snn_win and ler_std > 0:
        ratio = ler_std / max(ler_snn, 1e-9)
        print(f"  SNN wins: {ratio:.2f}× improvement over standard decoder")
        if abs(ler_snn - ler_raw) < 1e-6:
            print(f"  *** ZERO OVERHEAD: LER_SNN = LER_raw ***")
    else:
        print(f"  Standard decoder wins at this error rate")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
