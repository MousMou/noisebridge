"""
noisebridge.qaoa_correct
========================
SNN adaptada para circuitos QAOA/VQE con output de distribución ancha.

El SNN centering matrix estándar asume output esparso (pocos estados dominantes).
Para QAOA MaxCut el output ideal tiene ~2^(n-1) estados con probabilidades similares.
Este módulo implementa tres estrategias complementarias:

Estrategia 1 — Suavizado suave (smooth_correct)
    En lugar de amplificar/suprimir agresivamente, suaviza el ruido de medición
    preservando la estructura multimodo. Usa W_scale reducido y snn_factor bajo.

Estrategia 2 — SNN consciente del problema (problem_aware_correct)
    La corrección usa las puntuaciones del problema (corte MaxCut, energía VQE)
    como señal: amplifica estados con score > media, suprime estados con score < media.
    No necesita entrenamiento — usa el Hamiltoniano directamente como peso.

Estrategia 3 — Matriz de pesos basada en el grafo (graph_weight_correct)
    W[i,j] refleja la estructura del grafo: si los bitstrings i y j difieren
    en un número de aristas mayor al promedio, W[i,j] > 0 (se refuerzan).
    Solo factible en forma esparsa sobre los estados observados.

Referencia: DeepSeek proposal — "hacer que la SNN funcione en QAOA" (2026-05-13)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional


# ── Utilidades de scoring ─────────────────────────────────────────────────────

def maxcut_score(bits: str, edges: List[Tuple[int, int]]) -> int:
    """Número de aristas cortadas por el bitstring `bits`."""
    return sum(1 for i, j in edges
               if i < len(bits) and j < len(bits) and bits[i] != bits[j])


def maxcut_ratio(bits: str, edges: List[Tuple[int, int]]) -> float:
    """Corte normalizado ∈ [0, 1]."""
    return maxcut_score(bits, edges) / len(edges) if edges else 0.0


# ── Estrategia 1: Suavizado suave ─────────────────────────────────────────────

def smooth_correct(
    counts: Dict[str, int],
    n: int,
    smoothing: float = 0.15,
    T: int = 20,
) -> Dict[str, float]:
    """
    Suavizado de ruido de medición preservando distribución multimodo.

    En vez de amplificar/suprimir agresivamente (como en SNN-NR v2),
    aplica un filtro suave que sube las probabilidades ligeramente por debajo
    de la media y baja las que están muy por encima — reduciendo fluctuaciones
    sin colapsar la distribución a pocos estados.

    Parámetros
    ----------
    counts    : dict conteos raw
    n         : número de qubits
    smoothing : intensidad del suavizado ∈ [0, 1] (default 0.15)
    T         : pasos de difusión (default 20)

    Returns
    -------
    dict : distribución suavizada normalizada (solo estados observados)
    """
    total = sum(counts.values())
    if total == 0:
        return {}

    keys = list(counts.keys())
    x    = np.array([counts[k] / total for k in keys], dtype=float)
    mean = x.mean()

    # Filtro suave: mueve cada probabilidad hacia la media según smoothing
    # x_smooth[i] = x[i] - smoothing * (x[i] - mean)
    #             = (1 - smoothing) * x[i] + smoothing * mean
    x_smooth = (1.0 - smoothing) * x + smoothing * mean

    # Iteraciones de difusión para propagación local (opcional)
    for _ in range(T):
        global_mean = x_smooth.mean()
        x_smooth = (1.0 - smoothing) * x_smooth + smoothing * global_mean
        x_smooth = np.clip(x_smooth, 0.0, None)

    total_s = x_smooth.sum()
    if total_s < 1e-10:
        return {k: float(v / total) for k, v in zip(keys, x)}

    x_smooth /= total_s
    return {k: float(v) for k, v in zip(keys, x_smooth)}


# ── Estrategia 2: SNN consciente del problema (problem-aware) ─────────────────

def problem_aware_correct(
    counts: Dict[str, int],
    score_fn: Callable[[str], float],
    snn_factor: float = 0.40,
    T: int = 30,
    beta: float = 0.5,
) -> Dict[str, float]:
    """
    SNN LIF con pesos determinados por la función objetivo del problema.

    En vez del centering matrix (W[i,j] = delta(i,j) - 1/N),
    la corriente de entrada al neurón i es:

        current[i] = W_score * (score[i] - mean_score)

    donde score[i] es el valor de la función objetivo para el bitstring i
    (MaxCut ratio, energía VQE negativa, etc.).

    Los estados con score > media reciben corriente positiva → spikes → amplificados.
    Los estados con score < media reciben corriente negativa → suprimidos.

    Parámetros
    ----------
    counts     : dict conteos raw
    score_fn   : función que recibe un bitstring y devuelve su puntuación (mayor = mejor)
    snn_factor : mezcla SNN/raw ∈ [0, 1]
    T          : pasos LIF
    beta       : decaimiento de membrana

    Returns
    -------
    dict : distribución corregida normalizada
    """
    total = sum(counts.values())
    if total == 0:
        return {}

    keys   = list(counts.keys())
    x      = np.array([counts[k] / total for k in keys], dtype=float)
    scores = np.array([score_fn(k) for k in keys], dtype=float)

    # Normalizar scores a [0, 1]
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-10:
        # Todos los estados tienen el mismo score → sin señal
        return {k: float(v) for k, v in zip(keys, x)}

    scores_norm = (scores - s_min) / (s_max - s_min)
    mean_score  = scores_norm.mean()

    # Dinámica LIF: la corriente es proporcional al exceso de score sobre la media
    threshold   = 0.05
    W_score     = 1.2   # escala de la corriente basada en el score
    membrane    = np.zeros(len(x))
    accumulator = np.zeros(len(x))

    for _ in range(T):
        current   = W_score * (scores_norm - mean_score)  # ∈ (-1.2, 1.2)
        membrane  = beta * membrane + current
        spikes    = (membrane >= threshold).astype(float)
        membrane *= 1.0 - spikes
        accumulator += spikes

    snn_out  = accumulator / T
    combined = snn_factor * snn_out + (1.0 - snn_factor) * x
    combined = np.clip(combined, 0.0, None)

    total_c = combined.sum()
    if total_c < 1e-10:
        return {k: float(v) for k, v in zip(keys, x)}

    combined /= total_c
    return {k: float(v) for k, v in zip(keys, combined)}


# ── Estrategia 3: Matriz de pesos basada en el grafo ─────────────────────────

def graph_weight_correct(
    counts: Dict[str, int],
    edges: List[Tuple[int, int]],
    n: int,
    snn_factor: float = 0.45,
    beta: float = 0.5,
    T: int = 30,
) -> Dict[str, float]:
    """
    SNN con matriz de pesos codificando la estructura del grafo MaxCut.

    Para cada par de bitstrings observados (i, j), el peso W[i,j] refleja
    si i "mejora" a j en términos de corte:

        W[i,j] = (score[i] - score[j]) / max_score

    La corriente al neurón i es:
        current[i] = sum_j W[i,j] * x[j]
                   = sum_j (score[i] - score[j]) * x[j] / max_score
                   = score[i] - E_x[score]   (exceso de score sobre la media ponderada)

    Es decir: amplifica estados cuyo corte supera el corte medio de la distribución.
    Esparso: solo opera sobre los estados observados (O(shots^2) no O(2^n)^2).

    Parámetros
    ----------
    counts     : conteos raw
    edges      : lista de aristas del grafo MaxCut
    n          : número de qubits
    snn_factor : mezcla SNN/raw
    beta       : decaimiento membrana LIF
    T          : pasos LIF

    Returns
    -------
    dict : distribución corregida normalizada
    """
    total = sum(counts.values())
    if total == 0:
        return {}

    keys   = list(counts.keys())
    x      = np.array([counts[k] / total for k in keys], dtype=float)
    scores = np.array([maxcut_ratio(k, edges) for k in keys], dtype=float)

    max_score = scores.max() if scores.max() > 0 else 1.0

    # current[i] = score[i] - E_x[score] (exceso sobre la media ponderada de la dist.)
    # Esto es equivalente a W@x donde W[i,j] = (score[i]-score[j])/max_score
    # pero calculado directamente en O(states) no O(states^2)
    threshold  = 0.01
    membrane   = np.zeros(len(x))
    accumulator = np.zeros(len(x))

    for _ in range(T):
        weighted_mean_score = np.dot(scores, x)  # E_x[score] de la distribución actual
        current   = (scores - weighted_mean_score) / max_score
        membrane  = beta * membrane + current
        spikes    = (membrane >= threshold).astype(float)
        membrane *= 1.0 - spikes
        accumulator += spikes

    snn_out  = accumulator / T
    combined = snn_factor * snn_out + (1.0 - snn_factor) * x
    combined = np.clip(combined, 0.0, None)

    total_c = combined.sum()
    if total_c < 1e-10:
        return {k: float(v) for k, v in zip(keys, x)}

    combined /= total_c
    return {k: float(v) for k, v in zip(keys, combined)}


# ── Evaluación comparativa ─────────────────────────────────────────────────────

def evaluate_qaoa_corrections(
    counts: Dict[str, int],
    edges: List[Tuple[int, int]],
    n: int,
    snn_params: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """
    Evalúa las tres estrategias + SNN estándar sobre una distribución QAOA.

    Returns dict con claves: raw, smooth, problem_aware, graph_weight, standard_snn
    Cada valor: {"avg_ratio": float, "best_ratio": float, "top10_avg": float}
    """
    from noisebridge.correct import sparse_snn_correct

    def metrics(dist: Dict[str, float]) -> Dict:
        if not dist:
            return {"avg_ratio": 0.0, "best_ratio": 0.0, "top10_avg": 0.0}
        total = sum(dist.values())
        avg   = sum(p * maxcut_ratio(bs, edges) for bs, p in dist.items()) / total
        best  = max(maxcut_ratio(bs, edges) for bs in dist)
        top10 = sorted(dist.items(), key=lambda x: -x[1])[:10]
        t10a  = sum(maxcut_ratio(bs, edges) for bs, _ in top10) / len(top10)
        return {"avg_ratio": round(avg, 5), "best_ratio": round(best, 5),
                "top10_avg": round(t10a, 5)}

    total  = sum(counts.values()) or 1
    raw_d  = {bs: cnt / total for bs, cnt in counts.items()}
    score_fn = lambda bs: maxcut_ratio(bs, edges)

    results = {
        "raw":           metrics(raw_d),
        "smooth":        metrics(smooth_correct(counts, n, smoothing=0.20, T=15)),
        "problem_aware": metrics(problem_aware_correct(counts, score_fn)),
        "graph_weight":  metrics(graph_weight_correct(counts, edges, n)),
    }

    if snn_params:
        results["standard_snn"] = metrics(
            dict(zip(raw_d.keys(),
                     [sparse_snn_correct(counts, snn_params, n).get(k, 0.0)
                      for k in raw_d.keys()]))
        )

    return results
