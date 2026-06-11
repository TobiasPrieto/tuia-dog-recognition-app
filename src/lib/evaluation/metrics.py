"""Funciones auxiliares para evaluacion (provistas por la catedra).

Cubren las metricas pedidas en el TP:
  - Etapa 1: NDCG@10 (ndcg_at_k).
  - Etapa 2: precision / recall / F1 / specificity.
"""
from __future__ import annotations

import math
from typing import Sequence


def ndcg_at_k(relevances: Sequence[float], k: int = 10) -> float:
    """NDCG@k de un ranking.

    `relevances` son las relevancias en el orden devuelto por la busqueda
    (ej: 1.0 si el vecino es de la misma raza que la consulta, 0.0 si no).
    """
    rel = [float(r) for r in list(relevances)[:k]]
    if not rel:
        return 0.0
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(rel, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Retorna (precision, recall, f1) a partir de conteos TP/FP/FN."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def specificity(tn: int, fp: int) -> float:
    """Especificidad: TN / (TN + FP)."""
    return tn / (tn + fp) if (tn + fp) else 0.0
