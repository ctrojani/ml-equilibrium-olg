"""
Analytic benchmark for log utility case.

Goal: return equilibrium objects in a consistent dict format.
Later: compare ML outputs to this benchmark.
"""

from __future__ import annotations
from typing import Dict, Any

from src.model.log_analytic import solve_log


def solve_log_benchmark(params: Any) -> Dict[str, float]:
    """
    Analytic solution for log benchmark.

    Returns a dict with keys:
    - r, phi, pi, p_over_y
    """
    sol = solve_log(params)
    return {
        "r": float(sol.r),
        "phi": float(sol.phi),
        "pi": float(sol.pi),
        "p_over_y": float(sol.p_over_y),
    }
