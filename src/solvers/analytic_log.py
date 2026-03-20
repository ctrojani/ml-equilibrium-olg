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
    - beta, r, theta, phi, p_over_y, c_coeff, pi
    """
    sol = solve_log(params)
    return {
        "beta": float(sol.beta),
        "r": float(sol.r),
        "theta": float(sol.theta),
        "phi": float(sol.phi),
        "p_over_y": float(sol.p_over_y),
        "c_coeff": float(sol.c_coeff),
        "pi": float(sol.pi),
        "pi_coeff": float(sol.pi_coeff),
        "mu_s": float(sol.mu_s),
        "sigma_s": float(sol.sigma_s),
    }
