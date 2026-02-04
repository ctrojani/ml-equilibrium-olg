"""
Analytic benchmark for log utility case.

Goal: return equilibrium objects in a consistent dict format.
Later: compare ML outputs to this benchmark.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


def solve_log_benchmark(params: Any) -> Dict[str, float]:
    """
    Placeholder analytic solution for log benchmark.
    Replace the body with formulas from Ehling / professor notes.
    """
    # TODO: implement closed-form for r, phi, (maybe) risky share / p_over_y.
    out = {
        "r": float("nan"),
        "phi": float("nan"),
        "pi": float("nan"),
        "p_over_y": float("nan"),
    }
    return out
