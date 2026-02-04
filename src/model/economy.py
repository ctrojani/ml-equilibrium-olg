from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Any

from .parameters import Params


def simulate_output(params: Params) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Simulate aggregate output Y_t via geometric Brownian motion:
        dY/Y = mu_y dt + sigma_y dW
    """
    rng = np.random.default_rng(params.seed)
    n = int(params.T / params.dt) + 1
    t = np.linspace(0.0, params.T, n)

    dW = rng.standard_normal(size=n - 1) * np.sqrt(params.dt)
    logY = np.zeros(n)
    for k in range(1, n):
        logY[k] = logY[k - 1] + (params.mu_y - 0.5 * params.sigma_y**2) * params.dt + params.sigma_y * dW[k - 1]

    Y = np.exp(logY)

    extras = {
        "T": params.T,
        "dt": params.dt,
        "mean_log_growth": float(np.mean(np.diff(logY) / params.dt)),
        "std_log_growth": float(np.std(np.diff(logY) / np.sqrt(params.dt))),
    }
    return t, Y, extras
