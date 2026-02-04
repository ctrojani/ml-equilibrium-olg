from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # Output process
    mu_y: float = 0.02
    sigma_y: float = 0.15

    # Demographics
    nu: float = 0.02  # death/birth rate

    # Preferences (log utility case uses rho primarily)
    rho: float = 0.03

    # Endowment share
    omega: float = 0.5

    # Simulation controls (for sanity checks)
    T: float = 1.0
    dt: float = 1 / 252
    seed: int = 0
