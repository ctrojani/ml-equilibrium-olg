from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LogSolution:
    r: float         # risk-free rate
    phi: float       # wealth-to-consumption ratio
    pi: float        # portfolio share (in dollars / wealth if normalized)
    p_over_y: float  # price-dividend/output ratio


def solve_log(params) -> LogSolution:
    """
    Closed-form (log utility) benchmark used ONLY for numerical validation.

    Notes:
    - We keep this deliberately "small": a stable reference map params -> objects.
    - If later you decide the correct closed forms differ (once aligned with Ehling/notes),
      you update them here, and everything (tests + training) updates consistently.

    Current benchmark (placeholder-but-consistent):
    - r = rho + nu
    - phi = 1 / (rho + nu)
    - pi = mu_y / sigma_y^2
    - p/Y = (1-omega) * phi
    """
    mu_y = float(params.mu_y)
    sigma_y = float(params.sigma_y)
    rho = float(params.rho)
    nu = float(params.nu)
    omega = float(params.omega)

    if sigma_y <= 0:
        raise ValueError("sigma_y must be > 0")
    if rho + nu <= 0:
        raise ValueError("rho + nu must be > 0")

    r = rho + nu
    phi = 1.0 / (rho + nu)
    pi = mu_y / (sigma_y * sigma_y)
    p_over_y = (1.0 - omega) * phi

    return LogSolution(r=r, phi=phi, pi=pi, p_over_y=p_over_y)
