from __future__ import annotations

from dataclasses import dataclass
import math

from .parameters import Params


@dataclass(frozen=True)
class LogSolution:
    beta: float
    r: float
    theta: float
    phi: float
    p_over_y: float
    c_coeff: float
    pi: float
    mu_s: float
    sigma_s: float

    @property
    def pi_coeff(self) -> float:
        return self.pi

    def consumption(self, wealth):
        return self.c_coeff * wealth

    def risky_position(self, wealth):
        return self.pi * wealth


def solve_log(params: Params) -> LogSolution:
    rho = params.rho
    nu = params.nu
    omega = params.omega
    mu_y = params.mu_y
    sigma_y = params.sigma_y
    sigma_s = params.sigma_s

    if rho <= 0.0:
        raise ValueError("rho must be positive")
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    if sigma_y <= 0.0:
        raise ValueError("sigma_y must be positive")
    if sigma_s <= 0.0:
        raise ValueError("sigma_s must be positive")
    if not 0.0 <= omega <= 1.0:
        raise ValueError("omega must be in [0, 1]")

    disc = rho**2 + 4.0 * nu * (rho + nu) * (1.0 - omega)
    beta = (rho + 2.0 * nu - math.sqrt(disc)) / (2.0 * nu)

    phi = 1.0 / (rho + nu)
    theta = sigma_y
    r = rho + mu_y - sigma_y**2 + nu * (1.0 - beta)

    # easy benchmark convention:
    # choose mu_s consistently from theta = (mu_s - r)/sigma_s
    if params.mu_s is None:
        mu_s = r + theta * sigma_s
    else:
        mu_s = params.mu_s

    c_coeff = rho + nu
    pi = (mu_s - r) / (sigma_s**2)
    p_over_y = phi

    return LogSolution(
        beta=beta,
        r=r,
        theta=theta,
        phi=phi,
        p_over_y=p_over_y,
        c_coeff=c_coeff,
        pi=pi,
        mu_s=mu_s,
        sigma_s=sigma_s,
    )
