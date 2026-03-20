from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    mu_y: float = 0.02
    sigma_y: float = 0.15
    rho: float = 0.03
    nu: float = 0.02
    omega: float = 0.50

    # Risky asset coefficients for the log benchmark
    sigma_s: float = 0.15
    mu_s: float | None = None

    # Wealth grid for training
    w_min: float = 0.05
    w_max: float = 5.00
    n_train: int = 512
    n_test: int = 256

    # NN hyperparameters
    hidden_dim: int = 64
    num_layers: int = 3
    lr: float = 1e-3
    epochs: int = 2000
    batch_size: int = 128
    print_every: int = 200
    lambda_levels: float = 1.0
    lambda_rates: float = 1.0

    # Simulation controls used by the economy helpers
    T: float = 1.0
    dt: float = 1 / 252

    seed: int = 0
