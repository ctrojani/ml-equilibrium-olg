from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from src.ml.losses import structural_log_loss
from src.ml.networks import LogPolicyNet
from src.ml.trainer import train_policy
from src.model.log_analytic import solve_log
from src.model.parameters import Params


def test_solve_log_returns_consistent_benchmark_objects():
    params = Params()
    solution = solve_log(params)

    assert solution.beta < 1.0
    assert solution.r > 0.0
    assert solution.phi == pytest.approx(1.0 / (params.rho + params.nu))
    assert solution.p_over_y == pytest.approx(solution.phi)
    assert solution.pi == pytest.approx(solution.pi_coeff)


def test_policy_net_learns_log_benchmark_policy():
    params = replace(
        Params(),
        n_train=128,
        hidden_dim=32,
        num_layers=2,
        lr=5e-3,
        epochs=400,
        batch_size=64,
        print_every=1000,
    )
    solution = solve_log(params)

    torch.manual_seed(params.seed)
    wealth = torch.linspace(params.w_min, params.w_max, params.n_train).reshape(-1, 1)

    model = LogPolicyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    history = train_policy(
        model=model,
        optimizer=optimizer,
        loss_fn=structural_log_loss,
        w_train=wealth,
        epochs=params.epochs,
        batch_size=params.batch_size,
        print_every=params.print_every,
        c_coeff=solution.c_coeff,
        pi_coeff=solution.pi_coeff,
    )

    assert history[-1]["loss_total"] < 1e-4

    model.eval()
    with torch.no_grad():
        consumption_pred, risky_pred, _, _ = model(wealth)

    consumption_true = solution.consumption(wealth)
    risky_true = solution.risky_position(wealth)

    assert torch.max(torch.abs(consumption_pred - consumption_true)).item() < 2e-2
    assert torch.max(torch.abs(risky_pred - risky_true)).item() < 2e-2
