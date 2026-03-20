import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, ".cache", "matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model.parameters import Params
from src.model.log_analytic import solve_log
from src.ml.networks import LogPolicyNet
from src.ml.losses import structural_log_loss
from src.ml.trainer import train_policy


def evaluate_policy(model, wealth, solution):
    model.eval()
    with torch.no_grad():
        consumption_pred, risky_pred, c_rate_pred, risky_share_pred = model(wealth)

    consumption_true = solution.consumption(wealth)
    risky_true = solution.risky_position(wealth)

    metrics = {
        "consumption_mae": float(torch.mean(torch.abs(consumption_pred - consumption_true)).item()),
        "consumption_max_abs": float(torch.max(torch.abs(consumption_pred - consumption_true)).item()),
        "risky_mae": float(torch.mean(torch.abs(risky_pred - risky_true)).item()),
        "risky_max_abs": float(torch.max(torch.abs(risky_pred - risky_true)).item()),
        "c_rate_mae": float(torch.mean(torch.abs(c_rate_pred - solution.c_coeff)).item()),
        "c_rate_max_abs": float(torch.max(torch.abs(c_rate_pred - solution.c_coeff)).item()),
        "risky_share_mae": float(torch.mean(torch.abs(risky_share_pred - solution.pi_coeff)).item()),
        "risky_share_max_abs": float(torch.max(torch.abs(risky_share_pred - solution.pi_coeff)).item()),
    }
    return metrics, consumption_true, risky_true, consumption_pred, risky_pred


def main(params: Params | None = None):
    params = params or Params()
    solution = solve_log(params)

    torch.manual_seed(params.seed)
    wealth_train = torch.linspace(params.w_min, params.w_max, params.n_train).reshape(-1, 1)
    wealth_test = torch.linspace(params.w_min, params.w_max, params.n_test).reshape(-1, 1)

    model = LogPolicyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    history = train_policy(
        model=model,
        optimizer=optimizer,
        loss_fn=structural_log_loss,
        w_train=wealth_train,
        epochs=params.epochs,
        batch_size=params.batch_size,
        print_every=params.print_every,
        c_coeff=solution.c_coeff,
        pi_coeff=solution.pi_coeff,
        lambda_levels=params.lambda_levels,
        lambda_rates=params.lambda_rates,
    )

    metrics, consumption_true, risky_true, consumption_pred, risky_pred = evaluate_policy(
        model=model,
        wealth=wealth_test,
        solution=solution,
    )

    for key, value in metrics.items():
        print(f"{key:>20s}: {value:.8f}")

    root = Path(PROJECT_ROOT)
    plt.figure()
    plt.plot(wealth_test.numpy(), consumption_true.numpy(), label="true c(W)")
    plt.plot(wealth_test.numpy(), consumption_pred.numpy(), "--", label="predicted c(W)")
    plt.legend()
    plt.title("Consumption policy")
    plt.savefig(root / "plots_log_c.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(wealth_test.numpy(), risky_true.numpy(), label="true pi(W)")
    plt.plot(wealth_test.numpy(), risky_pred.numpy(), "--", label="predicted pi(W)")
    plt.legend()
    plt.title("Portfolio policy")
    plt.savefig(root / "plots_log_pi.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot([entry["loss_total"] for entry in history])
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(root / "plots_log_loss.png", dpi=150)
    plt.close()

    torch.save(model.state_dict(), root / "log_policy_model.pt")

    payload = {
        "params": asdict(params),
        "solution": asdict(solution),
        "metrics": metrics,
        "history": history,
    }
    with open(root / "log_policy_history.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("Training finished.")
    print("Saved plots: plots_log_c.png, plots_log_pi.png, plots_log_loss.png")
    print("Saved model: log_policy_model.pt")
    print("Saved history: log_policy_history.json")
    return payload


if __name__ == "__main__":
    main()
