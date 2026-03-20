from __future__ import annotations

import torch


def policy_loss(consumption_pred, risky_pred, consumption_true, risky_true):
    loss_c = torch.mean((consumption_pred - consumption_true) ** 2)
    loss_pi = torch.mean((risky_pred - risky_true) ** 2)
    return loss_c + loss_pi


def _safe_scale(target: torch.Tensor) -> torch.Tensor:
    scale = torch.amax(torch.abs(target.detach()))
    return torch.clamp(scale, min=1.0)


def structural_log_loss(
    wealth,
    consumption_pred,
    risky_pred,
    c_rate_pred,
    risky_share_pred,
    c_coeff,
    pi_coeff,
    lambda_levels=1.0,
    lambda_rates=1.0,
):
    consumption_true = c_coeff * wealth
    risky_true = pi_coeff * wealth

    c_rate_true = torch.full_like(c_rate_pred, c_coeff)
    risky_share_true = torch.full_like(risky_share_pred, pi_coeff)

    loss_c_level = torch.mean(
        ((consumption_pred - consumption_true) / _safe_scale(consumption_true)) ** 2
    )
    loss_pi_level = torch.mean(
        ((risky_pred - risky_true) / _safe_scale(risky_true)) ** 2
    )
    loss_c_rate = torch.mean((c_rate_pred - c_rate_true) ** 2)
    loss_pi_rate = torch.mean((risky_share_pred - risky_share_true) ** 2)

    total_loss = (
        lambda_levels * (loss_c_level + loss_pi_level)
        + lambda_rates * (loss_c_rate + loss_pi_rate)
    )

    return total_loss, {
        "loss_total": float(total_loss.detach().item()),
        "loss_c_level": float(loss_c_level.detach().item()),
        "loss_pi_level": float(loss_pi_level.detach().item()),
        "loss_c_rate": float(loss_c_rate.detach().item()),
        "loss_pi_rate": float(loss_pi_rate.detach().item()),
    }
