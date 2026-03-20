from __future__ import annotations

from typing import Any

import torch


def train_policy(
    model,
    optimizer,
    loss_fn,
    w_train,
    epochs=2000,
    batch_size=128,
    print_every=200,
    **loss_kwargs: Any,
):
    history = []
    n_train = w_train.shape[0]

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_train, device=w_train.device)

        epoch_metrics = {}
        n_batches = 0

        for start in range(0, n_train, batch_size):
            batch_idx = permutation[start : start + batch_size]
            wealth_batch = w_train[batch_idx]

            optimizer.zero_grad()
            outputs = model(wealth_batch)
            loss, metrics = loss_fn(wealth_batch, *outputs, **loss_kwargs)
            loss.backward()
            optimizer.step()

            for key, value in metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0.0) + float(value)
            n_batches += 1

        averaged_metrics = {
            key: value / max(n_batches, 1) for key, value in epoch_metrics.items()
        }
        history.append(averaged_metrics)

        if (
            epoch == 0
            or (epoch + 1) % print_every == 0
            or epoch == epochs - 1
        ):
            print(
                f"Epoch {epoch + 1}: "
                f"loss = {averaged_metrics['loss_total']:.8f}, "
                f"c_rate = {averaged_metrics['loss_c_rate']:.8f}, "
                f"pi_rate = {averaged_metrics['loss_pi_rate']:.8f}"
            )

    return history
