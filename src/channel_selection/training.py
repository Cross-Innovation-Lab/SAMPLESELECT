"""Minimal optimization helpers for Stages 1 and 2."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import ChannelSelectionModel, SelectorLosses


def train_stage1_step(
    model: ChannelSelectionModel,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    source_labels: torch.Tensor,
) -> torch.Tensor:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model.stage1_logits(inputs), source_labels)
    loss.backward()
    optimizer.step()
    return loss.detach()


def train_selector_step(
    model: ChannelSelectionModel,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    source_labels: torch.Tensor,
    contrastive_labels: torch.Tensor,
    *,
    group_shape: tuple[int, int],
) -> SelectorLosses:
    model.set_stage2_trainable()
    optimizer.zero_grad(set_to_none=True)
    losses = model.stage2_losses(
        inputs,
        source_labels,
        contrastive_labels,
        group_shape=group_shape,
    )
    losses.total.backward()
    optimizer.step()
    return SelectorLosses(*(value.detach() for value in losses.__dict__.values()))
