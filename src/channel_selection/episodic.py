"""Fresh-head few-shot evaluation with frozen representation modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .model import ChannelSelectionModel


@dataclass
class EpisodicResult:
    logits: torch.Tensor
    predictions: torch.Tensor


def evaluate_episode(
    model: ChannelSelectionModel,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    *,
    ways: int = 5,
    steps: int = 100,
    learning_rate: float = 0.01,
    inner_batch_size: int = 4,
    head_seed: int | None = None,
) -> EpisodicResult:
    """Fit a fresh linear head on support data and predict query labels.

    Query labels are deliberately absent from this API. The source-domain head
    is not used. Only the newly created episodic head is optimized.
    """
    if ways <= 1 or steps <= 0 or learning_rate <= 0.0 or inner_batch_size <= 0:
        raise ValueError("invalid episodic hyperparameters")
    if support_y.ndim != 1 or support_y.shape[0] != support_x.shape[0]:
        raise ValueError("support labels must have shape [num_support]")
    if support_y.min().item() < 0 or support_y.max().item() >= ways:
        raise ValueError("support labels must be in [0, ways)")

    model.freeze_representation()
    support_features = model.episodic_embedding(support_x)
    query_features = model.episodic_embedding(query_x)

    if head_seed is None:
        head = nn.Linear(model.embedding_dim, ways).to(support_features.device)
    else:
        devices = [support_features.device] if support_features.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(head_seed)
            head = nn.Linear(model.embedding_dim, ways).to(support_features.device)
    optimizer = torch.optim.SGD(
        head.parameters(),
        lr=learning_rate,
        momentum=0.9,
        dampening=0.9,
        weight_decay=0.001,
    )
    head.train()
    support_count = support_y.shape[0]
    for step in range(steps):
        start = (step * inner_batch_size) % support_count
        stop = min(support_count, start + inner_batch_size)
        indices = torch.arange(start, stop, device=support_features.device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(support_features[indices]), support_y[indices])
        loss.backward()
        optimizer.step()

    head.eval()
    with torch.no_grad():
        logits = head(query_features)
    return EpisodicResult(logits=logits, predictions=logits.argmax(dim=1))
