"""Losses used to train the channel selector."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def grouped_supervised_contrastive(
    features: torch.Tensor,
    class_labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss evaluated independently within each group.

    Args:
        features: Tensor with shape ``[groups, samples_per_group, dim]``.
        class_labels: Foreground class labels with shape
            ``[groups, samples_per_group]``. Each anchor must have at least one
            positive with the same label in its group.
        temperature: Contrastive temperature.
    """
    if features.ndim != 3:
        raise ValueError("features must have shape [G,M,D]")
    if class_labels.shape != features.shape[:2]:
        raise ValueError("class_labels must have shape [G,M]")
    if features.shape[-1] == 0 or temperature <= 0.0:
        raise ValueError("invalid feature dimension or temperature")
    if not torch.isfinite(features).all():
        raise ValueError("features contain NaN or Inf")

    z = F.normalize(features, dim=2)
    logits = torch.bmm(z, z.transpose(1, 2)) / temperature
    sample_count = features.shape[1]
    eye = torch.eye(sample_count, dtype=torch.bool, device=features.device)[None]
    same_class = class_labels[:, :, None].eq(class_labels[:, None, :])
    positives = same_class & ~eye
    negatives = ~same_class
    if not positives.any(dim=2).all():
        raise ValueError("every anchor needs at least one positive in its group")

    allowed = positives | negatives
    log_prob = logits.masked_fill(~allowed, -torch.inf)
    log_prob = log_prob - torch.logsumexp(log_prob, dim=2, keepdim=True)
    per_anchor = -(
        log_prob.masked_fill(~positives, 0.0).sum(dim=2)
        / positives.sum(dim=2)
    )
    return per_anchor.mean()
