"""Sample-conditioned channel scoring and sequential Gumbel Top-k selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SelectorOutput:
    """Outputs of the channel selector.

    The training mask is a soft vector in ``[0, 1]^C``. During evaluation the
    mask is deterministic and binary.
    """

    selected_map: torch.Tensor
    descriptor: torch.Tensor
    scores: torch.Tensor
    mask: torch.Tensor
    indices: torch.Tensor | None


class SampleConditionedChannelSelector(nn.Module):
    """Select a fixed fraction of channels independently for every sample.

    Args:
        channels: Number of feature-map channels ``C``.
        keep_ratio: Fraction of channels selected by Top-k.
        temperature: Gumbel-Softmax temperature used during training.
        hidden_multiplier: Width multiplier of the MLP scorer.
        dropout: Dropout probability in the scorer.
        detach_descriptor: Preserve the two-stage protocol by preventing the
            selector objective from updating the feature extractor through the
            scorer input. The feature extractor should also be frozen explicitly.
    """

    def __init__(
        self,
        channels: int,
        keep_ratio: float = 0.7,
        temperature: float = 0.3,
        *,
        hidden_multiplier: int = 4,
        dropout: float = 0.5,
        detach_descriptor: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        self.channels = int(channels)
        self.keep_ratio = float(keep_ratio)
        self.temperature = float(temperature)
        self.k = max(1, int(self.channels * self.keep_ratio))
        self.detach_descriptor = bool(detach_descriptor)

        hidden = self.channels * int(hidden_multiplier)
        self.scorer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.channels, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.channels),
        )

    def _soft_topk(self, scores: torch.Tensor) -> torch.Tensor:
        """Sequential soft Top-k sampling without replacement."""
        remaining = scores
        mask = torch.zeros_like(scores)
        for _ in range(self.k):
            draw = F.gumbel_softmax(
                remaining,
                tau=self.temperature,
                hard=False,
                dim=1,
            )
            mask = torch.maximum(mask, draw)
            chosen = draw.argmax(dim=1, keepdim=True)
            remaining = remaining.scatter(1, chosen, -torch.inf)
        return mask

    def _hard_topk(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices = scores.topk(self.k, dim=1).indices
        mask = torch.zeros_like(scores).scatter(1, indices, 1.0)
        return mask, indices

    def forward(self, feature_map: torch.Tensor) -> SelectorOutput:
        if feature_map.ndim != 4 or feature_map.shape[1] != self.channels:
            raise ValueError(
                f"expected feature map [B,{self.channels},H,W], "
                f"got {tuple(feature_map.shape)}"
            )

        descriptor = feature_map.mean(dim=(2, 3))
        scorer_input = descriptor.detach() if self.detach_descriptor else descriptor
        scores = self.scorer(scorer_input)
        if self.training:
            mask = self._soft_topk(scores)
            indices = None
        else:
            mask, indices = self._hard_topk(scores)
        selected_map = feature_map * mask[:, :, None, None]
        return SelectorOutput(selected_map, descriptor, scores, mask, indices)
