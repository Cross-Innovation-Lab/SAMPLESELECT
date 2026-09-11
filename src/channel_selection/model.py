"""Three-stage model wrapper for sample-conditioned channel selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .losses import grouped_supervised_contrastive
from .selector import SampleConditionedChannelSelector


@dataclass
class SelectorLosses:
    total: torch.Tensor
    classification: torch.Tensor
    contrastive: torch.Tensor


class ChannelSelectionModel(nn.Module):
    """Backbone, source head, and sample-conditioned channel selector.

    The backbone must return a feature map with shape ``[B,C,H,W]``. Stage 1
    learns the backbone and source head. Stage 2 freezes those modules and
    learns only the selector. Stage 3 freezes the representation modules and
    fits a fresh episodic linear head; see :mod:`channel_selection.episodic`.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        channels: int,
        spatial_size: tuple[int, int],
        source_classes: int,
        keep_ratio: float = 0.7,
        temperature: float = 0.3,
        lambda_contrastive: float = 0.02,
        contrastive_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if source_classes <= 1:
            raise ValueError("source_classes must be greater than one")
        if lambda_contrastive < 0.0:
            raise ValueError("lambda_contrastive must be non-negative")

        self.backbone = backbone
        self.channels = int(channels)
        self.spatial_size = tuple(int(v) for v in spatial_size)
        self.embedding_dim = self.channels * self.spatial_size[0] * self.spatial_size[1]
        self.source_head = nn.Linear(self.embedding_dim, int(source_classes))
        self.selector = SampleConditionedChannelSelector(
            channels=self.channels,
            keep_ratio=keep_ratio,
            temperature=temperature,
        )
        self.lambda_contrastive = float(lambda_contrastive)
        self.contrastive_temperature = float(contrastive_temperature)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(x)
        expected = (self.channels, *self.spatial_size)
        if tuple(feature_map.shape[1:]) != expected:
            raise ValueError(
                f"backbone must return [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(feature_map.shape)}"
            )
        return feature_map

    @staticmethod
    def flatten(feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map.reshape(feature_map.shape[0], -1)

    def stage1_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Stage 1: source-domain representation learning."""
        return self.source_head(self.flatten(self.feature_map(x)))

    def set_stage1_trainable(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)
        for parameter in self.source_head.parameters():
            parameter.requires_grad_(True)
        for parameter in self.selector.parameters():
            parameter.requires_grad_(False)

    def set_stage2_trainable(self) -> None:
        """Freeze the base and selector BN; optimize only scorer Linear layers.

        The selector BatchNorm affine parameters and running statistics remain
        frozen during Stage 2. This matches the formal SelectorOnly protocol.
        """
        self.backbone.eval()
        self.source_head.eval()
        for module in (self.backbone, self.source_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
        for parameter in self.selector.parameters():
            parameter.requires_grad_(False)
        self.selector.train()
        for module in self.selector.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()
            elif isinstance(module, nn.Linear):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def stage2_parameters(self) -> list[nn.Parameter]:
        """Return the exact Stage 2 optimizer whitelist."""
        self.set_stage2_trainable()
        return [parameter for parameter in self.selector.parameters() if parameter.requires_grad]

    def selected_components(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        feature_map = self.feature_map(x)
        selected = self.selector(feature_map)
        selected_descriptor = selected.selected_map.mean(dim=(2, 3))
        embedding = self.flatten(selected.selected_map)
        return {
            "feature_map": feature_map,
            "descriptor": selected.descriptor,
            "scores": selected.scores,
            "mask": selected.mask,
            "indices": selected.indices,
            "selected_map": selected.selected_map,
            "selected_descriptor": selected_descriptor,
            "embedding": embedding,
        }

    def stage2_losses(
        self,
        x: torch.Tensor,
        source_labels: torch.Tensor,
        contrastive_labels: torch.Tensor,
        *,
        group_shape: tuple[int, int],
    ) -> SelectorLosses:
        """Stage 2 objective ``L_sel = L_CE + lambda * L_con``.

        ``contrastive_labels`` contains the foreground/class identity used for
        positive pairs. Samples in each contrastive group may have different
        backgrounds.
        """
        components = self.selected_components(x)
        logits = self.source_head(components["embedding"])
        loss_ce = F.cross_entropy(logits, source_labels)

        groups, samples_per_group = group_shape
        batch_size = groups * samples_per_group
        if x.shape[0] != batch_size or contrastive_labels.numel() != batch_size:
            raise ValueError("batch size does not match group_shape")
        grouped_features = components["selected_descriptor"].reshape(
            groups, samples_per_group, self.channels
        )
        grouped_labels = contrastive_labels.reshape(groups, samples_per_group)
        loss_con = grouped_supervised_contrastive(
            grouped_features,
            grouped_labels,
            temperature=self.contrastive_temperature,
        )
        total = loss_ce + self.lambda_contrastive * loss_con
        return SelectorLosses(total, loss_ce, loss_con)

    def freeze_representation(self) -> None:
        """Freeze backbone and selector before few-shot evaluation."""
        self.eval()
        for module in (self.backbone, self.source_head, self.selector):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None

    @torch.no_grad()
    def episodic_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Stage 3 embedding; the source-domain head is intentionally bypassed."""
        return self.selected_components(x)["embedding"]
