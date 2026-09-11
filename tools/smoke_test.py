"""Run a small CPU-only Stage 1 -> Stage 2 -> Stage 3 sanity check."""

from __future__ import annotations

import torch

from channel_selection import ChannelSelectionModel, evaluate_episode
from channel_selection.reference_backbone import ReferenceBackbone
from channel_selection.training import train_selector_step, train_stage1_step


def main() -> None:
    torch.manual_seed(0)
    model = ChannelSelectionModel(
        ReferenceBackbone(channels=8, spatial_size=(2, 3)),
        channels=8,
        spatial_size=(2, 3),
        source_classes=4,
        keep_ratio=0.7,
        temperature=0.3,
    )

    model.set_stage1_trainable()
    stage1_optimizer = torch.optim.SGD(
        list(model.backbone.parameters()) + list(model.source_head.parameters()),
        lr=1e-2,
    )
    stage1_loss = train_stage1_step(
        model,
        stage1_optimizer,
        torch.randn(8, 1, 12, 12),
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
    )

    stage2_optimizer = torch.optim.Adam(model.stage2_parameters(), lr=1e-3)
    stage2_losses = train_selector_step(
        model,
        stage2_optimizer,
        torch.randn(8, 1, 12, 12),
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        group_shape=(2, 4),
    )

    episode = evaluate_episode(
        model,
        support_x=torch.randn(10, 1, 12, 12),
        support_y=torch.arange(5).repeat_interleave(2),
        query_x=torch.randn(7, 1, 12, 12),
        ways=5,
        steps=2,
        inner_batch_size=4,
        head_seed=0,
    )
    print(f"stage1_loss={stage1_loss.item():.6f}")
    print(f"stage2_loss={stage2_losses.total.item():.6f}")
    print(f"query_logits_shape={tuple(episode.logits.shape)}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
