import inspect
import json
from pathlib import Path

import torch
from torch import nn

from channel_selection import (
    ChannelSelectionModel,
    evaluate_episode,
    load_weights_checkpoint,
    save_weights_checkpoint,
)
from channel_selection.reference_backbone import ReferenceBackbone
from channel_selection.training import train_selector_step, train_stage1_step


def build_model() -> ChannelSelectionModel:
    return ChannelSelectionModel(
        ReferenceBackbone(channels=8, spatial_size=(2, 3)),
        channels=8,
        spatial_size=(2, 3),
        source_classes=4,
        keep_ratio=0.7,
        temperature=0.3,
    )


def test_selector_mask_and_shape():
    model = build_model()
    x = torch.randn(6, 1, 12, 12)
    model.selector.train()
    soft = model.selected_components(x)
    assert soft["selected_map"].shape == (6, 8, 2, 3)
    assert torch.all((soft["mask"] >= 0) & (soft["mask"] <= 1))

    model.selector.eval()
    hard = model.selected_components(x)["mask"]
    assert torch.all((hard == 0) | (hard == 1))
    assert torch.all(hard.sum(dim=1) == 5)

    indices = model.selected_components(x)["indices"]
    assert indices is not None
    assert indices.shape == (6, 5)
    assert torch.all(indices.sort(dim=1).values.diff(dim=1) != 0)


def test_stage2_freezes_base_and_updates_selector():
    model = build_model()
    model.set_stage2_trainable()
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert not any(p.requires_grad for p in model.source_head.parameters())
    linear_parameters = {
        id(parameter)
        for module in model.selector.modules()
        if isinstance(module, nn.Linear)
        for parameter in module.parameters()
    }
    batch_norm_parameters = {
        id(parameter)
        for module in model.selector.modules()
        if isinstance(module, nn.BatchNorm1d)
        for parameter in module.parameters()
    }
    assert all(
        parameter.requires_grad == (id(parameter) in linear_parameters)
        for parameter in model.selector.parameters()
    )
    assert batch_norm_parameters
    assert all(not module.training for module in model.selector.modules() if isinstance(module, nn.BatchNorm1d))

    x = torch.randn(8, 1, 12, 12)
    source_y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    pair_y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    base_before = {name: value.detach().clone() for name, value in model.backbone.state_dict().items()}
    selector_before = {
        name: value.detach().clone() for name, value in model.selector.state_dict().items()
    }
    optimizer = torch.optim.Adam(model.stage2_parameters(), lr=1e-3)
    losses = train_selector_step(
        model,
        optimizer,
        x,
        source_y,
        pair_y,
        group_shape=(2, 4),
    )
    assert torch.isfinite(losses.total)
    assert any(p.grad is not None for p in model.selector.parameters())
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(
        torch.equal(value, base_before[name])
        for name, value in model.backbone.state_dict().items()
    )
    assert any(
        not torch.equal(value, selector_before[name])
        for name, value in model.selector.state_dict().items()
        if "running_" not in name and "num_batches_tracked" not in name
    )
    assert all(
        torch.equal(value, selector_before[name])
        for name, value in model.selector.state_dict().items()
        if "running_" in name or "num_batches_tracked" in name
    )


def test_stage1_updates_backbone_and_source_head():
    model = build_model()
    model.set_stage1_trainable()
    optimizer = torch.optim.SGD(
        list(model.backbone.parameters()) + list(model.source_head.parameters()),
        lr=1e-2,
    )
    before = model.source_head.weight.detach().clone()
    loss = train_stage1_step(
        model,
        optimizer,
        torch.randn(6, 1, 12, 12),
        torch.tensor([0, 1, 2, 3, 0, 1]),
    )
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.source_head.weight.detach())


def test_episode_uses_fresh_head_only():
    model = build_model()
    support_x = torch.randn(10, 1, 12, 12)
    support_y = torch.arange(5).repeat_interleave(2)
    query_x = torch.randn(7, 1, 12, 12)
    result = evaluate_episode(
        model,
        support_x,
        support_y,
        query_x,
        ways=5,
        steps=2,
        inner_batch_size=4,
        head_seed=123,
    )
    assert result.logits.shape == (7, 5)
    assert result.predictions.shape == (7,)
    assert not any(p.requires_grad for p in model.parameters())
    assert "query_y" not in inspect.signature(evaluate_episode).parameters


def test_final_protocol_configuration():
    config_path = Path(__file__).parents[1] / "configs" / "default.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["feature_channels"] == 640
    assert config["feature_height"] == 4
    assert config["feature_width"] == 5
    assert config["selector"]["keep_ratio"] == 0.7
    assert config["selector"]["selected_channels"] == 448
    assert config["selector"]["temperature"] == 0.3
    assert config["selector"]["temperature_schedule"] == "fixed"
    assert config["selector"]["without_replacement"] is True
    assert config["selector"]["hard_during_training"] is False
    assert config["selector"]["selector_batch_norm"] == "frozen_eval"
    assert config["selector_training"]["lambda_contrastive"] == 0.02
    assert config["selector_training"]["contrastive_temperature"] == 0.07
    assert config["selector_training"]["group_shape"] == [16, 8]
    assert config["few_shot_evaluation"]["inner_batch_size"] == 4
    assert config["few_shot_evaluation"]["adaptation_steps"] == 100
    assert config["randomness_control"]["fixed_data_schedule_seed"] == 42
    assert config["randomness_control"]["fixed_selector_schedule_seed"] == 42
    assert config["randomness_control"]["paired_episode_head_initialization"] is True
    assert config["checkpointing"]["state_dict_only"] is True
    assert config["checkpointing"]["weights_only_load"] is True


def test_paper_scale_selector_parameter_freeze_contract():
    from channel_selection import SampleConditionedChannelSelector

    selector = SampleConditionedChannelSelector(channels=640)
    for parameter in selector.parameters():
        parameter.requires_grad_(False)
    for module in selector.modules():
        if isinstance(module, nn.Linear):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        elif isinstance(module, nn.BatchNorm1d):
            module.eval()

    total = sum(parameter.numel() for parameter in selector.parameters())
    trainable = sum(
        parameter.numel() for parameter in selector.parameters() if parameter.requires_grad
    )
    assert selector.k == 448
    assert total == 9_846_400
    assert trainable == 9_836_160
    assert total - trainable == 10_240


def test_eval_never_calls_gumbel(monkeypatch):
    model = build_model()
    x = torch.randn(6, 1, 12, 12)
    model.freeze_representation()

    def forbidden(*args, **kwargs):
        raise AssertionError("Gumbel sampling must not run during Stage 3")

    monkeypatch.setattr(torch.nn.functional, "gumbel_softmax", forbidden)
    first = model.selected_components(x)
    second = model.selected_components(x)
    assert torch.equal(first["mask"], second["mask"])
    assert torch.equal(first["indices"], second["indices"])


def test_episode_head_seed_is_reproducible():
    model = build_model()
    support_x = torch.randn(10, 1, 12, 12)
    support_y = torch.arange(5).repeat_interleave(2)
    query_x = torch.randn(7, 1, 12, 12)
    kwargs = dict(ways=5, steps=2, inner_batch_size=4, head_seed=987)
    first = evaluate_episode(model, support_x, support_y, query_x, **kwargs)
    second = evaluate_episode(model, support_x, support_y, query_x, **kwargs)
    assert torch.equal(first.logits, second.logits)
    assert torch.equal(first.predictions, second.predictions)


def test_checkpoint_is_state_dict_only_and_round_trips(tmp_path):
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    path = tmp_path / "model.pt"
    reference = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }

    save_weights_checkpoint(
        path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata={"epoch": 1, "complete": True},
    )
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert raw["format"] == "channel-selection-state-dict-v1"
    assert "optimizer_state_dict" in raw
    assert "scheduler_state_dict" in raw

    restored = build_model()
    payload = load_weights_checkpoint(path, restored)
    assert payload["metadata"] == {"epoch": 1, "complete": True}
    assert all(
        torch.equal(reference[name], tensor)
        for name, tensor in restored.state_dict().items()
    )


def test_checkpoint_rejects_custom_metadata(tmp_path):
    class PrivateObject:
        pass

    with torch.no_grad():
        model = build_model()
    try:
        save_weights_checkpoint(
            tmp_path / "rejected.pt", model, metadata={"value": PrivateObject()}
        )
    except TypeError:
        pass
    else:
        raise AssertionError("custom checkpoint metadata must be rejected")
