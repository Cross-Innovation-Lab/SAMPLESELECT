"""State-dict-only checkpoint helpers.

The helpers deliberately avoid serializing live model, optimizer, or scheduler
objects. Checkpoints created here can therefore be loaded with
``weights_only=True`` on supported PyTorch versions.
"""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import Any

import torch
from torch import nn


def _plain_metadata(value: Any) -> Any:
    """Return JSON-like metadata or reject executable/custom objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain_metadata(item) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("checkpoint metadata keys must be strings")
        return {key: _plain_metadata(item) for key, item in value.items()}
    raise TypeError(
        "checkpoint metadata may contain only JSON-like primitive values"
    )


def save_weights_checkpoint(
    path: str | PathLike[str],
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save tensor state and inert metadata, never live Python objects."""
    payload: dict[str, Any] = {
        "format": "channel-selection-state-dict-v1",
        "model_state_dict": model.state_dict(),
        "metadata": _plain_metadata(dict(metadata or {})),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        if not hasattr(scheduler, "state_dict"):
            raise TypeError("scheduler must provide state_dict()")
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def load_weights_checkpoint(
    path: str | PathLike[str],
    model: nn.Module,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a release checkpoint through PyTorch's restricted loader."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    if payload.get("format") != "channel-selection-state-dict-v1":
        raise ValueError("unsupported checkpoint format")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint is missing model_state_dict")
    model.load_state_dict(state, strict=strict)
    return payload
