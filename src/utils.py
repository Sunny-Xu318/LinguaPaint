from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import yaml
from torchvision.utils import save_image


def load_config(path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1.0, 1.0) + 1.0) * 0.5


def save_tensor_image(tensor: torch.Tensor, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(tensor.detach().cpu()), path)


def set_requires_grad(modules, flag: bool) -> None:
    if isinstance(modules, nn.Module):
        modules = [modules]
    for module in modules:
        for p in module.parameters():
            p.requires_grad_(flag)


class AverageMeter:
    def __init__(self) -> None:
        self.values: Dict[str, float] = {}
        self.count = 0

    def update(self, logs: Dict[str, object]) -> None:
        self.count += 1
        for key, value in logs.items():
            scalar = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
            self.values[key] = self.values.get(key, 0.0) + scalar

    def summary(self) -> Dict[str, float]:
        return {key: value / max(1, self.count) for key, value in self.values.items()}


class ModelEMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if v.is_floating_point():
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow.state_dict()

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow.load_state_dict(state)

    @property
    def module(self) -> nn.Module:
        return self.shadow


def clip_gradients(parameters: Iterable[torch.Tensor], max_norm: float) -> torch.Tensor:
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
