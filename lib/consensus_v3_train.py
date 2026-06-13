"""Loss и EMA для обучения Consensus V3."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class V3LossCfg:
    """Регуляризация при fine-tune от Stage-1."""
    anchor_lambda: float = 0.08
    trust_floor: float = 0.25
    trust_gamma: float = 1.5


@dataclass
class V5LossCfg:
    """RGB weighted MSE: сильнее штраф в low-trust (дыры), слабее в trusted warp."""
    hole_weight: float = 2.5
    valid_weight: float = 0.5
    trust_gamma: float = 1.5
    anchor_lambda: float = 0.05
    normalize_weights: bool = True


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.data, alpha=1.0 - d)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow.state_dict()


def v3_composite_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    base_init: torch.Tensor,
    trust: torch.Tensor,
    cfg: V3LossCfg,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    pred/target/base: NCHW [0,1], trust: N1HW в [0,1] (из 7-го канала входа или mask).
    """
    err = (pred - target) ** 2
    w = cfg.trust_floor + (1.0 - cfg.trust_floor) * trust.clamp(0, 1).pow(cfg.trust_gamma)
    recon = (w * err).mean()
    anchor = ((pred - base_init) ** 2).mean()
    loss = recon + cfg.anchor_lambda * anchor
    return loss, {
        "recon": float(recon.detach().cpu()),
        "anchor": float(anchor.detach().cpu()),
        "total": float(loss.detach().cpu()),
    }


def v5_weighted_rgb_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    base_init: torch.Tensor,
    trust: torch.Tensor,
    cfg: V5LossCfg,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    pred/target/base: NCHW [0,1], trust: N1HW.
    Вес растёт там, где trust низкий (дыры / ненадёжный warp).
    """
    err = (pred - target) ** 2
    t = trust.clamp(0, 1)
    hole = (1.0 - t).pow(cfg.trust_gamma)
    valid = t.pow(cfg.trust_gamma)
    w = cfg.hole_weight * hole + cfg.valid_weight * valid
    if cfg.normalize_weights:
        w = w / w.mean().clamp(min=1e-6)
    recon = (w * err).mean()
    anchor = ((pred - base_init) ** 2).mean()
    loss = recon + cfg.anchor_lambda * anchor
    return loss, {
        "recon": float(recon.detach().cpu()),
        "anchor": float(anchor.detach().cpu()),
        "total": float(loss.detach().cpu()),
        "w_hole_mean": float((cfg.hole_weight * hole).mean().detach().cpu()),
        "w_valid_mean": float((cfg.valid_weight * valid).mean().detach().cpu()),
    }
