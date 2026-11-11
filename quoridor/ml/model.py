"""Policy-value network (small AlphaZero-style ResNet)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import N_ACTIONS, N_PLANES, heuristic_value


@dataclass
class ModelConfig:
    channels: int = 64
    blocks: int = 6


class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    """Input: (B, N_PLANES=12, 9, 9) planes. Output: policy logits (B, 209), value (B,) in [-1, 1]."""

    def __init__(self, cfg: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        ch = cfg.channels
        self.stem = nn.Sequential(nn.Conv2d(N_PLANES, ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(ch), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(cfg.blocks)])
        self.p_head = nn.Sequential(nn.Conv2d(ch, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU(),
                                    nn.Flatten(), nn.Linear(4 * 81, N_ACTIONS))
        self.v_head = nn.Sequential(nn.Conv2d(ch, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(),
                                    nn.Flatten(), nn.Linear(2 * 81, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(self.stem(x))
        return self.p_head(h), self.v_head(h).squeeze(-1)


class Evaluator:
    """Inference wrapper: numpy planes + legal mask -> (priors, value).

    ``heuristic_mix`` blends the network value with the path-length heuristic:
    ``value = (1 - mix) * network + mix * heuristic``. A mix around 0.5 makes
    MCTS play sensibly while the value head is still untrained; 0 = pure network.
    """

    def __init__(self, net: PolicyValueNet, device: str = "cpu", heuristic_mix: float = 0.0) -> None:
        self.net = net.to(device).eval()
        self.device = device
        self.heuristic_mix = heuristic_mix

    @torch.no_grad()
    def __call__(self, planes: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        logits, value = self.net(x)
        logits = logits[0].float().cpu().numpy()
        logits[~mask] = -1e9
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        v = float(value[0])
        if self.heuristic_mix:
            v = (1 - self.heuristic_mix) * v + self.heuristic_mix * heuristic_value(planes)
        return p, v


def save_checkpoint(path, net: PolicyValueNet, **extra) -> None:
    torch.save({"model_config": asdict(net.cfg), "model_state": net.state_dict(), **extra}, path)


def load_model(path, device: str = "cpu") -> tuple[PolicyValueNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = PolicyValueNet(ModelConfig(**ckpt["model_config"]))
    net.load_state_dict(ckpt["model_state"])
    return net.to(device).eval(), ckpt
