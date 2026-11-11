"""Training loop: warm-up by imitation, then RL against the Depth-4 alpha-beta bot.

    python -m quoridor.ml.train --out runs/exp1            # start (or resume) a run
    python -m quoridor.ml.train --out runs/exp1 --preset smoke   # 2-minute pipeline check

Each iteration:
  1. play ``games_per_iter`` games: agent (MCTS + network) vs Depth-4 bot;
  2. add the positions to a replay buffer and take ``train_steps`` SGD steps;
  3. save ``latest.pt`` (+ the buffer) so a Colab disconnect loses little;
  4. every ``eval_every`` iterations: play ``eval_games`` against the
     deterministic Depth-4 bot from fixed seeded openings (half as Red, half
     as Blue) and log the win rate. A new best is saved as ``best.pt``.

Stopping condition: when the evaluation win rate reaches ``target_win_rate``
(default 0.95), the model is re-tested on ``confirm_games`` *new* openings
to rule out a lucky streak. If that also reaches the target, the model is
saved as ``milestone_95.pt`` and training stops (``--keep-training`` to go on).

Outputs in ``--out``: config.json, metrics.jsonl, latest.pt, best.pt,
milestone_95.pt, buffer.npz and TensorBoard logs in ``tb/``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .arena import wilson_lower_bound
from .encoding import N_ACTIONS, N_PLANES, mirror_batch
from .model import ModelConfig, PolicyValueNet, save_checkpoint
from .selfplay import (GameSettings, default_workers, run_eval, run_teacher, run_training,
                       serialize_model, worker_init)


@dataclass
class TrainConfig:
    # model
    channels: int = 64
    blocks: int = 6
    # opponent / games
    opponent_depth: int = 4
    teacher_depth: int = 4
    opening_plies: int = 6
    sims: int = 64
    temp_moves: int = 8
    learn_opponent_moves: bool = True
    # warm-up (imitation of alpha-beta games, run once at the start)
    warmup_games: int = 300
    warmup_steps: int = 3000
    # RL loop
    iterations: int = 100000
    games_per_iter: int = 48
    train_steps: int = 300
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    buffer_size: int = 250_000
    # evaluation + stopping
    eval_every: int = 2
    eval_games: int = 100
    eval_sims: int = 64
    eval_seed: int = 12345
    target_win_rate: float = 0.95
    confirm_games: int = 200
    confirm_seed: int = 777
    stop_at_target: bool = True
    # infrastructure
    workers: int = 0          # 0 = all CPU cores
    device: str = "auto"      # "auto" | "cuda" | "cpu"
    max_hours: Optional[float] = None  # stop cleanly before e.g. a Colab session limit
    seed: int = 0
    save_buffer: bool = True


PRESETS = {
    # Checks the whole pipeline end to end in a couple of minutes.
    "smoke": dict(channels=32, blocks=2, warmup_games=8, warmup_steps=40, games_per_iter=4,
                  train_steps=20, sims=8, eval_sims=8, eval_every=1, eval_games=6,
                  confirm_games=6, iterations=2, batch_size=64),
    # Free Colab (2 CPU cores): fewer games per iteration so progress shows up regularly.
    "colab-free": dict(games_per_iter=16, train_steps=150, eval_games=60, eval_every=3, sims=48, eval_sims=48),
    # Colab Pro / many-core machines.
    "full": dict(),
}


# ---------------------------------------------------------------------- #
class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.planes = np.zeros((capacity, N_PLANES, 9, 9), dtype=np.float16)
        self.pis = np.zeros((capacity, N_ACTIONS), dtype=np.float16)
        self.z = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.pos = 0

    def add(self, planes: np.ndarray, pis: np.ndarray, z: np.ndarray) -> None:
        for i in range(len(z)):
            self.planes[self.pos] = planes[i]
            self.pis[self.pos] = pis[i]
            self.z[self.pos] = z[i]
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=n)
        return self.planes[idx].astype(np.float32), self.pis[idx].astype(np.float32), self.z[idx]

    def save(self, path: Path) -> None:
        n = self.size
        np.savez_compressed(path, planes=self.planes[:n], pis=self.pis[:n], z=self.z[:n], pos=self.pos)

    def load(self, path: Path) -> None:
        d = np.load(path)
        n = min(len(d["z"]), self.capacity)
        self.planes[:n], self.pis[:n], self.z[:n] = d["planes"][:n], d["pis"][:n], d["z"][:n]
        self.size = n
        self.pos = int(d["pos"]) % self.capacity if n == self.capacity else n


class Trainer:
    def __init__(self, cfg: TrainConfig, out: Path) -> None:
        self.cfg = cfg
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        # Start the worker processes before torch touches threads or CUDA, so
        # forking is safe. Workers run games on CPU; the main process trains.
        self.workers = cfg.workers or default_workers()
        ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
        self.pool = ctx.Pool(self.workers, initializer=worker_init)
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
        torch.manual_seed(cfg.seed)
        self.np_rng = np.random.default_rng(cfg.seed)
        self.net = PolicyValueNet(ModelConfig(cfg.channels, cfg.blocks)).to(self.device)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.buffer = ReplayBuffer(cfg.buffer_size)
        self.iteration = 0
        self.version = 0
        self.best_win_rate = -1.0
        self.games_played = 0
        self.warmed_up = False
        self.start_time = time.time()
        self.gs = GameSettings(cfg.opponent_depth, cfg.teacher_depth, cfg.sims, cfg.eval_sims,
                               cfg.temp_moves, cfg.opening_plies, cfg.learn_opponent_moves)
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(str(out / "tb"))
        except Exception:  # tensorboard not installed: metrics.jsonl still works
            pass

    # ------------------------------------------------------------------ #
    def log(self, kind: str, **data) -> None:
        rec = {"time": round(time.time() - self.start_time, 1), "iteration": self.iteration,
               "games_played": self.games_played, "kind": kind, **data}
        with open(self.out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if self.tb is not None:
            for k, v in data.items():
                if isinstance(v, (int, float)) and v is not None:
                    self.tb.add_scalar(f"{kind}/{k}", v, self.iteration)
            self.tb.flush()

    def say(self, msg: str) -> None:
        hours = (time.time() - self.start_time) / 3600
        print(f"[{hours:5.2f}h it {self.iteration:4d}] {msg}", flush=True)

    def save(self, name: str, **extra) -> None:
        save_checkpoint(self.out / name, self.net, optimizer=self.opt.state_dict(), iteration=self.iteration,
                        best_win_rate=self.best_win_rate, games_played=self.games_played,
                        warmed_up=self.warmed_up, config=asdict(self.cfg), **extra)

    def resume(self) -> bool:
        path = self.out / "latest.pt"
        if not path.exists():
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["model_state"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.iteration = ckpt["iteration"]
        self.best_win_rate = ckpt.get("best_win_rate", -1.0)
        self.games_played = ckpt.get("games_played", 0)
        self.warmed_up = ckpt.get("warmed_up", True)
        if (self.out / "buffer.npz").exists():
            self.buffer.load(self.out / "buffer.npz")
        self.say(f"resumed from {path} (buffer {self.buffer.size} positions, best win rate {self.best_win_rate:.0%})")
        return True

    def model_blob(self) -> tuple[int, bytes]:
        self.version += 1
        return self.version, serialize_model(self.net)

    # ------------------------------------------------------------------ #
    def train_steps(self, steps: int) -> dict:
        self.net.train()
        tot_p = tot_v = 0.0
        for _ in range(steps):
            planes, pis, z = self.buffer.sample(self.cfg.batch_size, self.np_rng)
            if self.np_rng.random() < 0.5:
                planes, pis = mirror_batch(planes, pis)
            x = torch.from_numpy(np.ascontiguousarray(planes)).to(self.device)
            target_pi = torch.from_numpy(np.ascontiguousarray(pis)).to(self.device)
            target_z = torch.from_numpy(z).to(self.device)
            logits, v = self.net(x)
            p_loss = -(target_pi * F.log_softmax(logits, dim=1)).sum(1).mean()
            v_loss = F.mse_loss(v, target_z)
            loss = p_loss + v_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            tot_p += p_loss.item()
            tot_v += v_loss.item()
        self.net.eval()
        return {"policy_loss": tot_p / steps, "value_loss": tot_v / steps}

    def warmup(self) -> None:
        c = self.cfg
        if c.warmup_games <= 0:
            self.warmed_up = True
            return
        self.say(f"warm-up: {c.warmup_games} alpha-beta games on {self.workers} worker(s)...")
        seeds = [(c.seed * 10_000_019 + i, self.gs) for i in range(c.warmup_games)]
        done = 0
        for planes, pis, z, info in self.pool.imap_unordered(run_teacher, seeds):
            self.buffer.add(planes, pis, z)
            done += 1
            if done % max(1, c.warmup_games // 10) == 0:
                self.say(f"  warm-up games {done}/{c.warmup_games}")
        self.say(f"warm-up: training {c.warmup_steps} steps on {self.buffer.size} positions")
        losses = self.train_steps(c.warmup_steps)
        self.warmed_up = True
        self.log("warmup", **losses, positions=self.buffer.size)
        self.save("latest.pt")

    def play_training_games(self) -> dict:
        c = self.cfg
        version, blob = self.model_blob()
        base = (c.seed + 1) * 1_000_000_007 + self.games_played
        jobs = [(base + i, version, blob, self.gs) for i in range(c.games_per_iter)]
        wins = 0
        plies = 0
        for planes, pis, z, info in self.pool.imap_unordered(run_training, jobs):
            self.buffer.add(planes, pis, z)
            wins += info["agent_won"]
            plies += info["plies"]
        self.games_played += c.games_per_iter
        return {"train_win_rate": wins / c.games_per_iter, "avg_plies": plies / c.games_per_iter}

    def evaluate(self, n_games: int, seed: int) -> dict:
        version, blob = self.model_blob()
        jobs = [(i, seed, version, blob, self.gs) for i in range(n_games)]
        results = list(self.pool.imap_unordered(run_eval, jobs))
        wins = sum(r["won"] for r in results)
        draws = sum(r["winner"] is None for r in results)
        red_wins = sum(r["won"] for r in results if r["agent_side"] == 0)
        return {"win_rate": wins / n_games, "wins": wins, "games": n_games, "draws": draws,
                "wilson_low": wilson_lower_bound(wins, n_games),
                "win_rate_as_red": red_wins / max(1, (n_games + 1) // 2),
                "win_rate_as_blue": (wins - red_wins) / max(1, n_games // 2),
                "avg_plies": sum(r["plies"] for r in results) / n_games}

    # ------------------------------------------------------------------ #
    def run(self) -> str:
        c = self.cfg
        self.say(f"device={self.device}, workers={self.workers}, target={c.target_win_rate:.0%} "
                 f"vs Depth-{c.opponent_depth} bot")
        if not self.warmed_up:
            self.warmup()
        while self.iteration < c.iterations:
            if c.max_hours is not None and time.time() - self.start_time > c.max_hours * 3600:
                self.say("time budget reached: saving and stopping (run again with the same --out to resume)")
                self.save("latest.pt")
                return "time_budget"
            self.iteration += 1
            t0 = time.time()
            g = self.play_training_games()
            losses = self.train_steps(c.train_steps)
            self.log("train", **g, **losses, buffer=self.buffer.size, seconds=round(time.time() - t0, 1))
            self.say(f"train: agent won {g['train_win_rate']:.0%} of {c.games_per_iter} games, "
                     f"loss p={losses['policy_loss']:.3f} v={losses['value_loss']:.3f}, "
                     f"{time.time() - t0:.0f}s")
            self.save("latest.pt")
            if c.save_buffer and self.iteration % 5 == 0:
                self.buffer.save(self.out / "buffer.npz")

            if self.iteration % c.eval_every == 0:
                r = self.evaluate(c.eval_games, c.eval_seed)
                self.log("eval", **r)
                self.say(f"EVAL vs Depth-{c.opponent_depth}: win rate {r['win_rate']:.1%} "
                         f"({r['wins']}/{r['games']}, 95% CI low {r['wilson_low']:.1%}, "
                         f"red {r['win_rate_as_red']:.0%} / blue {r['win_rate_as_blue']:.0%})")
                if r["win_rate"] > self.best_win_rate:
                    self.best_win_rate = r["win_rate"]
                    self.save("best.pt", eval=r)
                    self.say(f"  new best -> {self.out / 'best.pt'}")
                if r["win_rate"] >= c.target_win_rate:
                    if self.confirm(r):
                        return "target_reached"
        self.say("finished all iterations")
        return "iterations_done"

    def confirm(self, first: dict) -> bool:
        c = self.cfg
        if c.confirm_games > 0:
            self.say(f"  target reached; confirming on {c.confirm_games} new openings...")
            r = self.evaluate(c.confirm_games, c.confirm_seed + self.iteration)
            self.log("confirm", **r)
            self.say(f"  confirmation: {r['win_rate']:.1%} ({r['wins']}/{r['games']})")
            if r["win_rate"] < c.target_win_rate:
                self.say("  not confirmed - training continues")
                return False
        else:
            r = first
        name = f"milestone_{round(c.target_win_rate * 100)}.pt"
        self.save(name, eval=first, confirm=r)
        self.log("milestone", win_rate=first["win_rate"], confirm_win_rate=r["win_rate"])
        self.say(f"*** MILESTONE: {first['win_rate']:.1%} vs Depth-{c.opponent_depth} bot -> {self.out / name} ***")
        return c.stop_at_target

    def close(self) -> None:
        if self.cfg.save_buffer and self.buffer.size:
            self.buffer.save(self.out / "buffer.npz")
        self.pool.close()
        self.pool.join()
        if self.tb is not None:
            self.tb.close()


# ---------------------------------------------------------------------- #
def parse_args(argv=None) -> tuple[TrainConfig, Path, bool]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="runs/quoridor", help="output directory (checkpoints, logs)")
    p.add_argument("--preset", choices=sorted(PRESETS), help="named set of defaults")
    p.add_argument("--fresh", action="store_true", help="ignore an existing latest.pt in --out")
    p.add_argument("--keep-training", action="store_true", help="don't stop when the target is reached")
    for f in fields(TrainConfig):
        if f.name == "stop_at_target":
            continue
        kind = {"int": int, "float": float, "str": str, "bool": None}.get(str(f.type).replace("Optional[", "").rstrip("]"))
        name = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            p.add_argument(name, type=lambda s: s.lower() in ("1", "true", "yes"), default=None,
                           metavar="BOOL")
        else:
            p.add_argument(name, type=kind or str, default=None)
    a = p.parse_args(argv)
    cfg = TrainConfig()
    if a.preset:
        for k, v in PRESETS[a.preset].items():
            setattr(cfg, k, v)
    for f in fields(TrainConfig):
        v = getattr(a, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    if a.keep_training:
        cfg.stop_at_target = False
    return cfg, Path(a.out), a.fresh


def main(argv=None) -> str:
    cfg, out, fresh = parse_args(argv)
    random.seed(cfg.seed)
    trainer = Trainer(cfg, out)
    try:
        if fresh or not trainer.resume():
            (out / "metrics.jsonl").unlink(missing_ok=True)
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        status = trainer.run()
    except KeyboardInterrupt:
        trainer.say("interrupted: saving latest.pt")
        trainer.save("latest.pt")
        status = "interrupted"
    finally:
        trainer.close()
    print(f"status: {status}")
    return status


if __name__ == "__main__":
    main()
