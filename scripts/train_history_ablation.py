"""Train controlled v6 memory-only and v7 memory+room-reward PPO runs."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from boltcrypt_wrapper import make_env
from wall_penalty_v2 import WallPenalty
from navigation_observation_v5 import NavigationObservation
from history_observation import HistoryObservation
from loop_penalty_v8 import PersistentLoopPenalty
from cell_novelty_v9 import DiminishingCellNovelty
from revisit_penalty_v10 import IncreasingRevisitPenalty

ROOT = Path(__file__).resolve().parent


class MapCycle(gym.Wrapper):
    def __init__(self, env, run_seed, spec):
        super().__init__(env)
        self.seeds = np.arange(spec["start"], spec["start"] + spec["count"])
        self.rng = np.random.default_rng(run_seed)
        self.used_seeds = []

    def reset(self, *, seed=None, options=None):
        chosen = int(self.rng.choice(self.seeds))
        self.used_seeds.append(chosen)
        return self.env.reset(seed=chosen, options=options)


class LogAndStop(BaseCallback):
    def __init__(self, deadline):
        super().__init__()
        self.deadline = deadline
        self.rows = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self.rows.append({"training_step": self.num_timesteps, "reward": ep["r"],
                                  "length": ep["l"],
                                  "success": int(info.get("audit", {}).get("end_reason") == "success")})
        return time.perf_counter() < self.deadline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["v6", "v7", "v8", "v9", "v10"], required=True)
    p.add_argument("--seed", type=int, choices=[41, 42, 43], required=True)
    args = p.parse_args()
    config_path = ROOT / f"experiment_config_{args.version}.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
    output = ROOT / f"runs_{args.version}" / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    discovery = cfg["exploration"]["discovery_reward"] if args.version in ("v7", "v9", "v10") else None
    env = make_env(cfg["generator_config"], cfg["max_episode_steps"], discovery)
    env = WallPenalty(env, cfg["wall_collision_reward"])
    if args.version == "v8":
        env = PersistentLoopPenalty(env, cfg["loop_reward"])
    if args.version in ("v9", "v10"):
        env = DiminishingCellNovelty(env, cfg["first_cell_bonus"])
    if args.version == "v10":
        env = IncreasingRevisitPenalty(env, cfg["revisit_penalty_scale"], cfg["revisit_penalty_cap"])
    env = HistoryObservation(NavigationObservation(env))
    env = MapCycle(env, args.seed, cfg["seeds"]["train"])
    started = time.perf_counter()
    limit = cfg["training"]["maximum_wall_minutes_per_run"] * 60
    logger = LogAndStop(started + limit - 30)
    model = PPO("MultiInputPolicy", env, n_steps=256, batch_size=64,
                n_epochs=2, seed=args.seed, device="cpu", verbose=0)
    model.learn(total_timesteps=cfg["training"]["steps_per_run"], callback=logger)
    model.save(output / "model")
    seconds = time.perf_counter() - started
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["training_step", "reward", "length", "success"])
        writer.writeheader(); writer.writerows(logger.rows)
    result = {"version": args.version, "seed": args.seed, "steps": model.num_timesteps,
              "wall_seconds": seconds, "training_episodes": len(logger.rows),
              "training_successes": sum(r["success"] for r in logger.rows),
              "within_20_minute_limit": seconds <= limit,
              "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "map_seeds_used": env.used_seeds}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    env.close()
    if seconds > limit:
        raise RuntimeError("20-minute limit exceeded")
    print(json.dumps({k: v for k, v in result.items() if k != "map_seeds_used"}))


if __name__ == "__main__":
    main()
