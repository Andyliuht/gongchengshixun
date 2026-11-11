"""Train E with B rewards and goal hints, under the frozen per-run budget."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

from stable_baselines3 import PPO
from boltcrypt_wrapper import make_env
from wall_penalty_v2 import WallPenalty
from navigation_observation_v5 import NavigationObservation
from history_observation import HistoryObservation
from cell_novelty_v9 import DiminishingCellNovelty
from revisit_penalty_v10 import IncreasingRevisitPenalty
from goal_observation_v14 import GoalObservation
from train_history_ablation import MapCycle, LogAndStop

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=[41, 42, 43], required=True)
    args = parser.parse_args()
    path = ROOT / "experiment_config_v13.json"
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    output = ROOT / "runs_v14" / f"E_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    env = make_env(cfg["generator_config"], cfg["max_episode_steps"], cfg["exploration"]["discovery_reward"])
    env = WallPenalty(env, cfg["wall_collision_reward"])
    env = DiminishingCellNovelty(env, cfg["first_cell_bonus"])
    env = IncreasingRevisitPenalty(env, cfg["revisit_penalty_scale"], cfg["revisit_penalty_cap"])
    env = GoalObservation(HistoryObservation(NavigationObservation(env)))
    env = MapCycle(env, args.seed, cfg["seeds"]["train"])
    started = time.perf_counter()
    limit = cfg["training"]["maximum_wall_minutes_per_run"] * 60
    callback = LogAndStop(started + limit - 30)
    model = PPO("MultiInputPolicy", env, n_steps=cfg["training"]["n_steps"],
                batch_size=cfg["training"]["batch_size"], n_epochs=cfg["training"]["n_epochs"],
                seed=args.seed, device=cfg["training"]["device"], verbose=0)
    model.learn(total_timesteps=cfg["training"]["steps_per_run"], callback=callback)
    model.save(output / "model")
    seconds = time.perf_counter() - started
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["training_step", "reward", "length", "success"])
        writer.writeheader()
        writer.writerows(callback.rows)
    summary = {"version": "v14", "group": "E", "seed": args.seed,
               "steps": model.num_timesteps, "wall_seconds": seconds,
               "training_episodes": len(callback.rows),
               "training_successes": sum(row["success"] for row in callback.rows),
               "within_20_minute_limit": seconds <= limit,
               "base_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "map_seeds_used": env.used_seeds}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    env.close()
    if seconds > limit:
        raise RuntimeError("20-minute limit exceeded")
    print(json.dumps({k: v for k, v in summary.items() if k != "map_seeds_used"}))


if __name__ == "__main__":
    main()
