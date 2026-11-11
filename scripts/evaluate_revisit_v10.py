"""Paired sampled evaluation of v9 and v10 without a guide."""
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from boltcrypt_wrapper import make_env
from wall_penalty_v2 import WallPenalty
from cell_novelty_v9 import DiminishingCellNovelty
from revisit_penalty_v10 import IncreasingRevisitPenalty
from navigation_observation_v5 import NavigationObservation
from history_observation import HistoryObservation

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)


def run(model, cfg, version, train_seed, map_seed):
    env = WallPenalty(make_env(cfg["generator_config"], cfg["max_episode_steps"],
                               cfg["exploration"]["discovery_reward"]), cfg["wall_collision_reward"])
    env = DiminishingCellNovelty(env, cfg["first_cell_bonus"])
    if version == "v10":
        env = IncreasingRevisitPenalty(env, cfg["revisit_penalty_scale"], cfg["revisit_penalty_cap"])
    env = HistoryObservation(NavigationObservation(env))
    obs, _ = env.reset(seed=map_seed)
    torch.manual_seed(100000 + map_seed + train_seed)
    start_room = tuple(map(int, obs["global_pos"]))
    positions = [(start_room, tuple(map(int, obs["agent_pos"])))]
    start_cells = {positions[0][1]}
    first_crossing = None
    loops = bouts = long_bouts = longest = streak = repeated = 0
    try:
        for step in range(1, cfg["max_episode_steps"] + 1):
            action = model.predict(obs, deterministic=False)[0]
            obs, _, terminated, truncated, info = env.step(action)
            room = tuple(map(int, obs["global_pos"]))
            current = (room, tuple(map(int, obs["agent_pos"])))
            if room == start_room:
                start_cells.add(current[1])
            if first_crossing is None and room != start_room:
                first_crossing = step
            repeated += int(info["audit"]["events"]["revisited_position"])
            is_loop = (len(positions) >= 3 and positions[-3] == positions[-1]
                       and positions[-2] == current and positions[-1] != current)
            if is_loop:
                loops += 1
                if streak == 0:
                    bouts += 1
                streak += 1
                longest = max(longest, streak)
            elif streak:
                long_bouts += int(streak >= 10)
                streak = 0
            positions.append(current)
            if terminated or truncated:
                long_bouts += int(streak >= 10)
                return {"version": version, "train_seed": train_seed, "map_seed": map_seed,
                        "success": int(terminated), "steps": step,
                        "left_start_room": int(first_crossing is not None),
                        "first_crossing_step": first_crossing or "",
                        "start_room_unique_cells": len(start_cells),
                        "revisited_positions": repeated, "two_cell_loop_events": loops,
                        "two_cell_loop_bouts": bouts, "long_bouts_10plus": long_bouts,
                        "longest_loop_event_streak": longest}
        raise AssertionError("Episode did not end")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-count", type=int, choices=[5, 40], default=5)
    args = parser.parse_args()
    rows = []
    for version in ("v9", "v10"):
        cfg = json.loads((ROOT / f"experiment_config_{version}.json").read_text(encoding="utf-8-sig"))
        for train_seed in (41, 42, 43):
            model = PPO.load(ROOT / f"runs_{version}" / f"seed{train_seed}" / "model.zip", device="cpu")
            for map_seed in range(20000, 20000 + args.map_count):
                rows.append(run(model, cfg, version, train_seed, map_seed))
    suffix = "" if args.map_count == 5 else "_40maps"
    with (OUT / f"revisit_v9_v10_validation{suffix}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for version in ("v9", "v10"):
        part = [r for r in rows if r["version"] == version]
        summary[version] = {"episodes": len(part),
            "left_start_room": sum(r["left_start_room"] for r in part),
            "successes": sum(r["success"] for r in part),
            "mean_two_cell_loop_events": round(np.mean([r["two_cell_loop_events"] for r in part]), 1),
            "mean_two_cell_loop_bouts": round(np.mean([r["two_cell_loop_bouts"] for r in part]), 1),
            "mean_long_bouts_10plus": round(np.mean([r["long_bouts_10plus"] for r in part]), 1),
            "episodes_with_multiple_long_bouts": sum(r["long_bouts_10plus"] >= 2 for r in part),
            "max_longest_loop_event_streak": max(r["longest_loop_event_streak"] for r in part),
            "mean_start_room_unique_cells": round(np.mean([r["start_room_unique_cells"] for r in part]), 2),
            "mean_revisited_positions": round(np.mean([r["revisited_positions"] for r in part]), 1)}
    (OUT / f"revisit_v9_v10_summary{suffix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
