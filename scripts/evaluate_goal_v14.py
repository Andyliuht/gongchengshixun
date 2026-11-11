"""Evaluate goal-only E on development maps and a fresh sealed v14 test split."""
import argparse
import csv
import json
import time
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
from goal_observation_v14 import GoalObservation
from mechanics_audit import MechanicsAudit

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "experiment_config_v13.json").read_text(encoding="utf-8-sig"))


def make_e_env():
    env = make_env(CFG["generator_config"], CFG["max_episode_steps"], CFG["exploration"]["discovery_reward"])
    env = WallPenalty(env, CFG["wall_collision_reward"])
    env = DiminishingCellNovelty(env, CFG["first_cell_bonus"])
    env = IncreasingRevisitPenalty(env, CFG["revisit_penalty_scale"], CFG["revisit_penalty_cap"])
    return MechanicsAudit(GoalObservation(HistoryObservation(NavigationObservation(env))))


def run(model, train_seed, map_seed):
    env = make_e_env()
    started = time.perf_counter()
    try:
        obs, _ = env.reset(seed=map_seed)
        torch.manual_seed(100000 + map_seed + train_seed)
        rooms_map = env.unwrapped.generator.grid  # audit only; not read by policy or goal selector
        locked_map = int(any(room.is_locked for room in rooms_map.values()))
        puzzle_map = int(any(room.puzzle_type.value == "boulder" for room in rooms_map.values()))
        start_room = tuple(map(int, obs["global_pos"]))
        seen_rooms = {start_room}
        positions = [(start_room, tuple(map(int, obs["agent_pos"]))) ]
        first_crossing = first_key = first_puzzle = None
        keys = puzzles = boulder_moves = walls = door_rejections = loops = loop_bouts = loop_streak = 0
        goal_counts = {k: 0 for k in ("none", "key", "puzzle", "frontier", "exit")}
        for step in range(1, CFG["max_episode_steps"] + 1):
            before_room = tuple(map(int, obs["global_pos"]))
            was_solved = bool(env.unwrapped.curr_room.is_solved)
            kind = env.env.last_goal[0]
            goal_counts[kind] += 1
            action = int(model.predict(obs, deterministic=False)[0])
            obs, reward, terminated, truncated, info = env.step(action)
            event = info["mechanics"]["event"]
            current = (tuple(map(int, obs["global_pos"])), tuple(map(int, obs["agent_pos"])))
            seen_rooms.add(current[0])
            if first_crossing is None and current[0] != start_room:
                first_crossing = step
            if event == "key_acquired":
                keys += 1
                if first_key is None: first_key = step
            if event == "boulder_moved": boulder_moves += 1
            if event == "door_rejected_unknown_cause": door_rejections += 1
            if (current[0] == before_room and not was_solved and env.unwrapped.curr_room.is_solved
                    and env.unwrapped.curr_room.puzzle_type.value == "boulder"):
                puzzles += 1
                if first_puzzle is None: first_puzzle = step
            walls += int(info["wall_penalty"]["confirmed_collision"])
            is_loop = (len(positions) >= 3 and positions[-3] == positions[-1]
                       and positions[-2] == current and positions[-1] != current)
            if is_loop:
                loops += 1
                loop_bouts += int(loop_streak == 0)
                loop_streak += 1
            else:
                loop_streak = 0
            positions.append(current)
            if terminated or truncated:
                return {"group": "E", "train_seed": train_seed, "map_seed": map_seed,
                        "locked_map": locked_map, "puzzle_map": puzzle_map,
                        "success": int(terminated), "steps": step,
                        "left_start_room": int(first_crossing is not None),
                        "first_crossing_step": first_crossing or "",
                        "rooms_visited": len(seen_rooms), "keys_acquired": keys,
                        "first_key_step": first_key or "", "puzzles_completed": puzzles,
                        "first_puzzle_step": first_puzzle or "", "boulder_moves": boulder_moves,
                        "wall_collisions": walls, "door_rejections": door_rejections,
                        "two_cell_loop_events": loops, "two_cell_loop_bouts": loop_bouts,
                        "rule_actions": 0, "overrides": 0,
                        "goal_counts": json.dumps(goal_counts, sort_keys=True),
                        "wall_seconds": round(time.perf_counter() - started, 3)}
        raise AssertionError("Episode did not end")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    args = parser.parse_args()
    seeds = range(20000, 20040) if args.split == "validation" else range(40000, 40040)
    out = ROOT / "reports" / f"full_v14_E_{args.split}.csv"
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for train_seed in (41, 42, 43):
        model = PPO.load(ROOT / "runs_v14" / f"E_seed{train_seed}" / "model.zip", device="cpu")
        for map_seed in seeds:
            rows.append(run(model, train_seed, map_seed))
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"split": args.split, "episodes": len(rows), "successes": sum(r["success"] for r in rows),
               "left_start_room": sum(r["left_start_room"] for r in rows),
               "key_acquisition_episodes": sum(r["keys_acquired"] > 0 for r in rows),
               "puzzle_completion_episodes": sum(r["puzzles_completed"] > 0 for r in rows),
               "mean_wall_collisions": round(float(np.mean([r["wall_collisions"] for r in rows])), 1),
               "mean_two_cell_loop_events": round(float(np.mean([r["two_cell_loop_events"] for r in rows])), 1),
               "wall_seconds": round(sum(r["wall_seconds"] for r in rows), 1),
               "by_seed": {str(s): {"successes": sum(r["success"] for r in rows if r["train_seed"] == s),
                                   "left_start_room": sum(r["left_start_room"] for r in rows if r["train_seed"] == s)}
                           for s in (41, 42, 43)}}
    (ROOT / "reports" / f"full_v14_E_{args.split}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
