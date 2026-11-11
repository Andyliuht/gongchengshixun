"""One-time sealed-test A/B/C/D evaluation for frozen full-dungeon v13."""
import csv
import json
from collections import Counter
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
from mechanics_audit import MechanicsAudit
from limited_full_guide import LimitedFullGuide

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "experiment_config_v13.json").read_text(encoding="utf-8-sig"))
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)


def make_group_env(group):
    boosted = group in ("B", "D")
    discovery = CFG["exploration"]["discovery_reward"] if boosted or group == "A2" else None
    env = make_env(CFG["generator_config"], CFG["max_episode_steps"], discovery)
    env = WallPenalty(env, CFG["wall_collision_reward"])
    if boosted:
        env = DiminishingCellNovelty(env, CFG["first_cell_bonus"])
        env = IncreasingRevisitPenalty(env, CFG["revisit_penalty_scale"], CFG["revisit_penalty_cap"])
    return MechanicsAudit(HistoryObservation(NavigationObservation(env)))


def run(model, group, train_seed, map_seed, save_trace=False):
    env = make_group_env(group)
    obs, _ = env.reset(seed=map_seed)
    torch.manual_seed(100000 + map_seed + train_seed)
    rooms_map = env.unwrapped.generator.grid  # audit-only truth; never sent to policy/guide
    locked_map = int(any(r.is_locked for r in rooms_map.values()))
    puzzle_map = int(any(r.puzzle_type.value == "boulder" for r in rooms_map.values()))
    controller = LimitedFullGuide(CFG["guidance"], obs) if group in ("C", "D") else None
    start = tuple(map(int, obs["global_pos"]))
    positions = [(start, tuple(map(int, obs["agent_pos"])))]
    seen_rooms = {start}
    first_crossing = first_key = first_puzzle = None
    keys = puzzles = boulder_moves = walls = door_rejections = loops = loop_bouts = loop_streak = overrides = 0
    reasons = Counter()
    triggers = Counter()
    trace = []
    try:
        for step in range(1, CFG["max_episode_steps"] + 1):
            before = obs
            before_room = tuple(map(int, before["global_pos"]))
            was_solved = bool(env.unwrapped.curr_room.is_solved)
            raw = int(model.predict(obs, deterministic=False)[0])
            if controller:
                action, reason, target, engaged, trigger = controller.decide(obs, raw)
                if engaged:
                    reasons[reason] += 1
                    triggers[trigger] += 1
                    overrides += int(action != raw)
            else:
                action, reason, target, engaged, trigger = raw, "ppo_only", None, False, None
            obs, reward, terminated, truncated, info = env.step(action)
            event = info["mechanics"]["event"]
            if controller:
                controller.observe_transition(before, action, obs, info["mechanics"])
            current = (tuple(map(int, obs["global_pos"])), tuple(map(int, obs["agent_pos"])))
            seen_rooms.add(current[0])
            if first_crossing is None and current[0] != start:
                first_crossing = step
            if event == "key_acquired":
                keys += 1
                if first_key is None: first_key = step
            if event == "boulder_moved":
                boulder_moves += 1
            if event == "door_rejected_unknown_cause":
                door_rejections += 1
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
            if save_trace:
                trace.append({"step": step, "before": positions[-2], "ppo_action": raw,
                              "action": action, "reason": reason, "trigger": trigger,
                              "rule_action": engaged, "target": target, "after": current,
                              "event": event, "wall": info["wall_penalty"]["confirmed_collision"],
                              "reward": round(float(reward), 4), "terminated": bool(terminated)})
            if terminated or truncated:
                return ({"group": group, "train_seed": train_seed, "map_seed": map_seed,
                         "locked_map": locked_map, "puzzle_map": puzzle_map,
                         "success": int(terminated), "steps": step,
                         "left_start_room": int(first_crossing is not None),
                         "first_crossing_step": first_crossing or "",
                         "rooms_visited": len(seen_rooms), "keys_acquired": keys,
                         "first_key_step": first_key or "", "puzzles_completed": puzzles,
                         "first_puzzle_step": first_puzzle or "", "boulder_moves": boulder_moves,
                         "wall_collisions": walls, "door_rejections": door_rejections,
                         "two_cell_loop_events": loops, "two_cell_loop_bouts": loop_bouts,
                         "rule_actions": controller.rule_actions if controller else 0,
                         "overrides": overrides,
                         "reason_counts": json.dumps(reasons, sort_keys=True),
                         "trigger_counts": json.dumps(triggers, sort_keys=True)}, trace)
        raise AssertionError("Episode did not end")
    finally:
        env.close()


def main():
    rows = []
    for group in "ABCD":
        model_group = "A" if group in "AC" else "B"
        for train_seed in (41, 42, 43):
            model = PPO.load(ROOT / "runs_v13" / f"{model_group}_seed{train_seed}" / "model.zip", device="cpu")
            for map_seed in range(30000, 30040):
                row, trace = run(model, group, train_seed, map_seed,
                                 save_trace=(train_seed == 41 and map_seed == 30000))
                rows.append(row)
                if trace:
                    (OUT / f"full_v13_test_trace_{group}_seed41_map30000.json").write_text(
                        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "full_v13_four_groups_test.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for group in "ABCD":
        part = [r for r in rows if r["group"] == group]
        summary[group] = {"episodes": len(part),
            "successes": sum(r["success"] for r in part),
            "left_start_room": sum(r["left_start_room"] for r in part),
            "key_acquisition_episodes": sum(r["keys_acquired"] > 0 for r in part),
            "puzzle_completion_episodes": sum(r["puzzles_completed"] > 0 for r in part),
            "mean_rooms_visited": round(float(np.mean([r["rooms_visited"] for r in part])), 2),
            "mean_steps": round(float(np.mean([r["steps"] for r in part])), 1),
            "mean_wall_collisions": round(float(np.mean([r["wall_collisions"] for r in part])), 1),
            "mean_two_cell_loop_events": round(float(np.mean([r["two_cell_loop_events"] for r in part])), 1),
            "rule_actions": sum(r["rule_actions"] for r in part),
            "overrides": sum(r["overrides"] for r in part),
            "by_seed": {str(s): {"successes": sum(r["success"] for r in part if r["train_seed"] == s),
                                 "left_start_room": sum(r["left_start_room"] for r in part if r["train_seed"] == s)}
                        for s in (41, 42, 43)},
            "by_map_type": {label: {"episodes": len([r for r in part if (r["locked_map"], r["puzzle_map"]) == pair]),
                                    "successes": sum(r["success"] for r in part if (r["locked_map"], r["puzzle_map"]) == pair)}
                            for label, pair in {"neither": (0, 0), "key_only": (1, 0),
                                                "puzzle_only": (0, 1), "both": (1, 1)}.items()}}
    (OUT / "full_v13_four_groups_test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
