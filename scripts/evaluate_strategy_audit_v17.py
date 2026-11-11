"""One-pass, predeclared six-group audit on new full-dungeon maps."""
import csv
import json
import time
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
from goal_observation_v14 import GoalObservation, GOALS
from mechanics_audit import MechanicsAudit
from limited_full_guide import LimitedFullGuide

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
PLAN = json.loads((ROOT / "experiment_config_v17_audit.json").read_text(encoding="utf-8"))
CFG = json.loads((ROOT / "experiment_config_v13.json").read_text(encoding="utf-8-sig"))
assert PLAN["max_episode_steps"] == CFG["max_episode_steps"] == 1000
DETAIL = REPORTS / "full_v17_decision_audit.csv"
EPISODES = REPORTS / "full_v17_episodes.csv"
SUMMARY = REPORTS / "full_v17_summary.json"
if any(path.exists() for path in (DETAIL, EPISODES, SUMMARY)):
    raise FileExistsError("v17 output already exists")


def make_group_env(group):
    discovery = CFG["exploration"]["discovery_reward"] if group in ("A2", "B", "D", "E") else None
    env = make_env(CFG["generator_config"], 1000, discovery)
    env = WallPenalty(env, CFG["wall_collision_reward"])
    if group in ("B", "D", "E"):
        env = DiminishingCellNovelty(env, CFG["first_cell_bonus"])
        env = IncreasingRevisitPenalty(env, CFG["revisit_penalty_scale"], CFG["revisit_penalty_cap"])
    env = HistoryObservation(NavigationObservation(env))
    if group == "E":
        env = GoalObservation(env)
    return MechanicsAudit(env)


def model_path(group, seed):
    if group == "A2": return ROOT / "runs_a2" / f"A2_seed{seed}" / "model.zip"
    if group == "E": return ROOT / "runs_v14" / f"E_seed{seed}" / "model.zip"
    source = "A" if group == "C" else "B" if group == "D" else group
    return ROOT / "runs_v13" / f"{source}_seed{seed}" / "model.zip"


def place(obs):
    return tuple(map(int, obs["global_pos"])), tuple(map(int, obs["agent_pos"]))


def visible_tile(obs, target, tile):
    if target is None: return False
    x, y = map(int, target)
    grid = obs["grid"]
    return 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1] and int(grid[y, x]) == tile


def rule_reason_supported(reason, target, obs, guide):
    if reason == "visible_key":
        return not int(obs["inventory"]) and visible_tile(obs, target, 6)
    if reason == "visible_boulder_plan":
        room = place(obs)[0]
        known_switches = guide.switches.get(room, set())
        return (isinstance(target, list) and bool(target) and
                all(tuple(p) in known_switches and
                    (visible_tile(obs, p, 4) or visible_tile(obs, p, 5)) for p in target))
    if reason == "visible_final_exit":
        return visible_tile(obs, target, 3)
    if reason == "exit_tile_reentry":
        return visible_tile(obs, place(obs)[1], 3)
    if reason in ("unexplored_door", "route_to_frontier"):
        return visible_tile(obs, target, 2)
    return False


def goal_supported(kind, target, obs, guide):
    if kind == "none": return target is None
    if kind == "key": return not int(obs["inventory"]) and visible_tile(obs, target, 6)
    if kind == "puzzle":
        return (tuple(target) in guide.switches.get(place(obs)[0], set()) and
                (visible_tile(obs, target, 4) or visible_tile(obs, target, 5)))
    if kind == "frontier": return visible_tile(obs, target, 2)
    if kind == "exit": return visible_tile(obs, target, 3)
    return False


def goal_vector_supported(kind, target, obs):
    actual = np.asarray(obs["high_level_goal"])
    expected = np.array([float(kind == name) for name in GOALS] +
                        ([0.0, 0.0] if target is None else
                         [(target[0] - place(obs)[1][0]) / 9.0,
                          (target[1] - place(obs)[1][1]) / 9.0]), dtype=np.float32)
    return bool(np.allclose(actual, expected, atol=1e-6))


detail_fields = ["group", "train_seed", "map_seed", "step", "source", "trigger", "reason", "goal",
                 "target", "ppo_action", "submitted_action", "action_in_space", "reason_supported",
                 "goal_supported", "goal_vector_supported", "trace_consistent", "move_accepted",
                 "progress_toward_goal", "event", "reward", "room_before", "pos_before", "room_after", "pos_after"]
episode_rows = []
with DETAIL.open("w", newline="", encoding="utf-8") as detail_file:
    writer = csv.DictWriter(detail_file, fieldnames=detail_fields)
    writer.writeheader()
    for group in PLAN["groups"]:
        for train_seed in PLAN["train_seeds"]:
            model = PPO.load(model_path(group, train_seed), device="cpu")
            for map_seed in PLAN["map_seeds"]:
                env = make_group_env(group)
                started = time.perf_counter()
                try:
                    obs, _ = env.reset(seed=map_seed)
                    torch.manual_seed(100000 + map_seed + train_seed)
                    rooms = env.unwrapped.generator.grid  # metadata only; never sent to model or guide
                    locked = int(any(room.is_locked for room in rooms.values()))
                    puzzle = int(any(room.puzzle_type.value == "boulder" for room in rooms.values()))
                    start_room = place(obs)[0]
                    controller = LimitedFullGuide(CFG["guidance"], obs) if group in ("C", "D") else None
                    visited_rooms = {start_room}
                    first_room = first_key = first_puzzle = first_exit = None
                    first_goal_key = first_goal_puzzle = first_goal_exit = None
                    walls = door_rejections = loops = loop_streak = loop_bouts = 0
                    rule_actions = rule_supported = rule_accepted = rule_space = 0
                    goal_steps = goal_supported_steps = goal_vector_steps = goal_approach = goal_away = 0
                    trace_checked = trace_consistent = 0
                    reasons = Counter()
                    positions = [place(obs)]
                    for step in range(1, 1001):
                        before = obs
                        before_place = place(before)
                        was_solved = bool(env.unwrapped.curr_room.is_solved)
                        raw = int(model.predict(before, deterministic=False)[0])
                        if controller:
                            action, reason, target, engaged, trigger = controller.decide(before, raw)
                            kind = None
                        elif group == "E":
                            action, reason, target, engaged, trigger = raw, "ppo_only", None, False, None
                            kind, goal_target = env.env.last_goal
                        else:
                            action, reason, target, engaged, trigger = raw, "ppo_only", None, False, None
                            kind = None
                        action = int(action)
                        in_space = int(env.action_space.contains(action))
                        if not in_space: raise AssertionError((group, train_seed, map_seed, step, action))
                        reason_ok = int(rule_reason_supported(reason, target, before, controller.guide)) if engaged else ""
                        if kind is not None:
                            goal_ok = int(goal_supported(kind, goal_target, before, env.env.guide))
                            vector_ok = int(goal_vector_supported(kind, goal_target, before))
                            goal_steps += 1
                            goal_supported_steps += goal_ok
                            goal_vector_steps += vector_ok
                            if kind == "key" and first_goal_key is None: first_goal_key = step
                            if kind == "puzzle" and first_goal_puzzle is None: first_goal_puzzle = step
                            if kind == "exit" and first_goal_exit is None: first_goal_exit = step
                        else:
                            goal_ok = vector_ok = ""
                        after, reward, terminated, truncated, info = env.step(action)
                        obs = after
                        after_place = place(after)
                        mechanics = info["mechanics"]
                        event = mechanics["event"]
                        if controller: controller.observe_transition(before, action, after, mechanics)
                        accepted = int(before_place != after_place or event == "boulder_moved" or terminated)
                        trace_ok = int(int(info["audit"]["action"]) == action and
                                       bool(info["audit"]["events"]["no_position_change"]) ==
                                       (before_place == after_place) and
                                       bool(mechanics["room_changed"]) == (before_place[0] != after_place[0]))
                        trace_checked += 1
                        trace_consistent += trace_ok
                        if engaged:
                            rule_actions += 1
                            rule_supported += reason_ok
                            rule_accepted += accepted
                            rule_space += in_space
                            reasons[reason] += 1
                        progress = ""
                        if kind in ("key", "frontier", "exit") and goal_target is not None and before_place[0] == after_place[0]:
                            old_dist = sum(abs(a-b) for a,b in zip(before_place[1], goal_target))
                            new_dist = sum(abs(a-b) for a,b in zip(after_place[1], goal_target))
                            progress = 1 if new_dist < old_dist else -1 if new_dist > old_dist else 0
                            goal_approach += int(progress == 1)
                            goal_away += int(progress == -1)
                        if engaged or kind is not None:
                            writer.writerow({"group": group, "train_seed": train_seed, "map_seed": map_seed,
                                             "step": step, "source": "rule" if engaged else "PPO_goal",
                                             "trigger": trigger or "", "reason": reason, "goal": kind or "",
                                             "target": json.dumps(target if engaged else goal_target),
                                             "ppo_action": raw, "submitted_action": action,
                                             "action_in_space": in_space, "reason_supported": reason_ok,
                                             "goal_supported": goal_ok, "goal_vector_supported": vector_ok,
                                             "trace_consistent": trace_ok, "move_accepted": accepted,
                                             "progress_toward_goal": progress, "event": event,
                                             "reward": round(float(reward), 5),
                                             "room_before": before_place[0], "pos_before": before_place[1],
                                             "room_after": after_place[0], "pos_after": after_place[1]})
                        visited_rooms.add(after_place[0])
                        if first_room is None and after_place[0] != start_room: first_room = step
                        if first_key is None and event == "key_acquired": first_key = step
                        if (first_puzzle is None and before_place[0] == after_place[0] and not was_solved
                                and env.unwrapped.curr_room.is_solved
                                and env.unwrapped.curr_room.puzzle_type.value == "boulder"):
                            first_puzzle = step
                        if terminated: first_exit = step
                        walls += int(info["wall_penalty"]["confirmed_collision"])
                        door_rejections += int(event == "door_rejected_unknown_cause")
                        is_loop = len(positions) >= 3 and positions[-3] == positions[-1] and positions[-2] == after_place and positions[-1] != after_place
                        if is_loop:
                            loops += 1
                            loop_bouts += int(loop_streak == 0)
                            loop_streak += 1
                        else: loop_streak = 0
                        positions.append(after_place)
                        if terminated or truncated: break
                    else: raise AssertionError("Episode not ended")
                    episode_rows.append({"group": group, "train_seed": train_seed, "map_seed": map_seed,
                                         "locked_map": locked, "puzzle_map": puzzle, "steps": step,
                                         "success": int(terminated), "first_room_step": first_room or "",
                                         "first_key_step": first_key or "", "first_puzzle_step": first_puzzle or "",
                                         "first_exit_step": first_exit or "", "first_goal_key_step": first_goal_key or "",
                                         "first_goal_puzzle_step": first_goal_puzzle or "",
                                         "first_goal_exit_step": first_goal_exit or "",
                                         "rooms_visited": len(visited_rooms), "wall_collisions": walls,
                                         "door_rejections": door_rejections, "two_cell_loops": loops,
                                         "loop_bouts": loop_bouts, "rule_actions": rule_actions,
                                         "rule_supported": rule_supported, "rule_accepted": rule_accepted,
                                         "rule_in_space": rule_space, "goal_steps": goal_steps,
                                         "goal_supported_steps": goal_supported_steps,
                                         "goal_vector_steps": goal_vector_steps,
                                         "goal_approach": goal_approach, "goal_away": goal_away,
                                         "trace_checked": trace_checked, "trace_consistent": trace_consistent,
                                         "rule_reasons": json.dumps(reasons, sort_keys=True),
                                         "evaluation_wall_seconds": round(time.perf_counter() - started, 3)})
                finally:
                    env.close()
            print(f"completed {group} seed {train_seed}", flush=True)

with EPISODES.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(episode_rows[0]))
    writer.writeheader()
    writer.writerows(episode_rows)
summary = {"protocol": PLAN["protocol"], "maps": PLAN["map_seeds"], "per_group": {}}
for group in PLAN["groups"]:
    part = [r for r in episode_rows if r["group"] == group]
    assert len(part) == 24
    summary["per_group"][group] = {
        "episodes": 24, "successes": sum(r["success"] for r in part),
        "left_start_room": sum(bool(r["first_room_step"]) for r in part),
        "key_episodes": sum(bool(r["first_key_step"]) for r in part),
        "puzzle_episodes": sum(bool(r["first_puzzle_step"]) for r in part),
        "mean_rooms_visited": round(sum(r["rooms_visited"] for r in part) / 24, 2),
        "mean_wall_collisions": round(sum(r["wall_collisions"] for r in part) / 24, 1),
        "mean_two_cell_loops": round(sum(r["two_cell_loops"] for r in part) / 24, 1),
        "rule_actions": sum(r["rule_actions"] for r in part),
        "rule_supported": sum(r["rule_supported"] for r in part),
        "rule_accepted": sum(r["rule_accepted"] for r in part),
        "goal_steps": sum(r["goal_steps"] for r in part),
        "goal_supported_steps": sum(r["goal_supported_steps"] for r in part),
        "goal_vector_steps": sum(r["goal_vector_steps"] for r in part),
        "goal_approach": sum(r["goal_approach"] for r in part),
        "goal_away": sum(r["goal_away"] for r in part),
        "trace_checked": sum(r["trace_checked"] for r in part),
        "trace_consistent": sum(r["trace_consistent"] for r in part),
        "evaluation_wall_seconds": round(sum(r["evaluation_wall_seconds"] for r in part), 1)}
SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False), flush=True)
