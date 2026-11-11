"""Evaluate fixed and procedural A-E models on identical seen/validation/test maps."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from evaluate_full_v13_test import run as run_abcd
from evaluate_goal_v14 import run as run_e

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
PLAN = json.loads((ROOT / "experiment_config_v16.json").read_text(encoding="utf-8-sig"))


def model_path(layout, group, seed):
    source = "A" if group == "C" else "B" if group == "D" else group
    if layout == "fixed":
        return ROOT / "runs_v16_fixed" / f"{source}_seed{seed}" / "model.zip"
    if source == "E":
        return ROOT / "runs_v14" / f"E_seed{seed}" / "model.zip"
    return ROOT / "runs_v13" / f"{source}_seed{seed}" / "model.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["seen", "validation", "test"], required=True)
    args = parser.parse_args()
    if args.split == "seen":
        map_seeds = [PLAN["seen_map_evaluation_seed"]]
    else:
        lo, hi = PLAN["development_map_seeds" if args.split == "validation" else "sealed_test_map_seeds"]
        map_seeds = range(lo, hi + 1)
    output = REPORTS / f"full_v16_generalization_{args.split}.csv"
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for layout in ("fixed", "procedural"):
        for group in "ABCDE":
            for seed in (41, 42, 43):
                model = PPO.load(model_path(layout, group, seed), device="cpu")
                for map_seed in map_seeds:
                    started = time.perf_counter()
                    if group == "E":
                        row = run_e(model, seed, map_seed)
                    else:
                        row, trace = run_abcd(model, group, seed, map_seed,
                                              save_trace=(map_seed == map_seeds[0] and seed == 41
                                                          and group in "CD"))
                        if trace:
                            (REPORTS / f"full_v16_{args.split}_trace_{layout}_{group}_seed41_map{map_seed}.json").write_text(
                                json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
                    row["training_layout"] = layout
                    row["evaluation_split"] = args.split
                    row["wall_seconds"] = round(time.perf_counter() - started, 3)
                    rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for layout in ("fixed", "procedural"):
        summary[layout] = {}
        for group in "ABCDE":
            part = [r for r in rows if r["training_layout"] == layout and r["group"] == group]
            summary[layout][group] = {
                "episodes": len(part),
                "successes": sum(int(r["success"]) for r in part),
                "left_start_room": sum(int(r["left_start_room"]) for r in part),
                "mean_rooms_visited": round(float(np.mean([int(r["rooms_visited"]) for r in part])), 2),
                "key_acquisition_episodes": sum(int(r["keys_acquired"]) > 0 for r in part),
                "puzzle_completion_episodes": sum(int(r["puzzles_completed"]) > 0 for r in part),
                "mean_wall_collisions": round(float(np.mean([int(r["wall_collisions"]) for r in part])), 1),
                "mean_two_cell_loops": round(float(np.mean([int(r["two_cell_loop_events"]) for r in part])), 1),
                "rule_actions": sum(int(r["rule_actions"]) for r in part),
                "evaluation_wall_seconds": round(sum(r["wall_seconds"] for r in part), 1),
                "by_seed": {str(seed): {
                    "successes": sum(int(r["success"]) for r in part if r["train_seed"] == seed),
                    "left_start_room": sum(int(r["left_start_room"]) for r in part if r["train_seed"] == seed)}
                    for seed in (41, 42, 43)}}
    (REPORTS / f"full_v16_generalization_{args.split}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
