"""Larger branch-rich maze; budget-matched PPO and E-style hint."""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

import traditional_maze_e as base

base.SIZE = 11
base.MAX_STEPS = 100
OUT = Path(__file__).resolve().parents[1] / 'outputs'
STEPS = 60000
SEEDS = (41, 42, 43)


def junctions(grid):
    return sum(grid[y, x] == 0 and sum(grid[y+dy, x+dx] == 0 for dx, dy in base.DIRS) >= 3
               for y in range(1, base.SIZE - 1) for x in range(1, base.SIZE - 1))


def selected(lo, hi, count):
    result = []
    for seed in range(lo, hi):
        grid, _, _ = base.build_maze(seed)
        if junctions(grid) >= 4:
            result.append(seed)
            if len(result) == count:
                break
    return result


train_maps = selected(1000, 5000, 200)
test_maps = selected(6000, 10000, 80)
assert len(train_maps) == 200 and len(test_maps) == 80
base.TRAIN_MAPS = np.array(train_maps)


def draw_example(seed):
    grid, start, goal = base.build_maze(seed)
    route = base.shortest_path(grid, start, goal)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    fig, ax = plt.subplots(figsize=(7, 7), layout='constrained')
    ax.imshow(grid, cmap='gray_r', vmin=0, vmax=1)
    ax.scatter(*start, c='#29b8b1', s=220, edgecolor='white', label='起点', zorder=5)
    ax.scatter(*goal, c='#e06576', marker='*', s=300, edgecolor='white', label='出口', zorder=5)
    ax.set_xticks(np.arange(-.5, 11, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 11, 1), minor=True)
    ax.grid(which='minor', linewidth=.5, color='#9eaab0')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'11×11 多岔路迷宫 · 种子 {seed}\n岔路口 {junctions(grid)} 个；最短路径 {len(route)-1} 步')
    ax.legend(loc='lower center', bbox_to_anchor=(.5, -.09), ncol=2, frameon=False)
    fig.savefig(OUT / f'多岔路迷宫示例_{seed}.png', dpi=180, facecolor='white')
    plt.close(fig)


def main():
    for seed in test_maps[:3]:
        draw_example(seed)
    rows, timings = [], []
    for condition, hint in (('baseline', False), ('E_hint', True)):
        for model_seed in SEEDS:
            model = PPO('MlpPolicy', base.MazeEnv(hint, model_seed), seed=model_seed,
                        n_steps=128, batch_size=64, n_epochs=2, learning_rate=3e-4,
                        device='cpu', verbose=0)
            started = time.perf_counter()
            model.learn(total_timesteps=STEPS)
            timings.append(dict(condition=condition, model_seed=model_seed,
                                seconds=round(time.perf_counter()-started, 1)))
            print('trained', condition, model_seed, timings[-1]['seconds'], flush=True)
            for map_seed in test_maps:
                env = base.MazeEnv(hint)
                obs, _ = env.reset(seed=map_seed)
                torch.manual_seed(100000 + model_seed + map_seed)
                total = 0.0
                for step in range(1, base.MAX_STEPS + 1):
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, done, truncated, _ = env.step(int(action))
                    total += reward
                    if done or truncated:
                        break
                rows.append(dict(condition=condition, model_seed=model_seed, map_seed=map_seed,
                                 junctions=junctions(env.grid), success=int(done), steps=step,
                                 reward=round(total, 3), unique_cells=len(env.visited)))
    with (OUT / '多岔路迷宫_逐图结果.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    summary = {c: {'wins': sum(r['success'] for r in rows if r['condition'] == c),
                   'episodes': 240, 'mean_unique_cells': round(float(np.mean([r['unique_cells'] for r in rows if r['condition'] == c])), 2)}
               for c in ('baseline', 'E_hint')}
    (OUT / '多岔路迷宫_结果.json').write_text(json.dumps({'protocol': {'size': 11, 'min_junctions': 4,
        'train_map_seeds': train_maps, 'test_map_seeds': test_maps, 'model_seeds': SEEDS,
        'steps_per_model': STEPS, 'max_episode_steps': base.MAX_STEPS},
        'summary': summary, 'train_seconds': timings}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary, flush=True)


if __name__ == '__main__':
    main()
