"""Matched B-style exploration-reward ablation on the E-style maze experiment."""
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from traditional_maze_e import MazeEnv, MODEL_SEEDS, TEST_MAPS, STEPS, MAX_STEPS

OUT = Path(__file__).resolve().parents[1] / 'outputs'


class ExplorationMaze(MazeEnv):
    def __init__(self, hint=False, train_seed=None):
        super().__init__(hint, train_seed)
        self.visit_counts = {}

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.visit_counts = {self.pos: 1}
        return obs, info

    def step(self, action):
        old = self.pos
        obs, reward, done, truncated, info = super().step(action)
        if self.pos != old:
            previous = self.visit_counts.get(self.pos, 0)
            if previous == 0:
                reward += 0.03  # first visit to a cell
            elif previous >= 2:
                reward -= min(0.01 * (previous - 1), 0.05)  # escalating revisit cost
            self.visit_counts[self.pos] = previous + 1
        return obs, reward, done, truncated, info


def main():
    rows = []
    timings = []
    for condition, hint in [('B_explore', False), ('B_plus_E', True)]:
        for model_seed in MODEL_SEEDS:
            env = ExplorationMaze(hint, model_seed)
            model = PPO('MlpPolicy', env, seed=model_seed, n_steps=128, batch_size=64,
                        n_epochs=2, learning_rate=3e-4, device='cpu', verbose=0)
            started = time.perf_counter()
            model.learn(total_timesteps=STEPS)
            timings.append(dict(condition=condition, model_seed=model_seed,
                                seconds=round(time.perf_counter()-started, 2)))
            print(condition, model_seed, timings[-1]['seconds'], flush=True)
            for map_seed in TEST_MAPS:
                test = ExplorationMaze(hint)
                obs, _ = test.reset(seed=map_seed)
                torch.manual_seed(100000 + model_seed + map_seed)
                reward_sum = 0.0
                for step in range(1, MAX_STEPS + 1):
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, done, truncated, _ = test.step(int(action))
                    reward_sum += reward
                    if done or truncated:
                        break
                rows.append(dict(condition=condition, model_seed=model_seed,
                                 map_seed=map_seed, success=int(done), steps=step,
                                 reward=round(reward_sum, 4), unique_cells=len(test.visited)))
    with (OUT / '传统迷宫_B与E组合逐图结果.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = {condition: dict(wins=sum(r['success'] for r in rows if r['condition'] == condition),
                               episodes=240,
                               mean_unique_cells=round(float(np.mean([r['unique_cells'] for r in rows
                                                       if r['condition'] == condition])), 2),
                               mean_reward=round(float(np.mean([r['reward'] for r in rows
                                                       if r['condition'] == condition])), 4))
               for condition in ('B_explore', 'B_plus_E')}
    (OUT / '传统迷宫_B与E组合结果.json').write_text(
        json.dumps(dict(protocol=dict(train_maps=[1000,1199], test_maps=[6000,6079],
                                      model_seeds=MODEL_SEEDS, steps_per_model=STEPS,
                                      max_episode_steps=MAX_STEPS), summary=summary,
                        train_seconds=timings), ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary, flush=True)


if __name__ == '__main__':
    main()
