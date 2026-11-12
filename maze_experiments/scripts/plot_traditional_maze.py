import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from traditional_maze_e import build_maze, shortest_path

out = Path(__file__).resolve().parents[1] / 'outputs'
rows = list(csv.DictReader((out / '传统迷宫_E组逐图结果.csv').open(encoding='utf-8-sig')))
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 2, figsize=(11, 4.7), layout='constrained')
colors = ['#758ea5', '#2bb5af']
names = ['基础 PPO', 'PPO + 目标提示']
counts = [sum(int(r['success']) for r in rows if r['condition'] == c)
          for c in ('baseline', 'goal_hint')]
for i, (name, count) in enumerate(zip(names, counts)):
    ax[0].bar(i, count / 240 * 100, color=colors[i], width=.55)
    ax[0].text(i, count / 240 * 100 + .6, f'{count}/240  ({count/240:.1%})',
               ha='center', fontsize=12, weight='bold')
ax[0].set_xticks(range(2), names)
ax[0].set_ylim(0, 26)
ax[0].set_ylabel('未见迷宫通关率 (%)')
ax[0].set_title('相同 80 张未见迷宫 × 3 个模型种子')
ax[0].grid(axis='y', alpha=.2)
ax[0].set_axisbelow(True)

for j, seed in enumerate((41, 42, 43)):
    values = [sum(int(r['success']) for r in rows if r['condition'] == c
                  and int(r['model_seed']) == seed) / 80 * 100
              for c in ('baseline', 'goal_hint')]
    ax[1].plot(range(2), values, marker='o', linewidth=2.3, label=f'种子 {seed}')
    for x, v in enumerate(values):
        ax[1].annotate(f'{v:.1f}%', (x, v), xytext=(5, 5),
                       textcoords='offset points', fontsize=9)
ax[1].set_xticks(range(2), names)
ax[1].set_xlim(-.15, 1.2)
ax[1].set_ylim(0, 29)
ax[1].set_ylabel('通关率 (%)')
ax[1].set_title('每个模型种子均有提升')
ax[1].legend(frameon=False)
ax[1].grid(axis='y', alpha=.2)
ax[1].set_axisbelow(True)
fig.suptitle('传统迷宫：E 式目标提示与弱 PPO 的配对比较', fontsize=17, weight='bold')
fig.savefig(out / '传统迷宫_E组通关率对比.png', dpi=200, facecolor='white')

grid, start, goal = build_maze(6000)
path = shortest_path(grid, start, goal)
waypoint = path[min(3, len(path)-1)]
fig, ax = plt.subplots(figsize=(6.4, 6.4), layout='constrained')
ax.imshow(grid, cmap='gray_r', vmin=0, vmax=1)
ax.plot([p[0] for p in path], [p[1] for p in path], color='#f0ad4e',
        linewidth=2, alpha=.8, label='最短路径（供生成提示）')
ax.scatter(*start, s=230, c='#2bb5af', marker='o', edgecolors='white', zorder=5, label='起点')
ax.scatter(*goal, s=260, c='#dc6575', marker='*', edgecolors='white', zorder=5, label='出口')
ax.scatter(*waypoint, s=180, c='#735bd1', marker='D', edgecolors='white', zorder=5, label='目标提示点')
ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
ax.grid(which='minor', color='#9bb1ba', linewidth=.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('未见迷宫示例：种子 6000\n浅色可走，深色为墙', fontsize=14)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.03), ncol=2, frameon=False)
fig.savefig(out / '传统迷宫_E组地图示例.png', dpi=200, facecolor='white')
