import csv
from pathlib import Path
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parents[1] / 'outputs'
rows = []
for filename in ('传统迷宫_E组逐图结果.csv', '传统迷宫_B与E组合逐图结果.csv'):
    rows += list(csv.DictReader((out / filename).open(encoding='utf-8-sig')))
conditions = ('baseline', 'B_explore', 'goal_hint', 'B_plus_E')
labels = ('基础 PPO', 'B 式探索奖励', 'E 式目标提示', 'B + E')
counts = [sum(int(r['success']) for r in rows if r['condition'] == c) for c in conditions]
colors = ('#8299ad', '#eeac59', '#27afa7', '#866fd3')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
ax.bar(range(4), [n / 240 * 100 for n in counts], color=colors, width=.65)
for i, n in enumerate(counts):
    ax.text(i, n / 240 * 100 + .45, f'{n}/240\n{n/240:.1%}', ha='center', weight='bold')
ax.set_xticks(range(4), labels)
ax.set_ylim(0, 25)
ax.set_ylabel('未见迷宫通关率 (%)')
ax.set_title('传统迷宫：探索奖励 B 与目标提示 E 的 2×2 消融', fontsize=15, weight='bold')
ax.grid(axis='y', alpha=.2)
ax.set_axisbelow(True)
ax.text(.5, -.18, '同一批 80 张未见迷宫 × 3 个模型种子；每个模型训练 30,000 步',
        ha='center', transform=ax.transAxes, color='#596a75')
fig.savefig(out / '传统迷宫_B与E组合对比.png', dpi=200, facecolor='white')
