# 传统迷宫补充实验

本目录是 BoltCrypt 主实验之外的**独立简化迷宫验证**，不使用 BoltCrypt 地图，也不代表原项目 E 组的直接效果。

## 内容

- `scripts/traditional_maze_e.py`：7×7 迷宫环境、基础 PPO 与 E 式最短路径路标提示训练/评估。两组动作均由 PPO 决定。
- `scripts/traditional_maze_b_e.py`：在同一环境加入 B 式首次访问奖励与递增重访惩罚，比较 B 与 B+E。
- `scripts/branch_maze_experiment.py`：11×11 至少 4 个岔路口的更复杂迷宫；比较基础 PPO 与 E 式提示。
- `scripts/plot_traditional_maze.py`、`scripts/plot_maze_b_e.py`：依据逐图 CSV 绘制 7×7 实验图。
- `outputs/`：实验报告、逐图 CSV、汇总 JSON 与地图/结果 PNG。测试结果是实际运行记录，未作人工改写。

## 复现

建议 Python 3.11，安装 `numpy`、`matplotlib`、`torch`、`gymnasium==0.29.1`、`stable-baselines3==2.3.2`。在本目录执行：

```powershell
python .\scripts\traditional_maze_e.py
python .\scripts\traditional_maze_b_e.py
python .\scripts\branch_maze_experiment.py
```

三个脚本各自写入 `outputs/`。运行需要数分钟，CPU 速度依机器而异。基础奖励包含到出口的最短路距离塑形；E 式提示由**完整可见迷宫的最短路径**计算，比 BoltCrypt 原 E 阶段提示更强。详细协议、结果与限制见 `outputs/` 中的三份报告。
