"""Budget-matched PPO versus E-style waypoint hint on observable random mazes."""
import csv
import json
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

OUT = Path(__file__).resolve().parents[1] / 'outputs'
OUT.mkdir(exist_ok=True)
TRAIN_MAPS = np.arange(1000, 1200)
TEST_MAPS = range(6000, 6080)
MODEL_SEEDS = (41, 42, 43)
STEPS = 30000
SIZE = 7
MAX_STEPS = 40
DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def shortest_path(grid, start, goal):
    queue = deque([start])
    parent = {start: None}
    while queue:
        p = queue.popleft()
        if p == goal:
            break
        for dx, dy in DIRS:
            q = (p[0]+dx, p[1]+dy)
            if 0 <= q[0] < SIZE and 0 <= q[1] < SIZE and not grid[q[1], q[0]] and q not in parent:
                parent[q] = p
                queue.append(q)
    if goal not in parent:
        return [start]
    path = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path[::-1]


def build_maze(seed):
    rng = np.random.default_rng(seed)
    grid = np.ones((SIZE, SIZE), dtype=np.uint8)
    current = (1, 1)
    stack = [current]
    visited = {current}
    grid[current[1], current[0]] = 0
    while stack:
        x, y = stack[-1]
        choices = [(x+2*dx, y+2*dy, dx, dy) for dx,dy in DIRS
                   if 1 <= x+2*dx < SIZE-1 and 1 <= y+2*dy < SIZE-1
                   and (x+2*dx,y+2*dy) not in visited]
        if not choices:
            stack.pop()
            continue
        nx, ny, dx, dy = choices[int(rng.integers(len(choices)))]
        grid[y+dy,x+dx] = 0
        grid[ny,nx] = 0
        visited.add((nx,ny))
        stack.append((nx,ny))
    cells = list(zip(*np.where(grid == 0)))
    a = cells[int(rng.integers(len(cells)))]
    start = (int(a[1]), int(a[0]))
    distances = [(len(shortest_path(grid,start,(int(yx[1]),int(yx[0])))),(int(yx[1]),int(yx[0]))) for yx in cells]
    longest = max(d for d,_ in distances)
    goal = [p for d,p in distances if d == longest][int(rng.integers(sum(d == longest for d,_ in distances)))]
    return grid,start,goal


class MazeEnv(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self, hint=False, train_seed=None):
        super().__init__()
        self.hint = hint
        self.rng = np.random.default_rng(train_seed)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-1, 1, shape=(SIZE*SIZE*3+8+7,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        map_seed = int(self.rng.choice(TRAIN_MAPS)) if seed is None else int(seed)
        self.grid,self.pos,self.goal = build_maze(map_seed)
        self.visited = {self.pos}
        self.steps = 0
        return self._obs(), {'map_seed': map_seed}

    def _obs(self):
        a = np.zeros((3,SIZE,SIZE),dtype=np.float32)
        a[0] = self.grid
        a[1,self.pos[1],self.pos[0]] = 1
        a[2,self.goal[1],self.goal[0]] = 1
        local = [float(self.grid[self.pos[1]+dy,self.pos[0]+dx]) for dx,dy in DIRS]
        local += [self.pos[0]/SIZE,self.pos[1]/SIZE,
                  (self.goal[0]-self.pos[0])/SIZE,(self.goal[1]-self.pos[1])/SIZE]
        extra = np.zeros(7,dtype=np.float32)
        if self.hint:
            path = shortest_path(self.grid,self.pos,self.goal)
            waypoint = path[min(3,len(path)-1)]
            extra[4 if len(path)==1 else 3] = 1
            extra[5:] = [(waypoint[0]-self.pos[0])/SIZE,(waypoint[1]-self.pos[1])/SIZE]
        return np.concatenate((a.ravel(),np.asarray(local,dtype=np.float32),extra))

    def step(self, action):
        dx,dy = DIRS[int(action)]
        nxt = (self.pos[0]+dx,self.pos[1]+dy)
        self.steps += 1
        old_distance = len(shortest_path(self.grid,self.pos,self.goal))
        reward = -0.005
        if self.grid[nxt[1],nxt[0]]:
            reward -= 0.02
        else:
            self.pos = nxt
            if self.pos in self.visited:
                reward -= 0.01
            self.visited.add(self.pos)
        new_distance = len(shortest_path(self.grid,self.pos,self.goal))
        reward += 0.05 * (old_distance-new_distance)
        done = self.pos == self.goal
        if done:
            reward += 1.0
        return self._obs(),reward,done,self.steps >= MAX_STEPS and not done,{}


def main():
    rows, timings = [], []
    for condition in ('baseline','goal_hint'):
        for model_seed in MODEL_SEEDS:
            env = MazeEnv(condition=='goal_hint',model_seed)
            model = PPO('MlpPolicy',env,seed=model_seed,n_steps=128,batch_size=64,n_epochs=2,
                        learning_rate=3e-4,device='cpu',verbose=0)
            started = time.perf_counter()
            model.learn(total_timesteps=STEPS)
            timings.append(dict(condition=condition,model_seed=model_seed,seconds=round(time.perf_counter()-started,2)))
            print(condition,model_seed,timings[-1]['seconds'],flush=True)
            for map_seed in TEST_MAPS:
                test = MazeEnv(condition=='goal_hint')
                obs,_ = test.reset(seed=map_seed)
                import torch
                torch.manual_seed(100000+model_seed+map_seed)
                reward_sum = 0.0
                for step in range(1,MAX_STEPS+1):
                    action,_ = model.predict(obs,deterministic=False)
                    obs,reward,done,truncated,_ = test.step(int(action))
                    reward_sum += reward
                    if done or truncated:
                        break
                rows.append(dict(condition=condition,model_seed=model_seed,map_seed=map_seed,
                                 success=int(done),steps=step,reward=round(reward_sum,4),
                                 unique_cells=len(test.visited)))
    with (OUT/'传统迷宫_E组逐图结果.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary={c:{'wins':sum(r['success'] for r in rows if r['condition']==c),
                'episodes':sum(r['condition']==c for r in rows),
                'mean_reward':round(float(np.mean([r['reward'] for r in rows if r['condition']==c])),4),
                'mean_unique_cells':round(float(np.mean([r['unique_cells'] for r in rows if r['condition']==c])),2)}
             for c in ('baseline','goal_hint')}
    (OUT/'传统迷宫_E组结果.json').write_text(json.dumps({'protocol':{'size':SIZE,'train_maps':[1000,1199],
        'test_maps':[6000,6079],'model_seeds':MODEL_SEEDS,'steps_per_model':STEPS,'max_episode_steps':MAX_STEPS},
        'summary':summary,'train_seconds':timings},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
