"""Adds local door direction and visible wall flags without hidden-map data."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class NavigationObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old = dict(env.observation_space.spaces)
        old["navigation"] = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict(old)

    def observation(self, obs):
        grid = obs["grid"]
        x, y = map(int, obs["agent_pos"])
        doors = [(a, b) for b in range(grid.shape[0]) for a in range(grid.shape[1])
                 if int(grid[b, a]) == 2]
        if doors:
            door_x, door_y = min(doors, key=lambda p: (abs(p[0]-x)+abs(p[1]-y), p[1], p[0]))
            relative = [(door_x-x)/9.0, (door_y-y)/9.0, 1.0]
        else:
            relative = [0.0, 0.0, 0.0]
        blocked = []
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x+dx, y+dy
            blocked.append(float(not (0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0])
                                 or int(grid[ny, nx]) in (1, 9)))
        return {**obs, "navigation": np.asarray(relative + blocked, dtype=np.float32)}
