"""Adds only observations derived from the agent's own episode history."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class HistoryObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old = dict(env.observation_space.spaces)
        old["visit_map"] = spaces.Box(0.0, 1.0, shape=(10, 10), dtype=np.float32)
        old["recent_history"] = spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Dict(old)
        self.visits = {}
        self.room_entries = {}
        self.last_action = None
        self.last_moved = True

    @staticmethod
    def place(obs):
        return tuple(map(int, obs["global_pos"])), tuple(map(int, obs["agent_pos"]))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        room, cell = self.place(obs)
        self.visits = {room: {cell: 1}}
        self.room_entries = {room: 1}
        self.last_action = None
        self.last_moved = True
        return self.observation(obs), info

    def step(self, action):
        old_room, old_cell = self.place(self._last_raw)
        obs, reward, terminated, truncated, info = self.env.step(action)
        room, cell = self.place(obs)
        if room != old_room:
            self.room_entries[room] = self.room_entries.get(room, 0) + 1
        room_visits = self.visits.setdefault(room, {})
        room_visits[cell] = room_visits.get(cell, 0) + 1
        self.last_action = int(action)
        self.last_moved = (room, cell) != (old_room, old_cell)
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, obs):
        room, _ = self.place(obs)
        grid = np.zeros((10, 10), dtype=np.float32)
        for (x, y), count in self.visits.get(room, {}).items():
            grid[y, x] = min(count, 5) / 5.0
        action_one_hot = [float(self.last_action == i) for i in range(4)]
        meta = np.asarray([min(self.room_entries.get(room, 1), 5) / 5.0,
                           float(self.last_moved), *action_one_hot], dtype=np.float32)
        self._last_raw = obs
        return {**obs, "visit_map": grid, "recent_history": meta}
