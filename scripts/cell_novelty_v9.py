"""Episode-level, per-room first-visit reward with diminishing magnitude."""
import math
import gymnasium as gym


class DiminishingCellNovelty(gym.Wrapper):
    def __init__(self, env, first_cell_bonus=0.025):
        super().__init__(env)
        if first_cell_bonus <= 0:
            raise ValueError("first_cell_bonus must be positive")
        self.first_cell_bonus = float(first_cell_bonus)
        self.visited = {}
        self.previous = None

    @staticmethod
    def place(obs):
        return tuple(map(int, obs["global_pos"])), tuple(map(int, obs["agent_pos"]))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        room, cell = self.place(obs)
        self.visited = {room: {cell}}
        self.previous = (room, cell)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        room, cell = self.place(obs)
        seen = self.visited.setdefault(room, set())
        entered_room = room != self.previous[0]
        first_visit = cell not in seen and not entered_room
        bonus = self.first_cell_bonus / math.sqrt(len(seen)) if first_visit else 0.0
        seen.add(cell)
        self.previous = (room, cell)
        out = dict(info)
        out["cell_novelty"] = {"first_visit": first_visit, "entered_room": entered_room,
                               "bonus": bonus, "visited_cells_in_room": len(seen)}
        return obs, reward + bonus, terminated, truncated, out
