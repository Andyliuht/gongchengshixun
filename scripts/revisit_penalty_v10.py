"""Bounded, increasing penalty for moving onto previously visited room cells."""
import math
import gymnasium as gym


class IncreasingRevisitPenalty(gym.Wrapper):
    def __init__(self, env, scale=0.03, cap=0.09):
        super().__init__(env)
        if scale <= 0 or cap <= 0:
            raise ValueError("scale and cap must be positive")
        self.scale = float(scale)
        self.cap = float(cap)
        self.counts = {}
        self.previous = None

    @staticmethod
    def place(obs):
        return tuple(map(int, obs["global_pos"])), tuple(map(int, obs["agent_pos"]))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.counts = {self.place(obs): 1}
        self.previous = self.place(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current = self.place(obs)
        moved = current != self.previous
        entered_room = current[0] != self.previous[0]
        prior = self.counts.get(current, 0)
        penalty = min(self.cap, self.scale * math.log2(prior + 1)) if moved and not entered_room and prior else 0.0
        self.counts[current] = prior + 1 if moved else max(prior, 1)
        self.previous = current
        out = dict(info)
        out["revisit_penalty"] = {"prior_visits": prior, "moved": moved,
                                  "entered_room": entered_room, "penalty": penalty}
        return obs, reward - penalty, terminated, truncated, out
