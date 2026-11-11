"""Penalize persistent two-cell oscillation, while allowing a single backtrack."""
from collections import deque
import gymnasium as gym


class PersistentLoopPenalty(gym.Wrapper):
    def __init__(self, env, loop_reward=-0.10):
        super().__init__(env)
        self.loop_reward = float(loop_reward)
        self.recent = deque(maxlen=4)

    @staticmethod
    def place(obs):
        return (tuple(map(int, obs["global_pos"])), tuple(map(int, obs["agent_pos"])))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.recent.clear()
        self.recent.append(self.place(obs))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current = self.place(obs)
        old = list(self.recent)
        # A->B->A->B: a sustained two-cell loop. A single A->B->A
        # is allowed because returning once can be necessary for navigation.
        loop = (len(old) >= 3 and old[-3] == old[-1]
                and old[-2] == current and old[-1] != current)
        self.recent.append(current)
        if loop:
            reward = min(reward, self.loop_reward)
        out = dict(info)
        out["loop_penalty"] = {"persistent_two_cell_loop": bool(loop)}
        return obs, reward, terminated, truncated, out
