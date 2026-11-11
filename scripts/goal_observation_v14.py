"""Observable high-level goal hints; never chooses or overrides an action."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from full_dungeon_guide import FullDungeonGuide
from local_navigation_guide import path_to


GOALS = ("none", "key", "puzzle", "frontier", "exit")


class GoalObservation(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        old = dict(env.observation_space.spaces)
        old["high_level_goal"] = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict(old)
        self.guide = FullDungeonGuide()
        self.last_goal = None
        self.goal_counts = {name: 0 for name in GOALS}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.guide = FullDungeonGuide()
        self.last_goal = None
        self.goal_counts = {name: 0 for name in GOALS}
        return self._augment(obs), info

    def step(self, action):
        before = self._last_raw
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.guide.observe_transition(before, int(action), obs)
        return self._augment(obs), reward, terminated, truncated, info

    def _select(self, obs):
        guide = self.guide
        guide.observe(obs)
        room = guide.room(obs)
        here = guide.local(obs)
        guide.switches.setdefault(room, set()).update(guide.targets(obs, 4))
        if not int(obs["inventory"]):
            keys = guide.targets(obs, 6)
            if keys:
                target = min(keys, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
                return "key", target
        if room not in guide.solved_rooms and guide.switches[room]:
            switches = sorted(guide.switches[room])
            target = min(switches, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
            return "puzzle", target
        exits = guide.targets(obs, 3)
        if exits:
            target = min(exits, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
            return "exit", target
        route = guide._route_to_frontier(room)
        if route:
            return "frontier", route[0][1]
        return "none", None

    def _augment(self, obs):
        self._last_raw = obs
        kind, target = self._select(obs)
        self.last_goal = (kind, target)
        self.goal_counts[kind] += 1
        flags = [float(kind == name) for name in GOALS]
        if target is None:
            relative = [0.0, 0.0]
        else:
            here = self.guide.local(obs)
            relative = [(target[0] - here[0]) / 9.0, (target[1] - here[1]) / 9.0]
        return {**obs, "high_level_goal": np.asarray(flags + relative, dtype=np.float32)}
