"""Shared BoltCrypt evaluation wrapper; logs only observable state transitions."""
import gymnasium as gym
import numpy as np


class BoltCryptAuditWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps=1000):
        super().__init__(env)
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        self.max_episode_steps = max_episode_steps
        self.steps = 0
        self.visited_rooms = set()
        self.visited_positions = set()
        self.previous = None
        self.finished = False

    @staticmethod
    def position(obs):
        return (tuple(map(int, obs["global_pos"])), tuple(map(int, obs["agent_pos"])))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.steps = 0
        self.finished = False
        self.previous = obs
        room, local = self.position(obs)
        self.visited_rooms = {room}
        self.visited_positions = {(room, local)}
        return obs, dict(info)

    def step(self, action):
        if self.finished:
            raise RuntimeError("Call reset() after an episode ends")
        old_room, old_local = self.position(self.previous)
        old_key = int(self.previous["inventory"])
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        room, local = self.position(obs)
        first_room = room not in self.visited_rooms
        repeated_position = (room, local) in self.visited_positions
        no_position_change = (room, local) == (old_room, old_local)
        events = {
            "entered_new_room": first_room,
            "revisited_position": repeated_position,
            "no_position_change": no_position_change,
            "key_acquired": old_key == 0 and int(obs["inventory"]) == 1,
        }
        # A stationary result is observable; its cause (wall, puzzle, moving
        # obstacle, etc.) is not reliably available from BoltCrypt's empty info.
        self.visited_rooms.add(room)
        self.visited_positions.add((room, local))
        if self.steps >= self.max_episode_steps and not terminated:
            truncated = True
        self.finished = bool(terminated or truncated)
        out = dict(info)
        out["audit"] = {
            "step": self.steps,
            "action": int(action),
            "room": list(room),
            "local_position": list(local),
            "events": events,
            "end_reason": "success" if terminated else "step_limit" if truncated else None,
        }
        self.previous = obs
        return obs, reward, terminated, truncated, out


def make_env(generator_config, max_episode_steps=1000, exploration_reward=None):
    import boltcrypt  # registers environment
    # Override BoltCrypt's registered 1000-step TimeLimit when a diagnostic
    # explicitly requests a different episode cap.
    env = gym.make("BoltCrypt-v0", generator_config=generator_config,
                   max_episode_steps=max_episode_steps)
    if exploration_reward is not None:
        from boltcrypt.wrapper.roomdiscoveryreward import RoomDiscoveryReward
        env = RoomDiscoveryReward(env, discovery_reward=exploration_reward)
    return BoltCryptAuditWrapper(env, max_episode_steps=max_episode_steps)
