"""Classify BoltCrypt interactions only from visible before/after observations."""
import gymnasium as gym
import numpy as np


class MechanicsAudit(gym.Wrapper):
    MOVES = ((0, 1), (0, -1), (1, 0), (-1, 0))

    def __init__(self, env):
        super().__init__(env)
        self.previous = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous = obs
        return obs, info

    def step(self, action):
        action = int(action)
        old = self.previous
        old_room = tuple(map(int, old["global_pos"]))
        ox, oy = map(int, old["agent_pos"])
        dx, dy = self.MOVES[action]
        tx, ty = ox + dx, oy + dy
        grid = old["grid"]
        attempted_tile = int(grid[ty, tx]) if 0 <= tx < grid.shape[1] and 0 <= ty < grid.shape[0] else None
        old_boxes = set(map(tuple, np.argwhere(grid == 5).tolist()))
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_room = tuple(map(int, obs["global_pos"]))
        new_pos = tuple(map(int, obs["agent_pos"]))
        stayed = old_room == new_room and (ox, oy) == new_pos
        same_room = old_room == new_room
        new_boxes = set(map(tuple, np.argwhere(obs["grid"] == 5).tolist())) if same_room else set()
        key_acquired = int(old["inventory"]) == 0 and int(obs["inventory"]) == 1
        if old_room != new_room:
            event = "room_transition"
        elif key_acquired:
            event = "key_acquired"
        elif old_boxes != new_boxes:
            event = "boulder_moved"
        elif terminated:
            event = "exit_reached"
        elif stayed and attempted_tile == 2:
            event = "door_rejected_unknown_cause"
        elif stayed and attempted_tile == 5:
            event = "boulder_push_rejected"
        elif stayed and attempted_tile in (1, 9):
            event = "wall_or_column_collision"
        elif stayed:
            event = "no_move_unknown_cause"
        else:
            event = "move"
        out = dict(info)
        out["mechanics"] = {"event": event, "attempted_tile": attempted_tile,
                            "room_changed": old_room != new_room, "position_changed": not stayed,
                            "key_acquired": key_acquired, "boulder_moved": same_room and old_boxes != new_boxes,
                            "terminated": bool(terminated)}
        self.previous = obs
        return obs, reward, terminated, truncated, out
