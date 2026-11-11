"""V2 reward: a confirmed wall/column collision costs -0.10 total."""
import gymnasium as gym


class WallPenalty(gym.Wrapper):
    DELTAS = ((0, 1), (0, -1), (1, 0), (-1, 0))
    BLOCKED_TILES = {1, 9}  # BoltCrypt TILE_WALL, TILE_COLUMN

    def __init__(self, env, collision_reward=-0.10):
        super().__init__(env)
        self.collision_reward = float(collision_reward)
        self.previous = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous = obs
        return obs, info

    def step(self, action):
        action = int(action)
        x, y = map(int, self.previous["agent_pos"])
        dx, dy = self.DELTAS[action]
        grid = self.previous["grid"]
        tx, ty = x + dx, y + dy
        target_tile = int(grid[ty, tx]) if 0 <= ty < grid.shape[0] and 0 <= tx < grid.shape[1] else None
        old_room = tuple(map(int, self.previous["global_pos"]))
        old_pos = (x, y)
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_room = tuple(map(int, obs["global_pos"]))
        new_pos = tuple(map(int, obs["agent_pos"]))
        collision = target_tile in self.BLOCKED_TILES and (new_room, new_pos) == (old_room, old_pos)
        if collision:
            reward = self.collision_reward
        out = dict(info)
        out["wall_penalty"] = {"confirmed_collision": collision, "target_tile": target_tile}
        self.previous = obs
        return obs, reward, terminated, truncated, out
