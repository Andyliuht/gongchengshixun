"""Observable-room DFS guide: remembers tried doors, never reads hidden map."""
from collections import deque


MOVES = ((0, 1), (0, -1), (1, 0), (-1, 0))
BLOCKED = {1, 5, 9}


def path_to(grid, start, goal):
    q = deque([(start, [])])
    seen = {start}
    while q:
        (x, y), path = q.popleft()
        if (x, y) == goal:
            return path
        for action, (dx, dy) in enumerate(MOVES):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]):
                continue
            if (nx, ny) in seen or int(grid[ny, nx]) in BLOCKED:
                continue
            seen.add((nx, ny))
            q.append(((nx, ny), path + [action]))
    return None


class LocalNavigationGuide:
    def __init__(self):
        self.doors = {}
        self.edges = {}
        self.pending = None
        self.last_room = None
        self.trace = []

    @staticmethod
    def room(obs):
        return tuple(map(int, obs["global_pos"]))

    @staticmethod
    def local(obs):
        return tuple(map(int, obs["agent_pos"]))

    @staticmethod
    def targets(obs, tile):
        grid = obs["grid"]
        return [(x, y) for y in range(grid.shape[0]) for x in range(grid.shape[1])
                if int(grid[y, x]) == tile]

    def observe(self, obs):
        room = self.room(obs)
        self.doors[room] = self.targets(obs, 2)
        if self.pending is not None and room != self.pending[0]:
            old_room, old_door = self.pending
            self.edges[(old_room, old_door)] = room
            dx, dy = room[0] - old_room[0], room[1] - old_room[1]
            if (dx, dy) == (1, 0):
                matches = [p for p in self.doors[room] if p[0] == 0]
            elif (dx, dy) == (-1, 0):
                edge = max(p[0] for p in self.doors[room])
                matches = [p for p in self.doors[room] if p[0] == edge]
            elif (dx, dy) == (0, 1):
                matches = [p for p in self.doors[room] if p[1] == 0]
            elif (dx, dy) == (0, -1):
                edge = max(p[1] for p in self.doors[room])
                matches = [p for p in self.doors[room] if p[1] == edge]
            else:
                raise RuntimeError(f"Non-adjacent room transition: {old_room} -> {room}")
            if len(matches) != 1:
                raise RuntimeError(f"Cannot identify reverse door for {old_room} -> {room}: {matches}")
            back = matches[0]
            self.edges[(room, back)] = old_room
            self.pending = None
        self.last_room = room

    def _route_to_frontier(self, start):
        q = deque([(start, [])])
        seen = {start}
        while q:
            room, route = q.popleft()
            for door in self.doors[room]:
                if (room, door) not in self.edges:
                    return route + [(room, door)]
            for door in self.doors[room]:
                neighbor = self.edges.get((room, door))
                if neighbor is not None and neighbor not in seen:
                    seen.add(neighbor)
                    q.append((neighbor, route + [(room, door)]))
        return None

    def act(self, obs, model_action=None):
        self.observe(obs)
        room = self.room(obs)
        local = self.local(obs)
        grid = obs["grid"]
        exits = self.targets(obs, 3)
        if exits:
            if local in exits:
                # BoltCrypt may place the agent on the exit tile while entering
                # the room, yet only sets done on a later step INTO that tile.
                for action, (dx, dy) in enumerate(MOVES):
                    nx, ny = local[0] + dx, local[1] + dy
                    if (0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]
                            and int(grid[ny, nx]) not in BLOCKED and int(grid[ny, nx]) != 2):
                        return action, "exit_tile_reentry", local
            choices = [(path_to(grid, local, target), target) for target in exits]
            choices = [(p, t) for p, t in choices if p is not None and p]
            if choices:
                path, target = min(choices, key=lambda item: len(item[0]))
                return path[0], "visible_final_exit", target
        route = self._route_to_frontier(room)
        if route:
            _, door = route[0]
            path = path_to(grid, local, door)
            if path:
                self.pending = (room, door)
                return path[0], "unexplored_door" if len(route) == 1 else "route_to_frontier", door
        return model_action, "model_fallback", None
