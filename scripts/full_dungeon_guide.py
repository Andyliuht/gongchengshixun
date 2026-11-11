"""Observable DFS guide with key search, rejected-door memory and visible boulder planning."""
from collections import deque

from local_navigation_guide import LocalNavigationGuide, MOVES, BLOCKED, path_to


def visible_boulder_plan(grid, start, switches, max_states=100000):
    """Solve one visible room; never reads generator or hidden dungeon state."""
    boxes = frozenset((x, y) for y in range(grid.shape[0]) for x in range(grid.shape[1])
                      if int(grid[y, x]) == 5)
    switches = frozenset(switches)
    if not boxes or not switches:
        return None
    if switches.issubset(boxes):
        return []
    inner = [(x, y) for y in range(grid.shape[0]) for x in range(grid.shape[1])
             if int(grid[y, x]) not in (1, 2)]
    width = max(x for x, _ in inner) + 2
    height = max(y for _, y in inner) + 2

    def walk_paths(player, state_boxes):
        queue = deque([player])
        paths = {player: []}
        while queue:
            x, y = queue.popleft()
            for action, (dx, dy) in enumerate(MOVES):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]):
                    continue
                if (nx, ny) in paths or (nx, ny) in state_boxes or int(grid[ny, nx]) in (1, 2, 9):
                    continue
                paths[(nx, ny)] = paths[(x, y)] + [action]
                queue.append((nx, ny))
        return paths

    queue = deque([(boxes, start, [])])
    seen = set()
    while queue and len(seen) < max_states:
        state_boxes, player, plan = queue.popleft()
        paths = walk_paths(player, state_boxes)
        key = (state_boxes, min(paths))
        if key in seen:
            continue
        seen.add(key)
        for bx, by in state_boxes:
            for action, (dx, dy) in enumerate(MOVES):
                behind = (bx - dx, by - dy)
                ahead = (bx + dx, by + dy)
                ax, ay = ahead
                if behind not in paths or ahead in state_boxes:
                    continue
                if not (2 <= ax <= width - 3 and 2 <= ay <= height - 3):
                    continue
                if int(grid[ay, ax]) in (1, 2, 3, 9):
                    continue
                new_boxes = frozenset((set(state_boxes) - {(bx, by)}) | {ahead})
                next_plan = plan + paths[behind] + [action]
                if switches.issubset(new_boxes):
                    return next_plan
                queue.append((new_boxes, (bx, by), next_plan))
    return None


class FullDungeonGuide(LocalNavigationGuide):
    def __init__(self):
        super().__init__()
        self.blocked_doors = set()
        self.switches = {}
        self.last_inventory = 0
        self.solved_rooms = set()

    @staticmethod
    def reverse_door(obs, direction):
        doors = LocalNavigationGuide.targets(obs, 2)
        dx, dy = MOVES[direction]
        if dx == 1:
            options = [p for p in doors if p[0] == 0]
        elif dx == -1:
            edge = max(p[0] for p in doors)
            options = [p for p in doors if p[0] == edge]
        elif dy == 1:
            options = [p for p in doors if p[1] == 0]
        else:
            edge = max(p[1] for p in doors)
            options = [p for p in doors if p[1] == edge]
        return options[0] if len(options) == 1 else None

    def observe_transition(self, before, action, after):
        """Update memory from an actually executed step, including PPO-led steps."""
        old_room = self.room(before)
        new_room = self.room(after)
        old_local = self.local(before)
        new_key = int(after["inventory"])
        if new_key > self.last_inventory:
            self.blocked_doors.clear()
        self.last_inventory = new_key
        self.switches.setdefault(old_room, set()).update(self.targets(before, 4))
        self.switches.setdefault(new_room, set()).update(self.targets(after, 4))
        if old_room != new_room:
            dx, dy = MOVES[int(action)]
            old_door = (old_local[0] + dx, old_local[1] + dy)
            back = self.reverse_door(after, int(action))
            self.doors[old_room] = self.targets(before, 2)
            self.doors[new_room] = self.targets(after, 2)
            self.edges[(old_room, old_door)] = new_room
            if back is not None:
                self.edges[(new_room, back)] = old_room
            self.pending = None
        else:
            dx, dy = MOVES[int(action)]
            tx, ty = old_local[0] + dx, old_local[1] + dy
            grid = before["grid"]
            if (self.local(after) == old_local and 0 <= tx < grid.shape[1]
                    and 0 <= ty < grid.shape[0] and int(grid[ty, tx]) == 2):
                self.blocked_doors.add((old_room, (tx, ty)))
                self.pending = None
        if self.switches.get(new_room) and new_room not in self.solved_rooms:
            boxes = set(self.targets(after, 5))
            if self.switches[new_room].issubset(boxes):
                self.solved_rooms.add(new_room)
                self.blocked_doors = {item for item in self.blocked_doors if item[0] != new_room}

    def _route_to_frontier(self, start):
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            room, route = queue.popleft()
            for door in self.doors.get(room, []):
                if (room, door) not in self.edges and (room, door) not in self.blocked_doors:
                    return route + [(room, door)]
            for door in self.doors.get(room, []):
                neighbor = self.edges.get((room, door))
                if neighbor is not None and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, route + [(room, door)]))
        return None

    def act(self, obs, model_action=None):
        self.observe(obs)
        room = self.room(obs)
        local = self.local(obs)
        self.last_inventory = int(obs["inventory"])
        self.switches.setdefault(room, set()).update(self.targets(obs, 4))
        if not int(obs["inventory"]):
            keys = self.targets(obs, 6)
            choices = [(path_to(obs["grid"], local, key), key) for key in keys]
            choices = [(path, key) for path, key in choices if path]
            if choices:
                path, target = min(choices, key=lambda item: len(item[0]))
                return path[0], "visible_key", target
        if room not in self.solved_rooms and self.switches[room]:
            plan = visible_boulder_plan(obs["grid"], local, self.switches[room])
            if plan:
                return plan[0], "visible_boulder_plan", list(self.switches[room])
            if plan == []:
                self.solved_rooms.add(room)
                self.blocked_doors = {item for item in self.blocked_doors if item[0] != room}
        return super().act(obs, model_action)
