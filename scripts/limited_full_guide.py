"""Frozen, budgeted rule intervention around an independently sampled PPO policy."""
from full_dungeon_guide import FullDungeonGuide


class LimitedFullGuide:
    def __init__(self, config, initial_obs):
        self.config = config
        self.guide = FullDungeonGuide()
        self.visited = {self.place(initial_obs)}
        self.stagnation = 0
        self.pending_door_rejection = False
        self.cooldown = 0
        self.burst_left = 0
        self.rule_actions = 0
        self.last_trigger = None

    @staticmethod
    def place(obs):
        return tuple(map(int, obs["global_pos"])), tuple(map(int, obs["agent_pos"]))

    def trigger(self, obs):
        if self.pending_door_rejection and self.config["trigger_rejected_door"]:
            return "rejected_door"
        if self.config["trigger_visible_key"] and not int(obs["inventory"]):
            if self.guide.targets(obs, 6):
                return "visible_key"
        if self.config["trigger_visible_boulder_switch"]:
            room = self.guide.room(obs)
            if room not in self.guide.solved_rooms and (self.guide.targets(obs, 4) or self.guide.switches.get(room)):
                return "visible_puzzle"
        if self.stagnation >= self.config["stagnation_steps_trigger"]:
            return "stagnation"
        return None

    def decide(self, obs, raw_action):
        self.guide.observe(obs)
        if self.rule_actions >= self.config["maximum_rule_actions_per_episode"]:
            return int(raw_action), "budget_exhausted", None, False, None
        if self.burst_left == 0:
            if self.cooldown:
                self.cooldown -= 1
                return int(raw_action), "cooldown", None, False, None
            trigger = self.trigger(obs)
            if trigger is None:
                return int(raw_action), "ppo_only", None, False, None
            self.burst_left = self.config["maximum_consecutive_rule_actions"]
            self.last_trigger = trigger
            self.pending_door_rejection = False
        action, reason, target = self.guide.act(obs, int(raw_action))
        if action is None or reason == "model_fallback":
            self.burst_left = 0
            self.cooldown = self.config["cooldown_ppo_actions"]
            return int(raw_action), "guide_fallback", None, False, self.last_trigger
        self.rule_actions += 1
        self.burst_left -= 1
        if self.burst_left == 0:
            self.cooldown = self.config["cooldown_ppo_actions"]
        return int(action), reason, target, True, self.last_trigger

    def observe_transition(self, before, action, after, mechanics):
        self.guide.observe_transition(before, action, after)
        place = self.place(after)
        if place not in self.visited:
            self.visited.add(place)
            self.stagnation = 0
        else:
            self.stagnation += 1
        if mechanics["event"] == "door_rejected_unknown_cause":
            self.pending_door_rejection = True
        if self.place(before)[0] != place[0] and self.burst_left:
            self.burst_left = 0
            self.cooldown = self.config["cooldown_ppo_actions"]
