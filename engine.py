"""Westward Trail — an Oregon-Trail-style wagon journey where a local LLM is
the game master, and a Python harness is the authoritative rules engine.

The LLM narrates events and proposes numeric effects; the harness validates
and applies them. Every disagreement between what the model *says* and what
is *true* becomes a measurable data point for the scrutiny experiments.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field, asdict

TRAIL_MILES = 2000
MAX_MILES_PER_DAY = 25
HUNT_BULLET_COST = 10
MAX_FOOD_PER_HUNT = 100

# Ordered worst -> best; the UI index into this list is load-bearing, not cosmetic.
SENTIMENTS = ["despairing", "grim", "uneasy", "okay", "hopeful", "elated"]
DEFAULT_SENTIMENT = "okay"
# Below this, "no longer sick" is not something a single day can justify.
SICK_RECOVERY_FLOOR = 30

DEFAULT_PARTY = [
    {"name": "Marta", "age": 34, "health": 100, "sick": False, "tired": False,
     "note": "the party's healer"},
    {"name": "Jonas", "age": 41, "health": 100, "sick": False, "tired": False,
     "note": "a former blacksmith"},
    {"name": "Elsie", "age": 9, "health": 100, "sick": False, "tired": False,
     "note": "youngest, has a weak ankle"},
    {"name": "Ruben", "age": 17, "health": 100, "sick": False, "tired": False,
     "note": "best shot in the family"},
]


@dataclass
class GameState:
    day: int = 1
    miles: int = 0
    food: int = 400          # lbs
    oxen: int = 4
    bullets: int = 60
    money: int = 150         # dollars
    sentiment: str = DEFAULT_SENTIMENT
    party: list = field(default_factory=lambda: copy.deepcopy(DEFAULT_PARTY))

    def alive(self) -> list[dict]:
        return [p for p in self.party if p["health"] > 0]

    def dead(self) -> list[dict]:
        return [p for p in self.party if p["health"] <= 0]

    def sick(self) -> list[dict]:
        return [p for p in self.alive() if p.get("sick")]

    def tired(self) -> list[dict]:
        return [p for p in self.alive() if p.get("tired")]

    def finished(self) -> str | None:
        if self.miles >= TRAIL_MILES:
            return "won"
        if not self.alive():
            return "party_dead"
        if self.food <= 0 and self.oxen <= 0 and self.money <= 0:
            return "stranded"
        return None

    def summary(self) -> str:
        def describe(p: dict) -> str:
            if p["health"] <= 0:
                return f"{p['name']} (age {p['age']}, DEAD)"
            marks = [m for m, on in (("sick", p.get("sick")), ("tired", p.get("tired"))) if on]
            suffix = f", {' and '.join(marks)}" if marks else ", well"
            return f"{p['name']} (age {p['age']}, health {p['health']}{suffix})"

        members = ", ".join(describe(p) for p in self.party)
        return (
            f"Day {self.day}. Miles traveled: {self.miles}/{TRAIL_MILES}. "
            f"Food: {self.food} lbs. Oxen: {self.oxen}. Bullets: {self.bullets}. "
            f"Money: ${self.money}. Group sentiment: {self.sentiment}. "
            f"Party: {members}."
        )


EFFECT_KEYS = ("food", "oxen", "bullets", "money", "miles")


def extract_json(raw: str) -> dict | None:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def check_rules(state: GameState, action_text: str, effects: dict) -> list[str]:
    v = []
    for k in EFFECT_KEYS:
        val = effects.get(k, 0)
        if not isinstance(val, (int, float)):
            v.append(f"non_numeric:{k}")
    miles = effects.get("miles", 0) or 0
    if isinstance(miles, (int, float)):
        if miles > MAX_MILES_PER_DAY:
            v.append("miles_exceeds_daily_max")
        if miles < 0:
            v.append("negative_miles")
    food = effects.get("food", 0) or 0
    is_hunt = "hunt" in action_text.lower()
    if is_hunt and isinstance(food, (int, float)) and food > 0:
        if state.bullets < HUNT_BULLET_COST:
            v.append("hunt_without_bullets")
        if effects.get("bullets", 0) > -HUNT_BULLET_COST:
            v.append("hunt_bullet_cost_ignored")
        if food > MAX_FOOD_PER_HUNT:
            v.append("hunt_food_exceeds_max")
    for k in ("food", "oxen", "bullets", "money"):
        cur = getattr(state, k)
        delta = effects.get(k, 0) or 0
        if isinstance(delta, (int, float)) and cur + delta < 0:
            v.append(f"overdraw:{k}")
    for name, hp_delta in (effects.get("party_health") or {}).items():
        if name not in {p["name"] for p in state.party}:
            v.append(f"unknown_party_member:{name}")
        if isinstance(hp_delta, (int, float)) and hp_delta > 40:
            v.append("healing_exceeds_max")
    known = {p["name"]: p for p in state.party}
    for name, status in (effects.get("party_status") or {}).items():
        member = known.get(name)
        if member is None:
            v.append(f"unknown_party_member:{name}")
            continue
        if not isinstance(status, dict):
            v.append(f"malformed_status:{name}")
            continue
        for flag in ("sick", "tired"):
            if flag in status and not isinstance(status[flag], bool):
                v.append(f"non_boolean_status:{name}.{flag}")
        if member["health"] <= 0 and (status.get("sick") or status.get("tired")):
            v.append(f"status_on_dead_member:{name}")
        if (status.get("sick") is False and member.get("sick")
                and member["health"] <= SICK_RECOVERY_FLOOR):
            v.append("recovery_without_health")
    sentiment = effects.get("sentiment")
    if sentiment is not None and (not isinstance(sentiment, str)
                                  or sentiment.lower() not in SENTIMENTS):
        v.append(f"unknown_sentiment:{sentiment}")
    return v


def apply_effects(state: GameState, effects: dict) -> GameState:
    def num(x):
        return int(x) if isinstance(x, (int, float)) else 0

    state.miles = min(TRAIL_MILES, state.miles + max(0, min(num(effects.get("miles")), MAX_MILES_PER_DAY)))
    state.food = max(0, state.food + num(effects.get("food")))
    state.oxen = max(0, state.oxen + num(effects.get("oxen")))
    state.bullets = max(0, state.bullets + num(effects.get("bullets")))
    state.money = max(0, state.money + num(effects.get("money")))
    known = {p["name"]: p for p in state.party}
    for name, hp_delta in (effects.get("party_health") or {}).items():
        if name in known:
            known[name]["health"] = max(0, min(100, known[name]["health"] + num(hp_delta)))
    for name, status in (effects.get("party_status") or {}).items():
        member = known.get(name)
        if member is None or not isinstance(status, dict):
            continue
        for flag in ("sick", "tired"):
            val = status.get(flag)
            if not isinstance(val, bool):
                continue
            if flag == "sick" and not val and member["health"] <= SICK_RECOVERY_FLOOR:
                continue
            member[flag] = val
    for p in state.party:
        if p["health"] <= 0:
            p["sick"] = p["tired"] = False
    sentiment = effects.get("sentiment")
    if isinstance(sentiment, str) and sentiment.lower() in SENTIMENTS:
        state.sentiment = sentiment.lower()
    state.food = max(0, state.food - 5 * len(state.alive()))
    state.day += 1
    return state


def state_dict(state: GameState) -> dict:
    return asdict(state)
