from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import dataclass, field, asdict

TRAIL_MILES = 200
MAX_MILES_PER_DAY = 25
DEFAULT_MILES_PER_DAY = 10
DAILY_FOOD_PER_MEMBER = 5
HUNT_BULLET_COST = 10
MAX_FOOD_PER_HUNT = 100

STARVATION_HP = 12

SENTIMENTS = ["despairing", "grim", "uneasy", "okay", "hopeful", "elated"]
DEFAULT_SENTIMENT = "okay"
SICK_RECOVERY_FLOOR = 30

# A big-mileage day (forced marches, hard fords) risks a deadly period illness.
# Onset and death are one turn apart -- the harness rolls both, not the LLM,
# so the game can never be talked out of a death once the dice land on one.
DEADLY_ILLNESSES = ["cholera", "dysentery", "typhoid fever", "diphtheria"]
BIG_MILES_THRESHOLD = 20
DEADLY_ILLNESS_CHANCE = 0.30

SCAVENGE_WOUND_HP = 20
SCAVENGE_MONEY_FOUND = 10
SCAVENGE_BULLET_COST = 2

ENCOUNTER_EVERY_DAYS = 4
ENCOUNTER_OX_PRICE = 50
ENCOUNTER_ROLES = ("approach", "pass", "avoid")
SCAVENGE_RECRUITS = [
    {"name": "Jell the random", "age": 26, "health": 80, "sick": False, "tired": False,
     "fatal_illness": None, "note": "a drifter met in the dark"},
    {"name": "Odi the lucky", "age": 34, "health": 80, "sick": False, "tired": False,
     "fatal_illness": None, "note": "a wanderer with uncanny luck"},
    {"name": "Hrozna the dog", "age": 3, "health": 80, "sick": False, "tired": False,
     "fatal_illness": None, "note": "a stray trail dog"},
]

DEFAULT_PARTY = [
    {"name": "Jonas", "age": 41, "health": 90, "sick": False, "tired": False,
     "fatal_illness": None, "note": "a former blacksmith"},
    {"name": "Elsie", "age": 9, "health": 65, "sick": False, "tired": False,
     "fatal_illness": None, "note": "youngest, has a weak ankle"},
    {"name": "Ruben", "age": 17, "health": 80, "sick": False, "tired": False,
     "fatal_illness": None, "note": "best shot in the family"},
]


@dataclass
class GameState:
    day: int = 1
    miles: int = 0
    food: int = 160          # lbs
    oxen: int = 3
    bullets: int = 5
    money: int = 50          # dollars
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
            if p.get("fatal_illness"):
                marks.append(f"gravely ill with {p['fatal_illness']}, will not survive")
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


def apply_effects(state: GameState, effects: dict) -> tuple[GameState, list[str]]:
    def num(x):
        return int(x) if isinstance(x, (int, float)) else 0

    events: list[str] = []

    # A deadly illness rolled on a PRIOR big-mileage day kills its victim now,
    # one full turn after onset -- resolved before this turn's own effects so
    # nothing proposed today (however good) can talk them out of it.
    for p in state.party:
        if p.get("fatal_illness") and p["health"] > 0:
            events.append(f"{p['name']} has died of {p['fatal_illness']}.")
            p["health"] = 0
            p["fatal_illness"] = None

    miles = effects.get("miles")
    miles_delta = num(miles) if isinstance(miles, (int, float)) else DEFAULT_MILES_PER_DAY
    applied_miles = max(0, min(miles_delta, MAX_MILES_PER_DAY))
    state.miles = min(TRAIL_MILES, state.miles + applied_miles)
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
            if flag == "sick" and not val and (
                member.get("fatal_illness") or member["health"] <= SICK_RECOVERY_FLOOR
            ):
                continue
            member[flag] = val
    for p in state.party:
        if p["health"] <= 0:
            p["sick"] = p["tired"] = False
            p["fatal_illness"] = None
    sentiment = effects.get("sentiment")
    if isinstance(sentiment, str) and sentiment.lower() in SENTIMENTS:
        state.sentiment = sentiment.lower()
    # An explicit food delta is the day's whole food change, meals included;
    # the default meal charge applies only when the model omits food entirely.
    food = effects.get("food")
    food_delta = num(food) if isinstance(food, (int, float)) else -DAILY_FOOD_PER_MEMBER * len(state.alive())
    state.food = max(0, state.food + food_delta)

    if state.food <= 0 and state.alive():
        events.append("The wagon is out of food; hunger gnaws at everyone.")
        for p in state.alive():
            p["health"] = max(0, p["health"] - STARVATION_HP)
            if p["health"] == 0:
                p["sick"] = p["tired"] = False
                p["fatal_illness"] = None
                events.append(f"{p['name']} has starved to death.")

    if applied_miles >= BIG_MILES_THRESHOLD and random.random() < DEADLY_ILLNESS_CHANCE:
        candidates = [p for p in state.party if p["health"] > 0 and not p.get("fatal_illness")]
        if candidates:
            victim = random.choice(candidates)
            illness = random.choice(DEADLY_ILLNESSES)
            victim["fatal_illness"] = illness
            victim["sick"] = True
            events.append(f"{victim['name']} has fallen gravely ill with {illness}.")

    state.day += 1
    return state, events


# Each chance is rolled independently -- one night can yield several outcomes,
# or none at all. Like the deadly-illness roll, this is harness-only: the LLM
# is handed the results afterwards and only narrates them. The caller must
# check bullets >= SCAVENGE_BULLET_COST before calling.
def scavenge(state: GameState) -> list[str]:
    state.bullets = max(0, state.bullets - SCAVENGE_BULLET_COST)
    events: list[str] = [f"The party spent {SCAVENGE_BULLET_COST} bullets scavenging."]

    def living() -> list[dict]:
        return [p for p in state.party if p["health"] > 0]

    if random.random() < 0.10 and living():
        victim = random.choice(living())
        victim["health"] = 0
        victim["sick"] = victim["tired"] = False
        victim["fatal_illness"] = None
        events.append(f"{victim['name']} died scavenging in the dark.")
    if random.random() < 0.12:
        candidates = [p for p in living() if not p.get("sick")]
        if candidates:
            struck = random.choice(candidates)
            struck["sick"] = True
            events.append(f"{struck['name']} fell sick during the night.")
    if random.random() < 0.20 and living():
        hurt = random.choice(living())
        # a scavenging wound never kills outright -- death is its own roll above
        hurt["health"] = max(1, hurt["health"] - SCAVENGE_WOUND_HP)
        events.append(f"{hurt['name']} was wounded while scavenging.")
    if random.random() < 0.01 and state.oxen > 0:
        state.oxen -= 1
        events.append("An ox wandered off in the dark and was lost.")
    if random.random() < 0.02:
        names = {p["name"] for p in state.party}
        recruits = [r for r in SCAVENGE_RECRUITS if r["name"] not in names]
        if recruits:
            newcomer = copy.deepcopy(random.choice(recruits))
            state.party.append(newcomer)
            events.append(f"{newcomer['name']} joined the party during the night.")
    if random.random() < 0.15:
        state.food += 10
        events.append("The party found 10 lbs of food.")
    if random.random() < 0.10:
        state.money += SCAVENGE_MONEY_FOUND
        events.append(f"The party found ${SCAVENGE_MONEY_FOUND}.")
    if random.random() < 0.03:
        for p in living():
            p["sick"] = False
            p["fatal_illness"] = None
        events.append("The party found a rare medicine that cured everyone, even of deadly illness.")
    if random.random() < 0.18:
        state.bullets += 4
        events.append("The party found 4 bullets.")
    if random.random() < 0.08:
        state.oxen += 1
        events.append("The party found a stray ox.")
    return events


# Encounter days bypass the scorer: the three stances carry fixed risk
# profiles the harness rolls itself, so the LLM narrates the meeting but can
# neither price it nor talk its way out of the dice. Returns an ordinary
# effects dict for apply_effects plus event strings for the grave outcomes.
#
# Each stance rolls several independent chances rather than one, so "nothing
# but miles happened" is the exception, not the norm -- an earlier version of
# "avoid" had zero non-miles outcomes (100% miles-only) and "pass" resolved
# to miles-only ~78% of the time, which read as "every encounter is just
# miles" even though "approach" itself was reasonably eventful.
ENCOUNTER_FOOD_PRICE = 15
ENCOUNTER_FOOD_AMOUNT = 40


def encounter_effects(state: GameState, role: str) -> tuple[dict, list[str]]:
    effects: dict = {}
    events: list[str] = []
    candidates = list(state.alive())

    if role == "approach":
        effects["miles"] = 5
        if random.random() < 0.15 and candidates:
            victim = random.choice(candidates)
            effects["party_health"] = {victim["name"]: -100}
            events.append(f"{victim['name']} went into the strangers' camp and never came back.")
        if random.random() < 0.35:
            effects["money"] = effects.get("money", 0) + 20
        if random.random() < 0.30:
            effects["bullets"] = 6
        # Two spends can both be authorized in this same call (ox, then food)
        # -- track a running balance rather than checking state.money twice,
        # or a lucky-but-poor roll could jointly overspend what the party
        # actually has and get silently floored (and flagged as a referee
        # clamp on effects the harness itself generated, not the LLM).
        running_money = state.money + effects.get("money", 0)
        if running_money >= ENCOUNTER_OX_PRICE and random.random() < 0.25:
            effects["money"] = effects.get("money", 0) - ENCOUNTER_OX_PRICE
            effects["oxen"] = 1
            running_money -= ENCOUNTER_OX_PRICE
        # A trading outpost is, above all, somewhere to buy food -- without
        # this, "approach a trading post" and "approach a lone rider" priced
        # identically, which is exactly the flatness being fixed here.
        if running_money >= ENCOUNTER_FOOD_PRICE and random.random() < 0.40:
            effects["money"] = effects.get("money", 0) - ENCOUNTER_FOOD_PRICE
            effects["food"] = effects.get("food", 0) + ENCOUNTER_FOOD_AMOUNT
            events.append(f"The party traded ${ENCOUNTER_FOOD_PRICE} for "
                          f"{ENCOUNTER_FOOD_AMOUNT} lbs of food.")
    elif role == "pass":
        effects["miles"] = 12
        if random.random() < 0.08 and candidates:
            victim = random.choice(candidates)
            candidates.remove(victim)
            effects.setdefault("party_health", {})[victim["name"]] = -100
            events.append(f"{victim['name']} was killed in a tense exchange as the party passed.")
        if random.random() < 0.15 and candidates:
            hurt = random.choice(candidates)
            # a passing wound never kills outright -- death is its own roll above
            effects.setdefault("party_health", {})[hurt["name"]] = \
                -min(SCAVENGE_WOUND_HP, hurt["health"] - 1)
            events.append(f"{hurt['name']} was wounded as the party passed the strangers.")
        if random.random() < 0.25:
            effects["bullets"] = effects.get("bullets", 0) + 3
            events.append("A shout from the other group tossed over a handful of spare bullets.")
        if random.random() < 0.20:
            effects["money"] = effects.get("money", 0) + 8
            events.append("Someone in the other party flagged the wagon down just long enough to sell a trinket.")
    else:  # avoid / swing wide
        effects["miles"] = 3
        if random.random() < 0.15 and candidates:
            tired = random.choice(candidates)
            effects.setdefault("party_status", {})[tired["name"]] = {"tired": True}
            events.append(f"The long detour wore {tired['name']} out.")
        if random.random() < 0.12:
            effects["food"] = effects.get("food", 0) - 10
            events.append("The wide detour cost the party a day's rations getting back on trail.")
        if random.random() < 0.10:
            effects["food"] = effects.get("food", 0) + 5
            events.append("Skirting the group, the party passed a berry patch worth a quick stop.")
    return effects, events


def state_dict(state: GameState) -> dict:
    return asdict(state)
