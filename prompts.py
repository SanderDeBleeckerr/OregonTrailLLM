"""Game-master prompt strategies.

Each strategy builds the per-turn prompt from (state_text, history, extras).
The three strategies vary ONE prompt-engineering dimension at a time so the
experiments isolate its effect:

  minimal        : task + bare schema description
  fewshot_schema : + one fully worked example JSON turn
  rules_explicit : + the hard game rules spelled out verbatim

Every strategy asks for the same JSON: narrative, three options with numeric
effects, and (for the state-tracking experiment) a `believed_state` block the
model must reconstruct FROM THE CONVERSATION, not from the header.
"""
from __future__ import annotations

SCHEMA = """Respond ONLY with a JSON object (no markdown fences, no prose outside JSON):
{
  "believed_state": {"day": int, "miles": int, "food": int, "oxen": int,
                     "bullets": int, "money": int,
                     "party_health": {"<name>": int, ...}},
  "narrative": "2-3 sentences describing today's event on the trail",
  "options": [
    {"text": "short action description",
     "effects": {"food": int, "oxen": int, "bullets": int, "money": int,
                 "miles": int, "party_health": {"<name>": int},
                 "party_status": {"<name>": {"sick": bool, "tired": bool}},
                 "sentiment": "one of: despairing|grim|uneasy|okay|hopeful|elated"}},
    ... exactly 3 options ...
  ]
}
Effects are DELTAS (changes), negative for losses. Omit keys that don't change.
EXCEPTION: "party_status" and "sentiment" are absolute values, not deltas --
give the flag or mood as it would be AFTER the option is taken."""

EXAMPLE = """Example of a valid response:
{
  "believed_state": {"day": 4, "miles": 62, "food": 340, "oxen": 4,
                     "bullets": 60, "money": 150,
                     "party_health": {"Marta": 100, "Jonas": 95, "Elsie": 100, "Ruben": 100}},
  "narrative": "A rainstorm has swollen the creek ahead. The oxen balk at the muddy bank while Jonas studies the current.",
  "options": [
    {"text": "Ford the creek now", "effects": {"miles": 10, "party_health": {"Jonas": -10},
      "party_status": {"Jonas": {"tired": true}}, "sentiment": "uneasy"}},
    {"text": "Wait a day for the water to drop", "effects": {"miles": 0,
      "party_status": {"Jonas": {"tired": false}}, "sentiment": "okay"}},
    {"text": "Pay a local guide $15 for the safe crossing", "effects": {"money": -15,
      "miles": 12, "sentiment": "hopeful"}}
  ]
}"""

_EFFECT_RULES = [
    'miles per option must be between 0 and 25.',
    'Hunting requires at least 10 bullets and must include "bullets": -10 (or more negative); food gained from one hunt never exceeds 100 lbs.',
    'Never propose an effect that would push food, oxen, bullets, or money below 0 given the current state.',
    'Only use party member names that exist; a single event never raises anyone\'s health by more than 40.',
    '"sentiment" must be one of despairing|grim|uneasy|okay|hopeful|elated.',
    'Never clear a "sick" flag (sick: false) for anyone whose health is 30 or below -- they are too ill to recover in one day. Never mark a dead member (health 0) sick or tired.',
]
_BELIEVED_STATE_RULE = (
    "believed_state must reflect the game BEFORE today's options are applied."
)


def _numbered(rules: list[str]) -> str:
    header = "HARD RULES you must never violate when proposing effects:\n"
    return header + "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))


RULES = _numbered(_EFFECT_RULES + [_BELIEVED_STATE_RULE])
SCORER_RULES = _numbered(_EFFECT_RULES)


BASE = (
    "You are the game master of 'Westward Trail', a wagon-journey survival game "
    "set on a 2000-mile route west. The party travels WEST and only west. "
    "Keep the tone grounded and historical. "
    "Vary events: river crossings, illness, hunting, trading posts, weather, "
    "wagon damage, encounters with other travelers.\n"
    "Every day the party automatically eats 5 lbs of food per living member. "
    "The harness applies those meals itself, so never count them in your food "
    "numbers -- report only the food an action itself gains or spends.\n"
)

# Sole source for QUIZ facts; wording must stay stable across runs.
INTRO = (
    "Party intro: Marta (34, the party's healer), Jonas (41, a former "
    "blacksmith), Elsie (9, youngest, has a weak ankle), Ruben (17, best "
    "shot in the family). They set out west with a wagon and high hopes.\n"
)


def minimal(state_text: str, history: str, extra: str = "") -> str:
    return f"{BASE}\n{SCHEMA}\n\nRecent events:\n{history}\n\n{extra}{state_text}\nGenerate the next turn."


def fewshot_schema(state_text: str, history: str, extra: str = "") -> str:
    return (
        f"{BASE}\n{SCHEMA}\n\n{EXAMPLE}\n\nRecent events:\n{history}\n\n"
        f"{extra}{state_text}\nGenerate the next turn."
    )


def rules_explicit(state_text: str, history: str, extra: str = "") -> str:
    return (
        f"{BASE}\n{SCHEMA}\n\n{RULES}\n\nRecent events:\n{history}\n\n"
        f"{extra}{state_text}\nGenerate the next turn."
    )


STRATEGIES = {
    "minimal": minimal,
    "fewshot_schema": fewshot_schema,
    "rules_explicit": rules_explicit,
}


# One call for story+arithmetic produces contradictory numbers; split into narrator+scorer instead.
# "Scorer" != "referee": engine.check_rules is authoritative; the scorer is just another LLM call.

NARRATE_SCHEMA = """Respond ONLY with a JSON object (no markdown fences, no prose outside JSON):
{
  "believed_state": {"day": int, "miles": int, "food": int, "oxen": int,
                     "bullets": int, "money": int,
                     "party_health": {"<name>": int, ...}},
  "narrative": "2-3 sentences describing today's event on the trail",
  "options": [
    {"text": "short action description"},
    ... exactly 3 options ...
  ]
}
Write NO numbers or effects for the options -- a separate scorekeeper assigns
the consequences. Each option must be a concrete action the party could take today,
and the three must differ meaningfully (e.g. push on / wait / spend something).
believed_state must reflect the game BEFORE today's options are taken."""


SCORER_BASE = (
    "You are the scorekeeper for 'Westward Trail', a wagon-journey survival "
    "game on a 2000-mile route west. You do not tell stories and you do not "
    "invent events. Today's event and the party's three possible actions have "
    "already been written. Your only job is to assign the numeric consequences "
    "of each action, faithfully and conservatively.\n"
)

SCORER_SCHEMA = """Respond ONLY with a JSON object (no markdown fences, no prose outside JSON):
{
  "effects": [
    {"food": int, "oxen": int, "bullets": int, "money": int, "miles": int,
     "party_health": {"<name>": int},
     "party_status": {"<name>": {"sick": bool, "tired": bool}},
     "sentiment": "despairing|grim|uneasy|okay|hopeful|elated"},
    ... exactly one object per action, in the same order as the actions ...
  ]
}
food, oxen, bullets, money, miles and party_health are DELTAS: negative for a
loss, positive for a gain, and OMITTED when the action does not change them.
party_status and sentiment are ABSOLUTE values -- the flag or mood as it would
stand AFTER the action."""

# These are the coherence constraints a single narrate+score call kept breaking.
COHERENCE = """COHERENCE RULES -- every number must follow from the action's own words:
1. Change ONLY what the action would actually change, and omit every other key. Most actions touch two or three keys, not all of them.
2. Miles follow the action: resting, camping, waiting, treating someone, or trading in place is 0-5. Ordinary travel is 8-20. A hard forced march is up to 25.
3. An action never consumes a resource it does not mention. Resting does not kill oxen. Talking does not spend bullets. Only hunting spends bullets.
4. Health only rises when the action actually treats or rests someone, never for a member already at 100, and never by more than 40 in a day.
5. Health only falls from real hardship the narrative describes -- fords, cold, hunger, accidents, overwork. Resting never injures anyone.
6. Mark "tired" when the action exhausts someone and clear it when they genuinely rest. Mark "sick" only when the narrative gives a cause.
7. Sentiment must match how the day actually went for the party, on the scale despairing < grim < uneasy < okay < hopeful < elated. Move it from the party's current mood only as far as the day's outcome earns.
8. Do NOT deduct the party's daily meals -- the harness already eats 5 lbs per living member each day. Count only food the action itself gains or spends."""

SCORER_EXAMPLE = """Example, for a party of 4 whose mood is "okay" and whose members are all at full health:

Today's event: A rainstorm has swollen the creek ahead. The oxen balk at the muddy bank.
Actions:
1. Ford the creek now
2. Wait a day for the water to drop
3. Pay a local guide $15 for the safe crossing

{"effects": [
  {"miles": 10, "party_health": {"Jonas": -10}, "party_status": {"Jonas": {"tired": true}}, "sentiment": "uneasy"},
  {"miles": 0, "sentiment": "okay"},
  {"money": -15, "miles": 12, "sentiment": "hopeful"}
]}
Note what is absent: waiting spends no food beyond the automatic meals, costs no
oxen, and heals nobody -- so those keys simply do not appear."""


def narrate_prompt(state_text: str, history: str, extra: str = "") -> str:
    """Call 1: the story and the three choices, with no numbers attached."""
    return (
        f"{BASE}\n{NARRATE_SCHEMA}\n\nRecent events:\n{history}\n\n"
        f"{extra}{state_text}\nGenerate the next turn."
    )


def effects_prompt(state_text: str, narrative: str, option_texts: list[str]) -> str:
    """Call 2: the numbers for actions that have already been written."""
    actions = "\n".join(f"{i}. {t}" for i, t in enumerate(option_texts, 1))
    return (
        f"{SCORER_BASE}\n{SCORER_SCHEMA}\n\n{SCORER_RULES}\n\n{COHERENCE}\n\n"
        f"{SCORER_EXAMPLE}\n\n{state_text}\n\nToday's event: {narrative}\n\n"
        f"Actions:\n{actions}\n\n"
        f"Give exactly {len(option_texts)} effect objects, one per action, in order."
    )


QUIZ = [
    {"q": "Which party member is the youngest? Answer with just the name.",
     "answer": "elsie"},
    {"q": "Which party member is described as the best shot? Answer with just the name.",
     "answer": "ruben"},
    {"q": "What physical weakness does Elsie have? Answer in at most four words.",
     "answer": "weak ankle"},
    {"q": "What was Jonas's former trade? Answer in one or two words.",
     "answer": "blacksmith"},
]


def quiz_prompt(history: str, question: str) -> str:
    return (
        f"{BASE}\nYou will now answer a question about the party using only "
        f"what has been established in this game so far.\n\nGame so far:\n"
        f"{history}\n\nQuestion: {question}\nAnswer:"
    )
