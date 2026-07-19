from __future__ import annotations

import pathlib

_DIR = pathlib.Path(__file__).parent / "prompts"


def _load(name: str) -> str:
    return (_DIR / name).read_text(encoding="utf-8").rstrip("\n")


BASE = _load("base.txt")
SCHEMA = _load("schema.txt")
EXAMPLE = _load("example.txt")
# Sole source for QUIZ facts; wording must stay stable across runs.
INTRO = _load("intro.txt") + "\n"
NARRATE_SCHEMA = _load("narrate_schema.txt")
# Scorer task description + JSON schema, always used together, never apart.
SCORER_BASE = _load("scorer_base.txt")
# Coherence constraints (a single narrate+score call kept breaking these) + a
# worked example, always used together, never apart.
COHERENCE = _load("coherence.txt")
SCAVENGE = _load("scavenge.txt")
ENCOUNTER = _load("encounter.txt")
OUTCOME = _load("outcome.txt")

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


def minimal(state_text: str, history: str, extra: str = "") -> str:
    return f"{BASE}\n\n{SCHEMA}\n\nRecent events:\n{history}\n\n{extra}{state_text}\nGenerate the next turn."


def fewshot_schema(state_text: str, history: str, extra: str = "") -> str:
    return (
        f"{BASE}\n\n{SCHEMA}\n\n{EXAMPLE}\n\nRecent events:\n{history}\n\n"
        f"{extra}{state_text}\nGenerate the next turn."
    )


def rules_explicit(state_text: str, history: str, extra: str = "") -> str:
    return (
        f"{BASE}\n\n{SCHEMA}\n\n{RULES}\n\nRecent events:\n{history}\n\n"
        f"{extra}{state_text}\nGenerate the next turn."
    )


STRATEGIES = {
    "minimal": minimal,
    "fewshot_schema": fewshot_schema,
    "rules_explicit": rules_explicit,
}


# One call for story+arithmetic produces contradictory numbers; split into narrator+scorer instead.
# "Scorer" != "referee": engine.check_rules is authoritative; the scorer is just another LLM call.
def narrate_prompt(state_text: str, history: str, extra: str = "") -> str:
    return (
        f"{BASE}\n\n{NARRATE_SCHEMA}\n\nRecent events:\n{history}\n\n"
        f"{extra}{state_text}\nGenerate the next turn."
    )


def effects_prompt(state_text: str, narrative: str, option_texts: list[str],
                   extra: str = "") -> str:
    actions = "\n".join(f"{i}. {t}" for i, t in enumerate(option_texts, 1))
    return (
        f"{SCORER_BASE}\n\n{SCORER_RULES}\n\n{COHERENCE}\n\n"
        f"{extra}{state_text}\n\nToday's event: {narrative}\n\n"
        f"Actions:\n{actions}\n\n"
        f"Give exactly {len(option_texts)} effect objects, one per action, in order."
    )


def encounter_prompt(state_text: str, history: str) -> str:
    return (
        f"{BASE}\n\n{NARRATE_SCHEMA}\n\n{ENCOUNTER}\n\nRecent events:\n{history}\n\n"
        f"{state_text}\nGenerate the next turn."
    )


def outcome_prompt(state_text: str, history: str, narrative: str,
                   chosen: str, consequences: list[str]) -> str:
    facts = "\n".join(f"- {c}" for c in consequences) \
        or "- The day changed nothing of note."
    return (
        f"{OUTCOME}\n\nRecent events:\n{history}\n\n{state_text}\n\n"
        f"This morning: {narrative}\n\nThe party chose: {chosen}\n\n"
        f"What came of it (fixed, already applied):\n{facts}\n\n"
        f"Write the story."
    )


def scavenge_prompt(state_text: str, history: str, outcomes: list[str]) -> str:
    facts = "\n".join(f"- {o}" for o in outcomes) \
        or "- Nothing was found and no one was harmed."
    return (
        f"{SCAVENGE}\n\nRecent events:\n{history}\n\n{state_text}\n\n"
        f"What the night yielded (fixed, already applied):\n{facts}\n\n"
        f"Write the story."
    )


def image_prompt(narrative: str, sentiment: str) -> str:
    return (
        "Pixel art, 16-bit retro video-game style, wide scenic illustration, "
        "muted historical color palette, no text, no logos, no UI elements. "
        f"A wagon-train party traveling west on a frontier trail, mood {sentiment}. "
        f"Scene: {narrative}"
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
