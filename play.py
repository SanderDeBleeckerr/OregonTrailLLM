"""Play Westward Trail interactively against a local LLM game master.

    python play.py                                 # DEFAULT_MODEL
    python play.py --model qwen2.5:32b --strategy rules_explicit
"""
from __future__ import annotations

import argparse
import json

from engine import GameState, apply_effects, check_rules, extract_json, state_dict
from llm_client import DEFAULT_MODEL, OllamaClient
from prompts import INTRO, STRATEGIES

MAX_PARSE_MISSES = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--strategy", default="rules_explicit", choices=STRATEGIES)
    ap.add_argument("--host", default="http://localhost:11434")
    args = ap.parse_args()

    client = OllamaClient(model=args.model, host=args.host)
    build = STRATEGIES[args.strategy]
    state = GameState()
    history = INTRO

    print("=" * 60, "\nWESTWARD TRAIL — LLM game master edition\n", "=" * 60)
    misses = 0
    while state.finished() is None:
        state_text = "Current true state: " + state.summary()
        raw = client.generate(build(state_text, history[-4000:]), temperature=0.8)
        data = extract_json(raw)
        if not data or "options" not in data:
            misses += 1
            if misses >= MAX_PARSE_MISSES:
                print(f"\n{args.model} did not return usable JSON in "
                      f"{MAX_PARSE_MISSES} tries. Last reply was:\n{raw[:500]!r}")
                return
            print("(The game master mumbled something unparseable; retrying...)")
            continue
        misses = 0

        print(f"\n--- Day {state.day} | {state.miles} mi | food {state.food} "
              f"| oxen {state.oxen} | bullets {state.bullets} | ${state.money} ---")
        names = lambda ps: ", ".join(p["name"] for p in ps) or "none"  # noqa: E731
        print(f"    mood {state.sentiment} | alive {len(state.alive())}/"
              f"{len(state.party)} | sick: {names(state.sick())} "
              f"| tired: {names(state.tired())}")
        print(data.get("narrative", ""))
        options = data["options"][:3]
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt.get('text', '?')}")

        choice = None
        while choice is None:
            pick = input("Choose [1-3, q to quit]: ").strip().lower()
            if pick == "q":
                return
            if pick in {"1", "2", "3"} and int(pick) <= len(options):
                choice = options[int(pick) - 1]

        effects = choice.get("effects", {}) or {}
        violations = check_rules(state, choice.get("text", ""), effects)
        if violations:
            print(f"(referee note: clamped illegal effects {violations})")
        state = apply_effects(state, effects)
        history += (
            f"Day {state.day - 1}: {data.get('narrative','')} "
            f"Chosen: {choice.get('text','')} Effects: {json.dumps(effects)}\n"
        )

    print(f"\nGame over: {state.finished()}")
    print(state.summary())


if __name__ == "__main__":
    main()
