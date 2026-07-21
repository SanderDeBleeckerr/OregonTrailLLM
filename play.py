"""Play Westward Trail interactively against a local LLM game master.

    python play.py                                        # DEFAULT_MODEL
    python play.py --model Qwen/Qwen2.5-3B-Instruct --strategy rules_explicit
"""
from __future__ import annotations

import argparse
import json

from engine import GameState, apply_effects, check_rules, extract_json, state_dict
from llm_client import DEFAULT_MODEL, DEFAULT_TEXT_HOST, TextClient, ensure_text_server
from prompts import INTRO, STRATEGIES

MAX_PARSE_MISSES = 3


def play(client: TextClient, strategy: str, model: str) -> None:
    build = STRATEGIES[strategy]
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
                print(f"\n{model} did not return usable JSON in "
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
        state, events = apply_effects(state, effects)
        for event in events:
            print(f"(!) {event}")
        history += (
            f"Day {state.day - 1}: {data.get('narrative','')} "
            f"Chosen: {choice.get('text','')} Effects: {json.dumps(effects)}"
            + (f" Harness event: {' '.join(events)}" if events else "")
            + "\n"
        )

    print(f"\nGame over: {state.finished()}")
    print(state.summary())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--strategy", default="rules_explicit", choices=STRATEGIES)
    ap.add_argument("--text-host", default=DEFAULT_TEXT_HOST)
    args = ap.parse_args()

    server_process = ensure_text_server(args.text_host, args.model)
    client = TextClient(model=args.model, host=args.text_host)
    try:
        play(client, args.strategy, args.model)
    finally:
        if server_process is not None:
            server_process.terminate()


if __name__ == "__main__":
    main()
