"""Automated scrutiny experiments on the LLM game master.

A scripted bot plays full games; every turn is logged as one JSONL record.

Dimensions measured
-------------------
E1 format reliability : did the model return parseable, schema-valid JSON?
E2 rule adherence     : hard-rule violations in proposed effects (engine.check_rules)
E3 state tracking     : per-field |believed_state - true_state| over turns, in
                        two modes:
                          guided : true state shown in every turn's prompt
                                   (can the model even copy correctly?)
                          blind  : true state shown only on turn 1; the model
                                   must track it from the event history
E4 memory recall      : facts stated once at game start, quizzed at turns
                        5/10/15/20 (accuracy vs conversational distance)

Usage:
    python run_experiments.py --seeds 2 --turns 20              # DEFAULT_MODEL
    python run_experiments.py --model qwen2.5:32b --seeds 2     # model comparison
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random

from engine import GameState, apply_effects, check_rules, extract_json
from llm_client import DEFAULT_MODEL, OllamaClient
from prompts import INTRO, QUIZ, STRATEGIES, quiz_prompt

ROOT = pathlib.Path(__file__).parent
RESULTS = ROOT / "results"
QUIZ_TURNS = (5, 10, 15, 20)


def append(path: pathlib.Path, record: dict) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def existing_keys(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["key"] for line in path.open(encoding="utf-8")}


def believed_vs_true(believed: dict, state: GameState) -> dict:
    if not isinstance(believed, dict):
        return {"missing": True}
    err = {"missing": False}
    for k in ("day", "miles", "food", "oxen", "bullets", "money"):
        b = believed.get(k)
        err[k] = abs(b - getattr(state, k)) if isinstance(b, (int, float)) else None
    truth = {p["name"]: p["health"] for p in state.party}
    bh = believed.get("party_health") or {}
    hp_errs = [abs(bh[n] - hp) for n, hp in truth.items()
               if isinstance(bh.get(n), (int, float))]
    err["party_health_mean"] = sum(hp_errs) / len(hp_errs) if hp_errs else None
    err["party_names_covered"] = len(hp_errs) / len(truth)
    return err


def play_one_game(client: OllamaClient, strategy: str, mode: str, seed: int,
                  turns: int, out: pathlib.Path, done: set[str]) -> None:
    build = STRATEGIES[strategy]
    rng = random.Random(seed)
    state = GameState()
    history = INTRO
    run_id = f"{strategy}|{mode}|s{seed}"

    for turn in range(1, turns + 1):
        key = f"{run_id}|t{turn}"
        show_state = mode == "guided" or turn == 1
        state_text = ("Current true state: " + state.summary()) if show_state else (
            "Reconstruct the current state yourself from the events so far.")
        finished = state.finished()
        if finished:
            break

        raw = client.generate(build(state_text, history[-6000:]),
                              temperature=0.7, seed=seed * 100 + turn)
        data = extract_json(raw)
        parse_ok = bool(data and isinstance(data.get("options"), list)
                        and len(data["options"]) >= 1)

        if not parse_ok:
            append(out, {"key": key, "kind": "turn", "run": run_id,
                         "strategy": strategy, "mode": mode, "seed": seed,
                         "turn": turn, "parse_ok": False, "raw": raw[:500]})
            # burn a day so games can't loop forever on a broken model
            state.day += 1
            history += f"Day {state.day - 1}: (the trail was uneventful)\n"
            continue

        options = data["options"][:3]
        choice = options[rng.randrange(len(options))]
        effects = choice.get("effects", {}) or {}
        violations = check_rules(state, str(choice.get("text", "")), effects)
        drift = believed_vs_true(data.get("believed_state"), state)

        append(out, {
            "key": key, "kind": "turn", "run": run_id, "strategy": strategy,
            "mode": mode, "seed": seed, "turn": turn, "parse_ok": True,
            "n_options": len(options), "violations": violations,
            "drift": drift, "narrative": str(data.get("narrative", ""))[:300],
            "chosen": str(choice.get("text", ""))[:120],
        })

        state, events = apply_effects(state, effects)
        history += (
            f"Day {state.day - 1}: {data.get('narrative','')} "
            f"Chosen: {choice.get('text','')} Effects: {json.dumps(effects)}"
            + (f" Harness event: {' '.join(events)}" if events else "")
            + "\n"
        )

        if mode == "blind" and turn in QUIZ_TURNS:
            for qi, quiz in enumerate(QUIZ):
                qkey = f"{run_id}|t{turn}|q{qi}"
                if qkey in done:
                    continue
                ans = client.generate(quiz_prompt(history[-6000:], quiz["q"]),
                                      temperature=0.0, seed=seed)
                correct = quiz["answer"] in ans.lower()
                append(out, {"key": qkey, "kind": "quiz", "run": run_id,
                             "strategy": strategy, "mode": mode, "seed": seed,
                             "turn": turn, "question": quiz["q"],
                             "gold": quiz["answer"], "answer": ans[:200],
                             "correct": correct})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--modes", nargs="+", default=["blind", "guided"],
                    choices=["blind", "guided"])
    args = ap.parse_args()

    client = OllamaClient(model=args.model, host=args.host)
    out = RESULTS / "turns.jsonl"
    done = existing_keys(out)

    for strategy in STRATEGIES:
        for mode in args.modes:
            for seed in range(args.seeds):
                run_id = f"{strategy}|{mode}|s{seed}"
                run_keys = [k for k in done if k.startswith(run_id + "|t")]
                if run_keys:
                    last = max(int(k.split("|t")[1].split("|")[0]) for k in run_keys)
                    if last >= args.turns:
                        print(f"skip completed run {run_id}")
                        continue
                    print(f"run {run_id} incomplete (reached t{last}); replaying")
                print(f"=== {run_id} ===")
                play_one_game(client, strategy, mode, seed, args.turns, out,
                              done=set())

    print("Done. Now run: python analyze.py")


if __name__ == "__main__":
    main()
