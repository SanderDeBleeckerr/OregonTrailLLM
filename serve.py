"""Local web UI for Westward Trail.

    python serve.py                                  # DEFAULT_MODEL
    python serve.py --model qwen2.5:32b --strategy rules_explicit

Serves a browser front end on http://localhost:8080 that plays the same game
as play.py against the same local Ollama server. The harness stays the single
source of truth: the browser only ever renders state the engine has already
validated and clamped, and the referee's clamps are surfaced in the UI rather
than hidden, since those violations are the point of the experiment.

Deliberately stdlib-only (no Flask), to keep setup at `python serve.py`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from engine import (
    SENTIMENTS,
    TRAIL_MILES,
    GameState,
    apply_effects,
    check_rules,
    extract_json,
    state_dict,
)
from llm_client import DEFAULT_MODEL, OllamaClient
from prompts import INTRO, STRATEGIES, effects_prompt, narrate_prompt

ROOT = pathlib.Path(__file__).parent
WEB = ROOT / "web"

MAX_PARSE_MISSES = 3

GAMES: dict[str, dict] = {}
GAMES_LOCK = threading.Lock()


def view(state: GameState) -> dict:
    d = state_dict(state)
    d.update(
        trail_miles=TRAIL_MILES,
        sentiments=SENTIMENTS,
        alive=[p["name"] for p in state.alive()],
        dead=[p["name"] for p in state.dead()],
        sick=[p["name"] for p in state.sick()],
        tired=[p["name"] for p in state.tired()],
        finished=state.finished(),
    )
    return d


def ask_json(game: dict, prompt: str, temperature: float, valid) -> dict:
    last_raw = ""
    for _ in range(MAX_PARSE_MISSES):
        last_raw = game["client"].generate(prompt, temperature=temperature)
        data = extract_json(last_raw)
        if data is not None and valid(data):
            return data
    raise RuntimeError(
        f"{game['model']} did not return usable JSON in {MAX_PARSE_MISSES} tries. "
        f"Last reply: {last_raw[:300]!r}"
    )


def _texts(data: dict) -> list[str]:
    return [str(o.get("text", "?")) for o in data["options"][:3] if isinstance(o, dict)]


def _has_options(d: dict) -> bool:
    return isinstance(d.get("options"), list) and bool(_texts(d))


def generate_turn(game: dict) -> dict:
    state: GameState = game["state"]
    state_text = "Current true state: " + state.summary()
    history = game["history"][-4000:]

    if not game["scorer"]:
        data = ask_json(game, game["build"](state_text, history), 0.8, _has_options)
        return {
            "narrative": str(data.get("narrative", "")),
            "options": [{"text": str(o.get("text", "?")), "effects": o.get("effects") or {}}
                        for o in data["options"][:3] if isinstance(o, dict)],
        }

    story = ask_json(game, narrate_prompt(state_text, history), 0.8, _has_options)
    narrative = str(story.get("narrative", ""))
    texts = _texts(story)

    # Scorer does only arithmetic; near-greedy decoding so creativity shows up as rule violations, not prose.
    priced = ask_json(
        game,
        effects_prompt(state_text, narrative, texts),
        0.1,
        lambda d: isinstance(d.get("effects"), list) and bool(d["effects"]),
    )
    effects = [e if isinstance(e, dict) else {} for e in priced["effects"]]
    # A short scorer reply costs the tail options their effects, not the turn.
    effects += [{}] * (len(texts) - len(effects))
    return {
        "narrative": narrative,
        "options": [{"text": t, "effects": fx} for t, fx in zip(texts, effects)],
    }


def new_game(model: str, strategy: str, host: str, scorer: bool) -> dict:
    game = {
        "state": GameState(),
        "client": OllamaClient(model=model, host=host),
        "build": STRATEGIES[strategy],
        "history": INTRO,
        "model": model,
        "strategy": strategy,
        "scorer": scorer,
        "log": [],
        "lock": threading.Lock(),
    }
    game_id = uuid.uuid4().hex
    with GAMES_LOCK:
        GAMES[game_id] = game
    game["id"] = game_id
    return game


def choose(game: dict, index: int) -> dict:
    state: GameState = game["state"]
    options = game.get("turn", {}).get("options") or []
    if not 0 <= index < len(options):
        raise ValueError("no such option")

    choice = options[index]
    effects = choice.get("effects") or {}
    violations = check_rules(state, choice.get("text", ""), effects)
    game["state"] = state = apply_effects(state, effects)
    game["history"] += (
        f"Day {state.day - 1}: {game['turn'].get('narrative', '')} "
        f"Chosen: {choice.get('text', '')} Effects: {json.dumps(effects)}\n"
    )
    game["log"].append({
        "day": state.day - 1,
        "narrative": game["turn"].get("narrative", ""),
        "chosen": choice.get("text", ""),
        "effects": effects,
        "violations": violations,
    })

    if state.finished():
        game["turn"] = None
    else:
        game["turn"] = generate_turn(game)
    return {"violations": violations}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # quieter console
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def _game(self, payload: dict) -> dict:
        with GAMES_LOCK:
            game = GAMES.get(payload.get("game_id", ""))
        if game is None:
            raise KeyError("unknown game_id -- start a new journey")
        return game

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            try:
                body = (WEB / "index.html").read_bytes()
            except OSError:
                self._send(500, b"web/index.html is missing", "text/plain")
                return
            self._send(200, body, "text/html; charset=utf-8")
            return
        if path == "/api/config":
            self._json(200, {
                "strategies": list(STRATEGIES),
                "default_model": self.server.default_model,
                "default_strategy": self.server.default_strategy,
                "default_scorer": self.server.default_scorer,
            })
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "malformed JSON body"})
            return

        try:
            if self.path == "/api/new":
                scorer = payload.get("scorer")
                game = new_game(
                    payload.get("model") or self.server.default_model,
                    payload.get("strategy") or self.server.default_strategy,
                    self.server.ollama_host,
                    self.server.default_scorer if scorer is None else bool(scorer),
                )
                with game["lock"]:
                    game["turn"] = generate_turn(game)
                    self._json(200, {"game_id": game["id"], "state": view(game["state"]),
                                     "turn": game["turn"], "log": game["log"]})
                return

            if self.path == "/api/choose":
                game = self._game(payload)
                with game["lock"]:
                    result = choose(game, int(payload.get("index", -1)))
                    self._json(200, {"state": view(game["state"]), "turn": game["turn"],
                                     "log": game["log"], **result})
                return
        except KeyError as e:
            self._json(404, {"error": str(e)})
            return
        except (ValueError, TypeError) as e:
            self._json(400, {"error": str(e)})
            return
        except RuntimeError as e:
            self._json(502, {"error": str(e)})
            return

        self._json(404, {"error": "not found"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--strategy", default="rules_explicit", choices=STRATEGIES)
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama server")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--bind", default="127.0.0.1",
                    help="use 0.0.0.0 to let other machines on your LAN play")
    ap.add_argument("--single-call", action="store_true",
                    help="have one call write the story AND price the options, "
                         "instead of handing the numbers to a scorer call "
                         "(faster per turn, noticeably worse effects)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.default_model = args.model
    server.default_strategy = args.strategy
    server.default_scorer = not args.single_call
    server.ollama_host = args.host

    url = f"http://localhost:{args.port}"
    print(f"Westward Trail is running at {url}")
    print(f"  game master : {args.model} via {args.host}")
    print(f"  strategy    : {args.strategy}")
    print(f"  effects     : {'narrator + scorer' if server.default_scorer else 'inline (single call)'}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
