# CLAUDE.md — Westward Trail: LLM Game Master

## Assignment context

Course: **Key Topics in AI** (KTAI), Open University of the Netherlands, Q1 2024-2025.
Assignment: **10 — Developing GPT/LLM based apps via prompt engineering and scrutinizing it**.

This is a solo project. Deliverables are a **max 5-page ICML-format report** (LaTeX, PDF) and a **zip/repo of code + sample data + test scripts**. Grading is explicitly NOT on accuracy but on **implementation quality, creative experiments, and reasonableness of ideas**.

## What the project is

An **Oregon Trail clone** where a local LLM (via Ollama) is the game master. A Python harness holds the authoritative game state and enforces hard rules; the LLM narrates events, proposes options with numeric consequences, and tries to track state. Because ground truth always lives in the harness, every turn yields measurable data for the scrutiny experiments.

Reference game: https://oregontrail.ws/games/the-oregon-trail/

## Architecture

```
engine.py            Authoritative game state, rule checks, effect clamping
prompts.py           3 GM strategies, narrator/scorer pair, intro, quiz questions
llm_client.py        Ollama HTTP client, think-block stripping, leak detection
models.json          GPU-detected model catalog (non-thinking models only)
setup_wizard.py      Detect GPU VRAM (or ask), auto-pull best model from catalog
serve.py             Local web server (stdlib http.server) — the "app" deliverable
web/index.html       Browser UI: state panel, narrative, options, journal
play.py              Interactive playable game, terminal version
run_experiments.py   Scripted-bot playthroughs, JSONL logging, resumable
analyze.py           Markdown + LaTeX tables, plots (drift, violations, quiz)
```

Both front ends drive the same `engine.py`. The browser holds no game logic —
it only renders state the harness has already validated.

## Turn generation: two LLM calls

`serve.py` splits each turn in two, because one call asked to narrate *and*
price its own options reliably produced numbers that contradicted the action it
had just written (a "brief rest" costing an ox, `+105` health on someone unhurt):

1. **narrator** (`prompts.narrate_prompt`, temp 0.8) — invents the day and the
   three option texts. Told to write no numbers at all.
2. **scorer** (`prompts.effects_prompt`, temp 0.1) — invents no events. Given
   the true state, the narrative, and the three option texts, it assigns each
   one's consequences under the hard rules plus `prompts.COHERENCE`.

`--single-call` restores one-call behavior for A/B comparison.

Vocabulary: the **scorer** is an LLM and only *proposes*. The **referee** is
`engine.check_rules` — code, authoritative, and the thing that clamps.

## Hard constraints

- **Ollama only.** All LLM calls go to `localhost:11434` via `/api/generate`. No cloud APIs, no API keys. This is why the web UI is served locally: a remotely hosted page could not reach the model.
- **Non-thinking models only.** Never auto-select deepseek-r1, qwen3, gpt-oss, qwq, magistral, or anything that emits `<think>` blocks. The model catalog (`models.json`) is the single source of truth for allowed models. Always strip `<think>`/`<thinking>` blocks defensively anyway.
- **num_ctx must be set explicitly.** Ollama defaults to 4096 tokens regardless of the model's native max. The game needs ≥8192 (16384 preferred). Always pass `options.num_ctx` in every API call. Never rely on the default.
- **The harness is the referee.** The LLM proposes effects; the harness validates and clamps them. The game must never enter an illegal state regardless of what the model outputs. Log every violation but don't crash.
- **Resumable experiments.** `run_experiments.py` logs to append-only JSONL. Completed runs are skipped on rerun. A Ctrl-C must never corrupt data.
- **No external datasets.** The test set is the game itself. No downloads required beyond the Ollama model.

## Prompt strategies (the "prompt engineering" deliverable)

Three strategies, each varying ONE dimension so results cleanly attribute effects:

| Strategy | What it adds over the previous |
|---|---|
| `minimal` | Task description + bare JSON schema |
| `fewshot_schema` | + one fully worked example JSON turn |
| `rules_explicit` | + hard game rules written out verbatim |

Every strategy asks for the same JSON output containing:
- `believed_state`: the model's reconstruction of the current game state (for drift measurement)
- `narrative`: 2-3 sentences describing today's event
- `options`: exactly 3 choices, each with an `effects` dict of numeric deltas

## Experiments (the "scrutinizing it" deliverable)

| ID | Research question | Method |
|---|---|---|
| E1 | Does the model produce schema-valid JSON reliably? | Parse rate per strategy; format-failure taxonomy |
| E2 | Does it respect hard game rules? | Violation rate and type taxonomy via `engine.check_rules` (miles cap, ammo cost, overdrafts, unknown party members, healing caps) |
| E3 | Can it track state over a long conversation? | Per-field \|believed − true\| over turns. Two modes: **guided** (true state shown every turn) vs **blind** (true state shown only on turn 1) |
| E4 | Does it remember facts from early in the conversation? | Party facts stated once at game start (Elsie's weak ankle, Jonas the ex-blacksmith, Ruben the best shot, Marta the healer), quizzed at turns 5/10/15/20 |

Additional logged metric: **thinking leakage rate** — how often `<think>` blocks appear in the raw response despite using a non-thinking model and/or disabling thinking.

## Game rules (enforced by the harness, optionally told to the LLM)

1. Miles per turn: 0–25.
2. Hunting requires ≥10 bullets and must include `bullets: -10` (or more negative); food gained from one hunt ≤100 lbs.
3. No effect may push food, oxen, bullets, or money below 0.
4. Only existing party member names are valid in `party_health`.
5. A single event never heals anyone by more than 40 HP.
6. Daily food consumption: 5 lbs × alive party members (applied by harness after effects). **This rule is stated in `prompts.BASE`** — it must be. While it was harness-only the model double-charged meals, and blind-mode food drift grew ~20/day no matter how good its recall was, measuring a hidden rule instead of context fidelity.
7. Trail length: 2000 miles. Game ends on arrival, party death, or starvation with no resources.
8. Per-member `sick` / `tired`: LLM-proposed **absolute booleans**, not deltas. `sick` may never be cleared for anyone at ≤30 HP; a dead member (0 HP) carries no flags.
9. Group `sentiment`: one of `despairing|grim|uneasy|okay|hopeful|elated`, default `okay`. Only membership in that scale is enforced. How far the mood may move in a day is deliberately **un**constrained — a cap was tried and only ever fired on days the model had good reason to swing, so it manufactured violations rather than catching them.

## Setup wizard behavior

1. Try `nvidia-smi` → parse VRAM in MiB. Else try `rocm-smi` (AMD). Else try `sysctl hw.memsize` (macOS, use 65% of unified RAM). Else fall back to an interactive menu asking the user's GPU tier.
2. Read `models.json`, filter to `min_vram_gb <= detected`, pick highest `quality_rank`.
3. Check if model is already pulled (`ollama list`). If not, `ollama pull <pull_tag>` with progress display.
4. Launch the game with the selected model and `num_ctx = recommended_num_ctx` from the catalog.
5. If VRAM is within 1 GB of `min_vram_gb`, halve `num_ctx` rather than downgrading the model.

## Code style

- Python 3.10+. Type hints on all function signatures.
- Only stdlib + matplotlib (for analysis plots). No langchain, no frameworks.
- Write no comments and no docstrings. Code should be self-explanatory; only add a comment when the WHY is genuinely non-obvious (a hidden constraint, a subtle invariant, a workaround for a specific bug).
- JSON parsing must handle: markdown fences, extra prose around JSON, partial/malformed output. Never crash on bad model output.
- Every experiment record is one JSON line in a `.jsonl` file. Records have a `key` field for deduplication.
- Print progress during experiments (`[e1] strategy item turn: result`).

## Report structure (ICML template, max 5 pages + appendix)

1. **Introduction** — prompt engineering for game AI, why scrutiny matters, overview of the Oregon Trail concept.
2. **Methods** — architecture (harness vs LLM split), prompt strategies (table), experiment design (E1–E4), model selection and quantization.
3. **Experiments & Results** — tables and plots from `analyze.py`. Key results: format reliability, violation taxonomy with concrete examples, drift curves (blind vs guided), memory decay curve. Optional: model comparison, quantization comparison.
4. **Discussion** — what broke (the systematic errors), why (state drift, arithmetic weakness, context limits), what would help (structured decoding, tool use, explicit state injection), limitations of the study.
5. **References** — Ollama, the models used, ICML template, Oregon Trail history, relevant LLM evaluation literature.

## Files that must not be committed

- `results/` (generated, large JSONL files)
- `__pycache__/`
- `.env` or any secrets (there shouldn't be any)

## Quick start for a new contributor

```bash
# 1. Install Ollama: https://ollama.com
# 2. The setup wizard handles model selection:
python setup_wizard.py

# 3. Play the game in the browser (http://localhost:8080):
python serve.py
#    ...or in the terminal:
python play.py

# 4. Run experiments (scripted bot, ~30-45 min):
python run_experiments.py --seeds 2 --turns 20

# 5. Generate tables and plots for the report:
python analyze.py
# -> results/tables.md, results/tables.tex, results/*.png
```
