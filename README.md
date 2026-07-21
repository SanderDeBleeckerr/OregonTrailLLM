# Westward Trail: Scrutinizing an LLM Game Master

Course: Key Topics in AI — Assignment 10 (*Developing GPT/LLM based apps via
prompt engineering and scrutinizing it*)

The app is an Oregon-Trail-style survival game where a **local LLM is the game
master**: it narrates events and proposes numeric consequences, while a Python
harness holds the authoritative game state and enforces the rules. Because
ground truth always lives in the harness, every turn yields measurable data on
how well the model narrates *within constraints* — the scrutiny half of the
assignment.

## 60-second quickstart

```bash
# 1. Install a CUDA build of torch (from PyTorch's index, not PyPI), then the rest:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# 2. Play — serve.py auto-starts the text server (first run downloads ~16 GB):
python serve.py --no-image
```

Opens http://localhost:8080. Python 3.10+ (tested on 3.13). The first turn is
slow while the model loads onto the GPU — that's normal, not a hang. Drop
`--no-image` once you're happy to also download SD-Turbo (~2.5 GB).

## Setup

The game logic (`engine.py`, `serve.py`, `play.py`) is pure stdlib; all the
model-hosting weight lives in two sidecar processes, `text_server.py` and
`image_server.py`, which need the packages below.

**Game master (required).** `serve.py` auto-starts `text_server.py`, which
runs `google/gemma-4-E4B-it` — the default in `llm_client.py`
(`DEFAULT_MODEL`, the single place to change it permanently) — through
HF transformers, quantized to 4-bit NF4 via bitsandbytes on CUDA (~6 GB
VRAM; the first run downloads the full bf16 weights, ~16 GB). On a different
GPU, pick another entry from the catalog in `configs/text-llm-options.json`
and pass its **`hf_id`** (not its `id`) to `--model`. Avoid reasoning models
(qwen3, deepseek-r1): they spend the token budget on chain-of-thought and
rarely reach the JSON the harness needs. The context budget is
`text_server.py --ctx` (default 16384).

**Scene images (optional).** `serve.py` also auto-starts `image_server.py`,
which renders each day's scene with SD-Turbo through diffusers + CUDA. The
first run downloads the checkpoint (~2.5 GB) from Hugging Face automatically.
`serve.py --no-image` skips it; the game is unaffected.

Torch's CUDA build does **not** come from PyPI — install it first from
PyTorch's index (see quickstart). The cu128 wheel covers NVIDIA GTX 10-series
through RTX 50-series; Blackwell (RTX 50) cards *require* it — older cu12x
wheels have no sm_120 kernels. For AMD, Apple Silicon, or CPU-only setups,
pick the matching install command at https://pytorch.org/get-started/locally/
(bitsandbytes NF4 is CUDA-only; `text_server.py` falls back to bf16/fp32).

**Analysis plots (optional).** `analyze.py` needs matplotlib, included in
`requirements.txt`.

## Play it

In the browser (recommended — shows the full state panel):

```bash
python serve.py --strategy rules_explicit
```

Opens http://localhost:8080 (`--port` to change). The page renders the
authoritative state — miles progress, resources, group sentiment, and each
party member's health with alive/sick/tired badges — alongside the game
master's narrative. Each option shows the effects the model *proposed*, before
`engine.check_rules` has judged them; anything it clamps is surfaced as a
referee note rather than hidden. Choose with the mouse or the `1`/`2`/`3` keys.
Add `--bind 0.0.0.0` to let others on your LAN play; the LLM still runs on this
machine. Stdlib only — no Flask needed.

Or in the terminal:

```bash
python play.py --strategy rules_explicit
```

## Run the scrutiny experiments

```bash
python run_experiments.py --seeds 2 --turns 20
python analyze.py
```

~3 strategies × 2 modes × 2 seeds × 20 turns ≈ 240 turn calls plus 96 memory
quizzes. Fully completed runs are skipped on rerun.

## Experiment design

| Experiment | Research question | Measurement |
|------------|-------------------|-------------|
| Format reliability | Does the model produce schema-valid JSON reliably? | parse rate per prompt strategy |
| Rule adherence | Does it respect hard game rules (mileage caps, ammo costs, no overdrafts)? | violation rate and taxonomy (`engine.check_rules`) |
| State tracking | Can it *track state* over a long conversation? | per-field \|believed − true\| per turn, **guided** (state shown every turn) vs **blind** (state shown only at turn 1) |
| Memory recall | Does it remember details of the conversation? | facts stated once at game start, quizzed at turns 5/10/15/20 |

### Two calls per turn: narrator, then scorer

A single call asked to write a story *and* price it does both jobs badly: it
reaches for dramatic numbers that contradict the action it just wrote — a
"brief rest" costing an ox, `+105` health on a member already at 100. The
narrative is fine; the arithmetic is decoration.

`serve.py` therefore splits the turn:

1. **narrator** (`prompts.narrate_prompt`, temp 0.8) — invents the day and the
   three choices, and is told to write *no numbers at all*.
2. **scorer** (`prompts.effects_prompt`, temp 0.1) — never invents events. It
   receives the true state, the narrative, and the three action texts, and
   assigns effects under the hard rules plus coherence rules
   (`prompts.COHERENCE`) that only *can* be written once an action exists:
   "resting never injures anyone", "an action never consumes a resource it does
   not mention".

Each call carries only the instructions its own job needs, and the numbers run
near-greedy — creativity in a scorer shows up as broken rules, not better prose.
`--single-call` restores the one-call behavior, and the setup screen has a
toggle, so the two are A/B comparable from the same build.

Note the vocabulary: the **scorer** is an LLM and merely *proposes*. The
**referee** is `engine.check_rules` — code, authoritative, and the thing that
clamps. Everything the scorer says still goes through it.

### Tracked state

The harness owns all of it; the LLM only ever *proposes* changes.

| Field | Owner | Notes |
|-------|-------|-------|
| day, miles, food, oxen, bullets, money | harness | LLM proposes deltas, harness clamps |
| daily food upkeep (5 lbs × living member) | harness | applied automatically; **stated in `BASE`** so the model neither double-charges meals nor drifts on food |
| per-member `health` (0–100) | harness | 0 = dead; a dead member is excluded from `alive()` |
| per-member `sick` / `tired` | LLM proposes | absolute booleans, not deltas |
| group `sentiment` | LLM proposes | `despairing < grim < uneasy < okay < hopeful < elated`, defaults to `okay`; only membership is enforced, not how far it moves per day |

Because `sick`/`tired`/`sentiment` are model-proposed, they widen the
rule-adherence error taxonomy with mood/illness coherence failures the numeric
rules can't catch:

- `unknown_sentiment:<x>` — a mood outside the scale
- `recovery_without_health` — cleared `sick` for a member at ≤30 health
- `status_on_dead_member:<name>`, `non_boolean_status:<name>.<flag>`,
  `malformed_status:<name>`

Prompt strategies isolate one variable each (`prompts.py`):

- `minimal` — task + bare schema
- `fewshot_schema` — + one worked example turn
- `rules_explicit` — + hard rules written out verbatim

## Project layout

```
engine.py            authoritative game state, rule checks, effect clamping
prompts.py           3 GM strategies, the narrator/scorer pair, intro, quizzes
text_server.py       text generation: HF transformers + torch, own port
image_server.py      scene images: SD-Turbo via diffusers + CUDA, own port
serve.py             local web server (stdlib) — the app's browser front end
web/index.html       the UI: state panel, narrative, options, journal
play.py              interactive game, terminal version
run_experiments.py   scripted-bot playthroughs, resumable JSONL logging
analyze.py           Markdown + LaTeX tables, drift/quiz/violation plots
results/             generated outputs
```

Both front ends drive the same `engine.py`, so anything the web UI shows is
state the harness already validated — the browser holds no game logic.

## Notes for the report

- The blind-mode state-tracking drift curves are the headline result: they
  quantify context fidelity degradation as the game (conversation) grows.
- The daily food upkeep used to live only in `engine.apply_effects`, invisible
  to the model. Blind-mode food drift therefore grew ~20/day no matter how good
  the model's recall was — it measured a hidden harness rule, not context
  fidelity. `BASE` now states the rule. Any food-drift numbers collected before
  that change are not comparable with numbers collected after it.
- The rule-adherence violation taxonomy is the "systematic and reproducible errors" the
  assignment asks for; `results/qualitative_samples.md` collects concrete
  cases for the discussion section.
- Rerun with a second model into a fresh `results/` for a model comparison.
