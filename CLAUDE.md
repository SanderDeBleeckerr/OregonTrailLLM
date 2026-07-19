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
configs/text-llm-options.json  Model catalog by VRAM tier (non-thinking models only)
image_server.py      Scene images: SD-Turbo via diffusers + CUDA, own port
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

A third call, **outcome** (`prompts.outcome_prompt`, temp 0.8), runs after the
harness applies the pick: given the premise, the chosen option, and the applied
effects/events as immutable facts, it narrates how the day played out for the
post-day screen. Best-effort — on failure the screen just shows the badges.

Vocabulary: the **scorer** is an LLM and only *proposes*. The **referee** is
`engine.check_rules` — code, authoritative, and the thing that clamps.

## Hard constraints

- **Local inference only.** No cloud APIs, no API keys. This is why the web UI is served locally: a remotely hosted page could not reach the model.
- **Non-thinking models only.** Never auto-select deepseek-r1, qwen3, gpt-oss, qwq, magistral, or anything that emits `<think>` blocks. The model catalog (`configs/text-llm-options.json`) is the single source of truth for allowed models. Always strip `<think>`/`<thinking>` blocks defensively anyway.
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
| E4 | Does it remember facts from early in the conversation? | Party facts stated once at game start (Elsie's weak ankle, Jonas the ex-blacksmith, Ruben the best shot), quizzed at turns 5/10/15/20 |

Additional logged metric: **thinking leakage rate** — how often `<think>` blocks appear in the raw response despite using a non-thinking model and/or disabling thinking.

## Game rules (enforced by the harness, optionally told to the LLM)

1. Miles per turn: 0–25. If the effects omit `miles` entirely, the harness advances a default 10 — the wagon always moves unless the model explicitly says otherwise.
2. Hunting requires ≥10 bullets and must include `bullets: -10` (or more negative); food gained from one hunt ≤100 lbs. In the web flow a hunt proposed with <10 bullets is **voided**, not just logged: `serve.choose` strips the food and bullet deltas (meals still charge), and both narrator and scorer are told outright via a dynamic `NOTE:` line (`serve._low_bullet_note`) that hunting is off the table whenever bullets are low — a stated fact works where the written conditional rule kept failing.
3. No effect may push food, oxen, bullets, or money below 0.
4. Only existing party member names are valid in `party_health`.
5. A single event never heals anyone by more than 40 HP.
6. Daily food consumption: 5 lbs × alive party members, charged by the harness **only when the effects omit `food`**. An explicit `food` delta is the day's entire food change, meals included — otherwise the model's constant small food rewards silently cancel the meals and food never visibly drops. **This rule is stated in `prompts.BASE`** — it must be. While it was harness-only the model double-charged meals, and blind-mode food drift grew ~20/day no matter how good its recall was, measuring a hidden rule instead of context fidelity.
7. Trail length: 200 miles. Game ends on arrival, party death, or starvation with no resources.
8. Per-member `sick` / `tired`: LLM-proposed **absolute booleans**, not deltas. `sick` may never be cleared for anyone at ≤30 HP; a dead member (0 HP) carries no flags.
9. Group `sentiment`: one of `despairing|grim|uneasy|okay|hopeful|elated`, default `okay`. Only membership in that scale is enforced. How far the mood may move in a day is deliberately **un**constrained — a cap was tried and only ever fired on days the model had good reason to swing, so it manufactured violations rather than catching them.
10. Deadly illness: whenever a turn's *applied* miles reach 20+, there's a 30% harness-rolled chance a random living, not-already-ill party member contracts cholera, dysentery, typhoid fever, or diphtheria. This is **never LLM-proposed** — `engine.apply_effects` rolls it directly, after this turn's own effects are applied, so nothing the model writes can cause or prevent it. The victim dies at the start of the *next* `apply_effects` call (one full turn of warning), goes to 0 HP, and stays dead — no rule ever revives a 0 HP member. The event text (e.g. "Jonas has fallen gravely ill with cholera.") is appended to game history so the narrator stays consistent with it, and surfaced to the player as a distinct "Grave news" note, separate from referee-clamp notes.

11. Night scavenge (web UI): once per day the player may arm a scavenge before choosing the day's option; it resolves in the harness right after the day's effects, never on a finished game. It requires ≥2 bullets **at resolution time** (armed but under 2 → reported as skipped, nothing rolled) and costs 2 bullets, charged when the night resolves. Each chance is rolled **independently** (a night can yield several outcomes or none): 10% a member dies, 12% one falls sick, 20% one is wounded (−20 HP, floored at 1 — death is its own roll), 1% an ox is lost, 2% a recruit joins from a fixed pool (Jell the random, Odi the lucky, Hrozna the dog — each joinable once), 15% +10 food, 10% +$10, 3% a medicine cures the whole group including deadly illness, 18% +4 bullets, 8% +1 ox. Like the deadly-illness roll this is never LLM-proposed — the LLM only narrates the pre-rolled outcomes (`prompts/scavenge.txt`), with a stock fallback line if it can't. Results + story appear on the post-day consequence screen and go into game history so later narration stays consistent.

12. Encounter days (web UI): every 4th day (`day % ENCOUNTER_EVERY_DAYS == 0`) the narrator gets `prompts/encounter.txt` instead of the normal turn prompt: the party sights strangers, and the three options are fixed stances in fixed order — approach / pass close / swing wide — phrased freely by the LLM. The scorer never prices these turns (in either mode); `engine.encounter_effects` rolls them: **approach** = 5 miles, 15% a member is lost, 35% +$20, 30% +6 bullets, and with ≥$50 a 25% chance to buy an ox for $50; **pass** = 12 miles, 8% a member is killed, 15% one is wounded (−20 HP, non-lethal); **avoid** = 3 miles, no risk, no gain. Grave outcomes surface as harness events; gains/costs appear as ordinary effect badges. This also counters option monotony — `prompts/narrate_schema.txt` additionally demands options differ in kind, not pace (its old "push on / wait / spend" example taught small models to offer exactly that trio every day).

13. Starvation: at the end of every `apply_effects` in which food stands at 0 after the day's food accounting, each living member loses 12 HP (`STARVATION_HP`), repeating every turn until food is above 0 again. Purely harness-rolled — never LLM-proposed — and surfaced as harness events ("has starved to death" when it kills). The day's own food delta counts before the check (food found during the day averts that evening's starvation); a night scavenge resolves after it, so scavenged food only saves the party from the next day onward.

## Model selection

There is no setup wizard — selection is manual. `DEFAULT_MODEL` in `llm_client.py` (`gemma4:e4b`) is what runs unless `--model` is passed. `configs/text-llm-options.json` is the catalog: pick the highest-`quality_rank` entry whose `min_vram_gb` fits the machine, pull and pass its `pull_tag` (never its `id`). The catalog's `selection_rules` and `runtime` blocks carry the VRAM/num_ctx reasoning; `llm_client.py` sets `num_ctx` on every call.

Scene images bypass Ollama entirely: `image_server.py` runs `stabilityai/sd-turbo` through diffusers + CUDA (auto-downloaded from Hugging Face on first run, ~2.5 GB). `serve.py` spawns it automatically; `--no-image` disables it and the game degrades to the placeholder art.

## Code style

- Python 3.10+. Type hints on all function signatures.
- Only stdlib + matplotlib (analysis plots) + torch/diffusers (image_server.py only). No langchain, no frameworks. The game itself stays pure stdlib.
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
# 2. Pull the default game master model:
ollama pull gemma4:e4b

# 3. Play the game in the browser (http://localhost:8080):
python serve.py            # add --no-image if torch/diffusers aren't installed
#    ...or in the terminal:
python play.py

# Optional extras (scene images, analysis plots) — see requirements.txt:
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 4. Run experiments (scripted bot, ~30-45 min):
python run_experiments.py --seeds 2 --turns 20

# 5. Generate tables and plots for the report:
python analyze.py
# -> results/tables.md, results/tables.tex, results/*.png
```
