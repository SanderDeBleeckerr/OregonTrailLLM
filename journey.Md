# Stage 1

Default phi-3 model is not interesting, tried qwen3:30b model which failed due to reasoning intermingling with expected output.

# stage 2

Disabled reasoning but this did not suffice so I made a list of suitable models.


# stage 3

Context window is too short for gemma4:e4b for keeping +10 turns interesting enough.
Changed to different model and removed ollama defaults context limit.
ollama pull hf.co/bartowski/Qwen2.5-14B-Instruct-GGUF:IQ4_XS

# Stage 4

The rewards dont seem fitting for the choices very often, saving an ox by resting costs you an ox according to the game. The LLM call will now be 1 LLM call for the options and 1 for the rewards allowing more prompt tuning for the rewards.

# Stage 5

Considering often showing a graphical image via a seperate model that displays the choice to take.

ollama pull x/z-image-turbo

Considering adding sounds for ambience noise or when a choise creates suspsense.

https://pixabay.com/music

Notice the game is a bit dull because you go right away to the next day not really showing the consequence of your action. Also the consequence of your action is already visible on the choise which is great for debugging but bad for immersion.

# Stage 6

Game runs too slow the second LLM call can be a faster model.

# Stage 7

Just copying oregon trail is not very creative.
Moving to something playfully funny like you're in past times trying to move your cart to the university of Harlem.

# Stage 8

Cleaned up project moved prompts to txt files so they re easier to read and manipulate

# Stage 9

Fixed turn taking slightly so effects are only shown / calculated after a decision is taken and the user gets to see the repercussions after his decision via an afterscreen.

# Stage 10 

Got Fable5 advice on prompts:

Your prompts are already well-factored, which actually makes this easy: each technique below can be added as one new entry in STRATEGIES (or one scorer variant) and your existing E1–E4 harness measures the effect for free. That's exactly the story the assignment wants — "here's the technique, here's the measured delta."

First, a real bug I found: prompts/scorer_base.txt is broken. The schema block it shows the model is invalid JSON — line 12 has a stray "event": {"<name>": {"fight": bool, "tired": bool}, with unbalanced braces, "event" isn't a field your engine accepts, and sentiment is described in the prose below but missing from the schema itself. You're asking the model to imitate a malformed example. Fixing this alone should move your scorer parse rate (E1) and is a great "small prompt defects have measurable cost" anecdote for the report.

Techniques ranked by expected payoff

1. Budget injection (targets E2 — highest potential). Rule 3 ("never push food/oxen/bullets/money below 0 given the current state") forces the model to do subtraction, which small quantized models are bad at. Instead, compute the caps in Python and inject them as literal text into effects_prompt: "Spendable today: food ≤ 340, bullets ≤ 60, money ≤ 150. Hunting is only allowed (bullets ≥ 10): yes." This converts an arithmetic constraint into a copy constraint. I'd expect overdraft violations to drop the most of any single change.

2. Instruction position (targets E1+E2, nearly free). In all three strategies the schema and rules sit at the top, then history and state push them far from the generation point. Small models weight the end of the prompt heavily. Add a strategy that repeats a one-line format reminder and the rules after the state, just before "Generate the next turn." Classic, well-documented effect, trivially A/B-able.

3. Structured self-check inside the JSON (a non-thinking-model substitute for chain-of-thought). You can't use <think> models, but you can force cheap reasoning by adding a field the model must fill before the effects, e.g. "rule_check": "one sentence confirming each action's miles, spends, and heals are legal". The model generates its check first, then the numbers, and autoregression means the numbers are conditioned on the check. Strip the field before applying. Measure: violation rate with vs. without, plus token cost.

4. Self-consistency for the scorer (targets E2/E5-ish, creative-experiment material). The scorer runs at temp 0.1 anyway — sample it 3× and majority-vote each effect key (or take the median of each numeric delta). Report violation rate vs. single-sample and the 3× latency cost. This is a textbook technique (Wang et al., self-consistency) applied somewhere unusual, which graders tend to like.

5. Delta-annotated history (targets E3 blind-mode drift). In blind mode the model must reconstruct state from prose history. If your history lines are narrative-only, drift is inevitable. Add a variant where each history entry appends the chosen option's actual deltas: "Day 4: forded the creek (miles +10, Jonas -10 hp)". The model no longer has to remember arithmetic — it can re-derive it from the transcript. This directly tests why drift happens (missing information vs. failed accumulation), which is a stronger scientific claim than just plotting drift.

6. Fact pinning (targets E4). E4 shows memory decay of the intro facts. The mitigation experiment: a variant that re-injects a two-line party bio ("Elsie, youngest, weak ankle; Jonas, ex-blacksmith; ...") into every turn near the prompt end. Quiz accuracy should go flat at ~100%, and you frame it as "decay is a context-salience problem, not a capacity problem." Keep the original as the control — don't replace it, or you destroy the E4 measurement.

7. Constrained decoding as a ceiling baseline (frames the whole report). Ollama's format parameter accepts a JSON schema and grammar-constrains the output. It's not prompt engineering, but running it gives you the upper bound on E1 — "prompting recovered X% of the reliability that constrained decoding gets for free" is a strong discussion-section sentence, and your CLAUDE.md discussion outline already promises to talk about structured decoding.

Smaller cleanups worth a sentence each

- quiz_prompt prepends the full BASE (tone, food rules, miles defaults) — all irrelevant to answering a recall question, and it dilutes the actual question. A minimal quiz preamble is a fairer memory test.
- schema.txt says "Respond ONLY with JSON" but the two absolute-vs-delta exceptions are easy to miss; consider a one-line worked contrast ("party_health is a delta: -10; party_status is absolute: sick: true").
- The narrator is told "write NO numbers" but nothing stops it writing "$15" in an option text (your own coherence example does exactly this!) — worth deciding whether narrator-priced options should be honored or forbidden, and logging how often it happens.

If you only have time for three: fix the scorer schema bug, add budget injection, and add delta-annotated history. Those hit E1, E2, and E3 respectively with clean one-variable attribution, matching the design philosophy your strategies already follow.