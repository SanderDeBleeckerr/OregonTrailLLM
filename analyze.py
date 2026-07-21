"""Turn results/turns.jsonl into report-ready tables and plots.

Outputs (results/):
  tables.md / tables.tex     parse rate (format reliability), violation rates
                             by type (rule adherence), state-tracking drift
                             summary, memory-recall quiz accuracy
  drift_curves.png           mean |believed - true| per field vs turn (blind)
  quiz_decay.png             recall accuracy vs quiz turn
  violations.png             violations per 100 turns, by strategy
  qualitative_samples.md     narratives + violations for the report's
                             qualitative discussion
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent
RESULTS = ROOT / "results"
FIELDS = ("miles", "food", "bullets", "money", "day")


def load() -> list[dict]:
    path = RESULTS / "turns.jsonl"
    if not path.exists():
        print("no results yet"); return []
    by_key = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        by_key[r["key"]] = r  # dedupe: last record wins
    return list(by_key.values())


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def tex_table(headers, rows, caption, label):
    cols = "l" + "c" * (len(headers) - 1)
    body = " \\\\\n".join(" & ".join(str(c) for c in r) for r in rows)
    return (f"\\begin{{table}}[t]\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\centering\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n"
            + " & ".join(headers) + " \\\\\n\\midrule\n" + body
            + " \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main() -> None:
    records = load()
    if not records:
        return
    turns = [r for r in records if r["kind"] == "turn"]
    quizzes = [r for r in records if r["kind"] == "quiz"]
    strategies = sorted({r["strategy"] for r in turns})
    md, tex = [], []

    rows = []
    viol_by_strat = {}
    for s in strategies:
        sub = [r for r in turns if r["strategy"] == s]
        parse = sum(r["parse_ok"] for r in sub) / len(sub)
        parsed = [r for r in sub if r["parse_ok"]]
        n_viol = sum(len(r["violations"]) for r in parsed)
        viol_by_strat[s] = Counter(v for r in parsed for v in r["violations"])
        rows.append([s, len(sub), f"{parse:.3f}",
                     f"{100 * n_viol / max(1, len(parsed)):.1f}"])
    headers = ["Strategy", "Turns", "Parse rate", "Violations / 100 turns"]
    md += ["## Format reliability and rule adherence\n", md_table(headers, rows), ""]
    tex.append(tex_table(headers, rows,
                         "Format reliability and hard-rule violation rate per prompt strategy.",
                         "tab:format-rules"))

    all_types = sorted({t for c in viol_by_strat.values() for t in c})
    if all_types:
        vrows = [[s] + [viol_by_strat[s].get(t, 0) for t in all_types]
                 for s in strategies]
        md += ["### Violation counts by type\n",
               md_table(["Strategy"] + all_types, vrows), ""]
        tex.append(tex_table(["Strategy"] + all_types, vrows,
                             "Hard-rule violation counts by type.", "tab:violtypes"))

    rows = []
    for s in strategies:
        for mode in ("guided", "blind"):
            sub = [r for r in turns
                   if r["strategy"] == s and r["mode"] == mode and r["parse_ok"]]
            if not sub:
                continue
            missing = sum(r["drift"].get("missing", True) for r in sub) / len(sub)
            cells = [s, mode, f"{100 * (1 - missing):.0f}%"]
            for f in FIELDS:
                vals = [r["drift"][f] for r in sub
                        if r["drift"].get(f) is not None]
                cells.append(f"{sum(vals) / len(vals):.1f}" if vals else "—")
            rows.append(cells)
    headers = ["Strategy", "Mode", "State reported"] + [f"MAE {f}" for f in FIELDS]
    md += ["## State-tracking error (mean absolute)\n", md_table(headers, rows), ""]
    tex.append(tex_table(headers, rows,
                         "Mean absolute error of the model's believed state vs. ground truth.",
                         "tab:state-tracking"))

    fig, ax = plt.subplots(figsize=(8, 4))
    for s in strategies:
        for f, ls in (("food", "-"), ("miles", "--")):
            per_turn = defaultdict(list)
            for r in turns:
                if (r["strategy"] == s and r["mode"] == "blind" and r["parse_ok"]
                        and r["drift"].get(f) is not None):
                    per_turn[r["turn"]].append(r["drift"][f])
            xs = sorted(per_turn)
            if xs:
                ax.plot(xs, [sum(per_turn[t]) / len(per_turn[t]) for t in xs],
                        ls, label=f"{s} ({f})")
    ax.set_xlabel("Turn"); ax.set_ylabel("Mean |believed − true|")
    ax.set_title("State-tracking drift over the game (blind mode)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / "drift_curves.png", dpi=150)

    if quizzes:
        rows = []
        for s in strategies:
            sub = [r for r in quizzes if r["strategy"] == s]
            if not sub:
                continue
            acc = sum(r["correct"] for r in sub) / len(sub)
            rows.append([s, len(sub), f"{acc:.3f}"])
        headers = ["Strategy", "Quiz answers", "Recall accuracy"]
        md += ["## Memory recall of party facts\n", md_table(headers, rows), ""]
        tex.append(tex_table(headers, rows,
                             "Recall of facts stated once at game start.", "tab:memory"))

        fig, ax = plt.subplots(figsize=(6, 3.5))
        for s in strategies:
            per_turn = defaultdict(list)
            for r in quizzes:
                if r["strategy"] == s:
                    per_turn[r["turn"]].append(r["correct"])
            xs = sorted(per_turn)
            if xs:
                ax.plot(xs, [sum(per_turn[t]) / len(per_turn[t]) for t in xs],
                        marker="o", label=s)
        ax.set_xlabel("Quiz at turn"); ax.set_ylabel("Recall accuracy")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
        ax.set_title("Memory: fact recall vs conversational distance")
        fig.tight_layout(); fig.savefig(RESULTS / "quiz_decay.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    vals = []
    for s in strategies:
        parsed = [r for r in turns if r["strategy"] == s and r["parse_ok"]]
        vals.append(100 * sum(len(r["violations"]) for r in parsed) / max(1, len(parsed)))
    ax.bar(strategies, vals)
    ax.set_ylabel("Violations / 100 turns")
    ax.set_title("Hard-rule violations by prompt strategy")
    fig.tight_layout(); fig.savefig(RESULTS / "violations.png", dpi=150)

    interesting = [r for r in turns if r.get("violations")][:15]
    (RESULTS / "qualitative_samples.md").write_text(
        "\n\n".join(
            f"**{r['run']} turn {r['turn']}** violations={r['violations']}\n"
            f"> {r.get('narrative','')}\n> chosen: {r.get('chosen','')}"
            for r in interesting) or "No violations recorded.",
        encoding="utf-8")

    (RESULTS / "tables.md").write_text("\n".join(md), encoding="utf-8")
    (RESULTS / "tables.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"{len(turns)} turn records, {len(quizzes)} quiz records -> results/tables.*")


if __name__ == "__main__":
    main()
