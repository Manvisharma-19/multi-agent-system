"""
Evaluation harness.

Runs the pipeline over a fixed set of topics and, for each, measures:
  * quality: first-draft score, final score, and the lift the refine loop added
  * grounding: an LLM judge scores faithfulness / relevance / coverage of the
    final report AGAINST its sources (faithfulness is the hallucination signal)
  * operations: LLM calls, tokens, cost (USD), and wall-clock time

Writes eval_results.csv and eval_report.md, and prints an aggregate table. These
are the numbers you put on your resume — run it and read them off.

    python evaluate.py            # full set
    python evaluate.py --quick    # 2 topics, faster
"""

import argparse
import asyncio
import csv
import statistics
import sys
import time

from graph import graph, initial_state
from agents import judge_chain, Judgement
from metrics import MetricsCallback
from reliability import ainvoke_safe

TOPICS = [
    "Retrieval-augmented generation evaluation methods",
    "Agentic AI in enterprise workflow automation",
    "Small language models vs large language models in production",
    "Prompt injection attacks and defenses in LLM applications",
    "Vector database options for semantic search",
    "Advances in fusion energy in 2025 and 2026",
]

FIELDS = [
    "topic", "revisions", "score_first", "score_final", "lift",
    "faithfulness", "relevance", "coverage", "sources",
    "llm_calls", "total_tokens", "cost_usd", "llm_latency_s", "wall_s",
]


async def eval_one(topic: str, quality_bar: int = 8, max_revisions: int = 1) -> dict:
    cb = MetricsCallback()
    state = initial_state(topic, quality_bar, max_revisions)

    t0 = time.perf_counter()
    final: dict = {}
    async for step in graph.astream(state, config={"callbacks": [cb]}):
        for _node, update in step.items():
            final.update(update)
    wall = time.perf_counter() - t0

    report = final.get("best_report") or final.get("report", "")
    sources = final.get("sources", "")
    history = final.get("score_history", []) or [None]

    # Trim sources sent to the judge (token saver); degrade gracefully if it fails.
    judged = await ainvoke_safe(
        judge_chain,
        {"topic": topic, "report": report, "sources": sources[:5000]},
        fallback=lambda: Judgement(faithfulness=0.0, relevance=0.0, coverage=0.0,
                                   unsupported_claims=[], notes="__judge_failed__"),
        label="judge",
    )
    failed = judged.notes == "__judge_failed__"

    m = cb.summary()
    return {
        "topic": topic,
        "revisions": final.get("revision"),
        "score_first": history[0],
        "score_final": final.get("best_score"),
        "lift": (final.get("best_score") or 0) - (history[0] or 0),
        "faithfulness": None if failed else round(judged.faithfulness, 3),
        "relevance": None if failed else round(judged.relevance, 3),
        "coverage": None if failed else round(judged.coverage, 3),
        "sources": len(final.get("seen_urls", [])),
        "llm_calls": m["llm_calls"],
        "total_tokens": m["total_tokens"],
        "cost_usd": m["cost_usd"],
        "llm_latency_s": m["llm_latency_s"],
        "wall_s": round(wall, 1),
    }


def _avg(rows, key):
    vals = [r[key] for r in rows if isinstance(r[key], (int, float))]
    return round(statistics.mean(vals), 3) if vals else 0


async def main(topics: list[str], delay: float = 25.0, revisions: int = 1):
    rows = []
    running_tokens = 0
    for i, topic in enumerate(topics, 1):
        print(f"[{i}/{len(topics)}] {topic}")
        try:
            row = await eval_one(topic, max_revisions=revisions)
            rows.append(row)
            running_tokens += row["total_tokens"]
            print(f"    score {row['score_first']}→{row['score_final']} "
                  f"(+{row['lift']}) · faithful {row['faithfulness']} · "
                  f"${row['cost_usd']} · {row['wall_s']}s")
            print(f"    running total: {running_tokens:,} tokens "
                  f"(~{running_tokens / 100000:.0%} of a 100k/day free tier)")
        except Exception as exc:  # keep going; a bad topic shouldn't kill the suite
            print(f"    FAILED: {exc}")

        # Pace between topics to stay under Groq's per-minute token limit.
        if i < len(topics) and delay > 0:
            print(f"    (waiting {delay:.0f}s to respect the per-minute rate limit…)")
            await asyncio.sleep(delay)

    if not rows:
        print("No successful runs.")
        return

    agg = {
        "runs": len(rows),
        "avg_lift": _avg(rows, "lift"),
        "avg_final_score": _avg(rows, "score_final"),
        "avg_faithfulness": _avg(rows, "faithfulness"),
        "avg_relevance": _avg(rows, "relevance"),
        "avg_revisions": _avg(rows, "revisions"),
        "avg_cost_usd": round(_avg(rows, "cost_usd"), 4),
        "avg_tokens": int(_avg(rows, "total_tokens")),
        "avg_wall_s": _avg(rows, "wall_s"),
    }

    with open("eval_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    with open("eval_report.md", "w") as f:
        f.write("# Evaluation report\n\n## Aggregate\n\n")
        for k, v in agg.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Per-topic\n\n| " + " | ".join(FIELDS) + " |\n")
        f.write("|" + "---|" * len(FIELDS) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[k]) for k in FIELDS) + " |\n")

    print("\n=== AGGREGATE ===")
    for k, v in agg.items():
        print(f"  {k:20s} {v}")
    print("\nWrote eval_results.csv and eval_report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="run only 2 topics")
    ap.add_argument("--limit", type=int, default=None, help="cap number of topics")
    ap.add_argument("--delay", type=float, default=25.0, help="seconds between topics (rate-limit pacing)")
    ap.add_argument("--revisions", type=int, default=1, help="max revisions per topic during eval")
    args = ap.parse_args()

    chosen = TOPICS[:2] if args.quick else TOPICS
    if args.limit:
        chosen = chosen[:args.limit]
    asyncio.run(main(chosen, delay=args.delay, revisions=args.revisions))