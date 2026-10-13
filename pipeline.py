"""
CLI entry point. Streams each node as it fires so you can watch the refine loop
climb from the terminal:

    python pipeline.py
"""

import asyncio

from graph import graph, initial_state
from metrics import MetricsCallback


async def run(topic: str, quality_bar: int = 8, max_revisions: int = 3,
              report_format: str = "report") -> dict:
    state = initial_state(topic, quality_bar, max_revisions, report_format)
    cb = MetricsCallback()
    final: dict = {}

    async for event in graph.astream(state, config={"callbacks": [cb]}):
        for node, update in event.items():
            final.update(update)
            if node == "search":
                print(f"[search]  found {len(update.get('urls', []))} sources")
            elif node == "read":
                print(f"[read]    scraped {len(update.get('sources',''))} chars")
            elif node == "write":
                print(f"[write]   draft #{update.get('revision')} ready")
            elif node == "gap_search":
                print(f"[gap]     +{update.get('new_source_count')} new sources for "
                      f"{update.get('gap_queries')}")
            elif node == "critic":
                print(f"[critic]  score {update.get('score')}/10  "
                      f"→ {'accept' if update.get('score') >= quality_bar else 'research more'}")

    best = final.get("best_report") or final.get("report", "")
    m = cb.summary()
    print("\n" + "=" * 60)
    print(f"Finished in {final.get('revision')} revision(s). "
          f"Score trajectory: {final.get('score_history')}  "
          f"Best: {final.get('best_score')}/10")
    print(f"Metrics: {m['llm_calls']} LLM calls · {m['total_tokens']:,} tokens · "
          f"${m['cost_usd']} · {m['llm_latency_s']}s model time")
    print("=" * 60 + "\n")
    print(best)
    print("\n--- CRITIC ---")
    print(final.get("critique"))
    return final


if __name__ == "__main__":
    topic = input("Enter a research topic: ").strip()
    fmt = input("Format [report/paper] (default report): ").strip() or "report"
    asyncio.run(run(topic, report_format=fmt))