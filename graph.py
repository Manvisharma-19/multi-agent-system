"""
The research graph.

    search ─▶ read ─▶ write ─▶ critic ─(pass / out of revisions)─▶ END
                        ▲                        │
                        │                (below bar, revisions left)
                        │                        ▼
                        └──────────────── gap_search
                        (rewrite with the NEW evidence + the feedback)

The key idea: when the Critic rejects a draft, control does NOT go straight back
to the Writer. It goes through `gap_search`, which reads the Critic's specific
complaints, turns them into fresh web queries (`gap_chain`), pulls and scrapes
*new* sources, and appends them. Only then does the Writer rewrite. So every
revision can improve substance, not just prose — which is what stops the score
from plateauing.

We also keep the best-scoring draft, so if the loop never clears the bar we still
return the strongest report rather than the last one.

Nodes are async so scraping fans out concurrently.
"""

import asyncio
from typing import TypedDict

from langgraph.graph import StateGraph, END

from tools import search_web, scrape_many
from agents import writer_chain, critic_chain, gap_chain, Critique, GapQueries, structure_for
from reliability import ainvoke_safe
from config import REPORT_FORMAT


class ResearchState(TypedDict, total=False):
    topic: str
    search_results: str
    urls: list[str]
    seen_urls: list[str]      # every URL used so far (for dedup across gap rounds)
    sources: str              # accumulated scraped evidence (grows each gap round)
    report: str
    critique: dict
    score: int
    revision: int             # drafts written so far
    max_revisions: int
    quality_bar: int
    report_format: str        # "report" (concise) or "paper" (full IEEE academic)
    score_history: list[int]  # score after each revision — the "climb"
    best_report: str          # highest-scoring draft seen
    best_score: int
    improving: bool           # did the latest revision beat the best so far?
    # transient fields surfaced to the UI on a gap round:
    gap_queries: list[str]
    new_urls: list[str]
    new_source_count: int


SCRAPE_LIMIT = 4  # pages scraped per (initial or gap) round


def _combine(scraped):
    return "\n\n".join(f"SOURCE: {url}\n{body}" for url, body in scraped)


# ── Nodes ─────────────────────────────────────────────────────────────────────
async def search_node(state: ResearchState) -> dict:
    text, urls = await asyncio.to_thread(search_web, state["topic"])
    return {"search_results": text, "urls": urls, "seen_urls": list(urls)}


async def read_node(state: ResearchState) -> dict:
    urls = state.get("urls", [])
    scraped = await scrape_many(urls, limit=SCRAPE_LIMIT)
    good = [(u, b) for u, b in scraped if b and not b.startswith("[")]
    if not urls:
        sources = "(No sources were found for this topic.)"
    elif not good:
        sources = "(Sources were found but none could be scraped.)\n\n" + _combine(scraped)
    else:
        sources = _combine(scraped)
    return {"sources": sources}


async def write_node(state: ResearchState) -> dict:
    feedback = ""
    if state.get("critique"):
        prev = state["critique"]
        fixes = "; ".join(prev.get("improvements", []))
        feedback = (
            f"This is a REVISION. Your previous draft scored {prev.get('score')}/10. "
            f"Additional sources have been gathered below to help — use them. "
            f"Address these specific issues: {fixes}"
        )
    research = (
        f"SEARCH RESULTS:\n{state.get('search_results', '')}\n\n"
        f"DETAILED SOURCE CONTENT:\n{state.get('sources', '')}"
    )
    report = await ainvoke_safe(
        writer_chain,
        {
            "topic": state["topic"],
            "research": research,
            "feedback": feedback,
            "structure": structure_for(state.get("report_format", REPORT_FORMAT)),
        },
        label="writer",
    )
    return {"report": report, "revision": state.get("revision", 0) + 1}


async def critic_node(state: ResearchState) -> dict:
    bar = state["quality_bar"]
    # If the structured critic keeps failing, accept the current draft rather than
    # crash or loop forever — a graceful, honest degradation.
    fallback = lambda: Critique(  # noqa: E731
        score=bar, strengths=[],
        improvements=["Automated review was unavailable for this draft."],
        verdict="Review unavailable; accepting the current draft.",
    )
    critique = await ainvoke_safe(
        critic_chain,
        {
            "topic": state["topic"],
            "report": state["report"],
            "sources": state.get("sources", "")[:4500],
        },
        fallback=fallback,
        label="critic",
    )
    data = critique.model_dump()
    prev_best = state.get("best_score", -1)
    improved = data["score"] > prev_best
    updates = {
        "critique": data,
        "score": data["score"],
        "score_history": state.get("score_history", []) + [data["score"]],
        "improving": improved,
    }
    if improved:
        updates["best_score"] = data["score"]
        updates["best_report"] = state["report"]
    return updates


async def gap_search_node(state: ResearchState) -> dict:
    """Turn the Critic's complaints into fresh queries, then search + scrape new sources."""
    improvements = state.get("critique", {}).get("improvements", [])[:2]
    gq = await ainvoke_safe(
        gap_chain,
        {"topic": state["topic"], "gaps": "; ".join(improvements)},
        fallback=lambda: GapQueries(queries=[state["topic"]]),
        label="gap",
    )
    queries = gq.queries[:2] or [state["topic"]]

    # Search each gap query concurrently, then keep only URLs we haven't seen.
    seen = set(state.get("seen_urls", []))
    results = await asyncio.gather(*[asyncio.to_thread(search_web, q) for q in queries])
    new_urls = []
    for _text, urls in results:
        for u in urls:
            if u not in seen:
                seen.add(u)
                new_urls.append(u)

    scraped = await scrape_many(new_urls, limit=SCRAPE_LIMIT)
    appended = state.get("sources", "") + "\n\n" + _combine(scraped)
    return {
        "sources": appended,
        "seen_urls": list(seen),
        "gap_queries": queries,
        "new_urls": new_urls[:SCRAPE_LIMIT],
        "new_source_count": len(scraped),
    }


# ── Routing ───────────────────────────────────────────────────────────────────
def route_after_critic(state: ResearchState) -> str:
    if state["score"] >= state["quality_bar"]:
        return "accept"
    if state.get("revision", 0) >= state["max_revisions"]:
        return "accept"
    # Early stop: if this revision didn't beat our best, more rounds won't help —
    # don't burn calls getting worse. Return the best draft we already have.
    if not state.get("improving", True):
        return "accept"
    return "research_more"


# ── Build ─────────────────────────────────────────────────────────────────────
def build_graph():
    b = StateGraph(ResearchState)
    b.add_node("search", search_node)
    b.add_node("read", read_node)
    b.add_node("write", write_node)
    b.add_node("critic", critic_node)
    b.add_node("gap_search", gap_search_node)

    b.set_entry_point("search")
    b.add_edge("search", "read")
    b.add_edge("read", "write")
    b.add_edge("write", "critic")
    b.add_conditional_edges(
        "critic", route_after_critic,
        {"research_more": "gap_search", "accept": END},
    )
    b.add_edge("gap_search", "write")  # rewrite with the newly gathered evidence
    return b.compile()


graph = build_graph()


def initial_state(topic: str, quality_bar: int = 8, max_revisions: int = 3,
                  report_format: str = REPORT_FORMAT) -> ResearchState:
    return {
        "topic": topic,
        "revision": 0,
        "quality_bar": quality_bar,
        "max_revisions": max_revisions,
        "report_format": report_format,
        "score_history": [],
        "seen_urls": [],
        "best_score": -1,
    }