"""
Tests.

The unit tests run offline (no API keys, no network) so they work in CI as a
regression guard — they cover the cost math, the metrics aggregation, and the
graph's routing logic. The integration test actually runs the pipeline and is
skipped unless RUN_LIVE=1 and keys are set.

    pytest                 # fast offline tests
    RUN_LIVE=1 pytest      # also runs one real end-to-end pipeline
"""

import os
import pytest

# Offline tests must import the graph (which constructs clients) without real
# credentials. A dummy key lets construction succeed; nothing here calls the API.
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from metrics import MetricsCallback, LLMCall, cost_of
from graph import route_after_critic
from reliability import parse_retry_after


# ── retry-after parsing (honors Groq's 429 hint) ──────────────────────────────
def test_parse_retry_after_minutes_seconds():
    assert parse_retry_after("Please try again in 37m15.168s") == pytest.approx(2235.168)


def test_parse_retry_after_milliseconds():
    assert parse_retry_after("try again in 830ms") == pytest.approx(0.83)


def test_parse_retry_after_seconds():
    assert parse_retry_after("try again in 32.5s") == pytest.approx(32.5)


def test_parse_retry_after_none():
    assert parse_retry_after("some unrelated error") is None


# ── cost math ─────────────────────────────────────────────────────────────────
def test_cost_of_known_model():
    # 1M in + 1M out on llama-3.3-70b-versatile = 0.59 + 0.79
    assert cost_of("llama-3.3-70b-versatile", 1_000_000, 1_000_000) == pytest.approx(1.38)


def test_cost_of_unknown_model_falls_back():
    # unknown model should not crash; falls back to the default price table entry
    assert cost_of("nope", 1000, 1000) >= 0


# ── metrics aggregation ───────────────────────────────────────────────────────
def test_metrics_summary_aggregates():
    cb = MetricsCallback()
    cb.calls = [
        LLMCall("llama-3.3-70b-versatile", 1000, 500, 0.4),
        LLMCall("llama-3.3-70b-versatile", 2000, 800, 0.6),
    ]
    s = cb.summary()
    assert s["llm_calls"] == 2
    assert s["input_tokens"] == 3000
    assert s["output_tokens"] == 1300
    assert s["total_tokens"] == 4300
    assert s["llm_latency_s"] == pytest.approx(1.0)
    assert s["cost_usd"] > 0


# ── routing logic ─────────────────────────────────────────────────────────────
def test_route_accepts_when_bar_met():
    st = {"score": 9, "quality_bar": 8, "revision": 1, "max_revisions": 3}
    assert route_after_critic(st) == "accept"


def test_route_researches_more_when_below_bar():
    st = {"score": 6, "quality_bar": 8, "revision": 1, "max_revisions": 3}
    assert route_after_critic(st) == "research_more"


def test_route_accepts_at_revision_cap():
    st = {"score": 6, "quality_bar": 8, "revision": 3, "max_revisions": 3}
    assert route_after_critic(st) == "accept"


def test_route_early_stops_when_not_improving():
    # below bar, revisions left, but the revision didn't beat the best → stop
    st = {"score": 6, "quality_bar": 8, "revision": 1, "max_revisions": 3, "improving": False}
    assert route_after_critic(st) == "accept"


# ── live integration (opt-in) ─────────────────────────────────────────────────
@pytest.mark.skipif(os.getenv("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to run the real pipeline")
@pytest.mark.asyncio
async def test_pipeline_end_to_end():
    from graph import graph, initial_state
    state = initial_state("What is retrieval-augmented generation?", quality_bar=8, max_revisions=1)
    final = {}
    async for step in graph.astream(state):
        for _n, u in step.items():
            final.update(u)
    assert final.get("report")
    assert final.get("score_history")
    assert 0 <= final.get("best_score", -1) <= 10