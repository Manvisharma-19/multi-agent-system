"""
Token / cost / latency tracing.

`MetricsCallback` is a LangChain callback handler. Attach it to a run
(`graph.astream(state, config={"callbacks": [cb]})`) and it records every LLM
call — model, input/output tokens, and wall-clock latency — by listening to the
callback events LangChain fires. `summary()` aggregates them and prices the
tokens using the table in config.py.

This is the "how much did this run cost and how long did it take" layer that
separates a demo from something you'd actually operate.
"""

import threading
import time
from dataclasses import dataclass

from langchain_core.callbacks import BaseCallbackHandler

from config import PRICING, DEFAULT_MODEL


@dataclass
class LLMCall:
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float


def cost_of(model: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICING.get(model) or PRICING.get(DEFAULT_MODEL, {"input": 0, "output": 0})
    return input_tokens / 1e6 * price["input"] + output_tokens / 1e6 * price["output"]


def _extract(response) -> tuple[str, int, int]:
    """Pull (model, input_tokens, output_tokens) out of an LLMResult across versions."""
    model, itok, otok = DEFAULT_MODEL, 0, 0
    lo = getattr(response, "llm_output", None) or {}
    if isinstance(lo, dict):
        model = lo.get("model_name") or lo.get("model") or model
        tu = lo.get("token_usage") or {}
        itok = tu.get("prompt_tokens") or 0
        otok = tu.get("completion_tokens") or 0
    if not (itok or otok):
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            um = getattr(msg, "usage_metadata", None) or {}
            itok = um.get("input_tokens") or 0
            otok = um.get("output_tokens") or 0
            rm = getattr(msg, "response_metadata", None) or {}
            model = rm.get("model_name", model)
        except Exception:
            pass
    return model, int(itok), int(otok)


class MetricsCallback(BaseCallbackHandler):
    def __init__(self):
        self.calls: list[LLMCall] = []
        self._starts: dict = {}
        self._lock = threading.Lock()

    # both fire depending on model type — capture start time for either
    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        with self._lock:
            self._starts[run_id] = time.perf_counter()

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        with self._lock:
            self._starts[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        end = time.perf_counter()
        with self._lock:
            start = self._starts.pop(run_id, end)
        model, itok, otok = _extract(response)
        self.calls.append(LLMCall(model, itok, otok, end - start))

    def summary(self) -> dict:
        itok = sum(c.input_tokens for c in self.calls)
        otok = sum(c.output_tokens for c in self.calls)
        cost = sum(cost_of(c.model, c.input_tokens, c.output_tokens) for c in self.calls)
        return {
            "llm_calls": len(self.calls),
            "input_tokens": itok,
            "output_tokens": otok,
            "total_tokens": itok + otok,
            "cost_usd": round(cost, 6),
            "llm_latency_s": round(sum(c.latency_s for c in self.calls), 2),
        }