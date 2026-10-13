"""
Reliability layer.

Every LLM call goes through `ainvoke_safe`, which:
  * retries transient Groq failures — rate limits (TPM), timeouts, connection
    drops, 5xx, and the intermittent malformed structured-output 400
    (`tool_use_failed`);
  * honors Groq's own "try again in Xs" hint on 429s instead of guessing;
  * gives up fast on a *daily* rate limit (a long wait means it's a budget
    problem, not a transient one), so the eval records the failure quickly;
  * falls back to a caller-supplied default when retries are exhausted, so one
    bad call degrades gracefully instead of crashing the whole run.
"""

import re

from tenacity import (
    AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter,
)

# Groq ships with langchain-groq; import its exception types defensively.
try:
    from groq import (
        RateLimitError, APIConnectionError, APITimeoutError,
        InternalServerError, BadRequestError,
    )
    _TRANSIENT = (APIConnectionError, APITimeoutError, InternalServerError)
except Exception:  # pragma: no cover - groq should always be present
    RateLimitError = BadRequestError = ()
    _TRANSIENT = ()

MAX_ATTEMPTS = 4
LONG_WAIT_CUTOFF_S = 120  # a required wait longer than this = daily budget, give up


def parse_retry_after(msg: str) -> float | None:
    """Extract the wait Groq suggests, e.g. 'try again in 37m15.168s' or '830ms'."""
    m = re.search(r"try again in\s+(\d+)m([\d.]+)s", msg)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.search(r"try again in\s+([\d.]+)ms", msg)
    if m:
        return float(m.group(1)) / 1000.0
    m = re.search(r"try again in\s+([\d.]+)s", msg)
    if m:
        return float(m.group(1))
    return None


def _is_retryable(exc: BaseException) -> bool:
    if _TRANSIENT and isinstance(exc, _TRANSIENT):
        return True
    if RateLimitError and isinstance(exc, RateLimitError):
        wait = parse_retry_after(str(exc))
        # short wait (per-minute limit) → worth retrying; long wait (per-day) → no
        return wait is None or wait <= LONG_WAIT_CUTOFF_S
    if BadRequestError and isinstance(exc, BadRequestError):
        # the stochastic structured-output failure is worth one more shot
        return "tool_use_failed" in str(exc) or "failed to call a function" in str(exc).lower()
    return False


def _wait(retry_state) -> float:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is not None:
        secs = parse_retry_after(str(exc))
        if secs is not None and secs <= LONG_WAIT_CUTOFF_S:
            return secs + 0.5
    # fall back to exponential backoff with jitter
    return wait_exponential_jitter(initial=1, max=20)(retry_state)


async def ainvoke_safe(chain, payload, *, fallback=None, label="llm"):
    """Invoke a LangChain runnable with retries; return `fallback` if all fail."""
    try:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            wait=_wait,
            stop=stop_after_attempt(MAX_ATTEMPTS),
            reraise=True,
        ):
            with attempt:
                return await chain.ainvoke(payload)
    except Exception as exc:
        if fallback is not None:
            print(f"[reliability] {label} failed after retries ({type(exc).__name__}); using fallback")
            return fallback() if callable(fallback) else fallback
        raise