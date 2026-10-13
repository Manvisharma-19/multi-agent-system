"""
Central configuration: which models each role uses, and per-model pricing so the
metrics layer can turn token counts into real dollar figures.

Prices are USD per 1,000,000 tokens. Verify against https://groq.com/pricing —
they change. (Llama 3.3 70B Versatile was $0.59 in / $0.79 out as of mid-2026.)
"""

WRITER_MODEL = "openai/gpt-oss-120b"
CRITIC_MODEL = "openai/gpt-oss-120b"
JUDGE_MODEL = "openai/gpt-oss-120b"
DEFAULT_MODEL = "openai/gpt-oss-120b"

WRITER_TEMPERATURE = 0.3
CRITIC_TEMPERATURE = 0.0
JUDGE_TEMPERATURE = 0.0

# Default report structure: "report" (concise) or "paper" (full IEEE academic).
# Overridable per-run via the CLI, the API (?report_format=), or the web UI.
REPORT_FORMAT = "report"

PRICING = {
    # model_name: {input $/1M, output $/1M}
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
}