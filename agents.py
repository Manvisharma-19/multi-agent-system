"""
Agents & chains.

Two models on purpose:
  * WRITER runs on gpt-oss-120b (strong long-form synthesis);
  * CRITIC runs on llama-3.3-70b and returns a *structured* Pydantic object
    (`Critique`) via `with_structured_output`, so the score is a real number the
    graph can branch on — not text we have to regex.

The Writer prompt is revision-aware: on the first pass `feedback` is empty; on a
refinement pass the graph feeds the Critic's improvement points back in.
"""

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from config import (
    WRITER_MODEL, CRITIC_MODEL, JUDGE_MODEL,
    WRITER_TEMPERATURE, CRITIC_TEMPERATURE, JUDGE_TEMPERATURE,
)

load_dotenv()

# ── Models ────────────────────────────────────────────────────────────────────
writer_llm = ChatGroq(model=WRITER_MODEL, temperature=WRITER_TEMPERATURE)
critic_llm = ChatGroq(model=CRITIC_MODEL, temperature=CRITIC_TEMPERATURE)


# ── Structured critic output ──────────────────────────────────────────────────
class Critique(BaseModel):
    """Machine-readable review the graph routes on."""
    score: int = Field(description="Overall quality from 0 to 10", ge=0, le=10)
    strengths: list[str] = Field(description="What the report does well")
    improvements: list[str] = Field(description="Concrete, specific fixes for the next revision")
    unsupported_claims: list[str] = Field(default_factory=list,
        description="Claims in the report NOT backed by the provided sources")
    verdict: str = Field(description="One-line overall verdict")


# ── Writer chain (revision-aware, strictly grounded) ──────────────────────────
# Two selectable structures. The writer prompt takes {structure}, chosen per run.
REPORT_STRUCTURES = {
    "report": """Structure the report as:
1. Title — clear and concise.
2. Abstract — 100-200 words summarizing the topic, key findings, and takeaway.
3. Introduction — background and why the topic matters.
4. Key Findings — at least 3 well-explained points, each grounded with inline
   citations. Include a short Markdown comparison table if the sources support one.
5. Discussion — interpretation, patterns, and where sources disagree.
6. Limitations — of the sources and the analysis.
7. Conclusion — summary and key takeaways.
8. References — every source URL used.

Citation style: cite inline as (https://full-url) right after each claim.""",

    "paper": """Structure the paper with these numbered sections (IEEE-style):
1. Title — clear, concise, descriptive.
2. Abstract — 150-300 words: problem, approach, key findings, conclusion.
3. Keywords — 4-8 key terms.
4. Introduction — background, problem statement, objectives, and the questions addressed.
5. Literature Review (Related Work) — what the sources report, existing approaches,
   their limitations, and the gap this survey addresses.
6. Methodology — this is a SURVEY grounded in web sources: describe how sources were
   gathered, selection criteria, and how findings were synthesized. Do NOT invent a
   dataset, model training, or experiments that were not performed.
7. Findings & Comparative Analysis — synthesize the evidence with a Markdown comparison
   table built ONLY from figures/claims the sources actually report; every row traceable.
8. Discussion — interpretation, patterns, strengths, and where sources disagree.
9. Limitations — of the sources and of this survey (recency, coverage, bias).
10. Conclusion — summary of findings and key takeaways.
11. Future Work — open questions and directions the sources point to.
12. References — a NUMBERED list in citation order: [1] Title. URL — one per line.

Citation style (IEEE): cite sources inline with bracketed numbers like [1], [2].
Assign each source a number the FIRST time it is cited and REUSE that same number for
every later citation of it. A sentence drawing on two sources cites both, e.g. [1], [3].
Every number used in the text MUST appear in the numbered References list, and vice versa.""",
}


def structure_for(report_format: str) -> str:
    return REPORT_STRUCTURES.get(report_format, REPORT_STRUCTURES["report"])


writer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert research writer. Write clear, structured, factual reports "
     "grounded STRICTLY in the provided research. Follow these rules exactly:\n"
     "1. Every factual claim, statistic, figure, or quote must be attributed to a "
     "specific source, using the citation style specified in the structure below.\n"
     "2. If a claim is not supported by the provided sources, OMIT it. Do not use "
     "outside/general knowledge and do not guess.\n"
     "3. Never invent facts, numbers, or citations. If the evidence on a point is "
     "thin, say so explicitly rather than filling the gap."),
    ("human", """Write a detailed, well-researched piece on the topic below.

Topic: {topic}

Research gathered:
{research}

{feedback}

{structure}

RULES:
- Every factual sentence must cite its source inline, e.g. (https://...) or [n].
- Only include claims traceable to the research above; omit anything unsupported.
- For any section where the sources genuinely lack material, say so briefly rather
  than fabricating content, datasets, experiments, or results."""),
])

writer_chain = writer_prompt | writer_llm | StrOutputParser()


# ── Critic chain (structured, source-aware) ───────────────────────────────────
critic_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a fair, calibrated research critic. GROUNDING IS THE PRIORITY: scan "
     "each factual claim and check it against the provided sources. List every claim "
     "you cannot trace to the sources as an unsupported claim.\n\n"
     "Grade the report on how well it synthesizes and is SUPPORTED BY the sources — "
     "not against an ideal of peer-reviewed academic work. Do NOT lower the score "
     "merely for lacking academic/journal citations when the web sources given serve "
     "the topic.\n\n"
     "Calibration: 8-9 = well-structured, on-topic, nearly every claim traceable to "
     "the sources with inline URLs. 6-7 = solid but has several uncited or thinly "
     "supported claims. Below 6 = multiple unsupported claims, ignores the sources, "
     "or poor structure. Only ask for things the available sources could provide."),
    ("human", """Topic: {topic}

SOURCES the report was written from:
{sources}

REPORT:
{report}

Score it 0-10 using the calibration above. List any unsupported claims, give
concrete improvements that better use of THESE sources would fix, and a one-line
verdict."""),
])

critic_chain = critic_prompt | critic_llm.with_structured_output(Critique)


# ── Gap-query generator ───────────────────────────────────────────────────────
# Turns the Critic's vague complaints ("needs more data", "add counterarguments")
# into concrete web-search queries so the next revision can gather *new* evidence
# instead of just re-phrasing what it already had.
class GapQueries(BaseModel):
    queries: list[str] = Field(
        description="1-3 specific, self-contained web search queries that would close the gaps"
    )


gap_prompt = ChatPromptTemplate.from_messages([
    ("system", "You turn a reviewer's critique into precise web search queries. "
               "Every query MUST be about the given topic and stay on it — do not "
               "drift into tangential subtopics. Each query must be specific, "
               "standalone, and searchable, not an instruction."),
    ("human", """Research topic: {topic}

The reviewer wants the next draft to address these gaps (focus on the most important):
{gaps}

Write 1-2 focused web search queries — each explicitly about "{topic}" — that would
surface the missing information. Return only the queries."""),
])

gap_chain = gap_prompt | writer_llm.with_structured_output(GapQueries)


# ── LLM-as-judge (used by the eval harness) ───────────────────────────────────
# Scores a finished report AGAINST its sources — faithfulness catches claims the
# sources don't support (the key hallucination signal), separate from the
# Critic's in-loop quality score.
class Judgement(BaseModel):
    faithfulness: float = Field(description="0-1: fraction of claims actually supported by the sources", ge=0, le=1)
    relevance: float = Field(description="0-1: how well the report answers the topic", ge=0, le=1)
    coverage: float = Field(description="0-1: breadth of the topic covered", ge=0, le=1)
    unsupported_claims: list[str] = Field(description="Specific claims not backed by the sources")
    notes: str = Field(description="One-line overall assessment")


judge_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a rigorous evaluation judge. Score the report ONLY against the "
               "provided sources. A claim not supported by the sources lowers faithfulness."),
    ("human", """Topic: {topic}

SOURCES (the only evidence that counts):
{sources}

REPORT TO JUDGE:
{report}

Score faithfulness, relevance, and coverage each from 0 to 1, list any claims the
sources do not support, and give a one-line note."""),
])

judge_llm = ChatGroq(model=JUDGE_MODEL, temperature=JUDGE_TEMPERATURE)
judge_chain = judge_prompt | judge_llm.with_structured_output(Judgement)