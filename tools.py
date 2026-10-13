"""
Tools for the research pipeline.

Two flavours are exposed on purpose:
  * plain functions (`search_web`, `scrape_many`) that the LangGraph nodes call
    directly — deterministic, structured, and async where it matters;
  * `@tool` wrappers kept around in case you want an LLM tool-calling agent to
    drive search itself.

The important upgrade over the original is `scrape_many`: instead of scraping a
single URL, the Reader node fans out across the top-N results concurrently with
httpx + asyncio.gather, so deeper research costs roughly the time of one page.
"""

import asyncio
import os

import httpx
from bs4 import BeautifulSoup
from tavily import TavilyClient
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from dotenv import load_dotenv

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Tags that never carry article content — stripped before extracting text.
_JUNK_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]

# Domains that rarely make good research sources (video, social, forums).
_JUNK_DOMAINS = (
    "youtube.com", "youtu.be", "reddit.com", "pinterest.", "facebook.com",
    "twitter.com", "x.com", "tiktok.com", "instagram.com", "quora.com",
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=12), reraise=True)
def search_web(query: str, max_results: int = 8) -> tuple[str, list[str]]:
    """Search the web via Tavily (advanced depth), filtering out weak sources.

    Retries transient search failures. Returns (formatted_summary_text, urls).
    Video/social/forum domains and near-empty snippets are dropped so the Writer
    and Reader work from substantive pages, which is what lets the report — and the
    Critic's score — actually improve.
    """
    res = tavily.search(query=query, max_results=max_results, search_depth="advanced")
    blocks, urls = [], []
    for r in res.get("results", []):
        url = r["url"]
        if any(d in url for d in _JUNK_DOMAINS):
            continue
        content = r.get("content", "")
        if len(content) < 80:  # skip thin/placeholder results
            continue
        urls.append(url)
        blocks.append(f"Title: {r['title']}\nURL: {url}\nSnippet: {content[:300]}")
    return "\n\n---\n\n".join(blocks), urls


async def _scrape_one(client: httpx.AsyncClient, url: str, char_limit: int) -> tuple[str, str]:
    """Fetch and clean a single page. Never raises — failures return a marker."""
    try:
        resp = await client.get(
            url,
            timeout=10.0,
            headers={"User-Agent": "Mozilla/5.0 (ResearchBot)"},
            follow_redirects=True,
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(_JUNK_TAGS):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return url, text[:char_limit]
    except Exception as exc:  # noqa: BLE001 — one bad URL shouldn't kill the run
        return url, f"[could not scrape: {exc}]"


async def scrape_many(urls: list[str], limit: int = 4, char_limit: int = 5500) -> list[tuple[str, str]]:
    """Scrape the top `limit` URLs concurrently.

    Returns a list of (url, cleaned_text). This is the async fan-out: N pages
    fetched in parallel rather than one at a time.
    """
    urls = urls[:limit]
    if not urls:
        return []
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(*[_scrape_one(client, u, char_limit) for u in urls])


# ── Optional @tool wrappers (for an LLM-driven search agent) ──────────────────
@tool
def web_search(query: str) -> str:
    """Search the web for recent, reliable information. Returns titles, URLs, and snippets."""
    text, _ = search_web(query)
    return text


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a single URL."""
    results = asyncio.run(scrape_many([url], limit=1))
    return results[0][1] if results else "[no content]"