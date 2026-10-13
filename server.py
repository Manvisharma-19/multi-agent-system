"""
FastAPI backend.

    GET /               -> serves static/index.html
    GET /research?topic -> Server-Sent Events stream of the graph running live

The frontend is served from static/index.html (read at request time), so there is
no giant embedded string to corrupt. A MetricsCallback is attached per request, so
the final "done" event carries token / cost / latency figures.

Run locally:
    uvicorn server:app --reload      # http://127.0.0.1:8000
On Render the start command binds $PORT.
"""

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, Response

from graph import graph, initial_state
from metrics import MetricsCallback
from pdf_export import markdown_to_pdf

app = FastAPI(title="ResearchMind")

# index.html lives next to this file, in ./static or alongside server.py.
_HERE = Path(__file__).parent
_INDEX_CANDIDATES = [_HERE / "static" / "index.html", _HERE / "index.html"]


def _load_index() -> str:
    for path in _INDEX_CANDIDATES:
        if path.exists():
            return path.read_text(encoding="utf-8")
    return (
        "<h1>index.html not found</h1>"
        "<p>Expected static/index.html next to server.py.</p>"
    )


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/")
async def index():
    return HTMLResponse(_load_index())


@app.get("/favicon.ico")
async def favicon():
    return HTMLResponse("", status_code=204)


@app.post("/download/pdf")
async def download_pdf(request: Request):
    body = await request.json()
    md = body.get("markdown", "")
    title = body.get("title", "Research Report")
    filename = (body.get("filename", "research_report") or "research_report").replace('"', "")
    pdf = markdown_to_pdf(md, title=title)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}.pdf"'},
    )


@app.get("/research")
async def research(request: Request, topic: str, quality_bar: int = 8,
                   max_revisions: int = 3, report_format: str = "report"):
    async def event_stream():
        state = initial_state(topic, quality_bar, max_revisions, report_format)
        cb = MetricsCallback()
        yield sse({"event": "start", "topic": topic, "quality_bar": quality_bar})

        final: dict = {}
        try:
            async for step in graph.astream(state, config={"callbacks": [cb]}):
                if await request.is_disconnected():
                    break
                for node, update in step.items():
                    final.update(update)
                    if node == "search":
                        yield sse({"event": "search", "count": len(update.get("urls", [])),
                                   "urls": update.get("urls", [])})
                    elif node == "read":
                        yield sse({"event": "read", "chars": len(update.get("sources", ""))})
                    elif node == "gap_search":
                        yield sse({"event": "gap",
                                   "queries": update.get("gap_queries", []),
                                   "new_urls": update.get("new_urls", []),
                                   "new_sources": update.get("new_source_count", 0)})
                    elif node == "write":
                        yield sse({"event": "write", "revision": update.get("revision"),
                                   "report": update.get("report", "")})
                    elif node == "critic":
                        yield sse({"event": "critic", "score": update.get("score"),
                                   "critique": update.get("critique"),
                                   "history": update.get("score_history", [])})
            yield sse({"event": "done",
                       "revisions": final.get("revision"),
                       "history": final.get("score_history", []),
                       "best_score": final.get("best_score"),
                       "report": final.get("best_report") or final.get("report", ""),
                       "critique": final.get("critique"),
                       "metrics": cb.summary()})
        except Exception as exc:
            yield sse({"event": "error", "message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")