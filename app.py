"""
Launcher: `python app.py` starts the real ResearchMind web UI
(FastAPI server in server.py + static/index.html) — the same UI as the
GitHub repo and the live demo.

Equivalent to:  uvicorn server:app --reload
Then open:      http://127.0.0.1:8000
"""
import webbrowser
import threading

import uvicorn

HOST, PORT = "127.0.0.1", 8000

if __name__ == "__main__":
    threading.Timer(2.0, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    uvicorn.run("server:app", host=HOST, port=PORT, reload=True)