"""
FastAPI web server for the 10-K Analyst Agent.

Exposes:
  GET  /          → HTML interface
  POST /analyze   → upload PDF, get analysis
  POST /ask       → ask question about already-uploaded PDF
  GET  /health    → health check

Run locally:
    uvicorn src.web.server:app --reload --port 8000

Then open http://localhost:8000
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from src.agent.agent import TenKAnalyst

app = FastAPI(
    title="10-K Analyst Agent",
    description="AI-powered SEC 10-K filing analysis with citation grounding",
    version="0.1.0",
)

# Allow MuleRun (and dev origins) to embed via iframe
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten before production
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add iframe-friendly headers (required by MuleRun)
@app.middleware("http")
async def add_iframe_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "ALLOWALL"
    response.headers["Content-Security-Policy"] = "frame-ancestors *;"
    return response


# In-memory agent cache (per-session in production, single agent for now)
_agent_cache: dict = {}


@app.get("/health")
def health():
    return {"status": "ok", "agent_loaded": "default" in _agent_cache}


@app.get("/", response_class=HTMLResponse)
def home():
    """Simple HTML interface."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>10-K Analyst Agent</title>
    <meta charset="utf-8">
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, system-ui, sans-serif;
            max-width: 760px;
            margin: 40px auto;
            padding: 20px;
            background: #fafafa;
            color: #1a1a1a;
        }
        h1 { font-size: 24px; margin-bottom: 8px; }
        .subtitle { color: #666; margin-bottom: 24px; }
        .panel {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
        }
        label { display: block; font-weight: 600; margin-bottom: 8px; }
        input[type=file] { margin-bottom: 12px; }
        textarea {
            width: 100%;
            min-height: 80px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-family: inherit;
            font-size: 14px;
        }
        button {
            background: #1a1a1a;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        button:hover:not(:disabled) { background: #333; }
        .answer {
            background: #f5f5f5;
            border-left: 3px solid #1a1a1a;
            padding: 16px;
            margin-top: 16px;
            border-radius: 4px;
            white-space: pre-wrap;
        }
        .citations {
            font-size: 12px;
            color: #666;
            margin-top: 8px;
        }
        .grounded { color: #0a7d35; }
        .not-grounded { color: #c93a3a; }
        .loading { color: #888; font-style: italic; }
    </style>
</head>
<body>
    <h1>10-K Analyst Agent</h1>
    <div class="subtitle">Upload a SEC 10-K filing → ask financial questions → get cited answers.</div>

    <div class="panel">
        <label for="pdf">1. Upload 10-K PDF</label>
        <input type="file" id="pdf" accept=".pdf">
        <button onclick="uploadPDF()" id="uploadBtn">Process filing</button>
        <div id="uploadStatus"></div>
    </div>

    <div class="panel">
        <label for="question">2. Ask a question</label>
        <textarea id="question" placeholder="e.g. What was iPhone revenue in fiscal 2024?"></textarea>
        <button onclick="askQuestion()" id="askBtn" disabled>Ask</button>
        <div id="answer"></div>
    </div>

<script>
async function uploadPDF() {
    const fileInput = document.getElementById('pdf');
    const status = document.getElementById('uploadStatus');
    const askBtn = document.getElementById('askBtn');
    const uploadBtn = document.getElementById('uploadBtn');

    if (!fileInput.files[0]) {
        status.innerHTML = '<span style="color:#c93a3a">Please select a PDF first.</span>';
        return;
    }

    uploadBtn.disabled = true;
    status.innerHTML = '<span class="loading">Processing PDF... this takes 30-60 seconds.</span>';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const res = await fetch('/analyze', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.success) {
            status.innerHTML = `<span style="color:#0a7d35">✓ Loaded: ${data.chunks} chunks across ${data.pages} pages.</span>`;
            askBtn.disabled = false;
        } else {
            status.innerHTML = `<span style="color:#c93a3a">Error: ${data.error}</span>`;
        }
    } catch (e) {
        status.innerHTML = `<span style="color:#c93a3a">Network error: ${e.message}</span>`;
    }
    uploadBtn.disabled = false;
}

async function askQuestion() {
    const q = document.getElementById('question').value.trim();
    const answerDiv = document.getElementById('answer');
    const askBtn = document.getElementById('askBtn');

    if (!q) return;

    askBtn.disabled = true;
    answerDiv.innerHTML = '<div class="loading">Thinking...</div>';

    try {
        const res = await fetch('/ask', {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: 'question=' + encodeURIComponent(q),
        });
        const data = await res.json();
        const groundedClass = data.grounded ? 'grounded' : 'not-grounded';
        const groundedLabel = data.grounded ? '✓ Grounded' : '✗ Not in filing';
        answerDiv.innerHTML = `
            <div class="answer">${data.answer}</div>
            <div class="citations">
                <span class="${groundedClass}">${groundedLabel}</span> ·
                Sources: ${data.citations.join(', ') || 'none'}
            </div>
        `;
    } catch (e) {
        answerDiv.innerHTML = `<span style="color:#c93a3a">Error: ${e.message}</span>`;
    }
    askBtn.disabled = false;
}
</script>
</body>
</html>
"""


@app.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)):
    """Upload a 10-K PDF and build the agent."""
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"success": False, "error": "Only PDF files supported"})

    try:
        # Save upload to temp
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Build agent (this is slow — 30-60s for embedding)
        agent = TenKAnalyst.from_pdf(tmp_path)
        _agent_cache["default"] = agent

        # Get stats
        chunk_count = len(agent.searcher.chunks)
        page_count = max(c.metadata.get("page_number", 0) for c in agent.searcher.chunks)

        os.unlink(tmp_path)

        return {
            "success": True,
            "chunks": chunk_count,
            "pages": page_count,
        }
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    """Ask a question against the loaded agent."""
    if "default" not in _agent_cache:
        return JSONResponse({
            "answer": "Please upload a 10-K PDF first.",
            "citations": [],
            "grounded": False,
        })

    agent = _agent_cache["default"]
    result = agent.ask(question)

    return {
        "answer": result.answer,
        "citations": result.citations,
        "grounded": result.is_grounded,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)