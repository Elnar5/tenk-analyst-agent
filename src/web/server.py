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

import logging
import time
from datetime import datetime

# Setup analytics logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | ANALYTICS | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
analytics_log = logging.getLogger("analytics")


def track_event(event: str, **kwargs):
    """Log an analytics event. View in HuggingFace Spaces logs tab."""
    extra = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    analytics_log.info(f"event={event} | {extra}") 
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
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
    """Editorial finance UI — Bloomberg/WSJ aesthetic."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>10-K Analyst — SEC Filings, Cited.</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --paper: #faf8f3;
            --ink: #1a1a1a;
            --ink-soft: #44423d;
            --ink-mute: #8a8780;
            --rule: #d6d2c4;
            --accent: #8b1e1e;
            --accent-soft: #f3ebe8;
            --grounded: #2d6f3f;
            --not-grounded: #8b1e1e;
            --serif: 'Crimson Pro', 'Iowan Old Style', Georgia, serif;
            --mono: 'JetBrains Mono', 'Menlo', monospace;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            background: var(--paper);
            color: var(--ink);
            font-family: var(--serif);
            font-size: 18px;
            line-height: 1.55;
            min-height: 100vh;
            background-image:
                radial-gradient(circle at 20% 10%, rgba(139, 30, 30, 0.02) 0%, transparent 40%),
                radial-gradient(circle at 80% 80%, rgba(45, 111, 63, 0.02) 0%, transparent 40%);
        }

        .grain {
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: 0.4;
            mix-blend-mode: multiply;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.08'/%3E%3C/svg%3E");
            z-index: 1;
        }

        .container {
            max-width: 880px;
            margin: 0 auto;
            padding: 60px 32px 80px;
            position: relative;
            z-index: 2;
        }

        /* Masthead */
        .masthead {
            border-top: 3px double var(--ink);
            border-bottom: 1px solid var(--rule);
            padding: 24px 0 20px;
            margin-bottom: 40px;
            display: flex;
            justify-content: space-between;
            align-items: baseline;
        }
        .masthead-left {
            font-family: var(--serif);
        }
        .nameplate {
            font-size: 38px;
            font-weight: 700;
            line-height: 1;
            letter-spacing: -0.02em;
            font-style: italic;
        }
        .nameplate em {
            font-style: normal;
            color: var(--accent);
        }
        .tagline {
            font-family: var(--mono);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            color: var(--ink-mute);
            margin-top: 8px;
        }
        .masthead-right {
            font-family: var(--mono);
            font-size: 11px;
            color: var(--ink-mute);
            text-align: right;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        .masthead-right .vol { font-weight: 700; color: var(--ink); }

        /* Lede */
        .lede {
            border-bottom: 1px solid var(--rule);
            padding-bottom: 32px;
            margin-bottom: 48px;
        }
        .deck {
            font-size: 22px;
            line-height: 1.5;
            color: var(--ink-soft);
            font-style: italic;
            max-width: 620px;
        }
        .deck strong {
            color: var(--ink);
            font-style: normal;
        }
        .byline {
            font-family: var(--mono);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--ink-mute);
            margin-top: 16px;
        }

        /* Section headers */
        .section-num {
            font-family: var(--mono);
            font-size: 11px;
            font-weight: 700;
            color: var(--accent);
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .section-head {
            font-family: var(--serif);
            font-size: 28px;
            font-weight: 600;
            line-height: 1.2;
            margin-bottom: 16px;
            letter-spacing: -0.01em;
        }

        /* Cards */
        .panel {
            background: rgba(255, 255, 255, 0.5);
            border: 1px solid var(--rule);
            padding: 28px 32px;
            margin-bottom: 32px;
            position: relative;
        }
        .panel::before {
            content: '';
            position: absolute;
            top: -1px;
            left: -1px;
            right: -1px;
            height: 3px;
            background: var(--ink);
        }

        /* Form */
        input[type=file] {
            font-family: var(--mono);
            font-size: 13px;
            padding: 10px 0;
            color: var(--ink-soft);
            display: block;
            margin: 12px 0 20px;
            width: 100%;
        }
        input[type=file]::file-selector-button {
            font-family: var(--mono);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            padding: 8px 16px;
            background: var(--ink);
            color: var(--paper);
            border: none;
            cursor: pointer;
            margin-right: 14px;
            transition: background 0.15s;
        }
        input[type=file]::file-selector-button:hover {
            background: var(--accent);
        }
        textarea {
            width: 100%;
            min-height: 90px;
            padding: 14px 16px;
            font-family: var(--serif);
            font-size: 17px;
            line-height: 1.5;
            color: var(--ink);
            background: var(--paper);
            border: 1px solid var(--rule);
            border-radius: 0;
            resize: vertical;
            transition: border-color 0.15s;
        }
        textarea:focus {
            outline: none;
            border-color: var(--ink);
        }
        textarea::placeholder {
            color: var(--ink-mute);
            font-style: italic;
        }
        button {
            font-family: var(--mono);
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            padding: 14px 28px;
            background: var(--ink);
            color: var(--paper);
            border: none;
            cursor: pointer;
            margin-top: 16px;
            transition: background 0.15s, transform 0.05s;
        }
        button:hover:not(:disabled) {
            background: var(--accent);
        }
        button:active:not(:disabled) {
            transform: translateY(1px);
        }
        button:disabled {
            background: var(--ink-mute);
            cursor: not-allowed;
        }

        /* Status */
        .status {
            font-family: var(--mono);
            font-size: 12px;
            margin-top: 14px;
            min-height: 18px;
        }
        .status.ok { color: var(--grounded); }
        .status.err { color: var(--accent); }
        .status.loading { color: var(--ink-mute); font-style: italic; }

        /* Answer */
        .answer-block {
            border-left: 3px solid var(--ink);
            padding: 20px 24px;
            margin-top: 24px;
            background: rgba(255, 255, 255, 0.7);
            position: relative;
        }
        .answer-meta {
            font-family: var(--mono);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--ink-mute);
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 14px;
            padding-bottom: 10px;
            border-bottom: 1px dashed var(--rule);
        }
        .badge {
            display: inline-block;
            padding: 3px 10px;
            font-weight: 700;
            letter-spacing: 0.1em;
            border: 1px solid;
        }
        .badge.grounded {
            color: var(--grounded);
            border-color: var(--grounded);
        }
        .badge.not-grounded {
            color: var(--accent);
            border-color: var(--accent);
        }
        .answer-text {
            font-size: 18px;
            line-height: 1.65;
            color: var(--ink);
            white-space: pre-wrap;
        }
        .citations-list {
            margin-top: 18px;
            padding-top: 14px;
            border-top: 1px solid var(--rule);
            font-family: var(--mono);
            font-size: 11px;
            color: var(--ink-mute);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .citations-list strong {
            color: var(--ink);
            margin-right: 8px;
        }

        /* Stats strip */
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0;
            border-top: 1px solid var(--rule);
            border-bottom: 1px solid var(--rule);
            margin: 48px 0;
            padding: 20px 0;
        }
        .stat {
            text-align: center;
            border-right: 1px dotted var(--rule);
            padding: 0 16px;
        }
        .stat:last-child { border-right: none; }
        .stat-num {
            font-family: var(--serif);
            font-size: 32px;
            font-weight: 700;
            color: var(--ink);
            line-height: 1;
            font-variant-numeric: tabular-nums;
        }
        .stat-label {
            font-family: var(--mono);
            font-size: 9px;
            color: var(--ink-mute);
            text-transform: uppercase;
            letter-spacing: 0.15em;
            margin-top: 8px;
        }

        /* Footer */
        .footer {
            margin-top: 80px;
            padding-top: 32px;
            border-top: 3px double var(--ink);
            font-family: var(--mono);
            font-size: 10px;
            color: var(--ink-mute);
            text-transform: uppercase;
            letter-spacing: 0.15em;
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
        }
        .footer a {
            color: var(--ink-soft);
            text-decoration: none;
            border-bottom: 1px solid var(--rule);
            transition: color 0.15s;
        }
        .footer a:hover {
            color: var(--accent);
            border-color: var(--accent);
        }

        /* Responsive */
        @media (max-width: 640px) {
            .container { padding: 32px 20px 60px; }
            .masthead {
                flex-direction: column;
                gap: 8px;
                align-items: flex-start;
            }
            .masthead-right { text-align: left; }
            .nameplate { font-size: 30px; }
            .deck { font-size: 18px; }
            .stats { grid-template-columns: repeat(2, 1fr); }
            .stat { border-right: none; border-bottom: 1px dotted var(--rule); padding: 12px; }
            .stat:nth-child(odd) { border-right: 1px dotted var(--rule); }
            .panel { padding: 20px; }
        }

        /* Animations */
        @keyframes fade-up {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .fade-in {
            animation: fade-up 0.4s ease-out;
        }
    </style>
</head>
<body>
    <div class="grain"></div>
    <div class="container">

        <header class="masthead">
            <div class="masthead-left">
                <div class="nameplate">The 10-K <em>Analyst</em></div>
                <div class="tagline">SEC Filings · Cited Answers · No Hallucinations</div>
            </div>
            <div class="masthead-right">
                <div>Vol. I · No. 1</div>
                <div class="vol">EST. 2026</div>
            </div>
        </header>

        <section class="lede">
            <p class="deck">
                Upload any annual report. Ask any question. <strong>Receive verifiable answers</strong>
                grounded in specific sections and pages — or an honest "Not found in the filing"
                when the data isn't there.
            </p>
            <div class="byline">— A retrieval-augmented agent, built for buy-side rigor.</div>
        </section>

        <div class="stats">
            <div class="stat">
                <div class="stat-num">92.3<span style="font-size:18px;">%</span></div>
                <div class="stat-label">Pass Rate</div>
            </div>
            <div class="stat">
                <div class="stat-num">100<span style="font-size:18px;">%</span></div>
                <div class="stat-label">Citation Precision</div>
            </div>
            <div class="stat">
                <div class="stat-num">100<span style="font-size:18px;">%</span></div>
                <div class="stat-label">Hallucination Guard</div>
            </div>
            <div class="stat">
                <div class="stat-num">0.69<span style="font-size:18px;">s</span></div>
                <div class="stat-label">Median Latency</div>
            </div>
        </div>

        <section class="panel">
            <div class="section-num">§ I</div>
            <h2 class="section-head">Submit the Filing</h2>
            <p style="color: var(--ink-soft); margin-bottom: 8px;">
                A 10-K filing in PDF format. Processing takes 30–60 seconds for the typical 100-page document.
            </p>
            <input type="file" id="pdf" accept=".pdf">
            <button onclick="uploadPDF()" id="uploadBtn">Process Filing</button>
            <div class="status" id="uploadStatus"></div>
        </section>

        <section class="panel">
            <div class="section-num">§ II</div>
            <h2 class="section-head">Pose the Question</h2>
            <p style="color: var(--ink-soft); margin-bottom: 16px;">
                Ask about revenue, risks, segments, R&amp;D, governance — anything disclosed in the document.
            </p>
            <textarea id="question" placeholder="e.g. What was iPhone revenue in fiscal 2024, and how did it compare to the prior year?"></textarea>
            <button onclick="askQuestion()" id="askBtn" disabled>Submit Question</button>
            <div id="answer"></div>
        </section>

        <footer class="footer">
            <div>© 2026 · Independent · Open Source</div>
            <div>
                <a href="https://github.com/Elnar5/tenk-analyst-agent" target="_blank">Source on GitHub</a>
            </div>
        </footer>

    </div>

<script>
async function uploadPDF() {
    const fileInput = document.getElementById('pdf');
    const status = document.getElementById('uploadStatus');
    const askBtn = document.getElementById('askBtn');
    const uploadBtn = document.getElementById('uploadBtn');

    if (!fileInput.files[0]) {
        status.className = 'status err';
        status.textContent = '✕  Please select a PDF file.';
        return;
    }

    uploadBtn.disabled = true;
    status.className = 'status loading';
    status.textContent = 'Parsing document, building index...';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const res = await fetch('/analyze' + window.location.search, { method: 'POST', body: formData });
        const data = await res.json();
        if (data.success) {
            status.className = 'status ok';
            status.textContent = `✓  Indexed: ${data.chunks.toLocaleString()} chunks across ${data.pages} pages.`;
            askBtn.disabled = false;
        } else {
            status.className = 'status err';
            status.textContent = `✕  ${data.error}`;
        }
    } catch (e) {
        status.className = 'status err';
        status.textContent = `✕  ${e.message}`;
    }
    uploadBtn.disabled = false;
}

async function askQuestion() {
    const q = document.getElementById('question').value.trim();
    const answerDiv = document.getElementById('answer');
    const askBtn = document.getElementById('askBtn');

    if (!q) return;

    askBtn.disabled = true;
    answerDiv.innerHTML = '<div class="status loading" style="margin-top:20px;">Reading the filing...</div>';

    try {
        const res = await fetch('/ask' + window.location.search, {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: 'question=' + encodeURIComponent(q),
        });
        const data = await res.json();
        const groundedClass = data.grounded ? 'grounded' : 'not-grounded';
        const groundedLabel = data.grounded ? 'Grounded' : 'Not Found';
        const cites = data.citations.length
            ? data.citations.map(c => c.replace(/[\\[\\]]/g, '')).join(' · ')
            : 'None';

        answerDiv.innerHTML = `
            <div class="answer-block fade-in">
                <div class="answer-meta">
                    <span class="badge ${groundedClass}">${groundedLabel}</span>
                    <span>Q. ${new Date().toLocaleTimeString('en-US', {hour:'2-digit',minute:'2-digit'})}</span>
                </div>
                <div class="answer-text">${escapeHtml(data.answer)}</div>
                <div class="citations-list"><strong>Sources:</strong> ${escapeHtml(cites)}</div>
            </div>
        `;
    } catch (e) {
        answerDiv.innerHTML = `<div class="status err" style="margin-top:20px;">✕  ${e.message}</div>`;
    }
    askBtn.disabled = false;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
</script>
</body>
</html>"""

@app.post("/analyze")
async def analyze_pdf(request: Request, file: UploadFile = File(...)):
    is_owner = request.query_params.get("source") == "owner"
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"success": False, "error": "Only PDF files supported"})
    if not is_owner:
        track_event("pdf_upload_started", filename=file.filename, size_kb=0)
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
        page_count = max(c.page_number for c in agent.searcher.chunks)

        os.unlink(tmp_path)

        if not is_owner:
            track_event(
                "pdf_upload_success",
                filename=file.filename,
                chunks=chunk_count,
                pages=page_count,
            )
        return {
            "success": True,
            "chunks": chunk_count,
            "pages": page_count,
        }
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/ask")
async def ask_question(request: Request, question: str = Form(...)):
    is_owner = request.query_params.get("source") == "owner"
    """Ask a question against the loaded agent."""
    if not is_owner:
        track_event("question_asked", question_length=len(question))
    if "default" not in _agent_cache:
        return JSONResponse({
            "answer": "Please upload a 10-K PDF first.",
            "citations": [],
            "grounded": False,
        })

    agent = _agent_cache["default"]
    result = agent.ask(question)
    if not is_owner:
        track_event(
            "question_answered",
            grounded=result.is_grounded,
            citations_count=len(result.citations),
            answer_length=len(result.answer),
        )
    return {
        "answer": result.answer,
        "citations": result.citations,
        "grounded": result.is_grounded,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)