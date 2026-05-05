FROM python:3.12-slim

WORKDIR /code

# System dependencies for pdfplumber and chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /code/src/
COPY data/sample_filings/ /code/data/sample_filings/

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Pre-download the embedding model at build time (avoids slow first request)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Start the FastAPI server on port 7860 (HuggingFace standard)
CMD ["uvicorn", "src.web.server:app", "--host", "0.0.0.0", "--port", "7860"]