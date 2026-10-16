# SpecX — single container: FastAPI API + static UI + ONNX models baked in.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    DATA_DIR=/app/data \
    PORT=7860

# tesseract is only used by the optional OCR bonus (ENABLE_OCR=true)
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

# Run as a non-root user (required by Hugging Face Spaces, good practice elsewhere)
RUN useradd -m -u 1000 app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app src ./src
COPY --chown=app:app web ./web
COPY --chown=app:app pyproject.toml README.md ./
RUN mkdir -p /app/data && chown -R app:app /app/data

USER app
# Download the embedding + reranker models at build time so cold starts are fast.
RUN python -c "from specx.index.models import warmup; warmup()"

EXPOSE 7860
CMD ["sh", "-c", "uvicorn specx.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
