# ════════════════════════════════════════════════════════════════
# Dockerfile — Market Movement Prediction API
# Financial Market Movement Prediction Project
# Role   : API Layer (Areesha)
# ════════════════════════════════════════════════════════════════

FROM python:3.10-slim

# ── System deps ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory = project root ─────────────────────────────
#    Keeps parents[1] resolving to /app correctly
WORKDIR /app

# ── Install PyTorch CPU-only FIRST (separate index URL) ──────────
#    CPU wheel = ~200 MB vs 530 MB GPU + 366 MB nvidia_cudnn
#    This is the step that caused your timeout
RUN pip install --no-cache-dir \
    torch==2.4.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Install remaining API dependencies ────────────────────────────
#    Uses api_requirements.txt — NOT the full project requirements.txt
COPY docker/api_requirements.txt .
RUN pip install --no-cache-dir -r api_requirements.txt

# ── Copy only what the API needs at runtime ───────────────────────
COPY api/                  ./api/
COPY src/models/           ./src/models/
COPY src/timeseries/       ./src/timeseries/


# ── Expose port ───────────────────────────────────────────────────
EXPOSE 8000

# ── Python env vars ───────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE

# ── Run from /app so parents[1] = /app = project root ────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]