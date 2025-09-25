# Development stage - lightweight, reuses local .venv
FROM python:3.12-slim as development

WORKDIR /app

# Install minimal runtime dependencies + PDM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir pdm

# Create app user matching host user
RUN adduser --disabled-password --gecos '' --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set environment for development
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PDM_VENV_IN_PROJECT=1

EXPOSE 8000 7860

# Copy and use start script
COPY --chown=appuser:appuser start.sh ./
RUN chmod +x start.sh

CMD ["./start.sh"]

# Production stage - self-contained with pre-built dependencies  
FROM python:3.12-slim as production

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libssl-dev \
    && pip install --no-cache-dir pdm \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and install
COPY pyproject.toml pdm.lock* ./
RUN pdm config python.use_venv true && \
    pdm install --without dev --without test --no-editable

# Clean up build dependencies and install runtime ones
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 libgomp1 \
    && apt-get remove -y gcc g++ libffi-dev libssl-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Create app user
RUN adduser --disabled-password --gecos '' --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set production environment
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000 7860

CMD ["python", "app.py"]