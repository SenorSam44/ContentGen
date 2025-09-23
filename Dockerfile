# Multi-stage build for minimal final image size
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies (will be removed in final stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PDM
RUN pip install --no-cache-dir pdm

# Copy dependency files first (for Docker layer caching)
COPY pyproject.toml pdm.lock* ./

# Install dependencies
RUN pdm config python.use_venv true && \
    pdm install --without dev --without test --no-editable

# Production stage - minimal runtime image
FROM python:3.12-slim as production

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY . .

# Create non-root user for security
RUN adduser --disabled-password --gecos '' --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose ports
EXPOSE 8000 7860

# Run the application
CMD ["python", "app.py"]