FROM python:3.11-slim
WORKDIR /app

# Install system dependencies (build tools + runtime libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install PDM globally
RUN pip install --no-cache-dir pdm

# Copy only dependency files first (for caching)
COPY pyproject.toml pdm.lock* ./

# Configure PDM to use project venv inside container
RUN pdm config python.use_venv true

# Install dependencies (this will pick torch from torch-cpu source)
RUN pdm install --without dev --without test

# Copy app code
COPY . .

RUN chmod -R 777 /app

# Expose ports
ARG APP_PORT=8000
ARG WS_PORT=7860
EXPOSE $APP_PORT $WS_PORT

# Default command
CMD ["./start.sh"]
