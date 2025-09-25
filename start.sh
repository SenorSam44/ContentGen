#!/bin/sh
set -e

# ---------------------------
# Function to check dependencies
# ---------------------------
check_dependencies() {
    if [ -f "/app/.venv/bin/python" ]; then
        /app/.venv/bin/python -c "
import sys
try:
    import gradio, flask, transformers, torch
    print('Dependencies verified')
    sys.exit(0)
except ImportError as e:
    print(f'Missing dependency: {e}')
    sys.exit(1)
" 2>/dev/null
        return $?
    else
        return 1
    fi
}

# ---------------------------
# Determine environment
# ---------------------------
DEV_MODE=false
if [ -f "/app/pyproject.toml" ] && [ -w "/app" ]; then
    DEV_MODE=true
    echo "Development mode detected"
fi

# ---------------------------
# Install PDM only if needed
# ---------------------------
if ! check_dependencies; then
    echo "Dependencies missing, installing via PDM..."
    if ! command -v pdm >/dev/null 2>&1; then
        pip install --no-cache-dir pdm
    fi
    pdm install --without dev --without test
else
    echo "Dependencies verified in existing .venv"
fi

# ---------------------------
# Set Python environment
# ---------------------------
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=random

# ---------------------------
# Start Flask and Gradio
# ---------------------------
APP_PORT=${APP_PORT:-8000}
WS_PORT=${WS_PORT:-7860}

echo "Starting Flask API on port $APP_PORT..."
pdm run python app.py &

if [ "$DEV_MODE" = true ]; then
    echo "Starting Gradio UI on port $WS_PORT..."
    pdm run python gradio_ui.py &
fi

# Wait for all background processes
wait
