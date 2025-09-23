#!/bin/sh
set -e

# Function to check if dependencies are installed
check_dependencies() {
    if [ -f "/app/.venv/bin/python" ]; then
        # Check if key packages are importable
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

# Check if we're in development mode (volumes mounted)
if [ -f "/app/pyproject.toml" ] && [ -w "/app" ]; then
    echo "Development mode detected"

    # Check if local .venv exists and is valid
    if check_dependencies; then
        echo "Using existing virtual environment"
    else
        echo "Installing/updating dependencies..."
        # Install PDM if not available
        if ! command -v pdm >/dev/null 2>&1; then
            pip install --no-cache-dir pdm
        fi
        pdm install --without dev --without test
    fi
else
    echo "Production mode - using pre-built environment"
    if ! check_dependencies; then
        echo "ERROR: Dependencies not found in production image!"
        exit 1
    fi
fi

# Set memory-efficient Python options
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=random

# Use exec to replace shell process (saves memory)
echo "Starting application..."
exec python app.py