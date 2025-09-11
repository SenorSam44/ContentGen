#!/bin/sh

# Check if .venv exists and has the expected structure
if [ -d "/app/.venv/bin" ] && [ -f "/app/.venv/bin/python" ]; then
    echo "Virtual environment found, checking dependencies..."
    
    # Check if dependencies are already installed by trying to import a key package
    if /app/.venv/bin/python -c "import $(head -1 pdm.lock | cut -d' ' -f1 | tr -d '"')" 2>/dev/null; then
        echo "Dependencies are already installed, skipping pdm install."
    else
        echo "Dependencies missing, running pdm install..."
        pdm install
    fi
else
    echo "Virtual environment not found or incomplete, creating new one..."
    pdm install
fi

# Run the application
exec pdm run python app.py
