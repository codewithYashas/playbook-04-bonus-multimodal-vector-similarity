#!/bin/bash
# ByteEats Vector Similarity Search Installation Script
# Installs the locked dependencies (uv.lock) with uv into .venv

set -e

echo "ByteEats Vector Similarity Search Installation"
echo "=============================================="
echo

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed."
    echo "Install it from https://docs.astral.sh/uv/getting-started/installation/"
    echo "  macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  Homebrew:    brew install uv"
    exit 1
fi

# uv downloads a Python 3.12+ interpreter if the system doesn't have one (see requires-python)
echo "Installing dependencies from uv.lock..."
uv sync --frozen

echo "Setting up directory structure..."
uv run --frozen python scripts/setup_directories.py

# Make the run_pipeline.sh executable
chmod +x run_pipeline.sh

echo "Installation complete!"
echo
echo "To run the API server:"
echo "  ./run_pipeline.sh --start-api"
echo
echo "To run the dashboard:"
echo "  ./run_pipeline.sh --start-dashboard"
echo
echo "To run any other script, prefix it with 'uv run', for example:"
echo "  uv run python analyze_embeddings.py"
echo
