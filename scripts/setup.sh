#!/usr/bin/env bash
# One-command setup: Ollama + models + Python package for ollama-swarm-router
# Run from repo root, or use the one-liner below to clone + setup.
#
# One-liner (clone + setup):
#   bash -c 'git clone https://github.com/arjunchandran1999/CvxBasedAgentOrchestration.git && cd CvxBasedAgentOrchestration && ./scripts/setup.sh'
#
# Or after cloning:
#   cd CvxBasedAgentOrchestration && ./scripts/setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f pyproject.toml ]] || [[ ! -d src/swarm ]]; then
  echo "Error: Must run from repo root (or clone first). Expected pyproject.toml and src/swarm/"
  exit 1
fi

# --- 1. Ollama ---
if ! command -v ollama &>/dev/null; then
  echo "[1/4] Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "[1/4] Ollama already installed ($(ollama --version 2>/dev/null || echo 'ollama'))"
fi

# --- 2. Ensure Ollama is running ---
echo "[2/4] Ensuring Ollama service..."
if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
  echo "Starting Ollama (first 'ollama pull' will start the server if needed)..."
  nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
  sleep 3
  for i in {1..20}; do
    if curl -sf http://localhost:11434/api/tags &>/dev/null; then break; fi
    sleep 1
  done
  if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo "Warning: Ollama not responding. Run 'ollama serve' in another terminal, then re-run this script."
    exit 1
  fi
fi

# --- 3. Required models (default + decomposer/planner/judge) ---
MODELS=(
  qwen2.5-coder:7b
  llama3.1:8b
  phi4:14b
  gemma2:2b
)
echo "[3/4] Pulling models: ${MODELS[*]}"
for m in "${MODELS[@]}"; do
  echo "  Pulling $m..."
  ollama pull "$m" || { echo "  Failed to pull $m"; exit 1; }
done

# --- 4. Python environment ---
echo "[4/4] Setting up Python package..."
REQUIRED_PY="3.10"
if ! command -v python3 &>/dev/null; then
  echo "Error: python3 not found. Install Python ${REQUIRED_PY}+ first."
  exit 1
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0")
if [[ "$(printf '%s\n' "$REQUIRED_PY" "$PY_VER" | sort -V | head -1)" != "$REQUIRED_PY" ]]; then
  echo "Warning: Python $PY_VER detected; $REQUIRED_PY+ recommended. Continuing anyway."
fi

if command -v uv &>/dev/null; then
  echo "Using uv..."
  uv venv
  source .venv/bin/activate
  uv pip install -e .
else
  echo "Using pip (install uv for faster setup: curl -LsSf https://astral.sh/uv/install.sh | sh)"
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e .
fi

echo ""
echo "Setup complete. Activate the environment and run:"
echo "  source .venv/bin/activate"
echo "  swarm run --routing lp --gpu_vram_gb 24 --query 'Hello, world!'"
echo ""
