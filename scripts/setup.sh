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

_err() { echo "Error: $*" >&2; }
_warn() { echo "Warning: $*" >&2; }
_info() { echo "$*"; }

_have() { command -v "$1" >/dev/null 2>&1; }

_can_write_dir() {
  local d="$1"
  mkdir -p "$d" >/dev/null 2>&1 || return 1
  local f="$d/.swarm_write_test.$$"
  ( : >"$f" ) >/dev/null 2>&1 || return 1
  rm -f "$f" >/dev/null 2>&1 || true
  return 0
}

_fs_avail_mb_for_path() {
  # Print available MB for the filesystem containing path (or 0 on failure)
  local p="$1"
  df -Pm "$p" 2>/dev/null | awk 'NR==2 {print $4}' 2>/dev/null || echo "0"
}

# --- Storage strategy (auto) ---
# RunPod-like hosts often have:
# - tiny / overlay (fills quickly)
# - /workspace with quotas (looks huge but may reject writes)
# - /dev/shm with lots of space (tmpfs, not persistent) -> best default for models & tmp
DEFAULT_MODELS_DIR=""
if _can_write_dir "/dev/shm" && [[ "$(_fs_avail_mb_for_path /dev/shm)" -ge 20000 ]]; then
  DEFAULT_MODELS_DIR="/dev/shm/ollama-models"
elif _can_write_dir "/workspace"; then
  DEFAULT_MODELS_DIR="/workspace/ollama-models"
else
  DEFAULT_MODELS_DIR="$REPO_ROOT/.ollama-models"
fi

# If set, the Ollama server will use this directory for model storage.
# Example:
#   OLLAMA_MODELS=/dev/shm/ollama-models ./scripts/setup.sh
export OLLAMA_MODELS="${OLLAMA_MODELS:-$DEFAULT_MODELS_DIR}"

# Venv location: avoid creating it on a full / overlay.
DEFAULT_VENV_DIR="$REPO_ROOT/.venv"
if [[ "$(_fs_avail_mb_for_path "$REPO_ROOT")" -lt 2048 ]]; then
  # Repo filesystem is tight (often / overlay). Prefer /workspace for venv (smaller), else /dev/shm.
  if _can_write_dir "/workspace"; then
    DEFAULT_VENV_DIR="/workspace/swarm-venv"
  elif _can_write_dir "/dev/shm"; then
    DEFAULT_VENV_DIR="/dev/shm/swarm-venv"
  fi
fi

# Allow placing the virtualenv outside the repo (useful on constrained root filesystems).
# Examples:
#   VENV_DIR=/dev/shm/swarm-venv ./scripts/setup.sh
#   VENV_DIR=/workspace/swarm-venv ./scripts/setup.sh
VENV_DIR="${VENV_DIR:-$DEFAULT_VENV_DIR}"

# Temp + caches: ensurepip/pip stage wheels in /tmp by default; on many containers /tmp is on / (full).
DEFAULT_TMPDIR=""
if _can_write_dir "/dev/shm" && [[ "$(_fs_avail_mb_for_path /dev/shm)" -ge 2048 ]]; then
  DEFAULT_TMPDIR="/dev/shm/swarm-tmp"
elif _can_write_dir "/workspace"; then
  DEFAULT_TMPDIR="/workspace/swarm-tmp"
else
  DEFAULT_TMPDIR="$REPO_ROOT/.tmp"
fi
mkdir -p "$DEFAULT_TMPDIR"
export TMPDIR="${TMPDIR:-$DEFAULT_TMPDIR}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$TMPDIR/xdg-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$TMPDIR/pip-cache}"

# Clean up a common accidental path that can fill / quickly.
if [[ -d "/MOUNTPOINT" ]]; then
  _warn "Found /MOUNTPOINT (likely created by placeholder). Removing to free space."
  rm -rf /MOUNTPOINT >/dev/null 2>&1 || true
fi

_info "Using:"
_info "  REPO_ROOT=$REPO_ROOT"
_info "  OLLAMA_MODELS=$OLLAMA_MODELS"
_info "  VENV_DIR=$VENV_DIR"
_info "  TMPDIR=$TMPDIR"

# --- 1. Ollama ---
if ! command -v ollama &>/dev/null; then
  echo "[1/4] Installing Ollama..."
  if ! _have zstd; then
    # Ollama install script uses zstd to extract tar.zst.
    if _have apt-get; then
      _info "Installing zstd (required by Ollama installer)..."
      apt-get update -y >/dev/null 2>&1 || true
      apt-get install -y zstd >/dev/null 2>&1 || _warn "Failed to install zstd via apt-get (check APT sources)."
    elif _have dnf; then
      dnf install -y zstd >/dev/null 2>&1 || _warn "Failed to install zstd via dnf."
    elif _have yum; then
      yum install -y zstd >/dev/null 2>&1 || _warn "Failed to install zstd via yum."
    elif _have pacman; then
      pacman -Sy --noconfirm zstd >/dev/null 2>&1 || _warn "Failed to install zstd via pacman."
    else
      _warn "zstd not found and no known package manager detected. Ollama install may fail."
    fi
  fi
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "[1/4] Ollama already installed ($(ollama --version 2>/dev/null || echo 'ollama'))"
fi

# --- 2. Ensure Ollama is running ---
echo "[2/4] Ensuring Ollama service..."
# Always restart to ensure the server picks up OLLAMA_MODELS.
pkill -f "ollama serve" >/dev/null 2>&1 || true
mkdir -p "$OLLAMA_MODELS"
echo "Using OLLAMA_MODELS=$OLLAMA_MODELS"
nohup env OLLAMA_MODELS="$OLLAMA_MODELS" ollama serve >"$TMPDIR/ollama-serve.log" 2>&1 &

sleep 2
for i in {1..30}; do
  if curl -sf http://localhost:11434/api/tags &>/dev/null; then break; fi
  sleep 1
done
if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
  echo "Error: Ollama not responding at http://localhost:11434 (see $TMPDIR/ollama-serve.log)"
  exit 1
fi

# --- 3. Required models (default + decomposer/planner/judge) ---
MODELS=(
  qwen2.5-coder:7b
  llama3.1:8b
  phi4:14b
  gemma2:2b
)
echo "[3/4] Pulling models: ${MODELS[*]}"

# Clean partial blobs (common after quota/disk issues).
rm -f "$OLLAMA_MODELS"/blobs/*partial >/dev/null 2>&1 || true

for m in "${MODELS[@]}"; do
  echo "  Pulling $m..."
  # Ensure pulls use the same model dir as the server.
  env OLLAMA_MODELS="$OLLAMA_MODELS" ollama pull "$m" || { echo "  Failed to pull $m"; exit 1; }
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

mkdir -p "$(dirname "$VENV_DIR")"
if command -v uv &>/dev/null; then
  echo "Using uv..."
  export UV_CACHE_DIR="${UV_CACHE_DIR:-${VENV_DIR}.uv-cache}"
  uv venv "$VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  uv pip install -e .
else
  echo "Using pip (install uv for faster setup: curl -LsSf https://astral.sh/uv/install.sh | sh)"
  # If ensurepip fails due to /tmp being on a full filesystem, TMPDIR above should prevent it.
  python3 -m venv "$VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  pip install --no-cache-dir -e .
fi

echo ""
echo "Setup complete. Activate the environment and run:"
echo "  source \"$VENV_DIR/bin/activate\""
echo "  swarm run --routing lp --gpu_vram_gb 24 --query 'Hello, world!'"
echo ""
