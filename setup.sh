#!/usr/bin/env bash
set -euo pipefail

echo "=== copypaste setup ==="

# ── Check Bun ────────────────────────────────────────────
if ! command -v bun &>/dev/null; then
  echo "✗ Bun not found. Install: curl -fsSL https://bun.sh/install | bash"
  exit 1
fi
echo "✓ Bun $(bun --version)"

# ── Check Python 3 ───────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "✗ python3 not found. Install Python 3.9+ first."
  exit 1
fi
echo "✓ $(python3 --version)"

# ── Create venv if missing ───────────────────────────────
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "→ Creating Python venv in $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"
echo "✓ venv activated"

# ── Install Python deps ─────────────────────────────────
echo "→ Installing Python dependencies..."

# Core ML (required for training + inference)
pip install -q \
  scikit-learn \
  numpy

# Transformer fine-tuning (optional but recommended)
pip install -q \
  torch \
  transformers \
  datasets

# ONNX export (optional)
pip install -q \
  skl2onnx \
  optimum \
  onnxruntime \
  2>/dev/null || echo "⚠ ONNX deps skipped (optional — install manually if needed)"

echo "✓ Python dependencies installed"

# ── Create dirs ──────────────────────────────────────────
mkdir -p data models reports logs tasks

# ── Summary ──────────────────────────────────────────────
echo ""
echo "=== Ready ==="
echo "  bun run start     — launch the TUI"
echo "  bun test          — run tests"
echo ""
echo "Note: activate the venv before running:"
echo "  source $VENV_DIR/bin/activate"
