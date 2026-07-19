#!/bin/bash
# Build the Bubby Rust core kernel and its Python bindings.
# Requires: cargo, rustc, python3.11+, maturin
set -e

cd "$(dirname "$0")"

echo "=== Building Bubby Core (Rust → Python) ==="

# Install maturin if not present
if ! command -v maturin &>/dev/null; then
    echo "Installing maturin..."
    pip3 install maturin
fi

# Build release wheel
echo "Building release wheel..."
maturin build --release

echo ""
echo "=== Build complete ==="
echo "Wheel in: $(pwd)/target/wheels/"
echo ""
echo "Install with: pip install target/wheels/bubby_core-*.whl"