#!/usr/bin/env bash
set -e
echo "=== Banking Intent — Inference ==="
MESSAGE=${1:-"I am still waiting on my card?"}
python scripts/inference.py --config configs/inference.yaml --message "$MESSAGE"
