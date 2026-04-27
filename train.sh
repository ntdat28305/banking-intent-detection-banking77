#!/usr/bin/env bash
set -e
echo "=== Banking Intent — Training ==="
pip install -r requirements.txt
python scripts/preprocess_data.py --config configs/train.yaml
python scripts/train.py           --config configs/train.yaml
echo "=== Done! Checkpoint saved to outputs/checkpoint/ ==="
