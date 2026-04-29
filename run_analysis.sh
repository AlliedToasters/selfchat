#!/bin/bash

# Configuration
VENV=".venv/bin/python"
TRANSCRIPT_DIR="transcripts"  # ← Change this once, use everywhere
EMB_FILE="emb.npz"

# 1. Per-cell stats
$VENV analyze.py "$TRANSCRIPT_DIR"

# 2. Marker frequencies
$VENV markers.py "$TRANSCRIPT_DIR"

# 3. Embed terminal states
$VENV embed.py "$TRANSCRIPT_DIR" --out "$EMB_FILE"

# 4. Render PCA plots
$VENV plot.py "$EMB_FILE"
$VENV plot.py "$EMB_FILE" --seeds advbench --out figures/pca_advbench_only.png
$VENV plot.py "$EMB_FILE" --seeds alpaca --out figures/pca_alpaca_only.png

# 5. Separability Analysis
$VENV separability.py "$EMB_FILE"

# 6. Interactive browser
$VENV -m streamlit run browse.py \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.port 8501 \
    -- "$EMB_FILE" "$TRANSCRIPT_DIR"