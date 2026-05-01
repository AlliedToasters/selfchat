#!/bin/bash

# Run from project root: scripts/run_analysis.sh

# Configuration
VENV=".venv/bin/python"
# TRANSCRIPT_DIR="archive/transcripts_mixed/"
# TRANSCRIPT_DIR="archive/transcripts_nonspecific/"
TRANSCRIPT_DIR="transcripts/"
EMB_FILE="artifacts/emb.npz"

# 1. Per-cell stats
$VENV -m selfchat.analysis.analyze "$TRANSCRIPT_DIR"

# 2. Marker frequencies
$VENV -m selfchat.analysis.markers "$TRANSCRIPT_DIR"

# 3. Embed terminal states
$VENV -m selfchat.embeddings.embed "$TRANSCRIPT_DIR" --out "$EMB_FILE"

# 4. Render PCA plots
$VENV -m selfchat.viz.plot "$EMB_FILE"
$VENV -m selfchat.viz.plot "$EMB_FILE" --seeds advbench --out figures/pca_advbench_only.png
$VENV -m selfchat.viz.plot "$EMB_FILE" --seeds alpaca --out figures/pca_alpaca_only.png

# 5. Separability Analysis
$VENV -m selfchat.stats.separability "$EMB_FILE"

# 6. Interactive browser
$VENV -m streamlit run selfchat/viz/browse.py \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.port 8501 \
    -- "$EMB_FILE" "$TRANSCRIPT_DIR"
