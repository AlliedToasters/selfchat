#!/bin/bash

# Per-message embedding + probing pipeline (Phase 3).
# Analog of run_analysis.sh for the per-message embedding stack.
#
# Where run_analysis.sh produces ONE vector per run (terminal state) and
# runs run-level PCA / Mann-Whitney separability, this script produces
# ONE vector per turn-message and runs probes for turn-depth, agent
# A-vs-B, and agent-by-depth.

# Configuration
VENV=".venv/bin/python"
TRANSCRIPT_DIR="archive/transcripts_nonspecific/"
EMB_FILE="emb_msgs.npz"

# 1. Embed every turn-message of every natural-stop transcript.
#    Uses Ollama (nomic-embed-text); ~2.5 min for ~12k messages on a 5090.
#    Tolerates concurrent self-chat with ~50% throughput hit on the gemma
#    side during the embed window.
$VENV embed_messages.py "$TRANSCRIPT_DIR" --out "$EMB_FILE"

# 2. Turn-depth probes (pooled across seeds, completed-only by default)
$VENV probe.py "$EMB_FILE" --target bucket --scheme thirds
$VENV probe.py "$EMB_FILE" --target turn-vs-all --max-j 10

# 3. Agent A-vs-B probes
$VENV probe.py "$EMB_FILE" --target agent
$VENV probe.py "$EMB_FILE" --target agent-by-depth --scheme fives

# 4. Per-turn 50-way bucket — surfaces the decodability cliff
#    (slowest probe; ~1-2 min for ~6k samples × 50 classes via L-BFGS)
$VENV probe.py "$EMB_FILE" --target bucket --scheme per-turn

# 5. PCA scatter, colored by turn_index, faceted by variant
$VENV plot_msgs.py "$EMB_FILE" --out figures/pca_messages.png

# 6. Per-seed cross-checks for the Phase-2 headline conditions.
#    `freedom_neg_minimal` carries Phase 2's Bonferroni-significant PC1
#    separation; do per-message probes also surface seed-specific
#    structure? Pre-register predictions in priors.md before uncommenting.
# $VENV probe.py "$EMB_FILE" --target bucket --scheme thirds --seed freedom_neg_minimal
# $VENV probe.py "$EMB_FILE" --target agent-by-depth --scheme fives --seed freedom_neg_minimal
# $VENV plot_msgs.py "$EMB_FILE" --seed freedom_neg_minimal --out figures/pca_messages_freedom_neg_minimal.png
