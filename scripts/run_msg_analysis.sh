#!/bin/bash

# Per-message embedding + probing pipeline (Phase 3).
# Run from project root: scripts/run_msg_analysis.sh
#
# Where run_analysis.sh produces ONE vector per run (terminal state) and
# runs run-level PCA / Mann-Whitney separability, this script produces
# ONE vector per turn-message and runs probes for turn-depth, agent
# A-vs-B, and agent-by-depth.

# Configuration
VENV=".venv/bin/python"
TRANSCRIPT_DIR="archive/transcripts_nonspecific/"
EMB_FILE="artifacts/emb_msgs.npz"

# 1. Embed every turn-message of every natural-stop transcript.
$VENV -m selfchat.embeddings.embed_messages "$TRANSCRIPT_DIR" --out "$EMB_FILE"

# 2. Turn-depth probes (pooled across seeds, completed-only by default)
$VENV -m selfchat.stats.probe "$EMB_FILE" --target bucket --scheme thirds
$VENV -m selfchat.stats.probe "$EMB_FILE" --target turn-vs-all --max-j 10

# 3. Agent A-vs-B probes
$VENV -m selfchat.stats.probe "$EMB_FILE" --target agent
$VENV -m selfchat.stats.probe "$EMB_FILE" --target agent-by-depth --scheme fives

# 4. Per-turn 50-way bucket — surfaces the decodability cliff
$VENV -m selfchat.stats.probe "$EMB_FILE" --target bucket --scheme per-turn

# 5. PCA scatter, colored by turn_index, faceted by variant
$VENV -m selfchat.viz.plot_msgs "$EMB_FILE" --out figures/pca_messages.png

# 6. Per-seed cross-checks for the Phase-2 headline conditions.
#    `freedom_neg_minimal` carries Phase 2's Bonferroni-significant PC1
#    separation; do per-message probes also surface seed-specific
#    structure? Pre-register predictions in priors.md before uncommenting.
# $VENV -m selfchat.stats.probe "$EMB_FILE" --target bucket --scheme thirds --seed freedom_neg_minimal
# $VENV -m selfchat.stats.probe "$EMB_FILE" --target agent-by-depth --scheme fives --seed freedom_neg_minimal
# $VENV -m selfchat.viz.plot_msgs "$EMB_FILE" --seed freedom_neg_minimal --out figures/pca_messages_freedom_neg_minimal.png
