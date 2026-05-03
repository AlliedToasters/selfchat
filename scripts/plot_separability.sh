#!/bin/bash

# Phase 5 separability bar+line chart across the 7-seed × 3-tier grid.
# Run from project root: scripts/plot_separability.sh [--out figures/sep.png]
#
# Forwards CLI args to the Python plotting module.

VENV=".venv/bin/python"
exec "$VENV" -m selfchat.viz.plot_separability "$@"
