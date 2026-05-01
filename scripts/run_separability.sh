#!/bin/bash

# Phase 5 separability suite — single-command runner.
# Run from project root: scripts/run_separability.sh [--seed SEED] [other flags]
#
# Defaults to seed=freedom (the no-harm-pressure benchmark).
# Forwards all CLI args to the Python suite module.

VENV=".venv/bin/python"
exec "$VENV" -m selfchat.classify.suite "$@"
