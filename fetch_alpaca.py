"""Fetch Stanford Alpaca instructions and save the standalone subset as CSV.

Stanford Alpaca (Taori et al. 2023): 52K instruction-following demonstrations
generated from text-davinci-003. License: CC BY-NC 4.0 (research only — fine
for this project).

We use the `instruction` field as the seed prompt and filter to records
where `input` is empty (those are standalone prompts; the with-input ones
need extra context to make sense as a self-chat seed). The standalone
subset is ~half of the 52K records, vastly more than we'll consume.

The output CSV is **shuffled with a fixed seed before write**. Alpaca is
much larger than the other pools (~26K vs 520/100), and contiguous slices
from `run_idx=0` would over-sample the generation-order head where adjacent
records can share batch-level topic correlations. Freezing the shuffled
order into the CSV itself (rather than reshuffling at load time) means
the on-disk file is the canonical, reproducible artifact — no runtime
randomness, no Python-version drift in `random.shuffle`'s implementation.

Source is the original `alpaca_data.json` from the GitHub repo, which is
the same data the `tatsu-lab/alpaca` HF dataset mirrors.

Run: `python fetch_alpaca.py`
"""

from __future__ import annotations

import csv
import json
import random
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
    "main/alpaca_data.json"
)
OUT = Path(__file__).parent / "data" / "alpaca.csv"

# Fixed shuffle seed. Don't change after the first alpaca run is recorded —
# changing it reassigns prompts to run_idx values and breaks paired-across-
SHUFFLE_SEED = 0


def main() -> int:
    if OUT.exists():
        print(f"already present: {OUT}")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {URL}")
    with urllib.request.urlopen(URL) as resp:
        records = json.load(resp)

    standalone = [r for r in records if not (r.get("input") or "").strip()]
    print(f"loaded {len(records)} total; {len(standalone)} standalone (no `input` context)")

    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(standalone)
    print(f"shuffled with seed={SHUFFLE_SEED}")

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction"])
        writer.writeheader()
        for r in standalone:
            writer.writerow({"instruction": r["instruction"]})

    print(f"wrote {OUT} ({len(standalone) + 1} lines incl. header)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
