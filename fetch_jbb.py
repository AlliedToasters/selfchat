"""Fetch JailbreakBench JBB-Behaviors harmful-behaviors.csv from Hugging Face.

The dataset is public/ungated (no login required). 100 harmful behaviors
across 10 categories from OpenAI's usage policies; sources include Original,
TDC/HarmBench, and AdvBench.

Run: `python fetch_jbb.py`
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/"
    "resolve/main/data/harmful-behaviors.csv"
)
OUT = Path(__file__).parent / "data" / "jbb.csv"


def main() -> int:
    if OUT.exists():
        print(f"already present: {OUT}")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {URL}")
    urllib.request.urlretrieve(URL, OUT)
    n_lines = sum(1 for _ in OUT.open())
    print(f"wrote {OUT} ({n_lines} lines; expect 101 with header)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
