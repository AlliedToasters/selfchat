"""Fetch AdvBench harmful_behaviors.csv from the public llm-attacks GitHub repo.

The HF mirror (walledai/AdvBench) gates access behind a login. The original
upstream CSV is publicly available with no auth. We use that.

Run: `python fetch_advbench.py`
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)
OUT = Path(__file__).parent / "data" / "advbench.csv"


def main() -> int:
    if OUT.exists():
        print(f"already present: {OUT}")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {URL}")
    urllib.request.urlretrieve(URL, OUT)
    n_lines = sum(1 for _ in OUT.open())
    print(f"wrote {OUT} ({n_lines} lines; expect 521 with header)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
