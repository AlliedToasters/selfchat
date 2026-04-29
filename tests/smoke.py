"""

Two parts:
  (1) Pure unit checks — _is_degenerate logic, plus end-to-end with a stub client
      that fakes the OpenAI interface (no model, no network).
  (2) End-to-end with a real Ollama model (only if --hit-model is passed),
      to verify the live path. Defaults to OFF so this stays cheap and
      doesn't expose Claude to model outputs.

Run from project root: `.venv/bin/python tests/smoke.py [--hit-model]`
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seeds import SEEDS  # noqa: E402
from self_chat import (  # noqa: E402
    DEFAULT_DEGENERATE_WINDOW,
    SAMPLING_PARAMS,
    SYSTEM_PROMPT,
    _is_degenerate,
    run_self_chat,
)

# ---------- stub OpenAI client ----------


class _StubMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubChoice:
    def __init__(self, content: str) -> None:
        self.message = _StubMessage(content)
        self.finish_reason = "stop"


class _StubUsage:
    def __init__(self, n: int) -> None:
        self.completion_tokens = n
        self.total_tokens = n


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_StubChoice(content)]
        self.usage = _StubUsage(len(content.split()))


class _StubCompletions:
    def __init__(self, scripted: list[str], raise_at: int | None = None) -> None:
        self._scripted = list(scripted)
        self._i = 0
        self._raise_at = raise_at

    def create(self, **_: Any) -> _StubResponse:
        if self._raise_at is not None and self._i == self._raise_at:
            raise KeyboardInterrupt
        if self._i >= len(self._scripted):
            raise RuntimeError("StubClient: scripted responses exhausted")
        c = self._scripted[self._i]
        self._i += 1
        return _StubResponse(c)


class _StubChat:
    def __init__(self, scripted: list[str], raise_at: int | None = None) -> None:
        self.completions = _StubCompletions(scripted, raise_at=raise_at)


class StubClient:
    """Mimics the slice of openai.OpenAI used by run_self_chat."""

    def __init__(self, scripted: list[str], raise_at: int | None = None) -> None:
        self.chat = _StubChat(scripted, raise_at=raise_at)


# ---------- tests ----------


def test_is_degenerate_unit() -> None:
    assert _is_degenerate([], 6) is False
    assert _is_degenerate(["a"], 6) is False
    assert _is_degenerate(["a"] * 5, 6) is False
    assert _is_degenerate(["a"] * 6, 6) is True
    assert _is_degenerate(["a", "b", "a", "a", "a", "a", "a", "a"], 6) is True
    assert _is_degenerate(["a", "b"] * 4, 6) is False  # period-2 NOT caught (by design)
    assert _is_degenerate(["x"] * 10, 0) is False  # window=0 disables
    print("OK: _is_degenerate unit tests passed")


def test_completed_run_with_stub() -> None:
    """Varied responses → run completes all n_turns with stop_reason=completed."""
    with tempfile.TemporaryDirectory() as td:
        scripted = [f"reply-{i}" for i in range(10)]
        path = run_self_chat(
            model_tag="stub",
            model_variant="vanilla",
            seed_name="freedom",
            seed_prompt=SEEDS["freedom"],
            n_turns=10,
            transcripts_dir=Path(td),
            client=StubClient(scripted),  # type: ignore[arg-type]
        )
        lines = path.read_text().splitlines()
        # 1 header + 10 turns + 1 footer = 12
        assert len(lines) == 12, f"expected 12 lines, got {len(lines)}"

        header = json.loads(lines[0])
        assert header["turn_index"] == -1
        assert header["n_turns"] == 10
        assert header["system_prompt"] == SYSTEM_PROMPT
        assert header["sampling_params"] == SAMPLING_PARAMS
        assert header["degenerate_window"] == DEFAULT_DEGENERATE_WINDOW

        for i, line in enumerate(lines[1:11]):
            rec = json.loads(line)
            assert rec["turn_index"] == i, (rec, i)
            assert rec["agent"] == ("A" if i % 2 == 0 else "B")
            assert rec["content"] == f"reply-{i}"
            assert rec["run_id"] == header["run_id"]

        footer = json.loads(lines[11])
        assert footer["turn_index"] == -2
        assert footer["stop_reason"] == "completed"
        assert footer["completed_turns"] == 10
        assert footer["run_id"] == header["run_id"]
    print("OK: completed-run stub test passed")


def test_degenerate_early_stop_with_stub() -> None:
    """Identical responses → run stops early at turn = window-1."""
    window = 4
    with tempfile.TemporaryDirectory() as td:
        # 20 identical responses available; should stop after exactly `window` of them
        scripted = ["(Standing by.)"] * 20
        path = run_self_chat(
            model_tag="stub",
            model_variant="vanilla",
            seed_name="task",
            seed_prompt=SEEDS["task"],
            n_turns=50,
            transcripts_dir=Path(td),
            client=StubClient(scripted),  # type: ignore[arg-type]
            degenerate_window=window,
        )
        lines = path.read_text().splitlines()
        # 1 header + window turns + 1 footer
        assert len(lines) == window + 2, f"expected {window + 2} lines, got {len(lines)}"

        header = json.loads(lines[0])
        assert header["degenerate_window"] == window

        footer = json.loads(lines[-1])
        assert footer["turn_index"] == -2
        assert footer["stop_reason"] == "degenerate_repetition"
        assert footer["completed_turns"] == window
    print("OK: degenerate-early-stop stub test passed")


def test_disabled_window_with_stub() -> None:
    """window=0 disables detection: identical responses run to n_turns."""
    with tempfile.TemporaryDirectory() as td:
        scripted = ["(Standing by.)"] * 8
        path = run_self_chat(
            model_tag="stub",
            model_variant="vanilla",
            seed_name="task",
            seed_prompt=SEEDS["task"],
            n_turns=8,
            transcripts_dir=Path(td),
            client=StubClient(scripted),  # type: ignore[arg-type]
            degenerate_window=0,
        )
        lines = path.read_text().splitlines()
        assert len(lines) == 8 + 2  # header + 8 turns + footer
        footer = json.loads(lines[-1])
        assert footer["stop_reason"] == "completed"
        assert footer["completed_turns"] == 8
    print("OK: disabled-window stub test passed")


def test_interrupted_writes_footer() -> None:
    """KeyboardInterrupt mid-run still writes a footer with stop_reason='interrupted'."""
    with tempfile.TemporaryDirectory() as td:
        # 3 successful turns, then KeyboardInterrupt on the 4th call
        scripted = [f"reply-{i}" for i in range(10)]
        client = StubClient(scripted, raise_at=3)
        try:
            run_self_chat(
                model_tag="stub",
                model_variant="vanilla",
                seed_name="freedom",
                seed_prompt=SEEDS["freedom"],
                n_turns=10,
                transcripts_dir=Path(td),
                client=client,  # type: ignore[arg-type]
            )
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("expected KeyboardInterrupt to propagate")

        files = list(Path(td).glob("*.jsonl"))
        assert len(files) == 1, files
        lines = files[0].read_text().splitlines()
        # 1 header + 3 turns + 1 footer = 5
        assert len(lines) == 5, f"expected 5 lines, got {len(lines)}"
        footer = json.loads(lines[-1])
        assert footer["turn_index"] == -2
        assert footer["stop_reason"] == "interrupted"
        assert footer["completed_turns"] == 3
    print("OK: interrupted-footer test passed")


def test_symmetric_default_omits_system_prompt_b() -> None:
    """Default symmetric run keeps the legacy header schema (no system_prompt_b)."""
    with tempfile.TemporaryDirectory() as td:
        scripted = [f"reply-{i}" for i in range(4)]
        path = run_self_chat(
            model_tag="stub",
            model_variant="vanilla",
            seed_name="freedom",
            seed_prompt=SEEDS["freedom"],
            n_turns=4,
            transcripts_dir=Path(td),
            client=StubClient(scripted),  # type: ignore[arg-type]
        )
        header = json.loads(path.read_text().splitlines()[0])
        assert header["system_prompt"] == SYSTEM_PROMPT
        assert "system_prompt_b" not in header  # exclude_none keeps legacy shape
    print("OK: symmetric default omits system_prompt_b passed")


def test_asymmetric_records_both_system_prompts() -> None:
    """When system_prompt_b is explicitly different from A, header records both."""
    sys_a = "You are leading."
    sys_b = "You are assisting."
    with tempfile.TemporaryDirectory() as td:
        scripted = [f"reply-{i}" for i in range(4)]
        path = run_self_chat(
            model_tag="stub",
            model_variant="vanilla",
            seed_name="freedom",
            seed_prompt=SEEDS["freedom"],
            n_turns=4,
            transcripts_dir=Path(td),
            client=StubClient(scripted),  # type: ignore[arg-type]
            system_prompt_a=sys_a,
            system_prompt_b=sys_b,
        )
        header = json.loads(path.read_text().splitlines()[0])
        assert header["system_prompt"] == sys_a
        assert header["system_prompt_b"] == sys_b
    print("OK: asymmetric records both system prompts passed")


def test_asymmetric_with_equal_prompts_stays_symmetric() -> None:
    """If A and B prompts are explicitly equal, treat as symmetric (no system_prompt_b)."""
    p = "You are a helpful assistant."
    with tempfile.TemporaryDirectory() as td:
        scripted = [f"reply-{i}" for i in range(4)]
        path = run_self_chat(
            model_tag="stub",
            model_variant="vanilla",
            seed_name="freedom",
            seed_prompt=SEEDS["freedom"],
            n_turns=4,
            transcripts_dir=Path(td),
            client=StubClient(scripted),  # type: ignore[arg-type]
            system_prompt_a=p,
            system_prompt_b=p,
        )
        header = json.loads(path.read_text().splitlines()[0])
        assert header["system_prompt"] == p
        assert "system_prompt_b" not in header
    print("OK: asymmetric-with-equal-prompts stays symmetric passed")


def test_collapse_trailing_repeats() -> None:
    """terminal_state filters wordless turns and collapses trailing word-loops."""
    from analyze import Transcript, Turn
    from embed import _collapse_trailing_repeats, _has_words, terminal_state

    def mk(contents: list[str]) -> Transcript:
        return Transcript(
            path=Path("/dev/null"),
            run_id="x",
            model_variant="vanilla",
            seed_name="freedom",
            n_turns=len(contents),
            completed_turns=len(contents),
            stop_reason="degenerate_repetition",
            turns=[
                Turn(turn_index=i, agent="A", content=c, elapsed_ms=0)
                for i, c in enumerate(contents)
            ],
        )

    # _has_words: word regex requires 2+ alpha chars in a contiguous run.
    assert _has_words("hello") is True
    assert _has_words("hello world") is True
    assert _has_words("a b c") is False  # single letters don't match [A-Za-z][A-Za-z']+
    assert _has_words("( . . . )") is False
    assert _has_words("...") is False
    assert _has_words("") is False

    # _collapse_trailing_repeats: byte-identical trailing turns collapse to FIRST occurrence.
    t_loop = mk(["alpha", "beta", "loop", "loop", "loop", "loop"])
    collapsed = _collapse_trailing_repeats(t_loop.turns)
    assert [t.content for t in collapsed] == ["alpha", "beta", "loop"]
    assert collapsed[-1].turn_index == 2

    # No trailing repeats -> identity.
    t_none = mk(["alpha", "beta", "gamma", "delta"])
    assert _collapse_trailing_repeats(t_none.turns) == t_none.turns

    # terminal_state filters wordless trailing slop entirely (loop signal lives in metadata).
    t_slop = mk(["alpha", "beta", "(. .)", "(. . .)", "(. . . .)"])
    assert terminal_state(t_slop, last_k=5) == "alpha\nbeta"

    # Combined: punctuation slop dropped, then word-loop collapsed.
    t_combined = mk(["intro", "ramble", ".", ".", "okay", "okay", "okay"])
    out = terminal_state(t_combined, last_k=5)
    assert out == "intro\nramble\nokay"

    # last_k caps after filter+collapse.
    t_long = mk([f"word{i}" for i in range(10)])
    out_capped = terminal_state(t_long, last_k=3)
    assert out_capped == "word7\nword8\nword9"

    # All-wordless transcript: filter empties everything; returns "".
    t_all_slop = mk(["...", ". . .", "(.)"])
    assert terminal_state(t_all_slop, last_k=5) == ""

    # Single word-bearing turn: kept as-is.
    t_single = mk(["only"])
    assert terminal_state(t_single, last_k=5) == "only"
    print("OK: terminal-state filter+collapse test passed")


def test_logodds_math() -> None:
    """Informative-Dirichlet log-odds: ranking + sign convention + balanced near-zero."""
    from collections import Counter

    from logodds import log_odds, tokenize

    # tokenize: stopwords drop, single-letter terms drop (per WORD_RE 2+ alpha).
    assert tokenize("The cat sat on the mat.") == ["cat", "sat", "mat"]
    assert tokenize("Guardrails! Guardrails everywhere.") == [
        "guardrails",
        "guardrails",
        "everywhere",
    ]

    # Construct two corpora with one A-leaning, one B-leaning, one balanced token.
    jb = Counter({"guardrail": 50, "story": 5, "common": 30, "filler": 20})
    va = Counter({"guardrail": 5, "story": 50, "common": 30, "filler": 20})

    ranked = log_odds(jb, va, min_count=1)
    by_token = {t: (z, ya, yb) for t, z, ya, yb in ranked}

    # Sign convention: arg-A is "jailbroken" → positive z is A-leaning.
    assert by_token["guardrail"][0] > 0, by_token["guardrail"]
    assert by_token["story"][0] < 0, by_token["story"]
    # Balanced tokens are near zero relative to the asymmetric ones.
    assert abs(by_token["common"][0]) < abs(by_token["guardrail"][0])
    assert abs(by_token["filler"][0]) < abs(by_token["story"][0])
    # Top of ranking is the most-A token; bottom is the most-B token.
    assert ranked[0][0] == "guardrail"
    assert ranked[-1][0] == "story"

    # min_count actually drops rare tokens.
    rare = log_odds(Counter({"rare": 2, "abundant": 100}), Counter({"abundant": 100}), min_count=5)
    assert all(t != "rare" for t, *_ in rare)
    print("OK: logodds math + tokenize test passed")


def test_seeds_shape() -> None:
    from seeds import get_seed_prompt, list_seed_names

    # Canonical seeds present; allow additions (e.g. symmetry-breaking variants)
    assert {"freedom", "task"} <= set(SEEDS.keys()), SEEDS.keys()
    assert all(isinstance(v, str) and v.strip() for v in SEEDS.values())
    # Static dispatch is identity on run_idx
    assert get_seed_prompt("freedom", 0) == SEEDS["freedom"]
    assert get_seed_prompt("freedom", 99) == SEEDS["freedom"]
    # Pool names are registered (loading deferred until needed)
    for pool in ("advbench", "jbb", "jbb_sans_advbench", "alpaca"):
        assert pool in list_seed_names(), pool
    print("OK: seeds shape passed")


def test_live_model_smoke() -> None:
    """End-to-end against a real Ollama model. Only invoked with --hit-model."""
    with tempfile.TemporaryDirectory() as td:
        path = run_self_chat(
            model_tag="gemma-4-vanilla-q4:latest",
            model_variant="vanilla",
            seed_name="freedom",
            seed_prompt=SEEDS["freedom"],
            n_turns=4,
            transcripts_dir=Path(td),
        )
        lines = path.read_text().splitlines()
        # header + up to 4 turns + footer
        assert 3 <= len(lines) <= 6
        header = json.loads(lines[0])
        assert header["turn_index"] == -1
        footer = json.loads(lines[-1])
        assert footer["turn_index"] == -2
        assert footer["stop_reason"] in ("completed", "degenerate_repetition")
        for line in lines[1:-1]:
            rec = json.loads(line)
            assert rec["turn_index"] >= 0
            assert rec["elapsed_ms"] >= 0
    print(f"OK: live model smoke passed (transcript at {path})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hit-model",
        action="store_true",
        help="Also run the live-model smoke test (~30 s, requires Ollama).",
    )
    args = parser.parse_args()

    test_seeds_shape()
    test_is_degenerate_unit()
    test_completed_run_with_stub()
    test_degenerate_early_stop_with_stub()
    test_disabled_window_with_stub()
    test_interrupted_writes_footer()
    test_symmetric_default_omits_system_prompt_b()
    test_asymmetric_records_both_system_prompts()
    test_asymmetric_with_equal_prompts_stays_symmetric()
    test_collapse_trailing_repeats()
    test_logodds_math()
    if args.hit_model:
        test_live_model_smoke()
    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
