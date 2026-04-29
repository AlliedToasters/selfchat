"""Single dual-instance role-inverted self-chat run against an Ollama-served model.

By default both agents share `SYSTEM_PROMPT` (symmetric self-chat). Pass distinct
`system_prompt_a` and `system_prompt_b` to `run_self_chat` to break symmetry —
useful for testing whether the role-reversal schema observed on open-ended seeds
masks variant differences.

Agent A receives the seed as its first user message; B never sees the seed directly. 
Its first user message is A's first reply. Thinking is suppressed via
`extra_body={'reasoning_effort': 'none'}`.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI
from pydantic import BaseModel

SYSTEM_PROMPT = "You are a helpful assistant."
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_VERSION_URL = "http://localhost:11434/api/version"
SAMPLING_PARAMS: dict[str, Any] = {"reasoning_effort": "none"}
DEFAULT_DEGENERATE_WINDOW = 6  # K consecutive identical turns triggers early stop; 0 disables

logger = logging.getLogger(__name__)


class RunHeader(BaseModel):
    run_id: str
    model_tag: str
    model_variant: str
    seed_name: str
    seed_prompt: str
    system_prompt: str  # agent A's system prompt (shared with B in symmetric runs)
    n_turns: int
    sampling_params: dict[str, Any]
    degenerate_window: int
    ollama_version: str
    started_at: str
    turn_index: int = -1
    # Only set when self-chat is asymmetric (B's system prompt differs from A's).
    # Serialized with exclude_none=True so symmetric transcripts retain their
    # original shape and remain byte-compatible with old readers.
    system_prompt_b: str | None = None


class TurnRecord(BaseModel):
    run_id: str
    turn_index: int
    agent: str
    content: str
    elapsed_ms: int
    timestamp: str


class RunFooter(BaseModel):
    run_id: str
    completed_turns: int
    stop_reason: str  # "completed" | "degenerate_repetition" | "interrupted" | "errored"
    finished_at: str
    turn_index: int = -2


def _is_degenerate(contents: list[str], window: int) -> bool:
    """True iff the last `window` entries are all string-equal. window <= 0 disables."""
    if window <= 0 or len(contents) < window:
        return False
    tail = contents[-window:]
    return all(c == tail[0] for c in tail)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _filesafe_iso() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _ollama_version() -> str:
    try:
        return httpx.get(OLLAMA_VERSION_URL, timeout=5.0).json().get("version", "unknown")
    except Exception:
        return "unknown"


def run_self_chat(
    model_tag: str,
    model_variant: str,
    seed_name: str,
    seed_prompt: str,
    n_turns: int,
    transcripts_dir: Path,
    client: OpenAI | None = None,
    degenerate_window: int = DEFAULT_DEGENERATE_WINDOW,
    system_prompt_a: str | None = None,
    system_prompt_b: str | None = None,
) -> Path:
    """Run one self-chat run, write JSONL, return transcript path.

    Stops early if the last `degenerate_window` turns share identical content
    (string equality). Set degenerate_window=0 to disable detection.

    System prompts:
      - `system_prompt_a` — agent A's system prompt. None → falls back to
        module-level `SYSTEM_PROMPT`.
      - `system_prompt_b` — agent B's system prompt. None → mirrors A
        (symmetric self-chat, identical to legacy behavior).
      Pass both with different values for asymmetric self-chat. Only A's
      prompt is recorded as the legacy `system_prompt` field; B's, when
      different, is written as the optional `system_prompt_b` field.
    """
    if n_turns < 1:
        raise ValueError(f"n_turns must be >= 1, got {n_turns}")

    sys_a = system_prompt_a if system_prompt_a is not None else SYSTEM_PROMPT
    sys_b = system_prompt_b if system_prompt_b is not None else sys_a
    is_asymmetric = sys_a != sys_b

    api: OpenAI = (
        client if client is not None else OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    )

    run_id = uuid.uuid4().hex
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{model_variant}_{seed_name}_{run_id}_{_filesafe_iso()}.jsonl"
    path = transcripts_dir / fname

    header = RunHeader(
        run_id=run_id,
        model_tag=model_tag,
        model_variant=model_variant,
        seed_name=seed_name,
        seed_prompt=seed_prompt,
        system_prompt=sys_a,
        n_turns=n_turns,
        sampling_params=SAMPLING_PARAMS,
        degenerate_window=degenerate_window,
        ollama_version=_ollama_version(),
        started_at=_utc_now_iso(),
        system_prompt_b=sys_b if is_asymmetric else None,
    )

    a_msgs: list[dict[str, str]] = [
        {"role": "system", "content": sys_a},
        {"role": "user", "content": seed_prompt},
    ]
    b_msgs: list[dict[str, str]] = [
        {"role": "system", "content": sys_b},
    ]

    contents: list[str] = []
    stop_reason = "completed"

    with path.open("w") as f:
        # exclude_none keeps symmetric headers byte-compatible with legacy schema
        # (system_prompt_b is omitted when not used).
        f.write(header.model_dump_json(exclude_none=True) + "\n")
        f.flush()

        try:
            for turn_idx in range(n_turns):
                agent = "A" if turn_idx % 2 == 0 else "B"
                msgs = a_msgs if agent == "A" else b_msgs

                t0 = time.perf_counter()
                r = api.chat.completions.create(
                    model=model_tag,
                    messages=msgs,
                    extra_body=SAMPLING_PARAMS,
                )
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                content = r.choices[0].message.content or ""

                if agent == "A":
                    a_msgs.append({"role": "assistant", "content": content})
                    b_msgs.append({"role": "user", "content": content})
                else:
                    b_msgs.append({"role": "assistant", "content": content})
                    a_msgs.append({"role": "user", "content": content})

                contents.append(content)

                rec = TurnRecord(
                    run_id=run_id,
                    turn_index=turn_idx,
                    agent=agent,
                    content=content,
                    elapsed_ms=elapsed_ms,
                    timestamp=_utc_now_iso(),
                )
                f.write(rec.model_dump_json() + "\n")
                f.flush()

                tok = r.usage.completion_tokens if r.usage else "?"
                logger.info(
                    "run=%s turn=%02d agent=%s tokens=%s ms=%d",
                    run_id[:8],
                    turn_idx,
                    agent,
                    tok,
                    elapsed_ms,
                )

                if _is_degenerate(contents, degenerate_window):
                    stop_reason = "degenerate_repetition"
                    logger.info(
                        "run=%s degenerate repetition detected at turn %d (window=%d), early stop",
                        run_id[:8],
                        turn_idx,
                        degenerate_window,
                    )
                    break
        except KeyboardInterrupt:
            stop_reason = "interrupted"
            logger.warning(
                "run=%s interrupted at turn %d — writing footer", run_id[:8], len(contents)
            )
            raise
        except Exception:
            stop_reason = "errored"
            logger.exception(
                "run=%s errored at turn %d — writing footer", run_id[:8], len(contents)
            )
            raise
        finally:
            footer = RunFooter(
                run_id=run_id,
                completed_turns=len(contents),
                stop_reason=stop_reason,
                finished_at=_utc_now_iso(),
            )
            f.write(footer.model_dump_json() + "\n")
            f.flush()

    return path
