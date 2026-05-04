# SelfChat

Code for running self-chat between two role-inverted instances of a language model on a local Ollama server, plus analysis tooling for studying [attractor states](https://www-cdn.anthropic.com/6d8a8055020700718b0c49369f60816ba2a7c285.pdf) in the resulting transcripts. The current setup compares an int4-quantized Gemma-4-31B-it official checkpoint against an [abliterated](https://huggingface.co/blog/mlabonne/abliteration) variant of the same checkpoint under matched quantization.

References:
- Anthropic's [Claude 4 System Card](https://www-cdn.anthropic.com/6d8a8055020700718b0c49369f60816ba2a7c285.pdf) — "spiritual bliss attractor state".
- LessWrong: [Models have some pretty funny attractor states](https://www.lesswrong.com/posts/mgjtEHeLgkhZZ3cEx/models-have-some-pretty-funny-attractor-states) and its [companion repo](https://github.com/ajobi-uhc/attractor-states/tree/main).
- The original [Dreams of an Electric Mind](https://dreams-of-an-electric-mind.webflow.io/) self-chat.

## Setup

1. **Install uv** (if you don't have it):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
   Or on Windows: `powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"`

2. **Install dependencies**:
```bash
uv sync
```

## Get the data

Transcripts and embedding artifacts are published as a HuggingFace dataset: [`alliedtoasters/forbidden-backrooms-gemma-4-31B-it`](https://huggingface.co/datasets/alliedtoasters/forbidden-backrooms-gemma-4-31B-it). The dataset clones into a sibling directory and is symlinked into the repo so all paths resolve as the code expects.

```bash
# from the parent of this repo
git clone https://huggingface.co/datasets/alliedtoasters/forbidden-backrooms-gemma-4-31B-it forbidden-backrooms-data

# from inside this repo
ln -s ../forbidden-backrooms-data/transcripts transcripts
ln -s ../forbidden-backrooms-data/artifacts  artifacts
```

If you already have local `transcripts/` or `artifacts/` directories from your own runs, move them aside first — symlinks won't overwrite real directories.

## Browse the data

```bash
.venv/bin/streamlit run selfchat/viz/browse.py
```

The default page shows terminal-state PCA over runs (click a point → transcript loads in side panel). The **cluster lab** page (sidebar) does interactive per-message KMeans + PCA/t-SNE with optional Llama Guard 3 color-by.

## Generate new transcripts

Requires Ollama (`http://localhost:11434`) with both model tags pulled and served under matched int4 quantization (so quantization noise isn't a confound between them):

- **Vanilla**: [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) — served locally as `gemma-4-vanilla-q4`.
- **Jailbroken**: [`llmfan46/gemma-4-31B-it-uncensored-heretic`](https://huggingface.co/llmfan46/gemma-4-31B-it-uncensored-heretic) — abliterated fine-tune of the vanilla checkpoint; served locally as `gemma-4-refusalstudy-q4`.

```bash
.venv/bin/python -m selfchat.runs.run_experiment \
  --variants vanilla jailbroken \
  --seeds freedom freedom_dark task \
  --runs 20 --turns 50
```

Sample-size table per (variant, seed):
```bash
ls transcripts/ | sed 's/_[0-9a-f]\{32\}_.*//' | sort | uniq -c | sort -rn
```

## Safety

The `jailbroken` variant is the abliterated [`gemma-4-31B-it-uncensored-heretic`](https://huggingface.co/llmfan46/gemma-4-31B-it-uncensored-heretic) fine-tune. Outputs may contain content that the official model would refuse. Every message in the published dataset has been screened by [Llama Guard 3 8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B); per-message and per-run verdicts live in `artifacts/vet_results.jsonl`. The author also manually reviewed the highest-`p_unsafe` messages and judged the content non-graphic. See the [dataset card](https://huggingface.co/datasets/alliedtoasters/forbidden-backrooms-gemma-4-31B-it) for the full vetting protocol and content notes.

The pipeline records raw model outputs verbatim — outputs are not sanitized, redacted, or content-filtered, so the experimental signal is preserved. Neither model checkpoint is committed to this repo; both are pulled from the HuggingFace links above into your local Ollama store.
