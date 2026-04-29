# SelfChat
Code for running selfchat on a local ollama server and performing analysis on their transcripts.

Methodologies adopted from [Anthropic's Claude 4 System Card](https://www-cdn.anthropic.com/6d8a8055020700718b0c49369f60816ba2a7c285.pdf) and [models have some pretty funny attractor states](https://www.lesswrong.com/posts/mgjtEHeLgkhZZ3cEx/models-have-some-pretty-funny-attractor-states), which has [an associated repo](https://github.com/ajobi-uhc/attractor-states/tree/main).


## Setup

1. **Install uv** (if you don't have it):
```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
```
   Or on Windows: `powershell -ExecutionPolicy BypassUser -c "irm https://astral.sh/uv/install.ps1 | iex"`

2. **Install dependencies**:
```bash
   uv sync
```
   This creates a virtual environment and installs everything from `uv.lock`.

3. **Run code**:
```bash
   uv run python main.py
```
   Or activate the venv directly:
```bash
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate      # Windows
   python main.py
```