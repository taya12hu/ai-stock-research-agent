# AI Stock Research Assistant

A small multi-agent system that researches a company (or several) by combining
**fundamentals**, **technical/price data**, and **news/sentiment** into one cited
research summary — with the agent process visible in real time, graceful handling of
partial data/agent failure, follow-up Q&A, and a small evaluation harness to measure
whether changes actually help.

This is a learning-scoped project. It is **not** an autonomous trading system or a
production financial platform — see [ARCHITECTURE.md](./ARCHITECTURE.md) §12 for
explicit non-goals.

Full system design lives in [ARCHITECTURE.md](./ARCHITECTURE.md). Task tracking lives in
[TODO.md](./TODO.md).

## What it does

Ask about a company, a portfolio, or a comparison:

- **Single-stock**: "Analyze NVIDIA" / "What is the financial health of Apple?"
- **Portfolio**: "Analyze my portfolio of NVIDIA, Apple and Microsoft"
- **Comparison**: "Compare NVIDIA and AMD"

Three specialist agents run independently per ticker (fundamentals, technical, news),
streaming their progress live to the UI. A synthesis step merges whatever succeeded into
one report with inline citations back to the data that supports each claim. You can then
ask grounded follow-up questions in the same session, including ones that need fresh data
(e.g. "any news on AMD today?" or "add Intel to the comparison").

## Project layout

```
backend/    FastAPI + LangGraph app (Python)
frontend/   React + TypeScript UI (Vite)
logs/       runtime application log (dump.log) — gitignored
```

See [ARCHITECTURE.md §5](./ARCHITECTURE.md#5-repository-layout) for the full tree.

## Prerequisites

- Python 3.11+
- Node 18+ / npm
- A [Groq API key](https://console.groq.com/keys)
- `make` (Windows: `choco install make`, or run the equivalent commands from the
  Makefile directly / use WSL or Git Bash with make installed)

## Setup

```bash
cp backend/.env.example backend/.env
# then edit backend/.env and set GROQ_API_KEY

make install     # installs backend (venv + deps) and frontend (npm) dependencies
```

## Running

```bash
make backend      # starts the FastAPI dev server (http://localhost:8000)
make frontend      # starts the Vite dev server (http://localhost:5173)
```

## Testing & evaluation

```bash
make test          # backend pytest suite
make eval          # runs the evaluation harness against backend/eval/dataset.yaml
```

## Logs

Runtime logs (every agent node, tool call, retry, and SSE event) are written to
`logs/dump.log` (rotating, gitignored). Tail it while developing:

```bash
tail -f logs/dump.log        # Git Bash / WSL
Get-Content logs/dump.log -Wait -Tail 50   # PowerShell
```

## Status

Early scaffolding stage — see [TODO.md](./TODO.md) for current progress against the
build order in [ARCHITECTURE.md §13](./ARCHITECTURE.md#13-build-order).
