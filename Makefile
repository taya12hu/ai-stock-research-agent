# Common commands for the AI Stock Research Assistant.
# Windows: `choco install make`, or run via Git Bash/WSL with make installed.
# Assumes `python -m venv` (Windows layout: .venv/Scripts) and Node/npm on PATH.

PYTHON       ?= python
VENV         := backend/.venv
VENV_BIN     := $(VENV)/Scripts
BACKEND_PY   := $(VENV_BIN)/python

.PHONY: install install-backend install-frontend backend frontend test eval lint clean

install: install-backend install-frontend

install-backend:
	$(PYTHON) -m venv $(VENV)
	$(BACKEND_PY) -m pip install --upgrade pip
	$(BACKEND_PY) -m pip install -e "./backend[dev]"

install-frontend:
	cd frontend && npm install

backend:
	$(BACKEND_PY) -m uvicorn app.main:app --reload --app-dir backend --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

test:
	$(BACKEND_PY) -m pytest backend/tests -v

eval:
	cd backend && .venv/Scripts/python.exe -m eval.run_eval

lint:
	$(BACKEND_PY) -m ruff check backend

clean:
	rm -rf $(VENV) backend/.pytest_cache
	find backend -type d -name __pycache__ -prune -exec rm -rf {} +
