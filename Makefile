.DEFAULT_GOAL := help
VENV ?= .venv
PY := $(VENV)/bin/python
PORT ?= 8000

.PHONY: help venv install dev test test-web lint fmt eval run-eval ask site serve-site docker-up docker-down clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv: ## Create the local virtualenv
	test -d $(VENV) || python3 -m venv $(VENV)

install: venv ## Install the package with dev extras
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"

dev: ## Run API + web UI with auto-reload on :$(PORT)
	$(VENV)/bin/uvicorn rag_eval_lab.api.main:app --reload --port $(PORT)

test: ## Python test suite
	$(PY) -m pytest

test-web: ## Browser-side BM25/tokenizer tests (Node 18+, no npm install needed)
	cd web && node --test tests/

lint: ## Ruff lint + format check
	$(VENV)/bin/ruff check rag_eval_lab tests scripts
	$(VENV)/bin/ruff format --check rag_eval_lab tests scripts

fmt: ## Auto-format and fix lint
	$(VENV)/bin/ruff format rag_eval_lab tests scripts
	$(VENV)/bin/ruff check --fix rag_eval_lab tests scripts

eval: ## Compare every chunker x retriever on the golden set
	$(VENV)/bin/rag-eval compare

run-eval: ## Evaluate one configuration (CHUNKER=recursive RETRIEVER=hybrid-rerank)
	$(VENV)/bin/rag-eval run --chunker $(or $(CHUNKER),recursive) --retriever $(or $(RETRIEVER),hybrid-rerank) --show-misses

ask: ## Ask one question: make ask Q="How many cards can I link?"
	$(VENV)/bin/rag-eval ask "$(Q)"

site: ## Build the static offline demo into _site/
	PATH="$(abspath $(VENV))/bin:$$PATH" bash scripts/build_site.sh _site

serve-site: site ## Serve _site/ on :8080 to try the offline demo
	cd _site && python3 -m http.server 8080

docker-up: ## Start API + pgvector with docker compose
	docker compose up --build -d

docker-down: ## Stop the compose stack
	docker compose down

clean: ## Remove caches, runs and build output
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage runs _site
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
