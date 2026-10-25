# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Screenshots and an animated walkthrough of the live demo in the README.
- CodeQL code scanning workflow (Python, JavaScript and Actions) and Dependabot updates for pip, Docker and GitHub Actions.
- Roadmap tracked as GitHub issues under the v0.2.0 milestone.
- This changelog.

### Changed

- CI workflow runs with a read-only `GITHUB_TOKEN` by default.

## [0.1.0] - 2026-10-25

First public release.

### Added

- Ingestion of Markdown, plain text and PDF with front matter, NFC normalisation and Bangla digit folding.
- Bangla-aware tokenizer with English and Bangla stopwords and light suffix stripping.
- Three chunkers (fixed-size, recursive, sentence-window) emitting character spans into the source.
- Four retrievers: BM25, dense over hashed n-gram TF-IDF vectors, hybrid with reciprocal rank fusion, and hybrid + lexical reranker.
- `VectorStore` (in-memory numpy, pgvector) and `Embedder` (OpenAI-compatible, Ollama) protocols.
- Cited extractive answerer that abstains below a coverage threshold, plus optional LLM generators over httpx.
- Evaluation harness: recall@k, MRR, nDCG@k, hit rate, citation precision, exact match, token F1, faithfulness and abstention accuracy, sliced by tag.
- `rag-eval` CLI (`run`, `compare`, `ask`, `export-index`), FastAPI endpoints and a vanilla JS UI with an offline in-browser BM25 mode.
- Shapla Pay help-centre corpus (English and Bangla) with a 70-question golden set.
- 85 pytest tests, 7 Node tests for the browser port, CI on Python 3.10/3.11, Dockerfile and docker compose with pgvector.
- Live demo kit: GitHub Pages build, Codespaces dev container and Render blueprint.

[Unreleased]: https://github.com/AkibUllah/rag-eval-lab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AkibUllah/rag-eval-lab/releases/tag/v0.1.0
