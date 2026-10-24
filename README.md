<!-- topics: rag, retrieval-augmented-generation, information-retrieval, evaluation, bm25, hybrid-search, pgvector, fastapi, python, bangla, llm -->
# RAG Eval Lab

Retrieval-augmented Q&A over a bilingual (English + Bangla) help centre, built around an evaluation harness that measures every chunker and retriever combination against a golden set before anything ships.

[![CI](https://github.com/AkibUllah/rag-eval-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/AkibUllah/rag-eval-lab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)
![License](https://img.shields.io/badge/license-MIT-green)

## Try it live

| | |
|---|---|
| Live demo | https://akibullah.github.io/rag-eval-lab/ — the real web UI in offline mode: a pre-tokenised chunk index and BM25 + the extractive answerer running in your browser (verified to return the same chunks and answers as the Python pipeline for all 70 golden questions). The Evaluation tab shows the comparison grid generated on the last push. |
| Run in your browser | [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/AkibUllah/rag-eval-lab?quickstart=1) — installs the package, runs `rag-eval compare` once and starts the API on forwarded port 8000: `/` is the UI with all four retrievers, `/docs` is Swagger. |
| One-click deploy | [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/AkibUllah/rag-eval-lab) — `render.yaml` deploys the Docker image in offline mode (no database, no keys). |
| Local | `docker compose up` (see Quick Start) |

## Why this exists

Most RAG demos I review are tuned by eyeballing three questions. In fintech support that is not good enough: a wrong cash-out fee or a made-up dispute deadline is a compliance problem, not a UX glitch. This repo treats retrieval as something you measure first and tune second. It ships a small but realistic knowledge base for a fictional Bangladeshi mobile wallet (Shapla Pay), a 70-question golden set with verbatim evidence, and a harness that scores chunking, retrieval and answer quality in one command. Everything runs offline with no API keys; real LLM and embedding providers plug in behind small protocols when you want them.

## Features

- **Ingestion**: Markdown, plain text and PDF (text layer via optional `pypdf`) with front matter; NFC normalisation, zero-width cleanup and Bangla digit folding (`৩০,০০০` and `30,000` produce the same term)
- **Bangla-aware tokenizer**: keeps vowel signs inside words (plain `\w` splits `টাকা`), English and Bangla stopwords, light suffix stripping (`-এর`, `-কে`, `-টি`, `-গুলো`)
- **Three chunkers**: fixed-size word windows with overlap, recursive (heading → paragraph → sentence, never crossing a section) and sentence-window (index one sentence, return it with neighbours); all emit character spans into the source
- **Four retrievers**: BM25 (own implementation), dense over hashed n-gram TF-IDF vectors (numpy only), hybrid with reciprocal rank fusion, and hybrid + a lexical reranker
- **Pluggable backends**: `VectorStore` protocol with in-memory numpy and pgvector adapters; `Embedder` protocol with OpenAI-compatible `/embeddings` and Ollama adapters
- **Cited answers**: every claim carries `[doc#chunk]`; the offline extractive answerer quotes the best-supported sentences and abstains below a coverage threshold; optional OpenAI-compatible, Anthropic Messages and Ollama generators over plain httpx
- **Evaluation harness**: recall@k, MRR, nDCG@k, hit rate, citation precision, exact match, token F1, overlap-based faithfulness (optional LLM judge) and abstention accuracy, sliced by tag (Bangla, paraphrase, numeric...)
- **CLI**: `rag-eval run`, `rag-eval compare` (markdown table + JSON report), `rag-eval ask`, `rag-eval export-index`
- **API + UI**: `POST /ingest`, `POST /ingest/files`, `POST /ask`, `GET /eval/latest`; a vanilla JS UI that shows the answer, cited chunks and scores, and the latest evaluation grid

## Architecture

```mermaid
flowchart LR
    subgraph Ingest
        F[".md / .txt / .pdf"] --> L[Loader + normaliser]
        L --> C{Chunker<br/>fixed · recursive · sentence}
    end
    C --> B[BM25 index]
    C --> E[Embedder<br/>hashed TF-IDF · OpenAI · Ollama]
    E --> V[(VectorStore<br/>numpy · pgvector)]
    Q[Question] --> B & V
    B & V --> RRF[Reciprocal rank fusion]
    RRF --> RR[Lexical reranker]
    RR --> G{Generator<br/>extractive · LLM}
    G --> A["Answer with [doc#chunk] citations"]
    subgraph Evaluation
        GS[Golden set<br/>70 questions + evidence] --> H[Harness]
        H --> M[Retrieval + answer metrics]
        M --> T[Markdown table · JSON report]
    end
    C -. spans .-> H
    A -. answers .-> H
```

Key decisions:

- **Evidence spans, not chunk ids.** The golden set stores the verbatim evidence passage for each question. A chunk is relevant when it covers at least 50% of that passage's character span, so one golden set scores every chunker fairly.
- **Metrics count passages once.** Two overlapping fixed-size chunks containing the same evidence do not double recall or nDCG; a chunk only gains if it covers a passage not seen at a higher rank.
- **Rank fusion, not score mixing.** RRF combines BM25 (unbounded scores) and cosine similarity (-1..1) by rank alone, so there is no weight to tune per corpus.
- **One tokenizer, two runtimes.** The browser demo is a port of the Python tokenizer, BM25 and answerer; stopwords and suffixes are exported with the index, and parity tests check the JS tokenizer over the whole corpus and the full browser pipeline against Python.
- **Offline is the default, not a mock.** The extractive answerer is a real baseline (it never invents text), and the same harness evaluates an LLM generator by switching one env var.

## Evaluation results

`rag-eval compare` over the bundled corpus (29 documents, 5 in Bangla) and golden set (70 questions: 55 English and 11 Bangla answerable, plus 4 deliberately unanswerable, one of them in Bangla), `k=5`, offline generator. Genuine output from this repo; latency is retrieval only, single process on a laptop CPU.

```
$ rag-eval compare
29 documents, 70 questions, k=5
```

| Chunker | Retriever | Chunks | Recall@5 | MRR | nDCG@5 | Hit@1 | Hit@5 | Cit. prec | Answer F1 | Faithful | Abstain acc | Ctx words | p50 ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | bm25 | 73 | 0.992 | 0.875 | 0.889 | 0.788 | **1.000** | 0.842 | 0.386 | **1.000** | **0.500** | 327 | 0.08 |
| fixed | dense | 73 | **1.000** | 0.860 | 0.881 | 0.758 | **1.000** | 0.825 | 0.378 | **1.000** | 0.250 | 337 | 0.58 |
| fixed | hybrid | 73 | 0.992 | 0.877 | 0.891 | 0.788 | **1.000** | 0.833 | 0.384 | **1.000** | 0.250 | 336 | 1.33 |
| fixed | hybrid-rerank | 73 | 0.977 | 0.854 | 0.870 | 0.742 | 0.985 | 0.808 | 0.377 | **1.000** | 0.250 | 339 | 4.66 |
| recursive | bm25 | 127 | 0.985 | **0.893** | **0.898** | 0.818 | 0.985 | 0.839 | 0.412 | **1.000** | **0.500** | 154 | 0.10 |
| recursive | dense | 127 | 0.985 | 0.853 | 0.869 | 0.758 | 0.985 | **0.853** | 0.400 | **1.000** | **0.500** | 156 | 0.64 |
| recursive | hybrid | 127 | 0.985 | **0.893** | **0.898** | 0.818 | 0.985 | 0.839 | 0.407 | **1.000** | **0.500** | 158 | 1.22 |
| recursive | hybrid-rerank | 127 | 0.985 | **0.893** | 0.897 | 0.818 | 0.985 | 0.817 | **0.414** | **1.000** | **0.500** | 163 | 3.24 |
| sentence | bm25 | 284 | 0.924 | 0.879 | 0.877 | **0.848** | 0.924 | 0.830 | 0.400 | **1.000** | **0.500** | 132 | 0.12 |
| sentence | dense | 284 | 0.939 | 0.856 | 0.864 | 0.803 | 0.939 | 0.839 | 0.395 | **1.000** | **0.500** | 133 | 0.95 |
| sentence | hybrid | 284 | 0.939 | 0.867 | 0.872 | 0.818 | 0.939 | 0.839 | 0.397 | **1.000** | **0.500** | 132 | 1.63 |
| sentence | hybrid-rerank | 284 | 0.939 | 0.867 | 0.873 | 0.818 | 0.939 | 0.816 | 0.398 | **1.000** | **0.500** | 134 | 2.92 |

Per-slice MRR for the recursive chunker (`rag-eval run --chunker recursive --retriever <r>`):

| Retriever | Lexical (36) | Paraphrase (17) | Bangla (11) |
|---|---:|---:|---:|
| bm25 | **0.951** | 0.806 | 0.909 |
| dense | 0.905 | 0.788 | 0.848 |
| hybrid | **0.951** | 0.806 | 0.909 |
| hybrid-rerank | 0.924 | **0.835** | **0.955** |

What the numbers say:

- **Chunking matters more than the retriever here.** Recursive chunks reach the same recall as fixed 80-word windows with less than half the context (154 vs 327 words for five chunks), which is what you pay for in LLM tokens. Sentence-window gets the best Hit@1 (0.848) because single sentences are precise, but loses recall (0.92–0.94): multi-sentence evidence and list items split across windows.
- **BM25 is hard to beat on a small, consistently worded help centre.** Hybrid ties BM25 on the recursive chunks; hashed n-gram vectors are not semantic, so they only add value where wording differs morphologically. With fixed windows the dense retriever is the only one with perfect recall.
- **The reranker trades lexical precision for paraphrase and Bangla.** Same overall MRR as BM25 (0.893), better on paraphrased (0.835 vs 0.806) and Bangla (0.955 vs 0.909) questions, worse on exact-wording ones, at roughly 30x the latency (still about 3 ms). That is why `recursive` + `hybrid-rerank` is the served default: support questions are mostly paraphrased.
- **Answer quality is bounded by the generator, not retrieval.** Evidence is in the top 5 for 98.5% of questions, yet token F1 against reference answers is about 0.41: the extractive baseline quotes full sentences, and on multi-evidence questions it often picks one of two needed facts. Faithfulness is 1.0 by construction (it only quotes), so it is the metric to watch when an LLM generator is plugged in, not this one.
- **Abstention is a threshold trade-off.** At the default `min_coverage=0.4` the answerer answers 90.9% of answerable questions and abstains on 2 of 4 unanswerable ones; `--min-coverage 0.45` moves that to 87.9% / 3 of 4. The threshold was chosen on this same set; with only four unanswerable questions, treat it as a starting point rather than a tuned value.
- **The one miss** (`q053`, escalation after support fails) is a vocabulary gap: the question never says "grievance" or "escalate", so every retriever returns the customer-support page. Query expansion or real embeddings are the fix to measure next.

Other checks that are part of the harness rather than the table: running the dense retriever against PostgreSQL + pgvector (`RAG_VECTOR_STORE=pgvector`) produces identical metrics to the in-memory store (127 rows of 8192-dim vectors), and the browser BM25 port returns the same top-5 chunk ids and answers as Python for all 70 questions.

### Metric definitions

| Metric | Definition |
|---|---|
| Recall@k | Share of a question's evidence passages covered by any of the top k chunks |
| MRR | Mean of 1 / rank of the first chunk covering any evidence |
| nDCG@k | Binary gain per rank, only for chunks covering a not-yet-covered passage; ideal = min(#passages, k) hits at the top |
| Hit@1 / Hit@k | Any evidence covered at rank 1 / within k |
| Cit. prec | Share of cited chunk ids that cover golden evidence (judged against all chunks, not only retrieved ones) |
| Answer F1 / EM | Token-level F1 and exact match against the reference answer, citations stripped, same tokenizer as retrieval |
| Faithful | Share of answer claims whose tokens overlap their cited chunks by at least 60% (`--judge` uses an LLM YES/NO check instead) |
| Abstain acc | Share of unanswerable questions where the generator abstained |
| Ctx words | Words handed to the generator across the top k chunks |

## Tech Stack

| Layer | Choice |
| --- | --- |
| API | FastAPI, Pydantic v2, Uvicorn |
| Retrieval | Own BM25, numpy hashed TF-IDF vectors, RRF, lexical reranker |
| Vector store | In-memory numpy; PostgreSQL 16 + pgvector via psycopg 3 (optional) |
| Providers | httpx adapters: OpenAI-compatible chat/embeddings, Anthropic Messages, Ollama |
| UI | Vanilla JS modules (no build step), client-side BM25 for the Pages demo |
| Tooling | pytest, `node --test`, ruff, Makefile, GitHub Actions (3.10/3.11 + Node 18 + docker build) |
| Runtime | Python 3.10+, Dockerfile (non-root, healthcheck), docker compose with pgvector |

## Quick Start

### Docker Compose (API + PostgreSQL/pgvector)

```bash
git clone https://github.com/AkibUllah/rag-eval-lab.git
cd rag-eval-lab
docker compose up --build -d
curl -s localhost:8000/health
# UI at http://localhost:8000/, Swagger at /docs
docker compose exec api rag-eval compare   # fills the UI's Evaluation tab
```

### Local (no database)

```bash
make install        # python3 -m venv .venv && pip install -e ".[dev]"
make eval           # rag-eval compare: prints the table above
make dev            # API + UI on http://127.0.0.1:8000
make site serve-site   # build and serve the offline demo on :8080
```

### Using real models

Everything above is offline. To generate answers with a real model or embed with a real embedding model, set environment variables (or `.env`, see `.env.example`):

```bash
# Ollama, fully local
RAG_LLM_PROVIDER=ollama RAG_LLM_MODEL=llama3.1 \
RAG_EMBEDDINGS_PROVIDER=ollama RAG_EMBEDDINGS_MODEL=bge-m3 rag-eval run --retriever hybrid

# Anthropic generator, evaluated with an LLM faithfulness judge
RAG_LLM_PROVIDER=anthropic RAG_LLM_MODEL=<model> RAG_LLM_API_KEY=... rag-eval run --judge

# Any OpenAI-compatible server (OpenAI, vLLM, LM Studio, LiteLLM)
RAG_LLM_PROVIDER=openai RAG_LLM_BASE_URL=http://localhost:4000/v1 RAG_LLM_MODEL=<model> make dev
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness plus the active chunker, retriever, generator and stores |
| `GET` | `/documents` | Indexed documents |
| `POST` | `/ingest` | Add documents as JSON (`{"documents": [{"title", "text", "lang"}], "replace": false}`); re-indexes lazily |
| `POST` | `/ingest/files` | Multipart upload of `.md`, `.txt` or `.pdf` files (size-capped) |
| `POST` | `/ask` | `{"question", "k"?, "chunker"?, "retriever"?}` → answer, citations, ranked chunks with scores, timings |
| `GET` | `/eval/latest` | Latest `rag-eval run` report with per-question detail |
| `GET` | `/eval/latest/compare` | Latest `rag-eval compare` grid |
| `GET` | `/` | Web UI |

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "দিনে কতবার ক্যাশ আউট করা যায়?"}' | jq '.answer, .citations'
# "দিনে সর্বোচ্চ ৩০,০০০ টাকা এবং সর্বোচ্চ ৫ বার ক্যাশ আউট করা যায়। [bn-cash-out#2]"
# ["bn-cash-out#2"]
```

CLI:

```bash
rag-eval run --chunker sentence --retriever bm25 --show-misses   # one configuration, saved to runs/
rag-eval compare --chunkers recursive,sentence --retrievers bm25,hybrid --markdown table.md
rag-eval ask "How many cards can I link?"
rag-eval export-index _site/index.json                           # browser demo index
```

## Project Structure

```
rag_eval_lab/
├── text.py               # normalisation, bilingual tokenizer, sentence splitter
├── documents.py          # Document/Chunk models, md/txt/pdf loaders
├── chunking.py           # fixed, recursive, sentence-window chunkers (char spans)
├── retrieval/
│   ├── bm25.py           # BM25 with an inverted index
│   ├── embeddings.py     # hashed TF-IDF embedder, OpenAI-compatible + Ollama adapters
│   ├── store.py          # VectorStore protocol: numpy, pgvector
│   ├── dense.py · hybrid.py (RRF) · rerank.py
│   └── __init__.py       # build_retriever(name)
├── generation/
│   ├── extractive.py     # offline answerer with abstention
│   └── llm.py            # providers, LLM answerer, LLM judge
├── evaluation/
│   ├── dataset.py        # golden set, evidence → span resolution
│   ├── metrics.py        # retrieval + answer metrics
│   └── runner.py         # run / compare / reports
├── pipeline.py           # RAGPipeline, KnowledgeBase (cached per configuration)
├── api/                  # FastAPI app + schemas
└── cli.py                # rag-eval
data/corpus/              # 29 Shapla Pay help articles (24 EN, 5 BN)
data/golden.jsonl         # 70 questions with verbatim evidence and reference answers
web/                      # UI, bm25.js (browser port), node tests
scripts/build_site.sh     # GitHub Pages build
tests/                    # 85 pytest tests
```

## Testing

```bash
make test        # pytest: 85 tests
make test-web    # node --test: 7 tests for the browser BM25/tokenizer/answerer
make lint        # ruff check + format check
```

Covered: Bangla tokenisation edge cases (vowel signs, digit folding, South Asian digit grouping, suffixes); loaders including a hand-built PDF; span exactness for every chunk of every chunker on the real corpus; BM25, hashed embeddings, RRF arithmetic, reranking and the pgvector SQL (fake connection); hand-worked recall/MRR/nDCG cases including the duplicate-chunk rule; citation and claim parsing, faithfulness; provider request shapes for OpenAI-compatible, Anthropic and Ollama over `httpx.MockTransport`; golden-set integrity (every evidence passage resolves); a retrieval regression guard (Hit@5 >= 0.9); the API end to end; the CLI; and JS/Python parity: the browser tokenizer over the full corpus, and the browser BM25 + answerer against Python for all 70 golden questions.

## Roadmap

- Query expansion (synonyms, multi-query) measured on the paraphrase slice, starting with the `q053` miss
- A held-out split of the golden set so thresholds are tuned on dev and reported on test
- Cross-encoder reranker behind the existing `Reranker` protocol, compared against the lexical one
- Real-embedding runs (bge-m3 via Ollama) in the README table next to the hashed baseline
- Persistent BM25 index so the API does not re-index on restart

## License

MIT — see [LICENSE](LICENSE).
