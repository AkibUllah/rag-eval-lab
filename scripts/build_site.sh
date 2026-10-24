#!/usr/bin/env bash
# Build the GitHub Pages offline demo: the web UI, a pre-tokenised chunk index for
# client-side BM25, and a fresh evaluation grid for the Evaluation tab.
#   scripts/build_site.sh [output dir]      (default: _site; needs `rag-eval` on PATH)
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-_site}"
tmp_runs="$(mktemp -d)"
trap 'rm -rf "$tmp_runs"' EXIT

rm -rf "$out"
mkdir -p "$out"
cp web/index.html web/app.js web/bm25.js web/styles.css "$out/"
rag-eval export-index "$out/index.json" --chunker recursive
rag-eval compare --runs-dir "$tmp_runs" --json "$out/eval.json" > /dev/null
echo "built $out: $(ls "$out" | tr '\n' ' ')"
