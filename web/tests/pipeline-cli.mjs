// Parity helper for tests/test_web_parity.py: loads an exported index, answers each
// question from stdin with the browser BM25 + extractive answerer, prints ids and answers.
import { readFileSync } from "node:fs";

import { BM25, extractiveAnswer, makeTokenizer } from "../bm25.js";

const index = JSON.parse(readFileSync(process.argv[2], "utf8"));
const tokenize = makeTokenizer(index.lexicon);
const bm25 = new BM25(index.chunks);

let raw = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (d) => (raw += d));
process.stdin.on("end", () => {
  const out = JSON.parse(raw).map((question) => {
    const hits = bm25.search(tokenize(question), 5);
    return { ids: hits.map((h) => h.chunk.id), answer: extractiveAnswer(question, hits, tokenize).text };
  });
  process.stdout.write(JSON.stringify(out));
});
