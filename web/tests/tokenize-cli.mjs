// Parity helper for tests/test_web_parity.py: reads {lexicon, texts} JSON on stdin and
// prints the browser tokeniser's output for each text.
import { makeTokenizer, splitSentences } from "../bm25.js";

let raw = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (d) => (raw += d));
process.stdin.on("end", () => {
  const { lexicon, texts } = JSON.parse(raw);
  const tokenize = makeTokenizer(lexicon);
  process.stdout.write(JSON.stringify(texts.map((t) => ({ tokens: tokenize(t), sentences: splitSentences(t) }))));
});
