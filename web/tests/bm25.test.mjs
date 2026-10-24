import assert from "node:assert/strict";
import { test } from "node:test";

import { ABSTAIN_TEXT, BM25, extractiveAnswer, makeTokenizer, normalize, splitSentences, stemEn } from "../bm25.js";

const lexicon = {
  en_stopwords: ["the", "is", "what", "a", "of", "at", "for", "how", "do", "i", "my"],
  bn_stopwords: ["কত", "কি"],
  bn_suffixes: ["ের", "র", "ে"],
};
const tokenize = makeTokenizer(lexicon);

const chunk = (id, text) => ({ id, title: "", heading: "", text, terms: tokenize(text) });
const chunks = [
  chunk("fees#0", "Cash out at an agent costs 1.39% using the app. The ATM charge is 1.5%."),
  chunk("pin#0", "After 3 wrong PIN attempts the account is locked for 24 hours."),
  chunk("bn#0", "ক্যাশ আউট চার্জ প্রতি হাজারে ১৩.৯০ টাকা।"),
];

test("normalize folds Bangla digits and strips zero-width characters", () => {
  assert.equal(normalize("৳৫,০০০\u200c"), "৳5,000");
});

test("tokenizer removes stopwords, digit separators and stems", () => {
  assert.deepEqual(tokenize("What is the fee for 25,000 payments?"), ["fee", "25000", "payment"]);
  assert.deepEqual(tokenize("এজেন্টের চার্জ কত"), ["এজেন্ট", "চার্জ"]);
});

test("stemEn mirrors the Python rules", () => {
  assert.equal(stemEn("policies"), "policy");
  assert.equal(stemEn("charging"), "charg");
  assert.equal(stemEn("boxes"), "box");
  assert.equal(stemEn("access"), "access");
  assert.equal(stemEn("fee"), "fee");
});

test("splitSentences handles the danda and list items", () => {
  assert.deepEqual(splitSentences("One. Two?\n- three\n- four। বাক্য"), ["One.", "Two?", "- three", "- four।", "বাক্য"]);
});

test("BM25 ranks the matching chunk first and skips unknown terms", () => {
  const bm25 = new BM25(chunks);
  assert.equal(bm25.search(tokenize("wrong PIN locked"), 3)[0].chunk.id, "pin#0");
  assert.equal(bm25.search(tokenize("ক্যাশ আউট চার্জ"), 3)[0].chunk.id, "bn#0");
  assert.deepEqual(bm25.search(tokenize("cryptocurrency"), 3), []);
});

test("extractive answer cites the supporting chunk", () => {
  const bm25 = new BM25(chunks);
  const q = "How long is the account locked after wrong PIN attempts?";
  const answer = extractiveAnswer(q, bm25.search(tokenize(q), 3), tokenize);
  assert.equal(answer.abstained, false);
  assert.deepEqual(answer.citations, ["pin#0"]);
  assert.match(answer.text, /24 hours\. \[pin#0\]$/);
});

test("extractive answer abstains when nothing matches", () => {
  const answer = extractiveAnswer("Do you offer cryptocurrency trading?", [], tokenize);
  assert.equal(answer.text, ABSTAIN_TEXT);
  assert.equal(answer.abstained, true);
});
