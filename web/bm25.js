// Client-side retrieval for the offline demo: a port of rag_eval_lab/text.py (tokeniser),
// retrieval/bm25.py and generation/extractive.py. Stopwords and Bangla suffixes are read
// from the exported index so the browser and the Python package share one source of truth.

const BANGLA_DIGITS = "০১২৩৪৫৬৭৮৯";
const ZERO_WIDTH_RE = /[\u200b\u200c\u200d\ufeff]/g;
const NUM_SEP_RE = /(?<=\d),(?=\d{2,3}\b)/g;
const TOKEN_RE = /[a-z0-9]+|[\u0980-\u09f1\u09f4-\u09ff]+/g;
const SENTENCE_RE = /(?<=[.!?।])\s+|\n\s*\n|\n(?=\s*(?:[-*]\s|\d+\.\s|#))/;
const MARKUP_RE = /^\s*(?:[-*]\s+|\d+\.\s+)|\*\*|__|`/g;
export const ABSTAIN_TEXT = "I could not find this in the knowledge base.";

export function normalize(text) {
  let out = text.normalize("NFC").replace(ZERO_WIDTH_RE, "");
  for (let i = 0; i < 10; i++) out = out.split(BANGLA_DIGITS[i]).join(String(i));
  return out;
}

export function stemEn(tok) {
  if (/^\d+$/.test(tok) || tok.length <= 3) return tok;
  if (tok.endsWith("ies") && tok.length > 4) return tok.slice(0, -3) + "y";
  if (tok.endsWith("ing") && tok.length > 5) return tok.slice(0, -3);
  if (tok.endsWith("ed") && tok.length > 4) return tok.slice(0, -2);
  if (tok.endsWith("es") && "sxz".includes(tok[tok.length - 3]) && tok.length > 4) return tok.slice(0, -2);
  if (tok.endsWith("s") && !tok.endsWith("ss")) return tok.slice(0, -1);
  return tok;
}

// Lengths are compared in UTF-16 code units, which equal code points for the Bengali block.
export function stemBn(tok, suffixes) {
  for (const suffix of suffixes) {
    if (tok.endsWith(suffix) && tok.length - suffix.length >= 2) return tok.slice(0, -suffix.length);
  }
  return tok;
}

export function makeTokenizer(lexicon) {
  const en = new Set(lexicon.en_stopwords);
  const bn = new Set(lexicon.bn_stopwords);
  const suffixes = lexicon.bn_suffixes;
  return function tokenize(text) {
    const lowered = normalize(text).toLowerCase().replace(NUM_SEP_RE, "");
    const out = [];
    for (const tok of lowered.match(TOKEN_RE) || []) {
      const bangla = tok[0] >= "\u0980";
      if ((bangla ? bn : en).has(tok)) continue;
      out.push(bangla ? stemBn(tok, suffixes) : stemEn(tok));
    }
    return out;
  };
}

export function splitSentences(text) {
  return text.split(SENTENCE_RE).map((s) => (s || "").trim()).filter(Boolean);
}

export class BM25 {
  constructor(chunks, { k1 = 1.5, b = 0.75 } = {}) {
    this.k1 = k1;
    this.b = b;
    this.chunks = chunks;
    this.postings = new Map();
    this.lengths = chunks.map((c) => c.terms.length);
    chunks.forEach((c, i) => {
      const tf = new Map();
      for (const t of c.terms) tf.set(t, (tf.get(t) || 0) + 1);
      for (const [t, n] of tf) {
        if (!this.postings.has(t)) this.postings.set(t, []);
        this.postings.get(t).push([i, n]);
      }
    });
    const n = chunks.length;
    this.avgdl = n ? this.lengths.reduce((a, x) => a + x, 0) / n : 0;
    this.idf = new Map();
    for (const [t, p] of this.postings) this.idf.set(t, Math.log(1 + (n - p.length + 0.5) / (p.length + 0.5)));
  }

  search(queryTerms, k = 5) {
    const acc = new Map();
    for (const term of new Set(queryTerms)) {
      const idf = this.idf.get(term);
      if (idf === undefined) continue;
      for (const [i, tf] of this.postings.get(term)) {
        const norm = 1 - this.b + (this.b * this.lengths[i]) / (this.avgdl || 1);
        acc.set(i, (acc.get(i) || 0) + (idf * tf * (this.k1 + 1)) / (tf + this.k1 * norm));
      }
    }
    return [...acc.entries()]
      .sort((a, b) => b[1] - a[1] || a[0] - b[0])
      .slice(0, k)
      .map(([i, score]) => ({ chunk: this.chunks[i], score: Math.round(score * 1e6) / 1e6 }));
  }
}

// Same algorithm and defaults as ExtractiveAnswerer in generation/extractive.py.
export function extractiveAnswer(question, hits, tokenize, opts = {}) {
  const { maxContexts = 3, maxSentences = 2, minCoverage = 0.4, contextWeight = 0.4, rankPrior = 0.08 } = opts;
  const qTerms = new Set(tokenize(question));
  const cands = [];
  hits.slice(0, maxContexts).forEach((hit, rank) => {
    const around = new Set([...hit.chunk.terms, ...tokenize(hit.chunk.text)]);
    for (const sentence of splitSentences(hit.chunk.text)) {
      if (sentence.trimStart().startsWith("#")) continue;
      const text = sentence.replace(MARKUP_RE, "").trim();
      if (text) cands.push({ text, id: hit.chunk.id, terms: new Set(tokenize(text)), around, rank });
    }
  });
  if (!qTerms.size || !cands.length) return { text: ABSTAIN_TEXT, citations: [], abstained: true };
  const df = new Map();
  for (const c of cands) for (const t of c.terms) if (qTerms.has(t)) df.set(t, (df.get(t) || 0) + 1);
  const weight = new Map([...qTerms].map((t) => [t, Math.log(1 + cands.length / (1 + (df.get(t) || 0)))]));
  const total = [...weight.values()].reduce((a, x) => a + x, 0) || 1;
  const coverage = (c) => {
    let got = 0;
    for (const t of qTerms) {
      if (c.terms.has(t)) got += weight.get(t);
      else if (c.around.has(t)) got += contextWeight * weight.get(t);
    }
    return got / total;
  };
  const score = (c) => coverage(c) - rankPrior * c.rank;
  const ranked = cands.map((c, i) => ({ c, i, s: score(c) })).sort((a, b) => b.s - a.s || a.i - b.i).map((x) => x.c);
  if (coverage(ranked[0]) < minCoverage) return { text: ABSTAIN_TEXT, citations: [], abstained: true };
  const best = score(ranked[0]);
  const picked = [ranked[0]];
  const covered = new Set([...ranked[0].terms].filter((t) => qTerms.has(t)));
  for (const cand of ranked.slice(1)) {
    if (picked.length >= maxSentences) break;
    const fresh = [...cand.terms].filter((t) => qTerms.has(t) && !covered.has(t));
    if (score(cand) >= 0.85 * best && fresh.length) {
      picked.push(cand);
      fresh.forEach((t) => covered.add(t));
    }
  }
  return {
    text: picked.map((c) => `${c.text} [${c.id}]`).join(" "),
    citations: [...new Set(picked.map((c) => c.id))],
    abstained: false,
  };
}
