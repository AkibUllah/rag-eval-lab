import { BM25, extractiveAnswer, makeTokenizer } from "./bm25.js";

const SAMPLES = [
  "What is the cash out charge when using the app?",
  "How long do I have to raise a dispute?",
  "My phone was stolen. What should I do?",
  "What profit rate does a 3 year DPS give?",
  "পিন ভুলে গেলে কীভাবে রিসেট করব?",
  "দিনে কতবার ক্যাশ আউট করা যায়?",
  "Does Shapla Pay offer cryptocurrency trading?",
];

const METRICS = [
  ["recall@k", "Recall@k"], ["mrr", "MRR"], ["ndcg@k", "nDCG@k"], ["hit@1", "Hit@1"],
  ["citation_precision", "Cit. prec"], ["f1", "Answer F1"], ["faithfulness", "Faithful"],
  ["abstain_accuracy", "Abstain acc"],
];

const $ = (id) => document.getElementById(id);
const state = { mode: null, offline: null };

const escapeHtml = (s) =>
  s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

async function getJson(url, init) {
  const res = await fetch(url, init);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

async function detectMode() {
  try {
    const health = await getJson("health", { headers: { Accept: "application/json" } });
    if (health.status === "ok") {
      state.mode = "live";
      $("mode").textContent = `live API · ${health.documents} docs · ${health.llm_provider} generator`;
      $("mode").classList.add("live");
      $("retriever").value = health.retriever;
      return;
    }
  } catch {
    /* no backend: fall through to the static index */
  }
  const index = await getJson("index.json");
  const tokenize = makeTokenizer(index.lexicon);
  state.mode = "offline";
  state.offline = { tokenize, bm25: new BM25(index.chunks), index };
  $("mode").textContent = `offline demo · ${index.chunks.length} chunks · BM25 in your browser`;
  const select = $("retriever");
  select.value = "bm25";
  [...select.options].forEach((o) => (o.disabled = o.value !== "bm25"));
}

async function ask(question, retriever) {
  if (state.mode === "live") {
    return getJson("ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, retriever }),
    });
  }
  const { tokenize, bm25, index } = state.offline;
  const t0 = performance.now();
  const hits = bm25.search(tokenize(question), 5);
  const t1 = performance.now();
  const answer = extractiveAnswer(question, hits, tokenize);
  const cited = new Set(answer.citations);
  return {
    answer: answer.text,
    abstained: answer.abstained,
    citations: answer.citations,
    generator: "extractive (browser)",
    chunker: index.chunker,
    retriever: "bm25",
    timings: { retrieval_ms: t1 - t0, generation_ms: performance.now() - t1 },
    hits: hits.map((h) => ({ ...h.chunk, score: h.score, cited: cited.has(h.chunk.id) })),
  };
}

function renderAnswer(data) {
  const html = escapeHtml(data.answer).replace(
    /\[([a-z0-9][a-z0-9-]*#\d+)\]/g,
    (_, id) => `<a class="cite" href="#hit-${id.replace("#", "-")}">${id}</a>`,
  );
  $("answer-text").innerHTML = html;
  const t = data.timings;
  $("answer-meta").textContent =
    `${data.generator} · ${data.chunker} chunks · ${data.retriever} · ` +
    `retrieval ${t.retrieval_ms.toFixed(1)} ms · generation ${t.generation_ms.toFixed(1)} ms` +
    (data.abstained ? " · abstained" : "");
  $("hits").innerHTML = data.hits
    .map(
      (h, i) => `<li class="hit${h.cited ? " cited" : ""}" id="hit-${h.id.replace("#", "-")}">
        <header><span>${i + 1}. <code>${escapeHtml(h.id)}</code> · ${escapeHtml(h.heading || h.title)}</span>
        <span>score ${Number(h.score).toFixed(4)}${h.cited ? " · cited" : ""}</span></header>
        <pre>${escapeHtml(h.text.trim())}</pre></li>`,
    )
    .join("");
  $("result").hidden = false;
}

async function onSubmit(event) {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question || !state.mode) return;
  $("error").hidden = true;
  try {
    renderAnswer(await ask(question, $("retriever").value));
  } catch (err) {
    $("error").textContent = `Request failed: ${err.message}`;
    $("error").hidden = false;
  }
}

async function loadEval() {
  try {
    const data = await getJson(state.mode === "live" ? "eval/latest/compare" : "eval.json");
    const k = data.k;
    const best = Object.fromEntries(
      METRICS.map(([key]) => [key, Math.max(...data.runs.map((r) => r.metrics[key] ?? -1))]),
    );
    const head = ["Chunker", "Retriever", "Chunks", ...METRICS.map(([, l]) => l.replace("k", k)), "Ctx words"];
    const rows = data.runs.map((r) => {
      const cells = METRICS.map(([key]) => {
        const v = r.metrics[key];
        const cls = v !== null && Math.abs(v - best[key]) < 1e-9 ? ' class="best"' : "";
        return `<td${cls}>${v === null ? "–" : v.toFixed(3)}</td>`;
      });
      return `<tr><td>${r.chunker}</td><td>${r.retriever}</td><td>${r.n_chunks}</td>${cells.join("")}<td>${Math.round(r.metrics.context_words)}</td></tr>`;
    });
    $("eval-table").innerHTML =
      `<thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody>`;
    const q = data.runs[0]?.metrics.questions ?? 0;
    $("eval-meta").textContent = `${data.runs.length} configurations · ${q} questions · k=${k} · generated ${data.created_at.slice(0, 10)}`;
  } catch {
    $("eval-meta").textContent = "No comparison report yet. Run `rag-eval compare` and reload.";
  }
}

function switchTab(name) {
  document.querySelectorAll(".tab[data-tab]").forEach((t) => {
    const active = t.dataset.tab === name;
    t.classList.toggle("active", active);
    t.setAttribute("aria-selected", String(active));
  });
  $("tab-ask").hidden = name !== "ask";
  $("tab-eval").hidden = name !== "eval";
  if (name === "eval") loadEval();
}

$("samples").innerHTML = SAMPLES.map((q) => `<button type="button">${escapeHtml(q)}</button>`).join("");
$("samples").addEventListener("click", (e) => {
  if (e.target.tagName !== "BUTTON") return;
  $("question").value = e.target.textContent;
  $("ask-form").requestSubmit();
});
$("ask-form").addEventListener("submit", onSubmit);
document.querySelectorAll(".tab[data-tab]").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));

detectMode().catch((err) => {
  $("mode").textContent = "unavailable";
  $("error").textContent = `Could not reach the API or load index.json (${err.message}).`;
  $("error").hidden = false;
});
