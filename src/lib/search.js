import DOC_EMBEDDINGS   from "../data/embeddings.json";
import QUERY_EMBEDDINGS  from "../data/query_embeddings.json";
import CORPUS_EMB        from "../data/corpus_embeddings.json";

const STOP_WORDS = new Set([
  "a","an","the","and","or","but","in","on","at","to","for","of","with",
  "by","from","is","are","was","were","be","been","being","have","has","had",
  "do","does","did","will","would","could","should","may","might","shall",
  "can","its","it","this","that","these","those","as","not","no","if","so",
]);

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOP_WORDS.has(t));
}

function cosineSim(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // already L2-normalised by bge
}

// ── TF-IDF ───────────────────────────────────────────────────────────────────
export function tfidfSearch(query, docs) {
  const qTokens = tokenize(query);
  const N = docs.length;
  const df = {};
  const docTokens = docs.map((d) => {
    const tokens = tokenize((d.title || "") + " " + d.text);
    new Set(tokens).forEach((t) => { df[t] = (df[t] || 0) + 1; });
    return tokens;
  });
  return docs
    .map((doc, i) => {
      const tf = {};
      const tokens = docTokens[i];
      tokens.forEach((t) => { tf[t] = (tf[t] || 0) + 1; });
      const dl = tokens.length;
      let score = 0;
      qTokens.forEach((t) => {
        if (!df[t]) return;
        score += (tf[t] || 0) / dl * Math.log((N + 1) / (df[t] + 1));
      });
      return { ...doc, score };
    })
    .sort((a, b) => b.score - a.score);
}

// ── BM25 ─────────────────────────────────────────────────────────────────────
export function bm25Search(query, docs, k1 = 1.5, b = 0.75) {
  const qTokens = tokenize(query);
  const N = docs.length;
  const df = {};
  const docTokensList = docs.map((d) => {
    const tokens = tokenize((d.title || "") + " " + d.text);
    new Set(tokens).forEach((t) => { df[t] = (df[t] || 0) + 1; });
    return tokens;
  });
  const avgdl = docTokensList.reduce((s, t) => s + t.length, 0) / N;
  return docs
    .map((doc, i) => {
      const tokens = docTokensList[i];
      const dl = tokens.length;
      const tf = {};
      tokens.forEach((t) => { tf[t] = (tf[t] || 0) + 1; });
      let score = 0;
      qTokens.forEach((t) => {
        if (!df[t]) return;
        const idf = Math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1);
        const tfNorm = ((tf[t] || 0) * (k1 + 1)) / ((tf[t] || 0) + k1 * (1 - b + b * dl / avgdl));
        score += idf * tfNorm;
      });
      return { ...doc, score };
    })
    .sort((a, b) => b.score - a.score);
}

// ── Semantic (pre-computed embeddings, articles dataset) ─────────────────────
export function semanticSearch(topicKey, docs) {
  const queryVec = QUERY_EMBEDDINGS[topicKey];
  if (!queryVec) return null;
  return docs
    .map((doc) => {
      const v = DOC_EMBEDDINGS[String(doc.id)];
      return { ...doc, score: v ? cosineSim(queryVec, v) : 0 };
    })
    .sort((a, b) => b.score - a.score);
}

// ── Semantic (corpus chunks dataset) ─────────────────────────────────────────
export function semanticSearchCorpus(topicKey, chunks) {
  const queryVec = CORPUS_EMB.queries[topicKey];
  if (!queryVec) return null;
  return chunks
    .map((chunk) => {
      const v = CORPUS_EMB.chunks[String(chunk.id)];
      return { ...chunk, score: v ? cosineSim(queryVec, v) : 0 };
    })
    .sort((a, b) => b.score - a.score);
}

// ── Hybrid BM25+Semantic (articles dataset) ──────────────────────────────────
export function hybridSearch(topicKey, query, docs, alpha = 0.5, rrfK = 60) {
  const bm25List = bm25Search(query, docs);
  const semList  = semanticSearch(topicKey, docs);
  if (!semList) return bm25List;
  const scores = {};
  bm25List.forEach((doc, rank) => {
    scores[doc.id] = (scores[doc.id] || 0) + alpha * (1 / (rrfK + rank + 1));
  });
  semList.forEach((doc, rank) => {
    scores[doc.id] = (scores[doc.id] || 0) + (1 - alpha) * (1 / (rrfK + rank + 1));
  });
  return docs.map((doc) => ({ ...doc, score: scores[doc.id] || 0 })).sort((a, b) => b.score - a.score);
}

// ── Hybrid BM25+Semantic (corpus chunks) with live alpha ─────────────────────
// alpha=1 → pure BM25   alpha=0 → pure Semantic   (Weaviate convention)
export function hybridSearchCorpus(topicKey, query, chunks, alpha = 0.5, rrfK = 60) {
  const bm25List = bm25Search(query, chunks);
  const semList  = semanticSearchCorpus(topicKey, chunks);
  if (!semList) return bm25List;
  const scores = {};
  bm25List.forEach((chunk, rank) => {
    scores[chunk.id] = (scores[chunk.id] || 0) + alpha * (1 / (rrfK + rank + 1));
  });
  semList.forEach((chunk, rank) => {
    scores[chunk.id] = (scores[chunk.id] || 0) + (1 - alpha) * (1 / (rrfK + rank + 1));
  });
  return chunks.map((c) => ({ ...c, score: scores[c.id] || 0 })).sort((a, b) => b.score - a.score);
}
