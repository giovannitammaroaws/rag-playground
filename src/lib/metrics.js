// ── Corpus metrics (relevant = explicit id list, spam excluded already) ───────
export function computeCorpusMetrics(rankedChunks, relevantIds, k) {
  const topK = rankedChunks.slice(0, k);
  const rel  = new Set(relevantIds);
  const relevantFound  = topK.filter((c) => rel.has(c.id)).length;
  const totalRelevant  = relevantIds.length;
  const precision = k > 0 ? relevantFound / k : 0;
  const recall    = totalRelevant > 0 ? relevantFound / totalRelevant : 0;
  const f1        = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  return { precision, recall, f1, relevantFound, totalRelevant };
}

// Spam articles are category-correct but content-worthless.
// A good retriever should NOT return them — exclude from ground truth.
const isRelevant = (doc, category) =>
  doc.category === category && !doc.note?.startsWith("keyword spam");

export function computeMetrics(rankedDocs, relevantCategory, k) {
  const topK = rankedDocs.slice(0, k);
  const totalRelevant = rankedDocs.filter((d) => isRelevant(d, relevantCategory)).length;

  const relevantFound = topK.filter((d) => isRelevant(d, relevantCategory)).length;

  const precision = k > 0 ? relevantFound / k : 0;
  const recall = totalRelevant > 0 ? relevantFound / totalRelevant : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  return {
    precision: parseFloat(precision.toFixed(3)),
    recall: parseFloat(recall.toFixed(3)),
    f1: parseFloat(f1.toFixed(3)),
    relevantFound,
    totalRelevant,
  };
}
