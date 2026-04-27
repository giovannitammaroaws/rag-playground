function extractSentences(text) {
  return text.split(/(?<=[.!?])\s+/).filter((s) => s.length > 20);
}

function buildAnswer(chunks, query) {
  const sentences = chunks.flatMap((c) => extractSentences(c.text));
  const top = sentences.slice(0, 4).join(" ");
  return `Based on the retrieved documents: ${top}`;
}

export function runMockLLM(rankedDocs, k, strategy) {
  const chunks = rankedDocs.slice(0, Math.min(k, 5));

  if (strategy === "stuffing") {
    const context = chunks.map((c) => c.text).join("\n\n");
    const answer = buildAnswer(chunks, "");
    return {
      strategy: "Stuffing",
      steps: [
        { label: "Context sent to LLM", content: context },
        { label: "LLM answer", content: answer },
      ],
      answer,
      llmCalls: 1,
    };
  }

  if (strategy === "mapreduce") {
    const mapped = chunks.map((c, i) => ({
      label: `Chunk ${i + 1} summary`,
      content: extractSentences(c.text)[0] || c.text.slice(0, 120),
    }));
    const combined = mapped.map((m) => m.content).join(" ");
    const answer = `After processing each document independently and combining results: ${combined}`;
    return {
      strategy: "Map Reduce",
      steps: [
        ...mapped,
        { label: "Reduce — final synthesis", content: answer },
      ],
      answer,
      llmCalls: chunks.length + 1,
    };
  }

  if (strategy === "refine") {
    const steps = [];
    let current = extractSentences(chunks[0].text)[0] || chunks[0].text.slice(0, 120);
    steps.push({ label: "Initial answer from chunk 1", content: current });
    for (let i = 1; i < chunks.length; i++) {
      const newInfo = extractSentences(chunks[i].text)[0] || chunks[i].text.slice(0, 120);
      current = `${current} Additionally, ${newInfo.charAt(0).toLowerCase()}${newInfo.slice(1)}`;
      steps.push({ label: `Refined with chunk ${i + 1}`, content: current });
    }
    return {
      strategy: "Refine",
      steps,
      answer: current,
      llmCalls: chunks.length,
    };
  }
}
