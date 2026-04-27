const METHODS = [
  {
    key: "tfidf",
    label: "TF-IDF",
    color: "#6366f1",
    bg: "#ede9fe",
    tagColor: "#6366f1",
    what: "TF-IDF stands for Term Frequency / Inverse Document Frequency. It scores documents by looking at two things: how often a query word appears in a document (TF), and how rare that word is across the whole corpus (IDF). Rare words that appear often in a document get a high score.",
    analogy: "You search for 'neural network' in a library. TF-IDF rewards books where that phrase appears often, but only if it is rare across all books. A word like 'the' gets almost zero score because it appears in every single book.",
    pros: [
      "Extremely fast. No training needed, works instantly on any text.",
      "Fully transparent. You can trace exactly why a document ranked high.",
      "Great for precise terms like model names, error codes, or technical jargon.",
    ],
    cons: [
      "No semantic understanding. The words 'car' and 'automobile' are treated as completely unrelated.",
      "Short documents with high keyword density rank higher than long, detailed ones.",
      "Breaks on synonyms, paraphrases, and natural language queries.",
    ],
    when: "Use TF-IDF when queries are precise and technical (product codes, error messages, specific model names) and when speed matters more than semantic understanding.",
  },
  {
    key: "bm25",
    label: "BM25",
    color: "#06b6d4",
    bg: "#cffafe",
    tagColor: "#0891b2",
    what: "BM25 (Best Match 25) is the upgraded version of TF-IDF. It fixes two well-known problems: term frequency saturation (extra occurrences of a word give diminishing returns) and document length normalization (longer documents are not penalized or rewarded just for being long).",
    analogy: "A short tweet says 'chocolate chocolate chocolate' and a 10-page cookbook chapter covers chocolate in depth. TF-IDF ranks the tweet higher because 'chocolate' is 30% of its words. BM25 sees through that: after a few occurrences the score stops climbing, and it adjusts for document length.",
    pros: [
      "Fixes TF-IDF's length bias so longer, richer documents get a fair chance.",
      "Term saturation stops keyword spamming from gaming the ranking.",
      "Still very fast, no GPU needed, works on any text.",
      "Industry standard for keyword retrieval, used by Elasticsearch, OpenSearch, and Solr.",
    ],
    cons: [
      "Still keyword-only. No semantic understanding.",
      "Fails when the user's words differ from the document's words (synonyms, paraphrases).",
      "Sensitive to typos and word variations.",
    ],
    when: "BM25 is the best pure keyword method in almost every case. Use it instead of TF-IDF by default. Most production RAG systems use BM25 as their keyword retrieval backbone.",
  },
  {
    key: "rrf",
    label: "Hybrid RRF",
    color: "#f59e0b",
    bg: "#fef9c3",
    tagColor: "#b45309",
    what: "Reciprocal Rank Fusion (RRF) is not a retrieval method by itself. It is a fusion strategy. It takes the ranked lists from two or more methods and merges them into a single list. Documents that rank well across multiple methods float to the top. Scores are based on rank position, not raw scores, so it works even when the two methods use completely different scales.",
    analogy: "Two experts recommend restaurants. One is a food critic (semantic search), the other is a Yelp analyst (keyword search). Any restaurant that both experts rank highly beats restaurants that only one of them likes, regardless of how different their rating systems are.",
    pros: [
      "Consistently outperforms any single method across diverse query types.",
      "Score-scale agnostic. No need to normalize scores from different systems.",
      "Simple to tune (one parameter: k, typically set to 60).",
      "Works with any two or more ranked lists.",
    ],
    cons: [
      "Requires running multiple retrieval methods, which is slower than running just one.",
      "Without a true semantic method (embedding model), the hybrid is still keyword-only underneath.",
      "A document ranked #1 by one method and #50 by another can score lower than documents ranked #10 in both.",
    ],
    when: "Use hybrid RRF in production RAG whenever you can run two methods. It is the standard approach in Weaviate, Pinecone, and most modern vector databases. In this playground it fuses TF-IDF and BM25. In a full system you would fuse keyword search with dense embedding search.",
  },
];

const POST_RETRIEVAL = [
  {
    key: "stuffing",
    label: "Stuffing",
    color: "#10b981",
    bg: "#dcfce7",
    what: "The simplest strategy. All retrieved chunks are concatenated and placed directly into the LLM prompt as context. The LLM reads everything at once and produces a single answer.",
    pros: [
      "One LLM call. Cheapest and fastest.",
      "The LLM can reason across all chunks at the same time.",
      "Works perfectly when chunks fit within the context window.",
    ],
    cons: [
      "Fails when total chunk length exceeds the context window limit.",
      "LLM performance degrades on very long contexts (lost-in-the-middle problem).",
    ],
    when: "Your default choice. Use stuffing whenever retrieved chunks fit comfortably inside the context window. For most use cases with K up to 10 short chunks, this is all you need.",
  },
  {
    key: "mapreduce",
    label: "Map Reduce",
    color: "#6366f1",
    bg: "#ede9fe",
    what: "Map Reduce runs the LLM once per chunk (the Map step), asking it to extract relevant information from that chunk alone. Then all individual answers are combined and fed to a final LLM call that synthesizes them into a single response (the Reduce step).",
    analogy: "You have 20 research papers to summarize. Instead of reading all 20 at once, you give each one to a different research assistant (Map). Each writes a one-page summary. You take those 20 summaries and write the final report yourself (Reduce).",
    pros: [
      "Handles any number of chunks regardless of context window size.",
      "Each chunk is processed independently so the Map step can run in parallel.",
      "Great for summarization tasks across many documents.",
    ],
    cons: [
      "Multiple LLM calls means higher cost and latency.",
      "Relationships between chunks can be lost because the LLM never sees them together.",
      "The Reduce step may miss nuance that was only visible across multiple chunks.",
    ],
    when: "Use Map Reduce when you have many long documents that collectively exceed the context window and the task is summarization or extraction rather than precise question answering.",
  },
  {
    key: "refine",
    label: "Refine",
    color: "#f59e0b",
    bg: "#fef9c3",
    what: "Refine processes chunks one at a time in sequence. The LLM reads the first chunk and produces an initial answer. It then reads the second chunk and updates the answer with new information. This continues until all chunks have been processed.",
    analogy: "You are writing a news article. You read the first source and write a draft. You read the second source and revise it. You read the third and revise again. By the end, your article reflects everything you read, processed in order.",
    pros: [
      "Produces high-quality answers by building context progressively.",
      "Handles more chunks than stuffing without hitting the context window.",
      "Information from later chunks can correct mistakes made on earlier ones.",
    ],
    cons: [
      "Strictly sequential. Cannot be parallelized.",
      "Slowest strategy: N chunks means N LLM calls in sequence.",
      "Earlier chunks have disproportionate influence on the final answer.",
      "Expensive for large document sets.",
    ],
    when: "Use Refine when answer quality matters more than speed or cost, and when the order of information is important, for example processing a long document chronologically or refining a legal analysis chunk by chunk.",
  },
];

function MethodCard({ method }) {
  return (
    <div className="hiw-card">
      <div className="hiw-card-header">
        <span className="hiw-tag" style={{ background: method.bg, color: method.tagColor }}>
          {method.label}
        </span>
      </div>

      <h3 className="hiw-card-title" style={{ color: method.color }}>{method.label}</h3>

      <section className="hiw-section">
        <h4 className="hiw-section-title">What is it?</h4>
        <p className="hiw-text">{method.what}</p>
      </section>

      {method.analogy && (
        <section className="hiw-section hiw-analogy">
          <h4 className="hiw-section-title">Think of it like this</h4>
          <p className="hiw-text">{method.analogy}</p>
        </section>
      )}

      <div className="hiw-pro-con">
        <section className="hiw-section">
          <h4 className="hiw-section-title hiw-pro">Pros</h4>
          <ul className="hiw-list hiw-list--pro">
            {method.pros.map((p, i) => <li key={i}>{p}</li>)}
          </ul>
        </section>
        <section className="hiw-section">
          <h4 className="hiw-section-title hiw-con">Cons</h4>
          <ul className="hiw-list hiw-list--con">
            {method.cons.map((c, i) => <li key={i}>{c}</li>)}
          </ul>
        </section>
      </div>

      <section className="hiw-section hiw-when">
        <h4 className="hiw-section-title">When to use it</h4>
        <p className="hiw-text">{method.when}</p>
      </section>
    </div>
  );
}

export default function HowItWorks() {
  return (
    <div className="hiw-root">

      <div className="hiw-hero">
        <h2 className="hiw-hero-title">How RAG Retrieval Works</h2>
        <p className="hiw-hero-sub">
          A RAG pipeline has two phases. First, <strong>retrieval</strong>: finding the right chunks
          from your corpus. Then, <strong>generation</strong>: feeding those chunks to an LLM to produce the
          final answer. Both phases have multiple strategies with real trade-offs.
        </p>
      </div>

      <div className="hiw-phase-label">
        <div className="hiw-phase-line" />
        <span>Phase 1 — Retrieval Methods</span>
        <div className="hiw-phase-line" />
      </div>

      <div className="hiw-cards">
        {METHODS.map((m) => <MethodCard key={m.key} method={m} />)}
      </div>

      <div className="hiw-phase-label" style={{ marginTop: 48 }}>
        <div className="hiw-phase-line" />
        <span>Phase 2 — Post-Retrieval Strategies</span>
        <div className="hiw-phase-line" />
      </div>

      <div className="hiw-cards">
        {POST_RETRIEVAL.map((m) => <MethodCard key={m.key} method={m} />)}
      </div>

    </div>
  );
}
