import { useState, useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis,
} from "recharts";
import POSITIONS_2D   from "./data/positions_2d.json";
import CORPUS_POS     from "./data/corpus_positions_2d.json";
import "./App.css";
import { ARTICLES, TOPICS } from "./data/articles.js";
import { CHUNKS, QUERIES } from "./data/corpus.js";
import { tfidfSearch, bm25Search, semanticSearch, hybridSearch,
         semanticSearchCorpus, hybridSearchCorpus } from "./lib/search.js";
import { computeMetrics, computeCorpusMetrics } from "./lib/metrics.js";
import { runMockLLM } from "./lib/mockLLM.js";

const GITHUB_URL = "https://github.com/giovannitammaroaws/rag-playground";
const K_OPTIONS = [3, 5, 10, 20];
const METHODS = [
  { key: "tfidf",    label: "TF-IDF",   color: "#6366f1" },
  { key: "bm25",     label: "BM25",     color: "#06b6d4" },
  { key: "semantic", label: "Semantic", color: "#10b981" },
  { key: "hybrid",   label: "Hybrid",   color: "#f59e0b" },
];
const CHART_COLORS = { precision: "#6366f1", recall: "#06b6d4", f1: "#10b981" };
const GEN_STRATEGIES = ["stuffing", "mapreduce", "refine"];
const GEN_LABELS = { stuffing: "Stuffing", mapreduce: "Map Reduce", refine: "Refine" };

const TOPIC_COLORS = {
  ai:      { bg: "#ede9fe", color: "#6366f1" },
  space:   { bg: "#cffafe", color: "#0891b2" },
  health:  { bg: "#dcfce7", color: "#15803d" },
  finance: { bg: "#fef9c3", color: "#b45309" },
  climate: { bg: "#d1fae5", color: "#059669" },
};

// ── KInput: free-text K field, applies on blur or Enter ──────────────────────
function KInput({ value, onChange, max = 100 }) {
  const [raw, setRaw] = useState(String(value));
  const commit = (str) => {
    const n = parseInt(str, 10);
    if (!isNaN(n) && n >= 1) {
      const clamped = Math.min(max, n);
      onChange(clamped);
      setRaw(String(clamped));
    } else {
      setRaw(String(value));
    }
  };
  return (
    <input type="text" inputMode="numeric" value={raw}
      onChange={(e) => setRaw(e.target.value)}
      onBlur={(e) => commit(e.target.value)}
      onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); commit(raw); } }}
      className="k-free-input" />
  );
}

// ── helpers ──────────────────────────────────────────────────────────────────

function ResultCard({ doc, rank, isRelevant }) {
  return (
    <div className={`result-card ${isRelevant ? "relevant" : "irrelevant"}`}>
      <div className="result-rank">#{rank + 1}</div>
      <div className="result-body">
        <div className="result-title">{doc.title}</div>
        <div className="result-text">{doc.text.slice(0, 100)}…</div>
      </div>
      {isRelevant !== null && (
        <div className={`result-badge ${isRelevant ? "badge-hit" : "badge-miss"}`}>
          {isRelevant ? "hit" : "miss"}
        </div>
      )}
    </div>
  );
}

function MetricBar({ label, value, color }) {
  return (
    <div className="metric-bar-row">
      <span className="metric-bar-label">{label}</span>
      <div className="metric-bar-track">
        <div className="metric-bar-fill" style={{ width: `${value * 100}%`, background: color }} />
      </div>
      <span className="metric-bar-value">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

// ── SVG diagrams ─────────────────────────────────────────────────────────────

function IconTFIDF() {
  return (
    <svg viewBox="0 0 80 56" className="method-icon">
      <rect x="4" y="36" width="12" height="16" rx="2" fill="#6366f1" opacity="0.3" />
      <rect x="20" y="20" width="12" height="32" rx="2" fill="#6366f1" opacity="0.6" />
      <rect x="36" y="8" width="12" height="44" rx="2" fill="#6366f1" />
      <rect x="52" y="28" width="12" height="24" rx="2" fill="#6366f1" opacity="0.4" />
      <text x="4" y="54" fontSize="7" fill="#94a3b8" fontFamily="monospace">tf</text>
      <text x="18" y="54" fontSize="7" fill="#94a3b8" fontFamily="monospace">nn</text>
      <text x="34" y="54" fontSize="7" fill="#94a3b8" fontFamily="monospace">ai</text>
      <text x="50" y="54" fontSize="7" fill="#94a3b8" fontFamily="monospace">ml</text>
    </svg>
  );
}

function IconBM25() {
  return (
    <svg viewBox="0 0 80 56" className="method-icon">
      <rect x="4" y="24" width="12" height="28" rx="2" fill="#ef4444" opacity="0.3" />
      <rect x="20" y="20" width="12" height="32" rx="2" fill="#06b6d4" opacity="0.9" />
      <rect x="36" y="16" width="12" height="36" rx="2" fill="#06b6d4" />
      <rect x="52" y="22" width="12" height="30" rx="2" fill="#06b6d4" opacity="0.7" />
      <line x1="2" y1="18" x2="70" y2="18" stroke="#94a3b8" strokeWidth="1" strokeDasharray="3 2" />
      <text x="72" y="21" fontSize="6" fill="#94a3b8">cap</text>
    </svg>
  );
}

function IconSemantic() {
  return (
    <svg viewBox="0 0 80 56" className="method-icon">
      <circle cx="20" cy="28" r="10" fill="#10b981" opacity="0.25" />
      <circle cx="20" cy="28" r="5" fill="#10b981" opacity="0.7" />
      <circle cx="48" cy="18" r="7" fill="#6366f1" opacity="0.3" />
      <circle cx="48" cy="18" r="3" fill="#6366f1" opacity="0.7" />
      <circle cx="58" cy="38" r="6" fill="#f59e0b" opacity="0.3" />
      <circle cx="58" cy="38" r="3" fill="#f59e0b" opacity="0.7" />
      <line x1="20" y1="28" x2="48" y2="18" stroke="#10b981" strokeWidth="1.5" strokeDasharray="3 2" />
      <line x1="20" y1="28" x2="58" y2="38" stroke="#94a3b8" strokeWidth="1" strokeDasharray="3 2" />
      <text x="13" y="46" fontSize="6" fill="#64748b">query</text>
    </svg>
  );
}

function IconHybrid() {
  return (
    <svg viewBox="0 0 80 56" className="method-icon">
      <rect x="2" y="10" width="18" height="6" rx="2" fill="#06b6d4" opacity="0.8" />
      <rect x="2" y="19" width="18" height="6" rx="2" fill="#06b6d4" opacity="0.5" />
      <rect x="2" y="28" width="18" height="6" rx="2" fill="#06b6d4" opacity="0.3" />
      <circle cx="35" cy="14" r="5" fill="#10b981" opacity="0.8" />
      <circle cx="35" cy="25" r="5" fill="#10b981" opacity="0.5" />
      <circle cx="35" cy="36" r="5" fill="#10b981" opacity="0.3" />
      <path d="M22 16 L29 15 M22 22 L29 24 M22 31 L29 35" stroke="#94a3b8" strokeWidth="1" />
      <path d="M41 14 L50 22 M41 36 L50 28" stroke="#94a3b8" strokeWidth="1" />
      <rect x="51" y="16" width="24" height="18" rx="4" fill="#f59e0b" opacity="0.9" />
      <text x="56" y="28" fontSize="7" fill="#fff" fontWeight="bold">RRF</text>
    </svg>
  );
}

function IconStuffing() {
  return (
    <svg viewBox="0 0 80 56" className="method-icon">
      <rect x="2" y="4" width="34" height="8" rx="2" fill="#10b981" opacity="0.4" />
      <rect x="2" y="15" width="34" height="8" rx="2" fill="#10b981" opacity="0.6" />
      <rect x="2" y="26" width="34" height="8" rx="2" fill="#10b981" opacity="0.8" />
      <path d="M38 20 L46 20" stroke="#94a3b8" strokeWidth="1.5" />
      <polygon points="46,17 50,20 46,23" fill="#94a3b8" />
      <rect x="51" y="10" width="26" height="28" rx="4" fill="#0f172a" />
      <text x="57" y="28" fontSize="9" fill="#fff" fontWeight="bold">LLM</text>
      <path d="M64 40 L64 48" stroke="#94a3b8" strokeWidth="1.5" />
      <polygon points="61,48 64,52 67,48" fill="#94a3b8" />
    </svg>
  );
}

function IconMapReduce() {
  return (
    <svg viewBox="0 0 80 56" className="method-icon">
      <rect x="2" y="4" width="16" height="8" rx="2" fill="#6366f1" opacity="0.6" />
      <rect x="2" y="15" width="16" height="8" rx="2" fill="#6366f1" opacity="0.6" />
      <rect x="2" y="26" width="16" height="8" rx="2" fill="#6366f1" opacity="0.6" />
      <rect x="22" y="4" width="14" height="8" rx="2" fill="#0f172a" />
      <rect x="22" y="15" width="14" height="8" rx="2" fill="#0f172a" />
      <rect x="22" y="26" width="14" height="8" rx="2" fill="#0f172a" />
      <path d="M38 8 L44 18 M38 19 L44 20 M38 30 L44 22" stroke="#94a3b8" strokeWidth="1" />
      <rect x="45" y="14" width="14" height="16" rx="3" fill="#0f172a" />
      <text x="48" y="25" fontSize="7" fill="#fff" fontWeight="bold">+</text>
      <path d="M60 22 L68 22" stroke="#94a3b8" strokeWidth="1.5" />
      <polygon points="68,19 72,22 68,25" fill="#94a3b8" />
    </svg>
  );
}

function IconRefine() {
  return (
    <svg viewBox="0 0 80 56" className="method-icon">
      <rect x="2" y="6" width="14" height="8" rx="2" fill="#f59e0b" opacity="0.6" />
      <path d="M17 10 L22 10" stroke="#94a3b8" strokeWidth="1.5" />
      <rect x="23" y="6" width="12" height="8" rx="2" fill="#0f172a" />
      <path d="M36 10 L42 10" stroke="#94a3b8" strokeWidth="1.5" />
      <rect x="43" y="6" width="14" height="8" rx="2" fill="#f59e0b" opacity="0.4" />
      <path d="M58 10 L64 20" stroke="#94a3b8" strokeWidth="1" strokeDasharray="2 2"/>
      <rect x="2" y="22" width="14" height="8" rx="2" fill="#f59e0b" opacity="0.8" />
      <path d="M17 26 L22 26" stroke="#94a3b8" strokeWidth="1.5" />
      <rect x="23" y="22" width="12" height="8" rx="2" fill="#0f172a" />
      <path d="M36 26 L42 26" stroke="#94a3b8" strokeWidth="1.5" />
      <rect x="43" y="22" width="14" height="8" rx="2" fill="#f59e0b" opacity="0.6" />
      <path d="M58 26 L64 36" stroke="#94a3b8" strokeWidth="1" strokeDasharray="2 2"/>
      <rect x="55" y="36" width="22" height="12" rx="3" fill="#0f172a" />
      <text x="59" y="45" fontSize="7" fill="#fff" fontWeight="bold">Final</text>
    </svg>
  );
}

// ── Section: Retrieval Methods ───────────────────────────────────────────────

function RetrievalSection() {
  const [highlighted, setHighlighted] = useState(null);
  const [q, setQ] = useState(QUERIES[0]);
  const relevantSet = new Set(q.relevant);

  const rankings = useMemo(() => {
    const sem = semanticSearchCorpus(q.key, CHUNKS);
    return {
      tfidf:    tfidfSearch(q.query, CHUNKS).slice(0, 4),
      bm25:     bm25Search(q.query, CHUNKS).slice(0, 4),
      semantic: (sem ?? bm25Search(q.query, CHUNKS)).slice(0, 4),
      hybrid:   hybridSearchCorpus(q.key, q.query, CHUNKS, 0.5).slice(0, 4),
    };
  }, [q]);

  const COLS = [
    { key: "tfidf",    label: "TF-IDF",   color: "#6366f1", tag: "term freq × IDF",                note: "Rewards keyword count — wrong-context spam cheats" },
    { key: "bm25",     label: "BM25",      color: "#06b6d4", tag: "+ length norm + saturation",      note: "Better than TF-IDF, still keyword-only" },
    { key: "semantic", label: "Semantic",  color: "#10b981", tag: "cosine similarity on embeddings",  note: "Finds meaning — spam sinks, synonym rises" },
    { key: "hybrid",   label: "Hybrid",    color: "#f59e0b", tag: "RRF(BM25 + Semantic)",            note: "Best of both worlds" },
  ];

  return (
    <section className="page-section" id="retrieval">
      <div className="section-header">
        <span className="section-tag">Phase 1</span>
        <h2 className="section-title">Same query, 4 different rankings</h2>
      </div>

      <div className="retrieval-query-bar">
        <span className="retrieval-query-icon">🔍</span>
        <span className="retrieval-query-text">{q.query}</span>
        <div className="retrieval-query-topics">
          {QUERIES.map((qo) => (
            <button key={qo.key} type="button"
              className={`retrieval-topic-btn ${q.key === qo.key ? "active" : ""}`}
              onClick={() => setQ(qo)}>{qo.label}</button>
          ))}
        </div>
      </div>

      <div className="retrieval-demo-grid">
        {COLS.map(({ key, label, color, tag, note }) => (
          <div key={key} className="retrieval-demo-col">
            <div className="retrieval-demo-header" style={{ borderTopColor: color }}>
              <span className="retrieval-demo-name" style={{ color }}>{label}</span>
              <span className="retrieval-demo-tag">{tag}</span>
              <span className="retrieval-demo-note">{note}</span>
            </div>
            <div className="retrieval-demo-rows">
              {rankings[key].map((chunk, rank) => {
                const isRelevant = relevantSet.has(chunk.id);
                const isSpam     = chunk.type === "spam";
                const isSynonym  = chunk.type === "synonym";
                const tc         = TYPE_COLORS[chunk.type];
                return (
                  <div key={chunk.id}
                    className={`retrieval-row ${highlighted === chunk.id ? "retrieval-row--hl" : ""} ${isSpam ? "retrieval-row--spam" : ""}`}
                    onMouseEnter={() => setHighlighted(chunk.id)}
                    onMouseLeave={() => setHighlighted(null)}>
                    <span className="retrieval-row-rank" style={{ color }}>#{rank + 1}</span>
                    <span className="retrieval-row-type" style={{ color: tc.badge, background: tc.bg }}>{chunk.type}</span>
                    <span className="retrieval-row-title">{chunk.text.slice(0, 55)}…</span>
                    <div className="retrieval-row-badges">
                      {isSpam    && <span className="dataset-note-badge dataset-note-badge--spam">spam</span>}
                      {isSynonym && <span className="dataset-note-badge dataset-note-badge--synonym">syn</span>}
                      <span className={`result-badge ${isRelevant ? "badge-hit" : "badge-miss"}`}>
                        {isRelevant ? "✓" : "✗"}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="retrieval-callout-row">
        <div className="retrieval-callout retrieval-callout--bad">
          <span className="retrieval-callout-icon">⚠</span>
          <strong>TF-IDF + BM25 trap:</strong> the <em>spam</em> chunk ranks #1 — it contains all query keywords but about roads, archaeology and kitchens. Keyword methods cannot tell the difference.
        </div>
        <div className="retrieval-callout retrieval-callout--good">
          <span className="retrieval-callout-icon">✓</span>
          <strong>Semantic advantage:</strong> the <em>synonym</em> chunk ("Layered computational architectures…") has zero query keywords yet Semantic finds it — BGE understands it is about AI.
        </div>
      </div>
    </section>
  );
}

// ── Section: Metrics ─────────────────────────────────────────────────────────

function MetricsSection() {
  const [k, setK]         = useState(3);
  const [topic, setTopic] = useState(QUERIES[0]);

  const bm25res = useMemo(() => bm25Search(topic.query, CHUNKS), [topic]);
  const semres  = useMemo(() => semanticSearchCorpus(topic.key, CHUNKS) ?? bm25res, [topic, bm25res]);

  const liveMetrics = useMemo(() => ({
    bm25:     computeCorpusMetrics(bm25res, topic.relevant, k),
    semantic: computeCorpusMetrics(semres,  topic.relevant, k),
  }), [bm25res, semres, topic, k]);

  const relevantSet = new Set(topic.relevant);
  const bm25top  = bm25res.slice(0, k);
  const semantop = semres.slice(0, k);

  return (
    <section className="page-section" id="metrics">
      <div className="section-header">
        <span className="section-tag">Evaluation</span>
        <h2 className="section-title">Precision, Recall and F1</h2>
      </div>

      <div className="metrics-layout">

        {/* Left: formula definitions */}
        <div className="metrics-defs">
          <div className="metrics-def-card" style={{ borderTopColor: "#6366f1" }}>
            <div className="metrics-def-badge" style={{ background: "#ede9fe", color: "#6366f1" }}>P@K</div>
            <div className="metrics-def-formula">relevant found / K</div>
            <div className="metrics-def-label">How clean is the result list?</div>
          </div>
          <div className="metrics-def-card" style={{ borderTopColor: "#06b6d4" }}>
            <div className="metrics-def-badge" style={{ background: "#cffafe", color: "#06b6d4" }}>R@K</div>
            <div className="metrics-def-formula">relevant found / total relevant</div>
            <div className="metrics-def-label">How many did you miss?</div>
          </div>
          <div className="metrics-def-card" style={{ borderTopColor: "#10b981" }}>
            <div className="metrics-def-badge" style={{ background: "#dcfce7", color: "#10b981" }}>F1</div>
            <div className="metrics-def-formula">2 × (P × R) / (P + R)</div>
            <div className="metrics-def-label">Balance between the two</div>
          </div>
          <div className="metrics-tradeoff">
            <div className="metrics-tradeoff-arrow">↑ K</div>
            <div>
              <div style={{ color: "#dc2626", fontWeight: 700 }}>Precision falls</div>
              <div style={{ color: "#15803d", fontWeight: 700 }}>Recall rises</div>
            </div>
          </div>
        </div>

        {/* Right: interactive */}
        <div className="metrics-live">

          {/* Topic selector */}
          <div className="metrics-live-controls">
            <span className="controls-label">Topic</span>
            <div className="topic-pills">
              {QUERIES.map((q) => (
                <button key={q.key} type="button"
                  className={`topic-pill ${topic.key === q.key ? "active" : ""}`}
                  onClick={() => setTopic(q)}>{q.label}</button>
              ))}
            </div>
          </div>

          {/* K free input */}
          <div className="metrics-live-controls">
            <span className="controls-label">K (top results retrieved)</span>
            <KInput value={k} onChange={setK} />
          </div>

          {/* Hit/miss strips */}
          {[
            { label: "BM25",     color: "#06b6d4", chunks: bm25top  },
            { label: "Semantic", color: "#10b981", chunks: semantop  },
          ].map(({ label, color, chunks }) => (
            <div key={label} className="metrics-strip-wrap">
              <div className="metrics-strip-label" style={{ color }}>{label}</div>
              <div className="metrics-strip">
                {chunks.map((chunk) => {
                  const hit = relevantSet.has(chunk.id);
                  const tc  = TYPE_COLORS[chunk.type];
                  return (
                    <div key={chunk.id} title={`#${chunk.id} ${chunk.type}`}
                      className={`metrics-cell ${hit ? "metrics-cell--hit" : "metrics-cell--miss"}`}>
                      {hit ? "✓" : "✗"}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}

          {/* Live bars */}
          <div className="metrics-live-bars">
            {[
              { key: "bm25",     label: "BM25",     color: "#06b6d4" },
              { key: "semantic", label: "Semantic", color: "#10b981" },
            ].map(({ key, label, color }) => {
              const m = liveMetrics[key];
              return (
                <div key={key} className="metrics-live-method">
                  <div className="metrics-live-name" style={{ color }}>{label}</div>
                  {[
                    { name: "Precision", val: m.precision, c: "#6366f1" },
                    { name: "Recall",    val: m.recall,    c: "#06b6d4" },
                    { name: "F1",        val: m.f1,        c: "#10b981" },
                  ].map(({ name, val, c }) => (
                    <div key={name} className="metric-bar-row">
                      <span className="metric-bar-label">{name}</span>
                      <div className="metric-bar-track">
                        <div className="metric-bar-fill" style={{ width: `${val * 100}%`, background: c }} />
                      </div>
                      <span className="metric-bar-value">{(val * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                  <div className="metric-card-foot">{m.relevantFound} of {m.totalRelevant} relevant found @ K={k}</div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}

// ── Section: Generation Strategies ──────────────────────────────────────────

function PipelineBox({ label, color, small }) {
  return (
    <div className={`pipe-box ${small ? "pipe-box--small" : ""}`} style={{ borderColor: color, color }}>
      {label}
    </div>
  );
}
function PipelineArrow({ vertical }) {
  return <div className={`pipe-arrow ${vertical ? "pipe-arrow--v" : ""}`}>→</div>;
}
function PipelineLLM({ color }) {
  return <div className="pipe-llm" style={{ background: color }}>LLM</div>;
}
function PipelineCalls({ n, color }) {
  return (
    <div className="pipe-calls" style={{ color }}>
      {n} LLM call{n > 1 ? "s" : ""}
    </div>
  );
}

function GenerationSection() {
  return (
    <section className="page-section" id="generation">
      <div className="section-header">
        <span className="section-tag">Phase 2</span>
        <h2 className="section-title">Generation: how the LLM uses retrieved chunks</h2>
        <p className="section-sub">Three strategies — each is a tradeoff between cost, quality, and context window size.</p>
      </div>

      <div className="gen-layout">

        {/* Stuffing */}
        <div className="gen-card">
          <div className="gen-card-top" style={{ borderTopColor: "#10b981" }}>
            <span className="gen-card-name" style={{ color: "#10b981" }}>Stuffing</span>
            <PipelineCalls n={1} color="#10b981" />
          </div>
          <div className="gen-pipeline gen-pipeline--stuffing">
            <div className="pipe-chunks">
              <PipelineBox label="Chunk 1" color="#10b981" />
              <PipelineBox label="Chunk 2" color="#10b981" />
              <PipelineBox label="Chunk 3" color="#10b981" />
            </div>
            <PipelineArrow />
            <PipelineLLM color="#10b981" />
            <PipelineArrow />
            <PipelineBox label="Answer" color="#10b981" />
          </div>
          <div className="gen-card-footer">
            <span className="gen-tag gen-tag--pro">Fast, cheap</span>
            <span className="gen-tag gen-tag--con">Fails if chunks overflow context window</span>
          </div>
        </div>

        {/* Map Reduce */}
        <div className="gen-card">
          <div className="gen-card-top" style={{ borderTopColor: "#6366f1" }}>
            <span className="gen-card-name" style={{ color: "#6366f1" }}>Map Reduce</span>
            <PipelineCalls n={4} color="#6366f1" />
          </div>
          <div className="gen-pipeline gen-pipeline--mapreduce">
            <div className="pipe-map-col">
              <div className="pipe-map-row">
                <PipelineBox label="Chunk 1" color="#6366f1" small />
                <PipelineArrow />
                <PipelineLLM color="#6366f1" />
                <PipelineArrow />
                <PipelineBox label="Sum 1" color="#6366f1" small />
              </div>
              <div className="pipe-map-row">
                <PipelineBox label="Chunk 2" color="#6366f1" small />
                <PipelineArrow />
                <PipelineLLM color="#6366f1" />
                <PipelineArrow />
                <PipelineBox label="Sum 2" color="#6366f1" small />
              </div>
              <div className="pipe-map-row">
                <PipelineBox label="Chunk 3" color="#6366f1" small />
                <PipelineArrow />
                <PipelineLLM color="#6366f1" />
                <PipelineArrow />
                <PipelineBox label="Sum 3" color="#6366f1" small />
              </div>
            </div>
            <PipelineArrow />
            <PipelineLLM color="#6366f1" />
            <PipelineArrow />
            <PipelineBox label="Answer" color="#6366f1" small />
          </div>
          <div className="gen-card-footer">
            <span className="gen-tag gen-tag--pro">Handles unlimited chunks, parallel map</span>
            <span className="gen-tag gen-tag--con">LLM never sees all chunks together</span>
          </div>
        </div>

        {/* Refine */}
        <div className="gen-card">
          <div className="gen-card-top" style={{ borderTopColor: "#f59e0b" }}>
            <span className="gen-card-name" style={{ color: "#f59e0b" }}>Refine</span>
            <PipelineCalls n={3} color="#f59e0b" />
          </div>
          <div className="gen-pipeline gen-pipeline--refine">
            <div className="pipe-refine-row">
              <PipelineBox label="Chunk 1" color="#f59e0b" small />
              <PipelineArrow />
              <PipelineLLM color="#f59e0b" />
              <PipelineArrow />
              <PipelineBox label="Draft 1" color="#f59e0b" small />
            </div>
            <div className="pipe-refine-connector" />
            <div className="pipe-refine-row">
              <PipelineBox label="Chunk 2" color="#f59e0b" small />
              <span className="pipe-merge">+</span>
              <PipelineLLM color="#f59e0b" />
              <PipelineArrow />
              <PipelineBox label="Draft 2" color="#f59e0b" small />
            </div>
            <div className="pipe-refine-connector" />
            <div className="pipe-refine-row">
              <PipelineBox label="Chunk 3" color="#f59e0b" small />
              <span className="pipe-merge">+</span>
              <PipelineLLM color="#f59e0b" />
              <PipelineArrow />
              <PipelineBox label="Final" color="#f59e0b" small />
            </div>
          </div>
          <div className="gen-card-footer">
            <span className="gen-tag gen-tag--pro">Each step corrects the previous one</span>
            <span className="gen-tag gen-tag--con">Sequential, cannot parallelize</span>
          </div>
        </div>
      </div>
    </section>
  );
}

// ── Section: Mock LLM ────────────────────────────────────────────────────────

function MockLLMSection({ results, k }) {
  const [retrieverKey, setRetrieverKey] = useState("hybrid");
  const [genStrategy, setGenStrategy] = useState("stuffing");

  const llmOutput = useMemo(() => {
    if (!results) return null;
    return runMockLLM(results[retrieverKey], k, genStrategy);
  }, [results, retrieverKey, genStrategy, k]);

  if (!results) return null;

  return (
    <section className="page-section" id="mock-llm">
      <div className="section-header">
        <span className="section-tag">Mock LLM</span>
        <h2 className="section-title">See How Retrieval Affects the Answer</h2>
        <p className="section-sub">Switch retriever and generation strategy to see how the answer changes.</p>
      </div>

      <div className="mock-controls">
        <div className="mock-control-group">
          <div className="mock-control-label">Retriever</div>
          <div className="mock-pills">
            {METHODS.map((m) => (
              <button key={m.key} type="button"
                className={`mock-pill ${retrieverKey === m.key ? "active" : ""}`}
                style={retrieverKey === m.key ? { background: m.color, borderColor: m.color } : {}}
                onClick={() => setRetrieverKey(m.key)}>
                {m.label}
              </button>
            ))}
          </div>
        </div>
        <div className="mock-control-group">
          <div className="mock-control-label">Generation strategy</div>
          <div className="mock-pills">
            {GEN_STRATEGIES.map((s) => (
              <button key={s} type="button"
                className={`mock-pill ${genStrategy === s ? "active" : ""}`}
                onClick={() => setGenStrategy(s)}>
                {GEN_LABELS[s]}
              </button>
            ))}
          </div>
        </div>
      </div>

      {llmOutput && (
        <div className="mock-output">
          <div className="mock-steps">
            {llmOutput.steps.map((step, i) => (
              <div key={i} className="mock-step">
                <div className="mock-step-label">{step.label}</div>
                <div className="mock-step-content">{step.content}</div>
              </div>
            ))}
          </div>
          <div className="mock-meta">
            <span>Strategy: <strong>{llmOutput.strategy}</strong></span>
            <span>LLM calls: <strong>{llmOutput.llmCalls}</strong></span>
            <span className="mock-disclaimer">Simulated output, no real LLM</span>
          </div>
        </div>
      )}
    </section>
  );
}

// ── Corpus chunk card ─────────────────────────────────────────────────────────

const TYPE_COLORS = {
  normal:  { bg: "#f0fdf4", border: "#6ee7b7", badge: "#15803d", text: "normal"  },
  long:    { bg: "#eff6ff", border: "#93c5fd", badge: "#1d4ed8", text: "long"    },
  synonym: { bg: "#f5f3ff", border: "#c4b5fd", badge: "#7c3aed", text: "synonym" },
  spam:    { bg: "#fff7ed", border: "#fdba74", badge: "#c2410c", text: "spam"    },
};

function ChunkCard({ chunk, rank, isRelevant, isHighlighted }) {
  const tc = TYPE_COLORS[chunk.type] || TYPE_COLORS.normal;
  return (
    <div className={`chunk-card ${isHighlighted ? "chunk-card--hl" : ""} ${isRelevant === false ? "chunk-card--miss" : ""}`}
      style={isHighlighted ? { borderColor: tc.border, background: tc.bg } : {}}>
      <div className="chunk-card-top">
        <span className="chunk-card-rank">#{rank + 1}</span>
        <span className="chunk-card-badge" style={{ background: tc.bg, color: tc.badge }}>
          {tc.text}
        </span>
        {isRelevant !== null && (
          <span className={`result-badge ${isRelevant ? "badge-hit" : "badge-miss"}`}>
            {isRelevant ? "✓" : "✗"}
          </span>
        )}
      </div>
    </div>
  );
}

// ── Main Playground Section ──────────────────────────────────────────────────

function PlaygroundSection() {
  const [activeQuery, setActiveQuery]   = useState(null);  // QUERIES[i]
  const [k, setK]                       = useState(3);
  const [alpha, setAlpha]               = useState(50);    // 0–100, maps to 0.0–1.0
  const [hoveredChunk, setHoveredChunk] = useState(null);

  const alphaNorm = alpha / 100;   // 0 = pure semantic, 1 = pure BM25
  const q = activeQuery;

  const results = useMemo(() => {
    if (!q) return null;
    const sem = semanticSearchCorpus(q.key, CHUNKS);
    return {
      tfidf:    tfidfSearch(q.query, CHUNKS),
      bm25:     bm25Search(q.query, CHUNKS),
      semantic: sem ?? bm25Search(q.query, CHUNKS),
      hybrid:   hybridSearchCorpus(q.key, q.query, CHUNKS, alphaNorm),
    };
  }, [q, alphaNorm]);

  const metrics = useMemo(() => {
    if (!results || !q) return null;
    const m = (arr) => computeCorpusMetrics(arr, q.relevant, k);
    return { tfidf: m(results.tfidf), bm25: m(results.bm25), semantic: m(results.semantic), hybrid: m(results.hybrid) };
  }, [results, q, k]);

  // Which chunk ids are in top-K for each method
  const topIds = useMemo(() => {
    if (!results) return {};
    const ids = (arr) => new Set(arr.slice(0, k).map((c) => c.id));
    return { tfidf: ids(results.tfidf), bm25: ids(results.bm25), semantic: ids(results.semantic), hybrid: ids(results.hybrid) };
  }, [results, k]);

  const relevantSet = q ? new Set(q.relevant) : new Set();

  const COLS = [
    { key: "tfidf",    label: "TF-IDF",   color: "#6366f1" },
    { key: "bm25",     label: "BM25",     color: "#06b6d4" },
    { key: "semantic", label: "Semantic", color: "#10b981" },
    { key: "hybrid",   label: "Hybrid",   color: "#f59e0b" },
  ];

  return (
    <section className="page-section" id="playground">

      {/* Controls: Topic + K */}
      <div className="pg-controls">
        <div className="pg-ctrl-group">
          <span className="controls-label">Topic</span>
          <div className="topic-pills">
            {QUERIES.map((q2) => (
              <button key={q2.key} type="button"
                className={`topic-pill ${activeQuery?.key === q2.key ? "active" : ""}`}
                onClick={() => setActiveQuery(q2)}>{q2.label}</button>
            ))}
          </div>
        </div>
        <div className="pg-ctrl-group">
          <span className="controls-label">K (top results)</span>
          <KInput value={k} onChange={setK} />
        </div>
      </div>

      {!q && <div className="pg-hint">← Pick a topic to start</div>}

      {q && results && (<>

        {/* Hero alpha slider */}
        <div className="pg-alpha-hero">
          <span className="pg-ah-label" style={{color:"#10b981"}}>◀ Semantic (α=0)</span>
          <input type="range" min="0" max="100" value={alpha}
            onChange={(e) => setAlpha(+e.target.value)}
            className="alpha-slider" style={{flex:1}} />
          <span className="pg-ah-label" style={{color:"#06b6d4"}}>BM25 (α=1) ▶</span>
          <span className="pg-ah-val">α = {alphaNorm.toFixed(2)}</span>
        </div>

        <div className="pg-body">

          {/* Left: source document chunk map */}
          <div className="pg-doc-panel">
            <div className="pg-doc-title">Document · {CHUNKS.length} chunks</div>
            {CHUNKS.map((chunk) => {
              const inTopBM25 = topIds.bm25?.has(chunk.id);
              const inTopSem  = topIds.semantic?.has(chunk.id);
              const inTopHyb  = topIds.hybrid?.has(chunk.id);
              const isRel     = relevantSet.has(chunk.id);
              const tc        = TYPE_COLORS[chunk.type];
              return (
                <div key={chunk.id}
                  className={`pg-chunk-block ${hoveredChunk === chunk.id ? "pg-chunk-block--hover" : ""}`}
                  style={{ borderLeftColor: inTopHyb ? tc.border : "#e2e8f0" }}
                  onMouseEnter={() => setHoveredChunk(chunk.id)}
                  onMouseLeave={() => setHoveredChunk(null)}>
                  <div className="pg-chunk-meta">
                    <span className="pg-chunk-id">#{chunk.id}</span>
                    <span className="pg-chunk-type" style={{ color: tc.badge }}>{tc.text}</span>
                    <div className="pg-chunk-hits">
                      {inTopBM25 && <span className="pg-hit" style={{color:"#06b6d4"}}>BM25</span>}
                      {inTopSem  && <span className="pg-hit" style={{color:"#10b981"}}>Sem</span>}
                      {inTopHyb  && <span className="pg-hit" style={{color:"#f59e0b"}}>Hyb</span>}
                    </div>
                    {!isRel && chunk.type === "spam" && <span className="pg-spam-tag">spam</span>}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Right: BM25 | HYBRID (live) | Semantic */}
          <div className="pg-results-panel">

            {/* 3 metric cards */}
            <div className="pg-three-metrics">
              <div className="pg-mini-mc" style={{borderTopColor:"#06b6d4"}}>
                <div className="pg-mini-mc-name" style={{color:"#06b6d4"}}>BM25 <span className="pg-mc-hint">α=1</span></div>
                <div className="pg-mini-mc-vals">
                  <span>P {(metrics.bm25.precision*100).toFixed(0)}%</span>
                  <span>·</span>
                  <span>R {(metrics.bm25.recall*100).toFixed(0)}%</span>
                  <span>·</span>
                  <span>F1 {(metrics.bm25.f1*100).toFixed(0)}%</span>
                </div>
              </div>
              <div className="pg-live-mc">
                <div className="pg-live-mc-name">HYBRID <span className="pg-mc-hint">α={alphaNorm.toFixed(2)}</span></div>
                <MetricBar label="P" value={metrics.hybrid.precision} color="#6366f1" />
                <MetricBar label="R" value={metrics.hybrid.recall}    color="#06b6d4" />
                <MetricBar label="F1" value={metrics.hybrid.f1}       color="#10b981" />
              </div>
              <div className="pg-mini-mc" style={{borderTopColor:"#10b981"}}>
                <div className="pg-mini-mc-name" style={{color:"#10b981"}}>Semantic <span className="pg-mc-hint">α=0</span></div>
                <div className="pg-mini-mc-vals">
                  <span>P {(metrics.semantic.precision*100).toFixed(0)}%</span>
                  <span>·</span>
                  <span>R {(metrics.semantic.recall*100).toFixed(0)}%</span>
                  <span>·</span>
                  <span>F1 {(metrics.semantic.f1*100).toFixed(0)}%</span>
                </div>
              </div>
            </div>

            {/* 3 result columns */}
            <div className="pg-three-cols">
              {[
                {key:"bm25",     label:"BM25",     color:"#06b6d4"},
                {key:"hybrid",   label:"HYBRID",   color:"#f59e0b", live:true},
                {key:"semantic", label:"Semantic", color:"#10b981"},
              ].map(({key, label, color, live}) => (
                <div key={key} className={`pg-tcol ${live ? "pg-tcol--live" : ""}`} style={{borderTopColor:color}}>
                  <div className="pg-tcol-label" style={{color}}>{label} <span className="pg-mc-hint">top {k}</span></div>
                  {results[key].slice(0, k).map((chunk, rank) => (
                    <ChunkCard key={chunk.id} chunk={chunk} rank={rank}
                      isRelevant={relevantSet.size > 0 ? relevantSet.has(chunk.id) : null}
                      isHighlighted={hoveredChunk === chunk.id}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
      </>)}
    </section>
  );
}

// ── Section: Embedding Space ──────────────────────────────────────────────────

const CHUNK_DEMO_TEXT = `The James Webb Space Telescope has transformed our understanding of the early universe. By capturing infrared light from objects too distant and faint for previous instruments, Webb reveals galaxies that formed just hundreds of millions of years after the Big Bang.

Its primary mirror, spanning 6.5 meters, is composed of 18 gold-coated hexagonal segments that unfold in orbit. This design allows Webb to observe wavelengths invisible to Hubble, including thermal emissions from forming planetary systems.

Early results have already challenged existing cosmological models. Galaxies observed at redshift z > 10 appear far more massive and structured than simulations predicted, suggesting star formation in the early universe was more efficient than previously believed.`;

const CHUNK_COLORS = ["#6366f1", "#06b6d4", "#f59e0b"];

function ChunkingDemo() {
  const [chunkSize, setChunkSize] = useState(1);
  const sentences = CHUNK_DEMO_TEXT.split("\n\n").filter(Boolean);
  const chunks = chunkSize === 1 ? sentences : chunkSize === 2
    ? [sentences.slice(0, 2).join(" "), sentences[2]]
    : [sentences.join(" ")];

  return (
    <div className="chunk-demo">
      <div className="chunk-demo-controls">
        <span className="chunk-demo-label">Chunk count</span>
        <div className="k-pills">
          {[1, 2, 3].map((n) => (
            <button key={n} type="button"
              className={`k-pill ${chunkSize === n ? "active" : ""}`}
              onClick={() => setChunkSize(n)}>
              {n === 1 ? "3 chunks" : n === 2 ? "2 chunks" : "1 chunk"}
            </button>
          ))}
        </div>
      </div>
      <div className="chunk-text-wrap">
        {chunks.map((chunk, i) => (
          <span key={i} className="chunk-span" style={{ borderColor: CHUNK_COLORS[i], background: CHUNK_COLORS[i] + "18" }}>
            <span className="chunk-label" style={{ background: CHUNK_COLORS[i], color: "#fff" }}>Chunk {i + 1}</span>
            {chunk}
          </span>
        ))}
      </div>
      <div className="chunk-vectors">
        {chunks.map((_, i) => (
          <div key={i} className="chunk-vector-row">
            <span className="chunk-vector-label" style={{ color: CHUNK_COLORS[i] }}>Chunk {i + 1} embedding</span>
            <div className="chunk-vector-bar">
              {Array.from({ length: 24 }).map((__, j) => (
                <div key={j} className="chunk-vector-cell"
                  style={{ opacity: 0.15 + Math.abs(Math.sin(i * 7 + j * 1.3)) * 0.85,
                    background: CHUNK_COLORS[i] }} />
              ))}
              <span className="chunk-vector-ellipsis">· · · 768 dims</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function EmbeddingCustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  if (!d.title) return null;
  return (
    <div className="embed-tooltip">
      <div className="embed-tooltip-title">{d.title}</div>
      {d.isSpam && <div className="embed-tooltip-tag embed-tooltip-tag--spam">keyword spam</div>}
      {d.isSynonym && <div className="embed-tooltip-tag embed-tooltip-tag--synonym">synonym</div>}
    </div>
  );
}

function EmbeddingSection() {
  const [activeTopicKey, setActiveTopicKey] = useState(null);

  const semRanking = useMemo(() => {
    if (!activeTopicKey) return new Set();
    const ranked = semanticSearch(activeTopicKey, ARTICLES);
    return new Set(ranked.slice(0, 5).map((d) => d.id));
  }, [activeTopicKey]);

  const scatterData = useMemo(() => {
    return TOPICS.map((t) => ({
      ...t,
      tc: TOPIC_COLORS[t.key],
      points: ARTICLES.filter((a) => a.category === t.key).map((a) => {
        const [x, y] = POSITIONS_2D.docs[String(a.id)];
        return {
          x, y, id: a.id, title: a.title,
          isSpam: a.note?.startsWith("keyword spam"),
          isSynonym: a.note?.startsWith("synonym"),
          isTop5: semRanking.has(a.id),
        };
      }),
    }));
  }, [semRanking]);

  const queryPoint = activeTopicKey ? (() => {
    const [x, y] = POSITIONS_2D.queries[activeTopicKey];
    return [{ x, y, title: `Query: ${TOPICS.find(t => t.key === activeTopicKey)?.label}` }];
  })() : [];

  return (
    <section className="page-section" id="embedding">
      <div className="section-header">
        <span className="section-tag">How Semantic Works</span>
        <h2 className="section-title">From Text to Vectors to Retrieval</h2>
        <p className="section-sub">Text is encoded into dense 768-dim vectors. The scatter plot shows all 25 articles projected to 2D via PCA. Pick a topic to see where its query lands and which articles are closest.</p>
      </div>

      <div className="embed-layout">

        {/* Left: chunking demo */}
        <div className="embed-left">
          <div className="embed-panel-title">1. Text is split into chunks and embedded</div>
          <ChunkingDemo />
        </div>

        {/* Right: scatter plot */}
        <div className="embed-right">
          <div className="embed-panel-title">2. Embeddings live in vector space. Semantic retrieval = nearest neighbors.</div>

          <div className="embed-topic-pills">
            {TOPICS.map((t) => (
              <button key={t.key} type="button"
                className={`topic-pill ${activeTopicKey === t.key ? "active" : ""}`}
                style={activeTopicKey === t.key ? { background: TOPIC_COLORS[t.key].color, borderColor: TOPIC_COLORS[t.key].color } : {}}
                onClick={() => setActiveTopicKey(activeTopicKey === t.key ? null : t.key)}>
                {t.label}
              </button>
            ))}
          </div>

          <div className="embed-scatter-wrap">
            <ResponsiveContainer width="100%" height={280}>
              <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: -20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="x" type="number" domain={[-1.1, 1.1]} tick={false} axisLine={false} />
                <YAxis dataKey="y" type="number" domain={[-1.1, 1.1]} tick={false} axisLine={false} />
                <ZAxis range={[60, 60]} />
                <Tooltip content={<EmbeddingCustomTooltip />} />
                {scatterData.map((group) => (
                  <Scatter key={group.key} name={group.label} data={group.points}
                    fill={group.tc.color}
                    shape={(props) => {
                      const { cx, cy, payload } = props;
                      const isTop5 = payload.isTop5;
                      const r = payload.isSpam ? 7 : payload.isSynonym ? 7 : 6;
                      return (
                        <g>
                          {isTop5 && <circle cx={cx} cy={cy} r={r + 7} fill="none" stroke={group.tc.color} strokeWidth={2} opacity={0.5} />}
                          <circle cx={cx} cy={cy} r={r}
                            fill={group.tc.color}
                            fillOpacity={isTop5 ? 1 : 0.45}
                            stroke={payload.isSpam ? "#dc2626" : payload.isSynonym ? "#059669" : "none"}
                            strokeWidth={payload.isSpam || payload.isSynonym ? 1.5 : 0}
                          />
                        </g>
                      );
                    }}
                  />
                ))}
                {queryPoint.length > 0 && (
                  <Scatter name="Query" data={queryPoint} fill="#0f172a"
                    shape={(props) => {
                      const { cx, cy } = props;
                      const s = 10;
                      return <polygon points={`${cx},${cy-s} ${cx+s*0.6},${cy+s*0.4} ${cx-s*0.6},${cy+s*0.4}`}
                        fill="#0f172a" />;
                    }}
                  />
                )}
              </ScatterChart>
            </ResponsiveContainer>

            <div className="embed-legend">
              {scatterData.map((g) => (
                <span key={g.key} className="embed-legend-item">
                  <span className="embed-legend-dot" style={{ background: g.tc.color }} />{g.label}
                </span>
              ))}
              <span className="embed-legend-item">
                <span className="embed-legend-tri" />Query
              </span>
              {activeTopicKey && (
                <span className="embed-legend-item">
                  <span className="embed-legend-ring" style={{ borderColor: TOPIC_COLORS[activeTopicKey].color }} />Top-5 semantic
                </span>
              )}
              <span className="embed-legend-item">
                <span className="embed-legend-dot" style={{ background: "#fff", border: "1.5px solid #dc2626" }} />spam
              </span>
              <span className="embed-legend-item">
                <span className="embed-legend-dot" style={{ background: "#fff", border: "1.5px solid #059669" }} />synonym
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// ── Section: HNSW ────────────────────────────────────────────────────────────
// Pre-defined HNSW graph for 12 corpus chunks (hand-crafted for demo clarity).
// Layer 2 = entry points (3 nodes), Layer 1 = routing (6 nodes), Layer 0 = all nodes.
const HNSW_LAYERS = [
  // layer 2 — top, coarse, long edges
  { nodes: [0, 4, 8], edges: [[0,4],[0,8],[4,8]] },
  // layer 1 — mid
  { nodes: [0, 1, 4, 5, 8, 9], edges: [[0,1],[4,5],[8,9],[0,4],[4,8]] },
  // layer 0 — all
  { nodes: [0,1,2,3,4,5,6,7,8,9,10,11], edges: [[0,1],[1,2],[2,3],[4,5],[5,6],[6,7],[8,9],[9,10],[10,11],[0,4],[4,8],[1,5],[5,9]] },
];

const QUERY_KEYS = ["ai","climate","space"];
const TOPIC_LABEL = { ai:"AI", climate:"Climate", space:"Space" };

function HNSWSection() {
  const [queryKey, setQueryKey]   = useState("ai");
  const [layer,    setLayer]      = useState(2);
  const [step,     setStep]       = useState(null); // null = idle, 0..N = animating

  // Search path per query (hand-crafted to match actual graph)
  const PATHS = {
    ai:      [8, 4, 0, 1],   // start at entry 8 (space), go to 4 (climate entry), go to 0 (AI entry), refine to 1
    climate: [0, 4, 5],
    space:   [0, 8, 9],
  };
  const path = PATHS[queryKey];

  const W = 340, H = 260;
  const getPos = (id) => {
    const [x, y] = CORPUS_POS.chunks[String(id)];
    return { x: ((x + 1) / 2) * (W - 40) + 20, y: ((y + 1) / 2) * (H - 40) + 20 };
  };
  const queryPos = (() => {
    const [x, y] = CORPUS_POS.queries[queryKey];
    return { x: ((x + 1) / 2) * (W - 40) + 20, y: ((y + 1) / 2) * (H - 40) + 20 };
  })();

  const curLayer = HNSW_LAYERS[layer];
  const visitedNodes = step !== null ? new Set(path.slice(0, step + 1)) : new Set();
  const currentNode  = step !== null ? path[step] : null;

  const CHUNK_TOPIC_COLOR = { ai:"#6366f1", climate:"#059669", space:"#0891b2" };
  function chunkColor(id) {
    const c = CHUNKS[id];
    return CHUNK_TOPIC_COLOR[c?.topic] || "#94a3b8";
  }

  return (
    <section className="page-section" id="hnsw">
      <div className="section-header">
        <span className="section-tag">Vector Index</span>
        <h2 className="section-title">HNSW — how vector search stays fast at scale</h2>
        <p className="section-sub">Without an index, every query compares against every vector (O·N). HNSW builds a multi-layer graph: top layers skip long distances fast, bottom layers find the precise nearest neighbor.</p>
      </div>

      <div className="hnsw-layout">

        {/* Left: controls + explanation */}
        <div className="hnsw-left">
          <div className="hnsw-step-card">
            <div className="hnsw-step-num">1</div>
            <div>
              <strong>Coarse layer (top)</strong>
              <div className="hnsw-step-desc">Few nodes, long edges. Entry point chosen here.</div>
            </div>
          </div>
          <div className="hnsw-step-card">
            <div className="hnsw-step-num">2</div>
            <div>
              <strong>Routing layer (mid)</strong>
              <div className="hnsw-step-desc">Greedily walk toward the query in embedding space.</div>
            </div>
          </div>
          <div className="hnsw-step-card">
            <div className="hnsw-step-num">3</div>
            <div>
              <strong>Precise layer (bottom)</strong>
              <div className="hnsw-step-desc">All nodes, short edges. Return top-K nearest neighbors.</div>
            </div>
          </div>

          <div className="hnsw-controls">
            <div>
              <div className="controls-label" style={{marginBottom:4}}>Query</div>
              <div className="topic-pills">
                {QUERY_KEYS.map((k2) => (
                  <button key={k2} type="button"
                    className={`topic-pill ${queryKey === k2 ? "active" : ""}`}
                    onClick={() => { setQueryKey(k2); setStep(null); }}>
                    {TOPIC_LABEL[k2]}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <div className="controls-label" style={{marginBottom:4}}>Layer</div>
              <div className="k-pills">
                {[2,1,0].map((l) => (
                  <button key={l} type="button"
                    className={`k-pill ${layer === l ? "active" : ""}`}
                    onClick={() => setLayer(l)}>
                    L{l}
                  </button>
                ))}
              </div>
            </div>
            <button className="hnsw-search-btn"
              onClick={() => setStep(step === null ? 0 : step < path.length - 1 ? step + 1 : null)}>
              {step === null ? "▶ Start search" : step < path.length - 1 ? "▶ Next step" : "↺ Reset"}
            </button>
          </div>
        </div>

        {/* Right: graph viz */}
        <div className="hnsw-right">
          <svg width={W} height={H} className="hnsw-svg">
            {/* edges */}
            {curLayer.edges.map(([a, b], i) => {
              const pa = getPos(a), pb = getPos(b);
              const isActive = visitedNodes.has(a) && visitedNodes.has(b);
              return <line key={i} x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y}
                stroke={isActive ? "#f59e0b" : "#e2e8f0"} strokeWidth={isActive ? 2 : 1}
                strokeDasharray={layer === 2 ? "6 3" : layer === 1 ? "3 2" : "none"} />;
            })}

            {/* search path arrows */}
            {step !== null && path.slice(0, step).map((id, i) => {
              const pa = getPos(id), pb = getPos(path[i + 1]);
              if (!pb) return null;
              return <line key={`path-${i}`} x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y}
                stroke="#f59e0b" strokeWidth={2.5} markerEnd="url(#arr)" />;
            })}
            <defs>
              <marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                <path d="M0,0 L0,6 L6,3 z" fill="#f59e0b" />
              </marker>
            </defs>

            {/* nodes */}
            {curLayer.nodes.map((id) => {
              const p = getPos(id);
              const isVisited = visitedNodes.has(id);
              const isCurrent = id === currentNode;
              return (
                <g key={id}>
                  {isCurrent && <circle cx={p.x} cy={p.y} r={16} fill="#fef9c3" stroke="#f59e0b" strokeWidth={2} />}
                  <circle cx={p.x} cy={p.y} r={10}
                    fill={isVisited ? chunkColor(id) : "#fff"}
                    stroke={chunkColor(id)} strokeWidth={2}
                    fillOpacity={isVisited ? 0.9 : 0.4} />
                  <text x={p.x} y={p.y + 4} textAnchor="middle" fontSize="8" fontWeight="700"
                    fill={isVisited ? "#fff" : chunkColor(id)}>{id}</text>
                </g>
              );
            })}

            {/* query star */}
            <polygon
              points={[0,1,2,3,4].map(i => {
                const a = (i * 72 - 90) * Math.PI / 180;
                const r = i%2===0 ? 9 : 4;
                return `${queryPos.x + r*Math.cos(a)},${queryPos.y + r*Math.sin(a)}`;
              }).join(" ")}
              fill="#0f172a" />
            <text x={queryPos.x} y={queryPos.y + 18} textAnchor="middle" fontSize="8" fill="#0f172a" fontWeight="700">Q</text>
          </svg>

          <div className="hnsw-legend-row">
            {["ai","climate","space"].map(t => (
              <span key={t} className="embed-legend-item">
                <span className="embed-legend-dot" style={{background: CHUNK_TOPIC_COLOR[t]}} />{TOPIC_LABEL[t]}
              </span>
            ))}
            <span className="embed-legend-item"><span className="hnsw-legend-star">★</span>Query</span>
            {step !== null && <span className="hnsw-step-status">Step {step + 1}/{path.length} — visiting chunk #{path[step]}</span>}
          </div>
        </div>
      </div>
    </section>
  );
}

// ── Section: Dataset ─────────────────────────────────────────────────────────

function DatasetSection() {
  const [filter, setFilter] = useState("all");

  const visible = useMemo(() =>
    filter === "all" ? ARTICLES : ARTICLES.filter((a) => a.category === filter),
    [filter]
  );

  return (
    <section className="page-section" id="dataset">
      <div className="section-header">
        <span className="section-tag">Dataset</span>
        <h2 className="section-title">25 Articles, 5 Topics</h2>
        <p className="section-sub">Each topic has normal articles, a keyword-spam article (BM25 trap), and a synonym article (semantic advantage). Filter to inspect the ground truth before running a query.</p>
      </div>

      <div className="dataset-filters">
        <button type="button"
          className={`dataset-filter-pill ${filter === "all" ? "active" : ""}`}
          onClick={() => setFilter("all")}>
          All ({ARTICLES.length})
        </button>
        {TOPICS.map((t) => {
          const tc = TOPIC_COLORS[t.key] || {};
          return (
            <button key={t.key} type="button"
              className={`dataset-filter-pill ${filter === t.key ? "active" : ""}`}
              style={filter === t.key ? { background: tc.color, borderColor: tc.color, color: "#fff" } : {}}
              onClick={() => setFilter(t.key)}>
              {t.label} ({ARTICLES.filter((a) => a.category === t.key).length})
            </button>
          );
        })}
      </div>

      <div className="dataset-grid">
        {visible.map((article) => {
          const tc = TOPIC_COLORS[article.category] || { bg: "#f1f5f9", color: "#64748b" };
          const isSpam = article.note?.startsWith("keyword spam");
          const isSynonym = article.note?.startsWith("synonym");
          return (
            <div key={article.id} className={`dataset-card ${isSpam ? "dataset-card--spam" : ""} ${isSynonym ? "dataset-card--synonym" : ""}`}>
              <div className="dataset-card-header">
                <span className="dataset-id">#{article.id}</span>
                <div style={{ display: "flex", gap: 4 }}>
                  {isSpam && <span className="dataset-note-badge dataset-note-badge--spam">spam</span>}
                  {isSynonym && <span className="dataset-note-badge dataset-note-badge--synonym">synonym</span>}
                  <span className="dataset-topic-badge" style={{ background: tc.bg, color: tc.color }}>
                    {article.category}
                  </span>
                </div>
              </div>
              <div className="dataset-card-title">{article.title}</div>
              <div className="dataset-card-text">{article.text}</div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

// ── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-left">
          <div className="app-logo-mark" aria-hidden="true">
            <svg viewBox="0 0 56 56" fill="none">
              <rect width="56" height="56" rx="14" fill="#0f172a" />
              <rect x="10" y="28" width="10" height="18" rx="3" fill="#6366f1" />
              <rect x="23" y="18" width="10" height="28" rx="3" fill="#06b6d4" />
              <rect x="36" y="10" width="10" height="36" rx="3" fill="#f59e0b" />
            </svg>
          </div>
          <div>
            <h1 className="app-title">RAG Playground</h1>
            <p className="app-subtitle">TF-IDF vs BM25 vs Semantic vs Hybrid, with live Precision, Recall and F1</p>
          </div>
        </div>
        <nav className="app-nav-links">
          <a href="#playground">Playground</a>
          <a href="#retrieval">Retrieval</a>
          <a href="#embedding">Embeddings</a>
          <a href="#hnsw">HNSW</a>
          <a href="#metrics">Metrics</a>
          <a href="#generation">Generation</a>
        </nav>
      </header>

      <main className="app-content">
        <PlaygroundSection />
        <RetrievalSection />
        <EmbeddingSection />
        <HNSWSection />
        <MetricsSection />
        <GenerationSection />
      </main>

      <footer className="app-footer">
        <div className="footer-rule" />
        <div className="footer-body">
          <span className="footer-built">Built by <strong>Giovanni Tammaro</strong> with</span>
          {/* React logo */}
          <svg className="footer-logo" viewBox="0 0 40 40" aria-label="React">
            <circle cx="20" cy="20" r="3.2" fill="#61dafb"/>
            <ellipse cx="20" cy="20" rx="18" ry="6.5" fill="none" stroke="#61dafb" strokeWidth="1.6"/>
            <ellipse cx="20" cy="20" rx="18" ry="6.5" fill="none" stroke="#61dafb" strokeWidth="1.6" transform="rotate(60 20 20)"/>
            <ellipse cx="20" cy="20" rx="18" ry="6.5" fill="none" stroke="#61dafb" strokeWidth="1.6" transform="rotate(120 20 20)"/>
          </svg>
          <span className="footer-stack-label">React</span>
          <span className="footer-plus">+</span>
          {/* Vite logo */}
          <svg className="footer-logo" viewBox="0 0 32 32" aria-label="Vite">
            <path d="M29.88 6.57 17.08 28.75a1.16 1.16 0 0 1-2.03-.01L2.1 6.56A1.16 1.16 0 0 1 3.27 4.9l12.64 2.37 12.58-2.37a1.16 1.16 0 0 1 1.39 1.67Z" fill="#bd34fe"/>
            <path d="M22.27 2.05 15.95 3.3l-6.32-1.24A1.16 1.16 0 0 0 8.37 3.5l7.57 13.1 7.57-13.1a1.16 1.16 0 0 0-1.24-1.45Z" fill="#ffd62e"/>
          </svg>
          <span className="footer-stack-label">Vite</span>
          <span className="footer-sep-v" />
          <a className="footer-gh" href={GITHUB_URL} target="_blank" rel="noreferrer">
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path fill="currentColor" d="M8 0a8 8 0 0 0-2.53 15.59c.4.07.55-.17.55-.38v-1.33c-2.24.49-2.71-.95-2.71-.95-.36-.93-.9-1.17-.9-1.17-.73-.5.06-.5.06-.5.8.05 1.22.83 1.22.83.71 1.22 1.87.87 2.33.66.07-.52.28-.87.5-1.07-1.79-.2-3.67-.9-3.67-3.96 0-.88.31-1.6.82-2.16-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.47 7.47 0 0 1 4 0c1.53-1.03 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.16 0 3.07-1.88 3.75-3.67 3.96.29.25.54.73.54 1.48v2.2c0 .21.14.45.55.38A8 8 0 0 0 8 0Z"/>
            </svg>
            GitHub
          </a>
        </div>
      </footer>
    </div>
  );
}
