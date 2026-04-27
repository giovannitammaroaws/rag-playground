import { useState, useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { ARTICLES, TOPICS } from "../data/articles.js";
import { tfidfSearch, bm25Search, rrfSearch } from "../lib/search.js";
import { computeMetrics } from "../lib/metrics.js";

const K_OPTIONS = [3, 5, 10, 20];
const METHODS = [
  { key: "tfidf", label: "TF-IDF", color: "#6366f1" },
  { key: "bm25",  label: "BM25",   color: "#06b6d4" },
  { key: "rrf",   label: "Hybrid (RRF)", color: "#f59e0b" },
];

const CHART_COLORS = {
  precision: "#6366f1",
  recall: "#06b6d4",
  f1: "#10b981",
};

function ResultCard({ doc, rank, isRelevant }) {
  return (
    <div className={`result-card ${isRelevant ? "relevant" : "irrelevant"}`}>
      <div className="result-rank">#{rank + 1}</div>
      <div className="result-body">
        <div className="result-title">{doc.title}</div>
        <div className="result-text">{doc.text.slice(0, 110)}…</div>
      </div>
      <div className={`result-badge ${isRelevant ? "badge-hit" : "badge-miss"}`}>
        {isRelevant ? "✓ relevant" : "✗ off-topic"}
      </div>
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

export default function RAGPlayground() {
  const [query, setQuery] = useState("");
  const [k, setK] = useState(5);
  const [activeTopic, setActiveTopic] = useState(null);

  const handleTopic = (topic) => {
    setQuery(topic.query);
    setActiveTopic(topic.key);
  };

  const handleQueryChange = (e) => {
    setQuery(e.target.value);
    setActiveTopic(null);
  };

  const results = useMemo(() => {
    if (!query.trim()) return null;
    return {
      tfidf: tfidfSearch(query, ARTICLES),
      bm25:  bm25Search(query, ARTICLES),
      rrf:   rrfSearch(query, ARTICLES),
    };
  }, [query]);

  const relevantCategory = useMemo(() => {
    if (!activeTopic) return null;
    return TOPICS.find((t) => t.key === activeTopic)?.key ?? null;
  }, [activeTopic]);

  const metrics = useMemo(() => {
    if (!results || !relevantCategory) return null;
    return {
      tfidf: computeMetrics(results.tfidf, relevantCategory, k),
      bm25:  computeMetrics(results.bm25, relevantCategory, k),
      rrf:   computeMetrics(results.rrf, relevantCategory, k),
    };
  }, [results, relevantCategory, k]);

  const chartData = useMemo(() => {
    if (!metrics) return null;
    return METHODS.map(({ key, label }) => ({
      name: label,
      Precision: metrics[key].precision,
      Recall: metrics[key].recall,
      F1: metrics[key].f1,
    }));
  }, [metrics]);

  return (
    <div className="playground-root">

      {/* ── Hero ── */}
      <div className="pg-hero">
        <div className="pg-hero-text">
          <h2 className="pg-hero-title">Search 78 news articles with three different retrieval methods.</h2>
          <p className="pg-hero-sub">Pick a topic, type a query, and see how TF-IDF, BM25, and Hybrid RRF rank the same articles differently. The charts show Precision, Recall, and F1 in real time.</p>
        </div>
        <div className="pg-steps">
          <div className="pg-step"><span className="pg-step-num">1</span><span>Pick a topic shortcut below</span></div>
          <div className="pg-step-arrow">›</div>
          <div className="pg-step"><span className="pg-step-num">2</span><span>Choose how many results to retrieve (K)</span></div>
          <div className="pg-step-arrow">›</div>
          <div className="pg-step"><span className="pg-step-num">3</span><span>Compare results and metrics across all three methods</span></div>
        </div>
      </div>

      {/* ── Controls ── */}
      <div className="controls-panel">
        <div className="controls-section">
          <label className="controls-label">Query</label>
          <input
            className="query-input"
            type="text"
            value={query}
            onChange={handleQueryChange}
            placeholder="Type a query or pick a topic below…"
          />
        </div>

        <div className="controls-section">
          <label className="controls-label">Quick topics</label>
          <div className="topic-pills">
            {TOPICS.map((t) => (
              <button
                key={t.key}
                type="button"
                className={`topic-pill ${activeTopic === t.key ? "active" : ""}`}
                onClick={() => handleTopic(t)}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>

        <div className="controls-section controls-section--inline">
          <label className="controls-label">Top-K</label>
          <div className="k-pills">
            {K_OPTIONS.map((v) => (
              <button
                key={v}
                type="button"
                className={`k-pill ${k === v ? "active" : ""}`}
                onClick={() => setK(v)}
              >
                K = {v}
              </button>
            ))}
          </div>
        </div>

        {!activeTopic && query.trim() && (
          <p className="controls-hint">
            Select a topic shortcut to enable precision / recall metrics (ground-truth labels required).
          </p>
        )}
      </div>

      {/* ── No query yet ── */}
      {!query.trim() && (
        <div className="empty-state">
          <div className="empty-icon">⚡</div>
          <div className="empty-title">Pick a topic or type a query to start</div>
          <div className="empty-sub">Three retrieval methods run in parallel , compare their results instantly.</div>
        </div>
      )}

      {/* ── Results + Charts ── */}
      {results && (
        <>
          {/* Metrics chart */}
          {chartData && (
            <div className="chart-panel">
              <div className="chart-header">
                <span className="chart-title">Metrics @ K={k}</span>
                <span className="chart-sub">Ground truth: all <strong>{relevantCategory}</strong> articles ({ARTICLES.filter(a => a.category === relevantCategory).length} total)</span>
              </div>

              <div className="chart-grid">
                {/* Bar chart */}
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey="name" tick={{ fontSize: 13, fontWeight: 700 }} />
                      <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 12 }} />
                      <Tooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} />
                      <Legend />
                      <Bar dataKey="Precision" fill={CHART_COLORS.precision} radius={[4, 4, 0, 0]} />
                      <Bar dataKey="Recall"    fill={CHART_COLORS.recall}    radius={[4, 4, 0, 0]} />
                      <Bar dataKey="F1"        fill={CHART_COLORS.f1}        radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Metric cards */}
                <div className="metric-cards">
                  {METHODS.map(({ key, label, color }) => (
                    <div key={key} className="metric-card" style={{ borderLeftColor: color }}>
                      <div className="metric-card-title" style={{ color }}>{label}</div>
                      <MetricBar label="Precision" value={metrics[key].precision} color={CHART_COLORS.precision} />
                      <MetricBar label="Recall"    value={metrics[key].recall}    color={CHART_COLORS.recall} />
                      <MetricBar label="F1"        value={metrics[key].f1}        color={CHART_COLORS.f1} />
                      <div className="metric-card-foot">
                        {metrics[key].relevantFound}/{k} relevant · {metrics[key].totalRelevant} in corpus
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Result columns */}
          <div className="results-grid">
            {METHODS.map(({ key, label, color }) => (
              <div key={key} className="results-col">
                <div className="results-col-header" style={{ borderBottomColor: color }}>
                  <span className="results-col-label" style={{ color }}>{label}</span>
                  <span className="results-col-k">top {k}</span>
                </div>
                {results[key].slice(0, k).map((doc, rank) => (
                  <ResultCard
                    key={doc.id}
                    doc={doc}
                    rank={rank}
                    isRelevant={relevantCategory ? doc.category === relevantCategory : null}
                  />
                ))}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
