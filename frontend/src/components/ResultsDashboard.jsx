import { useState, useEffect } from "react";
import { API_BASE } from "../App";

const CH4_RESULTS = {
  C1: { condition:"C1", classifier:"RF", pipeline:"Lexical", n_features:14, accuracy:0.914, precision:0.908, recall:0.932, f1_score:0.920, auc_roc:0.945, false_positive_rate:0.082, mcc:0.831 },
  C2: { condition:"C2", classifier:"XGB", pipeline:"Lexical", n_features:14, accuracy:0.928, precision:0.925, recall:0.941, f1_score:0.933, auc_roc:0.958, false_positive_rate:0.065, mcc:0.856 },
  C3: { condition:"C3", classifier:"RF", pipeline:"Structural", n_features:14, accuracy:0.946, precision:0.951, recall:0.948, f1_score:0.949, auc_roc:0.972, false_positive_rate:0.041, mcc:0.892 },
  C4: { condition:"C4", classifier:"XGB", pipeline:"Structural", n_features:14, accuracy:0.958, precision:0.962, recall:0.959, f1_score:0.960, auc_roc:0.981, false_positive_rate:0.032, mcc:0.915 },
  C5: { condition:"C5", classifier:"RF", pipeline:"Combined", n_features:25, accuracy:0.972, precision:0.975, recall:0.971, f1_score:0.973, auc_roc:0.989, false_positive_rate:0.021, mcc:0.944 },
  C6: { condition:"C6", classifier:"XGB", pipeline:"Combined", n_features:25, accuracy:0.984, precision:0.986, recall:0.983, f1_score:0.984, auc_roc:0.994, false_positive_rate:0.014, mcc:0.968 },
};

const COMPARISONS = {
  lexical: {
    label: "Lexical Pipeline (RF vs XGB)",
    rf: "C1",
    xgb: "C2",
    f1_gain: 0.013,
    significant: false,
    p_value: 0.084,
    interpretation: "No significant difference: RF entropy based splitting performs comparably to XGBoost gradient boosting on lexical features alone"
  },
  structural: {
    label: "Structural Pipeline (RF vs XGB)",
    rf: "C3",
    xgb: "C4",
    f1_gain: 0.011,
    significant: true,
    p_value: 0.012,
    interpretation: "XGBoost significantly outperforms RF when non linear structural metadata is present due to its L1/L2 regularisation and second order gradient optimisation"
  },
  combined: {
    label: "Combined Pipeline (RF vs XGB)",
    rf: "C5",
    xgb: "C6",
    f1_gain: 0.011,
    significant: true,
    p_value: 0.0003,
    interpretation: "XGBoost superiority is highly statistically significant on the combined pipeline (p < 0.001) confirming XGBoost as the optimal classifier for the full feature space"
  }
};

const PIPELINE_GAINS = [
  { from: "C1 (Lexical RF)", to: "C3 (Structural RF)", f1_gain: 0.029, note: "Structural features improve RF by +0.029 F1 over lexical alone" },
  { from: "C2 (Lexical XGB)", to: "C4 (Structural XGB)", f1_gain: 0.027, note: "Structural features improve XGBoost by +0.027 F1 over lexical alone" },
  { from: "C3 (Structural RF)", to: "C5 (Combined RF)", f1_gain: 0.024, note: "Adding lexical to structural improves RF further by +0.024 F1" },
  { from: "C4 (Structural XGB)", to: "C6 (Combined XGB)", f1_gain: 0.024, note: "Adding lexical to structural improves XGBoost by +0.024 F1" },
  { from: "C1 (Lexical RF)", to: "C6 (Combined XGB)", f1_gain: 0.064, note: "Full improvement from weakest to strongest condition is +0.064 F1 across both feature and classifier dimensions" },
];

const metrics = ["accuracy","precision","recall","f1_score","auc_roc","false_positive_rate","mcc"];
const metricLabels = { accuracy:"Accuracy", precision:"Precision", recall:"Recall", f1_score:"F1 Score", auc_roc:"AUC ROC", false_positive_rate:"FPR", mcc:"MCC" };

const pipelineColor = { Lexical:"#6366f1", Structural:"#8b5cf6", Combined:"#06b6d4" };
const classifierBg = { RF:"#1e1b4b", XGB:"#0c2340" };

const RISK = {
  HIGH: { bg: "rgba(224,89,91,0.1)", border: "rgba(224,89,91,0.3)", text: "#e0595b" },
  MEDIUM: { bg: "rgba(217,140,58,0.1)", border: "rgba(217,140,58,0.3)", text: "#d98c3a" },
  LOW: { bg: "rgba(80,184,122,0.1)", border: "rgba(80,184,122,0.3)", text: "#50b87a" },
};

function safeVal(v) {
  return v !== null && v !== undefined && !Number.isNaN(v) ? v : 0;
}

function Bar({ value, max = 1, min = 0, isLower = false }) {
  const v = safeVal(value);
  const pct = max !== min ? ((v - min) / (max - min)) * 100 : 50;
  const good = isLower ? v < 0.05 : v > 0.95;
  const warn = isLower ? v > 0.07 : v < 0.93;
  const color = good ? "#16a34a" : warn ? "#dc2626" : "#6366f1";
  try {
    return (
      <div style={{ position: "relative", height: 6, background: "#1e2235", borderRadius: 3, marginTop: 4, overflow: "hidden" }}>
        <div style={{ width: `${Math.min(Math.max(pct, 0), 100)}%`, height: "100%", background: color, borderRadius: 3 }} />
      </div>
    );
  } catch { return null; }
}

function ComparisonCard({ label, rfId, xgbId, data, f1_gain, significant, p_value, interpretation }) {
  const rf = data[rfId];
  const xgb = data[xgbId];
  if (!rf || !xgb) return null;
  try {
    return (
      <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.25rem", marginBottom: "1rem" }}>
        <h4 style={{ margin: "0 0 0.75rem", fontSize: "0.85rem", color: "#e2e8f0" }}>{label}</h4>
        <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: "1rem", alignItems: "center", marginBottom: "0.75rem" }}>
          <div style={{ textAlign: "center", padding: "0.75rem", background: "#0f1117", borderRadius: 8 }}>
            <p style={{ margin: 0, fontSize: "0.7rem", color: "#64748b" }}>Random Forest</p>
            <p style={{ margin: "0.15rem 0", fontSize: "1.1rem", fontWeight: 700, color: "#818cf8" }}>{rf.f1_score.toFixed(3)}</p>
            <p style={{ margin: 0, fontSize: "0.65rem", color: "#475569" }}>F1 Score</p>
          </div>
          <div style={{ textAlign: "center" }}>
            <span style={{ fontWeight: 700, fontSize: "0.9rem", color: f1_gain >= 0 ? "#6ee7b7" : "#fca5a5" }}>
              +{f1_gain.toFixed(3)}
            </span>
            <p style={{ margin: "2px 0 0", fontSize: "0.65rem", color: "#475569" }}>F1 Gain</p>
          </div>
          <div style={{ textAlign: "center", padding: "0.75rem", background: "#0f1117", borderRadius: 8 }}>
            <p style={{ margin: 0, fontSize: "0.7rem", color: "#64748b" }}>XGBoost</p>
            <p style={{ margin: "0.15rem 0", fontSize: "1.1rem", fontWeight: 700, color: "#38bdf8" }}>{xgb.f1_score.toFixed(3)}</p>
            <p style={{ margin: 0, fontSize: "0.65rem", color: "#475569" }}>F1 Score</p>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", padding: "0.5rem 0.75rem", background: "#0f1117", borderRadius: 6, flexWrap: "wrap" }}>
          <span style={{ padding: "2px 8px", borderRadius: 10, fontSize: "0.7rem", fontWeight: 700, background: significant ? "#052e16" : "#1e1b4b", color: significant ? "#6ee7b7" : "#818cf8" }}>
            {significant ? "Statistically Significant" : "Not Significant"}
          </span>
          <span style={{ fontSize: "0.72rem", color: "#64748b" }}>McNemar p = {p_value}</span>
          <span style={{ fontSize: "0.72rem", color: "#5c6078", flex: 1 }}>{interpretation}</span>
        </div>
      </div>
    );
  } catch { return null; }
}

function AnalysisRow({ r, i }) {
  try {
    if (r.mode === "batch") {
      const summary = r.summary || { total: 0, phishing: 0, legitimate: 0, errors: 0 };
      const resultsList = r.results || [];
      return (
        <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.25rem", marginBottom: "1rem" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.75rem" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span style={{ fontWeight: 600, fontSize: "0.85rem", color: "#e2e8f0" }}>Batch Analysis</span>
            </div>
            {r.timestamp && <span style={{ fontSize: "0.68rem", color: "#475569" }}>{new Date(r.timestamp).toLocaleTimeString()}</span>}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "0.5rem" }}>
            {[
              { label: "Total", value: safeVal(summary.total), c: "#6366f1" },
              { label: "Phishing", value: safeVal(summary.phishing), c: "#e0595b" },
              { label: "Legitimate", value: safeVal(summary.legitimate), c: "#50b87a" },
              { label: "Errors", value: safeVal(summary.errors), c: "#d98c3a" },
            ].map(s => (
              <div key={s.label} style={{ textAlign: "center", padding: "0.5rem", background: "#0f1117", borderRadius: 8 }}>
                <p style={{ fontSize: "0.65rem", color: "#64748b", margin: 0 }}>{s.label}</p>
                <p style={{ fontSize: "1.2rem", fontWeight: 700, color: s.c, margin: "0.1rem 0 0" }}>{s.value}</p>
              </div>
            ))}
          </div>
          {resultsList.length > 0 && (
            <div style={{ marginTop: "0.75rem", maxHeight: 200, overflowY: "auto" }}>
              {resultsList.slice(0, 10).map((item, j) => {
                if (!item) return null;
                const rc = item.error ? RISK.HIGH : RISK[item.risk_level] || RISK.LOW;
                return (
                  <div key={j} style={{ display: "flex", alignItems: "center", gap: "0.5rem", padding: "0.35rem 0", borderBottom: j < Math.min(resultsList.length, 10) - 1 ? "1px solid #2d3148" : "none" }}>
                    <span style={{ fontSize: "0.8rem", flexShrink: 0 }}>
                      {item.error ? "\u2717" : item.prediction === "phishing" ? "\u26A0" : "\u2713"}
                    </span>
                    <span style={{ fontFamily: "monospace", flex: 1, fontSize: "0.72rem", color: "#94a3b8", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.url || ""}</span>
                    {!item.error && (
                      <>
                        <span style={{ padding: "2px 8px", borderRadius: 12, background: rc.bg, color: rc.text, border: `1px solid ${rc.border}`, fontSize: "0.65rem", fontWeight: 600 }}>{item.risk_level}</span>
                        <span style={{ fontSize: "0.72rem", color: rc.text, fontWeight: 600, minWidth: 40, textAlign: "right" }}>{safeVal(item.phishing_probability) > 0 ? (item.phishing_probability * 100).toFixed(1) + "%" : ""}</span>
                      </>
                    )}
                  </div>
                );
              })}
              {resultsList.length > 10 && <p style={{ fontSize: "0.68rem", color: "#475569", textAlign: "center", marginTop: "0.4rem" }}>Plus {resultsList.length - 10} more</p>}
            </div>
          )}
        </div>
      );
    }

    const rc = RISK[r.risk_level] || RISK.LOW;
    return (
      <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderLeft: `3px solid ${rc.text}`, borderRadius: 10, padding: "1.25rem", marginBottom: "0.75rem" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ fontSize: "1rem" }}>{r.prediction === "phishing" ? "\u26A0" : "\u2713"}</span>
            <strong style={{ fontSize: "0.85rem", color: rc.text }}>{(r.prediction || "").toUpperCase()}</strong>
            {r.risk_level && (
              <span style={{ padding: "2px 8px", borderRadius: 12, background: rc.bg, color: rc.text, border: `1px solid ${rc.border}`, fontSize: "0.65rem", fontWeight: 700 }}>
                {r.risk_level}
              </span>
            )}
          </div>
          {r.timestamp && <span style={{ fontSize: "0.68rem", color: "#475569" }}>{new Date(r.timestamp).toLocaleTimeString()}</span>}
        </div>
        <p style={{ fontFamily: "monospace", color: "#94a3b8", fontSize: "0.75rem", wordBreak: "break-all", margin: "0 0 0.5rem" }}>{r.url || ""}</p>
        <div style={{ display: "flex", gap: "1.25rem", flexWrap: "wrap" }}>
          {[
            { label: "Probability", value: safeVal(r.phishing_probability) > 0 ? `${(r.phishing_probability * 100).toFixed(1)}%` : "" },
            { label: "Confidence", value: safeVal(r.confidence) > 0 ? `${(r.confidence * 100).toFixed(1)}%` : "" },
            { label: "Pipeline", value: r.pipeline },
            { label: "Classifier", value: r.classifier },
            { label: "Condition", value: r.condition },
          ].filter(m => m.value).map(m => (
            <div key={m.label}>
              <span style={{ fontSize: "0.62rem", color: "#475569", display: "block" }}>{m.label}</span>
              <span style={{ fontSize: "0.78rem", fontWeight: 600, color: "#e2e8f0" }}>{m.value}</span>
            </div>
          ))}
        </div>
      </div>
    );
  } catch { return null; }
}

export default function ResultsDashboard({ analysisResults, onClearResults }) {
  const [liveResults, setLiveResults] = useState(null);
  const [display, setDisplay] = useState("chapter4");
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/results`)
      .then(r => r.json())
      .then(d => setLiveResults(d))
      .catch(() => {});
  }, []);

  const results = display === "live" && liveResults ? liveResults.conditions : null;
  const data = results
    ? Object.fromEntries(Object.entries(results).map(([k, v]) => [k, { ...CH4_RESULTS[k], ...v, pipeline: CH4_RESULTS[k]?.pipeline }]))
    : CH4_RESULTS;

  const metricsList = [
    { row: data.C1, c: data.C2, label: "Lexical" },
    { row: data.C3, c: data.C4, label: "Structural" },
    { row: data.C5, c: data.C6, label: "Combined" },
  ];

  const bestF1 = Object.values(data).reduce((best, r) => (r.f1_score > (best?.f1_score || 0) ? r : best), null);

  return (
    <div>
      {error && (
        <div style={{ marginBottom: "1rem", padding: "0.75rem 1rem", background: "#450a0a", border: "1px solid #991b1b", borderRadius: 8, color: "#fca5a5", fontSize: "0.85rem" }}>
          {error}
        </div>
      )}

      {analysisResults && analysisResults.length > 0 && (
        <div style={{ marginBottom: "2rem" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem", flexWrap: "wrap", gap: "0.75rem" }}>
            <div>
              <h2 style={{ margin: 0, fontSize: "1.05rem", color: "#e2e8f0" }}>URL Analysis Results</h2>
              <p style={{ margin: "0.15rem 0 0", fontSize: "0.78rem", color: "#64748b" }}>
                {analysisResults.length} analysis{analysisResults.length !== 1 ? "s" : ""} recorded this session
              </p>
            </div>
            <button onClick={onClearResults} style={{ padding: "0.35rem 0.8rem", background: "#dc2626", border: "none", borderRadius: 6, color: "#fff", cursor: "pointer", fontSize: "0.75rem", fontWeight: 600 }}>Clear history</button>
          </div>
          {analysisResults.map((r, i) => (
            <AnalysisRow key={r.timestamp || i} r={r} i={i} />
          ))}
        </div>
      )}

      {!analysisResults?.length && (
        <div style={{ marginBottom: "2rem", padding: "2rem", background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, textAlign: "center" }}>
          <p style={{ fontSize: "0.9rem", color: "#94a3b8", marginBottom: "0.5rem" }}>No URL analyses yet</p>
          <p style={{ fontSize: "0.78rem", color: "#64748b" }}>Go to the URL Analysis tab to analyse a URL. Results will appear here for cross reference against published chapter 4 findings.</p>
        </div>
      )}

      {/* Comparative Analysis Section */}
      <div style={{ marginBottom: "1.5rem" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.5rem" }}>
          <div>
            <h2 style={{ margin: 0, color: "#e2e8f0", fontSize: "1.1rem" }}>Comparative Analysis: All 6 Conditions</h2>
            <p style={{ margin: "0.15rem 0 0", fontSize: "0.8rem", color: "#64748b" }}>
              Six condition controlled architecture: 3 pipelines (Lexical, Structural, Combined) x 2 classifiers (Random Forest, XGBoost)
            </p>
          </div>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            {["chapter4", "live"].map(m => (
              <button key={m} onClick={() => setDisplay(m)} disabled={m === "live" && !liveResults}
                style={{ padding: "0.4rem 0.9rem", background: display === m ? "#6366f1" : "#1a1d2e", border: "1px solid #2d3148", borderRadius: 6, color: display === m ? "#fff" : "#64748b", cursor: m === "live" && !liveResults ? "not-allowed" : "pointer", fontSize: "0.8rem", opacity: m === "live" && !liveResults ? 0.4 : 1 }}>
                {m === "chapter4" ? "Published Results" : "Live Training"}
              </button>
            ))}
          </div>
        </div>

        {/* Executive Summary */}
        {bestF1 && (
          <div style={{ background: "linear-gradient(135deg, #0f1117, #1a1d2e)", border: "1px solid #2d3148", borderRadius: 10, padding: "1.25rem", marginBottom: "1.5rem" }}>
            <h3 style={{ margin: "0 0 0.75rem", fontSize: "0.9rem", color: "#94a3b8" }}>Executive Summary</h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem" }}>
              <div style={{ padding: "0.75rem", background: "#0f1117", borderRadius: 8 }}>
                <p style={{ margin: 0, fontSize: "0.65rem", color: "#475569" }}>Best Condition</p>
                <p style={{ margin: "0.15rem 0", fontSize: "1rem", fontWeight: 700, color: "#a5b4fc" }}>{bestF1.condition}</p>
                <p style={{ margin: 0, fontSize: "0.7rem", color: "#64748b" }}>{bestF1.pipeline} + {bestF1.classifier}</p>
              </div>
              <div style={{ padding: "0.75rem", background: "#0f1117", borderRadius: 8 }}>
                <p style={{ margin: 0, fontSize: "0.65rem", color: "#475569" }}>Best F1 Score</p>
                <p style={{ margin: "0.15rem 0", fontSize: "1.3rem", fontWeight: 700, color: "#6ee7b7" }}>{bestF1.f1_score.toFixed(3)}</p>
                <p style={{ margin: 0, fontSize: "0.7rem", color: "#64748b" }}>Overall best performing condition</p>
              </div>
              <div style={{ padding: "0.75rem", background: "#0f1117", borderRadius: 8 }}>
                <p style={{ margin: 0, fontSize: "0.65rem", color: "#475569" }}>Best AUC ROC</p>
                <p style={{ margin: "0.15rem 0", fontSize: "1.3rem", fontWeight: 700, color: "#6ee7b7" }}>{bestF1.auc_roc.toFixed(3)}</p>
                <p style={{ margin: 0, fontSize: "0.7rem", color: "#64748b" }}>Best AUC ROC across all conditions</p>
              </div>
              <div style={{ padding: "0.75rem", background: "#0f1117", borderRadius: 8 }}>
                <p style={{ margin: 0, fontSize: "0.65rem", color: "#475569" }}>FPR at Best</p>
                <p style={{ margin: "0.15rem 0", fontSize: "1.1rem", fontWeight: 700, color: bestF1.false_positive_rate < 0.02 ? "#6ee7b7" : "#fbbf24" }}>{(bestF1.false_positive_rate * 100).toFixed(1)}%</p>
                <p style={{ margin: 0, fontSize: "0.7rem", color: "#64748b" }}>False positive rate at best condition</p>
              </div>
            </div>
          </div>
        )}

        {/* Classifier Comparison Cards */}
        <h3 style={{ margin: "0 0 1rem", fontSize: "0.9rem", color: "#94a3b8" }}>Classifier Comparison by Pipeline</h3>
        {Object.values(COMPARISONS).map(cmp => (
          <ComparisonCard key={cmp.label} {...cmp} data={data} />
        ))}

        {/* Pipeline Improvement Summary */}
        <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.25rem", marginBottom: "1.5rem" }}>
          <h3 style={{ margin: "0 0 0.75rem", fontSize: "0.9rem", color: "#94a3b8" }}>Pipeline Improvement Summary</h3>
          <p style={{ fontSize: "0.78rem", color: "#64748b", marginBottom: "1rem" }}>
            Each pipeline addition demonstrates measurable F1 improvement. The combined pipeline with XGBoost delivers the highest performance across all metrics.
          </p>
          {PIPELINE_GAINS.map((gain, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.6rem", padding: "0.5rem 0.75rem", background: "#0f1117", borderRadius: 6 }}>
              <div style={{ fontSize: "0.72rem", color: "#94a3b8", minWidth: 220 }}>
                {gain.from} <span style={{ color: "#475569" }}>to</span> {gain.to}
              </div>
              <div style={{ flex: 1, height: 8, background: "#1e2235", borderRadius: 4, overflow: "hidden" }}>
                <div style={{ width: `${Math.min(gain.f1_gain * 1000, 100)}%`, height: "100%", background: gain.f1_gain > 0.05 ? "#6ee7b7" : "#6366f1", borderRadius: 4 }} />
              </div>
              <span style={{ fontSize: "0.75rem", fontWeight: 600, color: "#6ee7b7", minWidth: 50, textAlign: "right" }}>+{gain.f1_gain.toFixed(3)}</span>
              <span style={{ fontSize: "0.68rem", color: "#5c6078", flex: 1 }}>{gain.note}</span>
            </div>
          ))}
        </div>

        {/* Performance Comparison Table */}
        <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, overflow: "hidden", marginBottom: "1.5rem" }}>
          <div style={{ padding: "0.75rem 1rem", borderBottom: "1px solid #2d3148" }}>
            <h3 style={{ margin: 0, fontSize: "0.85rem", color: "#e2e8f0" }}>Detailed Metrics by Condition</h3>
          </div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem" }}>
              <thead>
                <tr style={{ background: "#0f1117" }}>
                  <th style={{ padding: "0.75rem 1rem", textAlign: "left", color: "#64748b", fontWeight: 600 }}>Condition</th>
                  <th style={{ padding: "0.75rem", textAlign: "left", color: "#64748b", fontWeight: 600 }}>Pipeline</th>
                  <th style={{ padding: "0.75rem", textAlign: "left", color: "#64748b", fontWeight: 600 }}>Classifier</th>
                  {metrics.map(m => (
                    <th key={m} style={{ padding: "0.75rem", textAlign: "right", color: "#64748b", fontWeight: 600, whiteSpace: "nowrap" }}>{metricLabels[m]}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.values(data).map((row, i) => {
                  if (!row) return null;
                  return (
                    <tr key={row.condition} style={{ background: i % 2 === 0 ? "#1a1d2e" : "#1e2235", borderBottom: "1px solid #0f1117" }}>
                      <td style={{ padding: "0.75rem 1rem", fontWeight: 700, color: "#a5b4fc" }}>{row.condition}</td>
                      <td style={{ padding: "0.75rem" }}>
                        <span style={{ padding: "2px 8px", borderRadius: 12, background: `${(pipelineColor[row.pipeline] || "#6366f1")}22`, color: pipelineColor[row.pipeline] || "#6366f1", fontSize: "0.75rem", fontWeight: 600 }}>
                          {row.pipeline}
                        </span>
                      </td>
                      <td style={{ padding: "0.75rem" }}>
                        <span style={{ padding: "2px 8px", borderRadius: 12, background: classifierBg[row.classifier] || "#1e2235", color: row.classifier === "XGB" ? "#38bdf8" : "#818cf8", fontSize: "0.75rem", fontWeight: 600 }}>
                          {row.classifier}
                        </span>
                      </td>
                      {metrics.map(m => {
                        const val = row[m];
                        const isFPR = m === "false_positive_rate";
                        const vals = Object.values(data).filter(Boolean).map(r => r[m]).filter(v => v !== null && v !== undefined);
                        const maxV = vals.length > 0 ? Math.max(...vals) : 0;
                        const minV = vals.length > 0 ? Math.min(...vals) : 0;
                        const isBest = !isFPR ? val === maxV : val === minV;
                        return (
                          <td key={m} style={{ padding: "0.75rem", textAlign: "right" }}>
                            <span style={{ fontWeight: isBest ? 700 : 400, color: isBest ? "#6ee7b7" : "#e2e8f0" }}>
                              {safeVal(val) !== null && safeVal(val) !== undefined ? Number(val).toFixed(3) : ""}
                            </span>
                            <Bar value={val} isLower={isFPR} max={maxV} min={minV} />
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div style={{ padding: "0.5rem 1rem", background: "#0f1117", fontSize: "0.72rem", color: "#475569" }}>
            Green highlights = best value per metric. FPR: lower is better. All others: higher is better.
          </div>
        </div>

        {/* F1 Progression and FPR Charts */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem", marginBottom: "1.5rem" }}>
          <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.25rem" }}>
            <h3 style={{ margin: "0 0 1rem", fontSize: "0.9rem", color: "#94a3b8" }}>F1 Score Progression by Pipeline</h3>
            {["Lexical", "Structural", "Combined"].map(pipeline => {
              const rfRow = Object.values(data).find(r => r?.pipeline === pipeline && r?.classifier === "RF");
              const xgbRow = Object.values(data).find(r => r?.pipeline === pipeline && r?.classifier === "XGB");
              return (
                <div key={pipeline} style={{ marginBottom: "1rem" }}>
                  <p style={{ margin: "0 0 4px", fontSize: "0.78rem", color: pipelineColor[pipeline] || "#6366f1", fontWeight: 600 }}>{pipeline}</p>
                  {[{label:"RF", row:rfRow, color:"#818cf8"}, {label:"XGB", row:xgbRow, color:"#38bdf8"}].map(({label, row, color}) => (
                    <div key={label} style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: 4 }}>
                      <span style={{ width: 32, fontSize: "0.72rem", color: "#64748b" }}>{label}</span>
                      <div style={{ flex: 1, background: "#0f1117", borderRadius: 4, height: 18, position: "relative", overflow: "hidden" }}>
                        <div style={{ width: `${safeVal(row?.f1_score) * 100}%`, height: "100%", background: color, borderRadius: 4 }} />
                      </div>
                      <span style={{ width: 42, textAlign: "right", fontSize: "0.78rem", fontWeight: 600, color }}>{row?.f1_score?.toFixed(3) || ""}</span>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>

          <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.25rem" }}>
            <h3 style={{ margin: "0 0 1rem", fontSize: "0.9rem", color: "#94a3b8" }}>FPR Reduction by Pipeline (XGBoost)</h3>
            {["Lexical","Structural","Combined"].map(pipeline => {
              const row = Object.values(data).find(r => r?.pipeline === pipeline && r?.classifier === "XGB");
              const fpr = row?.false_positive_rate ?? 0;
              return (
                <div key={pipeline} style={{ marginBottom: "1rem" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontSize: "0.8rem", color: pipelineColor[pipeline] || "#6366f1" }}>{pipeline} XGB</span>
                    <span style={{ fontSize: "0.8rem", fontWeight: 700, color: fpr < 0.03 ? "#6ee7b7" : fpr < 0.06 ? "#fbbf24" : "#fca5a5" }}>{(fpr * 100).toFixed(1)}%</span>
                  </div>
                  <div style={{ background: "#0f1117", borderRadius: 4, height: 12 }}>
                    <div style={{ width: `${fpr * 100 * 10}%`, height: "100%", background: fpr < 0.03 ? "#16a34a" : fpr < 0.06 ? "#d97706" : "#dc2626", borderRadius: 4 }} />
                  </div>
                </div>
              );
            })}
            <p style={{ fontSize: "0.75rem", color: "#475569", marginTop: "0.75rem", lineHeight: 1.5 }}>
              FPR dropped from 6.5 percent (Lexical) to 3.2 percent (Structural) to <strong style={{color:"#6ee7b7"}}>1.4 percent (Combined)</strong>, demonstrating structural features are required for production grade FPR.
            </p>
          </div>
        </div>

        {/* Key Findings */}
        <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.25rem" }}>
          <h3 style={{ margin: "0 0 1rem", fontSize: "0.9rem", color: "#94a3b8" }}>Key Findings: Feature Classifier Interaction</h3>
          <div style={{ display: "grid", gap: "0.75rem" }}>
            {[
              { icon: "\u2460", title: "Lexical Features Favor Both Classifiers Equally", desc: "McNemars test confirms no statistically significant difference between RF and XGBoost on the isolated lexical pipeline (p = 0.084). RF entropy based splitting is inherently optimised for high dimensional string derived categorical features, performing comparably to XGBoost gradient optimisation." },
              { icon: "\u2461", title: "Structural Features Reveal XGBoost Advantage", desc: "XGBoost significantly outperforms RF on structural metadata (p = 0.012). XGBoost L1/L2 regularisation and second order gradient optimisation effectively model the sparse, continuous, non linear structural features." },
              { icon: "\u2462", title: "Combined Pipeline Delivers Maximum Performance", desc: "Both classifiers achieve their highest scores on the combined pipeline. XGBoost reaches F1 = 0.984 with AUC ROC = 0.994 and FPR = 1.4 percent, representing production grade phishing detection performance." },
              { icon: "\u2463", title: "Pipeline Improvement is Additive", desc: "Each feature category contributes measurable improvement. Moving from Lexical to Structural improves F1 by +0.029 (RF) and +0.027 (XGB). Adding the remaining category to reach Combined yields an additional +0.024 for both classifiers." },
              { icon: "\u2464", title: "Feature Category Gap Favors Structural", desc: "Structural features individually demonstrate higher discriminative power per feature than lexical features, because infrastructural metadata (domain age, SSL validity, DNS TTL) is significantly harder for attackers to manipulate than URL strings." },
            ].map((f, i) => (
              <div key={i} style={{ display: "flex", gap: "0.75rem", padding: "0.75rem", background: "#0f1117", borderRadius: 8 }}>
                <span style={{ fontSize: "1rem", color: "#6366f1", flexShrink: 0 }}>{f.icon}</span>
                <div>
                  <strong style={{ fontSize: "0.82rem", color: "#e2e8f0" }}>{f.title}</strong>
                  <p style={{ margin: "0.2rem 0 0", fontSize: "0.78rem", color: "#64748b", lineHeight: 1.5 }}>{f.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
