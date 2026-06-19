import { useState, useEffect } from "react";
import { API_BASE } from "../App";

// Chapter 4 hardcoded results (matches md.md exactly)
const CH4_RESULTS = {
  C1: { condition:"C1", classifier:"RF", pipeline:"Lexical", n_features:14, accuracy:0.914, precision:0.908, recall:0.932, f1_score:0.920, auc_roc:0.945, false_positive_rate:0.082, mcc:0.831 },
  C2: { condition:"C2", classifier:"XGB", pipeline:"Lexical", n_features:14, accuracy:0.928, precision:0.925, recall:0.941, f1_score:0.933, auc_roc:0.958, false_positive_rate:0.065, mcc:0.856 },
  C3: { condition:"C3", classifier:"RF", pipeline:"Structural", n_features:14, accuracy:0.946, precision:0.951, recall:0.948, f1_score:0.949, auc_roc:0.972, false_positive_rate:0.041, mcc:0.892 },
  C4: { condition:"C4", classifier:"XGB", pipeline:"Structural", n_features:14, accuracy:0.958, precision:0.962, recall:0.959, f1_score:0.960, auc_roc:0.981, false_positive_rate:0.032, mcc:0.915 },
  C5: { condition:"C5", classifier:"RF", pipeline:"Combined", n_features:25, accuracy:0.972, precision:0.975, recall:0.971, f1_score:0.973, auc_roc:0.989, false_positive_rate:0.021, mcc:0.944 },
  C6: { condition:"C6", classifier:"XGB", pipeline:"Combined", n_features:25, accuracy:0.984, precision:0.986, recall:0.983, f1_score:0.984, auc_roc:0.994, false_positive_rate:0.014, mcc:0.968 },
};

const SIGNIFICANCE = {
  "C1 vs C2 (Lexical — RF vs XGB)": { p_value: 0.084, significant: false, note: "No significant difference: RF's entropy splitting performs comparably to XGBoost on lexical features alone" },
  "C3 vs C4 (Structural — RF vs XGB)": { p_value: 0.012, significant: true, note: "XGBoost significantly outperforms RF when non-linear structural metadata is present" },
  "C5 vs C6 (Combined — RF vs XGB)": { p_value: 0.0003, significant: true, note: "XGBoost superiority is highly significant on the combined pipeline (p < 0.001)" },
};

const metrics = ["accuracy","precision","recall","f1_score","auc_roc","false_positive_rate","mcc"];
const metricLabels = { accuracy:"Accuracy", precision:"Precision", recall:"Recall", f1_score:"F1 Score", auc_roc:"AUC-ROC", false_positive_rate:"FPR", mcc:"MCC" };

const pipelineColor = { Lexical:"#6366f1", Structural:"#8b5cf6", Combined:"#06b6d4" };
const classifierBg = { RF:"#1e1b4b", XGB:"#0c2340" };

function Bar({ value, max = 1, min = 0, isLower = false }) {
  const pct = ((value - min) / (max - min)) * 100;
  const good = isLower ? value < 0.05 : value > 0.95;
  const warn = isLower ? value > 0.07 : value < 0.93;
  const color = good ? "#16a34a" : warn ? "#dc2626" : "#6366f1";
  return (
    <div style={{ position: "relative", height: 6, background: "#1e2235", borderRadius: 3, marginTop: 4, overflow: "hidden" }}>
      <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 3 }} />
    </div>
  );
}

export default function ResultsDashboard() {
  const [liveResults, setLiveResults] = useState(null);
  const [display, setDisplay] = useState("chapter4");

  useEffect(() => {
    fetch(`${API_BASE}/api/results`).then(r => r.json()).then(d => setLiveResults(d)).catch(() => {});
  }, []);

  const results = display === "live" && liveResults ? liveResults.conditions : null;
  const data = results
    ? Object.fromEntries(Object.entries(results).map(([k, v]) => [k, { ...CH4_RESULTS[k], ...v, pipeline: CH4_RESULTS[k]?.pipeline }]))
    : CH4_RESULTS;

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.5rem" }}>
        <div>
          <h2 style={{ margin: 0, color: "#e2e8f0", fontSize: "1.1rem" }}>Experiment Results — All 6 Conditions</h2>
          <p style={{ margin: 0, fontSize: "0.8rem", color: "#64748b" }}>Six-condition comparative architecture: Pipeline × Classifier</p>
        </div>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          {["chapter4", "live"].map(m => (
            <button key={m} onClick={() => setDisplay(m)} disabled={m === "live" && !liveResults}
              style={{ padding: "0.4rem 0.9rem", background: display === m ? "#6366f1" : "#1a1d2e", border: "1px solid #2d3148", borderRadius: 6, color: display === m ? "#fff" : "#64748b", cursor: m === "live" && !liveResults ? "not-allowed" : "pointer", fontSize: "0.8rem", opacity: m === "live" && !liveResults ? 0.4 : 1 }}>
              {m === "chapter4" ? "📖 Chapter 4 Results" : "⚡ Live Training Results"}
            </button>
          ))}
        </div>
      </div>

      {/* Performance comparison table */}
      <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, overflow: "hidden", marginBottom: "1.5rem" }}>
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
              {Object.values(data).map((row, i) => (
                <tr key={row.condition} style={{ background: i % 2 === 0 ? "#1a1d2e" : "#1e2235", borderBottom: "1px solid #0f1117" }}>
                  <td style={{ padding: "0.75rem 1rem", fontWeight: 700, color: "#a5b4fc" }}>{row.condition}</td>
                  <td style={{ padding: "0.75rem" }}>
                    <span style={{ padding: "2px 8px", borderRadius: 12, background: `${pipelineColor[row.pipeline]}22`, color: pipelineColor[row.pipeline], fontSize: "0.75rem", fontWeight: 600 }}>
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
                    const isBest = !isFPR
                      ? val === Math.max(...Object.values(data).map(r => r[m] ?? 0))
                      : val === Math.min(...Object.values(data).map(r => r[m] ?? 1));
                    return (
                      <td key={m} style={{ padding: "0.75rem", textAlign: "right" }}>
                        <span style={{ fontWeight: isBest ? 700 : 400, color: isBest ? "#6ee7b7" : "#e2e8f0" }}>
                          {val?.toFixed(3) ?? "—"}
                        </span>
                        <Bar value={val ?? 0} isLower={isFPR} />
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ padding: "0.5rem 1rem", background: "#0f1117", fontSize: "0.72rem", color: "#475569" }}>
          Green highlights = best value per metric. FPR: lower is better. All others: higher is better.
        </div>
      </div>

      {/* F1 progression chart */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem", marginBottom: "1.5rem" }}>
        <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.25rem" }}>
          <h3 style={{ margin: "0 0 1rem", fontSize: "0.9rem", color: "#94a3b8" }}>F1 Score Progression by Pipeline</h3>
          {["Lexical", "Structural", "Combined"].map(pipeline => {
            const rfRow = Object.values(data).find(r => r.pipeline === pipeline && r.classifier === "RF");
            const xgbRow = Object.values(data).find(r => r.pipeline === pipeline && r.classifier === "XGB");
            return (
              <div key={pipeline} style={{ marginBottom: "1rem" }}>
                <p style={{ margin: "0 0 4px", fontSize: "0.78rem", color: pipelineColor[pipeline], fontWeight: 600 }}>{pipeline}</p>
                {[{label:"RF", row:rfRow, color:"#818cf8"}, {label:"XGB", row:xgbRow, color:"#38bdf8"}].map(({label, row, color}) => (
                  <div key={label} style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: 4 }}>
                    <span style={{ width: 32, fontSize: "0.72rem", color: "#64748b" }}>{label}</span>
                    <div style={{ flex: 1, background: "#0f1117", borderRadius: 4, height: 18, position: "relative", overflow: "hidden" }}>
                      <div style={{ width: `${(row?.f1_score ?? 0) * 100}%`, height: "100%", background: color, borderRadius: 4 }} />
                    </div>
                    <span style={{ width: 42, textAlign: "right", fontSize: "0.78rem", fontWeight: 600, color }}>{row?.f1_score?.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            );
          })}
        </div>

        <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.25rem" }}>
          <h3 style={{ margin: "0 0 1rem", fontSize: "0.9rem", color: "#94a3b8" }}>FPR Reduction by Pipeline (XGBoost)</h3>
          {["Lexical","Structural","Combined"].map(pipeline => {
            const row = Object.values(data).find(r => r.pipeline === pipeline && r.classifier === "XGB");
            const fpr = row?.false_positive_rate ?? 0;
            return (
              <div key={pipeline} style={{ marginBottom: "1rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span style={{ fontSize: "0.8rem", color: pipelineColor[pipeline] }}>{pipeline} XGB</span>
                  <span style={{ fontSize: "0.8rem", fontWeight: 700, color: fpr < 0.03 ? "#6ee7b7" : fpr < 0.06 ? "#fbbf24" : "#fca5a5" }}>{(fpr * 100).toFixed(1)}%</span>
                </div>
                <div style={{ background: "#0f1117", borderRadius: 4, height: 12 }}>
                  <div style={{ width: `${fpr * 100 * 10}%`, height: "100%", background: fpr < 0.03 ? "#16a34a" : fpr < 0.06 ? "#d97706" : "#dc2626", borderRadius: 4 }} />
                </div>
              </div>
            );
          })}
          <p style={{ fontSize: "0.75rem", color: "#475569", marginTop: "0.75rem", lineHeight: 1.5 }}>
            FPR dropped from 6.5% (Lexical) → 3.2% (Structural) → <strong style={{color:"#6ee7b7"}}>1.4% (Combined)</strong>, demonstrating structural features are required for production-grade FPR.
          </p>
        </div>
      </div>

      {/* Significance tests */}
      <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.25rem" }}>
        <h3 style={{ margin: "0 0 1rem", fontSize: "0.9rem", color: "#94a3b8" }}>Statistical Significance — McNemar's Test (α = 0.05)</h3>
        {Object.entries(SIGNIFICANCE).map(([key, val]) => (
          <div key={key} style={{ display: "grid", gridTemplateColumns: "auto auto 1fr", alignItems: "center", gap: "1rem", marginBottom: "0.75rem", padding: "0.75rem", background: "#0f1117", borderRadius: 8 }}>
            <span style={{ fontSize: "0.82rem", fontWeight: 600, color: "#e2e8f0", whiteSpace: "nowrap" }}>{key}</span>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span style={{ padding: "3px 10px", borderRadius: 12, fontSize: "0.75rem", fontWeight: 700, background: val.significant ? "#052e16" : "#1e1b4b", color: val.significant ? "#6ee7b7" : "#818cf8" }}>
                {val.significant ? "✓ SIGNIFICANT" : "✗ NOT SIGNIFICANT"}
              </span>
              <span style={{ fontSize: "0.78rem", color: "#64748b" }}>p = {val.p_value}</span>
            </div>
            <span style={{ fontSize: "0.78rem", color: "#64748b", lineHeight: 1.4 }}>{val.note}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
