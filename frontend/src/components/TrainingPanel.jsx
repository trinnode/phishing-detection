import { useState, useEffect, useRef } from "react";
import { API_BASE } from "../App";

export default function TrainingPanel({ onTrainingComplete }) {
  const [status, setStatus] = useState(null);
  const [polling, setPolling] = useState(false);
  const [fastMode, setFastMode] = useState(true);
  const [dataDir, setDataDir] = useState("");
  const pollRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/models/status`).then(r => r.json()).then(setStatus).catch(() => {});
    return () => clearInterval(pollRef.current);
  }, []);

  async function startTraining() {
    const resp = await fetch(`${API_BASE}/api/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fast_mode: fastMode, data_dir: dataDir || null }),
    });
    if (resp.ok) {
      setPolling(true);
      pollRef.current = setInterval(async () => {
        const r = await fetch(`${API_BASE}/api/train/status`);
        const d = await r.json();
        setStatus(prev => ({ ...prev, training: d }));
        if (d.done || d.error) {
          clearInterval(pollRef.current);
          setPolling(false);
          if (d.done) onTrainingComplete?.();
          fetch(`${API_BASE}/api/models/status`).then(r => r.json()).then(s => setStatus(prev => ({ ...prev, ...s })));
        }
      }, 2000);
    }
  }

  const trainingState = status?.training;

  return (
    <div style={{ maxWidth: 800 }}>
      <h2 style={{ color: "#e2e8f0", fontSize: "1.1rem", marginBottom: "0.25rem" }}>Training Pipeline</h2>
      <p style={{ color: "#64748b", fontSize: "0.85rem", marginBottom: "1.5rem" }}>
        Trains all 6 experimental conditions with SMOTE, nested 10-fold CV, and GridSearchCV.
      </p>

      {/* Config card */}
      <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.5rem", marginBottom: "1.5rem" }}>
        <h3 style={{ margin: "0 0 1rem", fontSize: "0.9rem", color: "#94a3b8" }}>Configuration</h3>

        <div style={{ marginBottom: "1rem" }}>
          <label style={{ display: "flex", alignItems: "center", gap: "0.75rem", cursor: "pointer" }}>
            <input type="checkbox" checked={fastMode} onChange={e => setFastMode(e.target.checked)} style={{ accentColor: "#6366f1", width: 16, height: 16 }} />
            <div>
              <span style={{ color: "#e2e8f0", fontWeight: 600, fontSize: "0.9rem" }}>Fast Mode</span>
              <p style={{ margin: 0, fontSize: "0.78rem", color: "#64748b" }}>Reduced hyperparameter grid — completes in ~5 min. Uncheck for full GridSearch (~hours).</p>
            </div>
          </label>
        </div>

        <div style={{ marginBottom: "1.25rem" }}>
          <label style={{ fontSize: "0.8rem", color: "#94a3b8", display: "block", marginBottom: "0.4rem" }}>
            Real Dataset Directory (optional)
          </label>
          <input value={dataDir} onChange={e => setDataDir(e.target.value)}
            placeholder="/path/to/data/  (leave empty for synthetic demo dataset)"
            style={{ width: "100%", padding: "0.65rem 0.9rem", background: "#0f1117", border: "1px solid #2d3148", borderRadius: 6, color: "#e2e8f0", fontSize: "0.85rem", boxSizing: "border-box" }} />
          <p style={{ fontSize: "0.75rem", color: "#475569", margin: "0.4rem 0 0" }}>
            Expects: <code>phishtank.csv</code>, <code>openphish.txt</code>, <code>tranco.csv</code> in that folder.
            If absent, a synthetic 41,250-sample dataset is auto-generated.
          </p>
        </div>

        <button onClick={startTraining} disabled={polling}
          style={{ padding: "0.75rem 2rem", background: polling ? "#374151" : "#6366f1", border: "none", borderRadius: 8, color: "#fff", cursor: polling ? "not-allowed" : "pointer", fontWeight: 600, fontSize: "0.9rem" }}>
          {polling ? "⏳ Training in progress..." : "▶ Start Training"}
        </button>
      </div>

      {/* Live training log */}
      {trainingState && (
        <div style={{ background: "#0a0c14", border: "1px solid #2d3148", borderRadius: 10, padding: "1.25rem", marginBottom: "1.5rem", fontFamily: "monospace" }}>
          <p style={{ margin: "0 0 0.75rem", fontSize: "0.8rem", color: "#64748b", fontFamily: "inherit" }}>
            Status: <span style={{ color: trainingState.done ? "#6ee7b7" : trainingState.error ? "#fca5a5" : "#fbbf24" }}>
              {trainingState.done ? "COMPLETE" : trainingState.error ? "ERROR" : "RUNNING..."}
            </span>
          </p>
          {trainingState.log?.map((line, i) => (
            <p key={i} style={{ margin: "2px 0", fontSize: "0.78rem", color: "#94a3b8" }}>» {line}</p>
          ))}
          {trainingState.error && (
            <p style={{ color: "#fca5a5", fontSize: "0.8rem", marginTop: "0.5rem" }}>✗ {trainingState.error}</p>
          )}
        </div>
      )}

      {/* Model status grid */}
      <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.25rem" }}>
        <h3 style={{ margin: "0 0 1rem", fontSize: "0.9rem", color: "#94a3b8" }}>Model Status</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: "0.75rem" }}>
          {["C1","C2","C3","C4","C5","C6"].map(cid => {
            const trained = status?.models?.[cid];
            const labels = { C1:"Lexical RF", C2:"Structural RF", C3:"Combined RF", C4:"Lexical XGB", C5:"Structural XGB", C6:"Combined XGB" };
            return (
              <div key={cid} style={{ padding: "0.75rem", background: "#0f1117", borderRadius: 8, border: `1px solid ${trained ? "#065f46" : "#1e2235"}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontWeight: 700, color: "#a5b4fc" }}>{cid}</span>
                  <span style={{ fontSize: "0.7rem", padding: "2px 6px", borderRadius: 10, background: trained ? "#052e16" : "#1e2235", color: trained ? "#6ee7b7" : "#475569" }}>
                    {trained ? "✓ Ready" : "Not trained"}
                  </span>
                </div>
                <p style={{ margin: "4px 0 0", fontSize: "0.75rem", color: "#64748b" }}>{labels[cid]}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Pipeline reference */}
      <div style={{ marginTop: "1.5rem", background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.25rem" }}>
        <h3 style={{ margin: "0 0 0.75rem", fontSize: "0.9rem", color: "#94a3b8" }}>What happens during training</h3>
        {[
          ["1. Dataset Load", "PhishTank + OpenPhish + Tranco loaded (or synthetic 41,250-sample dataset generated)"],
          ["2. Feature Extraction", "14 lexical features and 14 structural features extracted per URL"],
          ["3. Preprocessing", "80/20 stratified split → Min-Max scaling → SMOTE applied to training partition only"],
          ["4. Nested CV", "Stratified 10-fold outer CV with 5-fold GridSearchCV inner loop per condition"],
          ["5. 6 Conditions", "C1–C6 trained sequentially: 3 pipelines × 2 classifiers"],
          ["6. Evaluation", "All metrics computed on untouched holdout set + McNemar's test for significance"],
          ["7. Save", "Best models serialised to backend/models/saved/ as .pkl files"],
        ].map(([step, desc]) => (
          <div key={step} style={{ display: "flex", gap: "0.75rem", marginBottom: "0.6rem" }}>
            <span style={{ fontWeight: 700, color: "#6366f1", whiteSpace: "nowrap", fontSize: "0.82rem", minWidth: 120 }}>{step}</span>
            <span style={{ color: "#64748b", fontSize: "0.82rem", lineHeight: 1.4 }}>{desc}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
