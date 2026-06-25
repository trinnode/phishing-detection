import { useState } from "react";
import { API_BASE } from "../App";

const CONDITION_OPTIONS = [
  { value: "C6", label: "C6 Combined XGBoost (Best, F1 0.984)" },
  { value: "C5", label: "C5 Combined Random Forest (F1 0.973)" },
  { value: "C4", label: "C4 Structural XGBoost (F1 0.960)" },
  { value: "C3", label: "C3 Structural Random Forest (F1 0.949)" },
  { value: "C2", label: "C2 Lexical XGBoost (F1 0.933)" },
  { value: "C1", label: "C1 Lexical Random Forest (F1 0.920)" },
];

const SAMPLE_URLS = [
  "http://paypal-secure-update.tk/login.php?redirect=account",
  "https://www.github.com/features",
  "http://192.168.1.1/secure/banking/verify?token=abc123",
  "https://stackoverflow.com/questions",
  "http://amazon-prize-winner-click-here.ml/claim",
  "https://www.bbc.co.uk/news",
];

const riskColors = {
  HIGH: { bg: "#450a0a", border: "#991b1b", text: "#fca5a5", badge: "#dc2626" },
  MEDIUM: { bg: "#431407", border: "#9a3412", text: "#fdba74", badge: "#ea580c" },
  LOW: { bg: "#052e16", border: "#15803d", text: "#86efac", badge: "#16a34a" },
};

export default function URLAnalyzer({ modelsReady, onResult }) {
  const [url, setUrl] = useState("");
  const [condition, setCondition] = useState("C6");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [batchMode, setBatchMode] = useState(false);
  const [batchUrls, setBatchUrls] = useState("");
  const [batchResults, setBatchResults] = useState(null);

  const endpoint = modelsReady ? "/api/predict" : "/api/predict/demo";

  async function analyzeURL() {
    if (!url.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const resp = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url.trim(), condition }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Prediction failed");
      setResult(data);
      onResult?.({ ...data, timestamp: Date.now(), mode: 'single' });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function analyzeBatch() {
    const urls = batchUrls.split("\n").map(u => u.trim()).filter(Boolean);
    if (!urls.length) return;
    setLoading(true);
    setError(null);
    setBatchResults(null);
    try {
      const resp = await fetch(`${API_BASE}/api/predict/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls, condition }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Batch prediction failed");
      setBatchResults(data);
      onResult?.({ ...data, timestamp: Date.now(), mode: 'batch' });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const colors = result ? riskColors[result.risk_level] || riskColors.LOW : null;

  return (
    <div>
      {!modelsReady && (
        <div style={{ background: "#1e1b4b", border: "1px solid #4338ca", borderRadius: 8, padding: "0.75rem 1rem", marginBottom: "1.5rem", fontSize: "0.85rem", color: "#a5b4fc" }}>
          Demo Mode: Models not yet trained. Using heuristic lexical scoring. Go to Training Pipeline tab to train all 6 conditions.
        </div>
      )}

      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1.5rem" }}>
        {["single", "batch"].map(m => (
          <button key={m} onClick={() => { setBatchMode(m === "batch"); setResult(null); setBatchResults(null); }}
            style={{ padding: "0.5rem 1rem", background: (batchMode ? m === "batch" : m === "single") ? "#6366f1" : "#1e2235", border: "1px solid #2d3148", borderRadius: 6, color: "#e2e8f0", cursor: "pointer", fontSize: "0.85rem" }}>
            {m === "single" ? "Single URL" : "Batch Analysis"}
          </button>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: "1.5rem", alignItems: "start" }}>
        <div>
          {!batchMode ? (
            <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.5rem" }}>
              <label style={{ fontSize: "0.8rem", color: "#94a3b8", display: "block", marginBottom: "0.5rem" }}>URL to Analyse</label>
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <input
                  value={url}
                  onChange={e => setUrl(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && analyzeURL()}
                  placeholder="https://example.com/path"
                  style={{ flex: 1, padding: "0.75rem 1rem", background: "#0f1117", border: "1px solid #2d3148", borderRadius: 6, color: "#e2e8f0", fontSize: "0.9rem", outline: "none" }}
                />
                <button onClick={analyzeURL} disabled={loading || !url.trim()}
                  style={{ padding: "0.75rem 1.5rem", background: "#6366f1", border: "none", borderRadius: 6, color: "#fff", cursor: "pointer", fontWeight: 600, fontSize: "0.9rem", opacity: loading ? 0.6 : 1 }}>
                  {loading ? "..." : "Analyse"}
                </button>
              </div>

              <div style={{ marginTop: "0.75rem" }}>
                <p style={{ fontSize: "0.75rem", color: "#475569", margin: "0 0 0.4rem" }}>Sample URLs:</p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem" }}>
                  {SAMPLE_URLS.map(s => (
                    <button key={s} onClick={() => setUrl(s)}
                      style={{ padding: "3px 8px", background: "#0f1117", border: "1px solid #2d3148", borderRadius: 4, color: "#94a3b8", cursor: "pointer", fontSize: "0.7rem", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {s.replace('https://', '').replace('http://', '').substring(0, 30)}…
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.5rem" }}>
              <label style={{ fontSize: "0.8rem", color: "#94a3b8", display: "block", marginBottom: "0.5rem" }}>URLs (one per line, max 100)</label>
              <textarea value={batchUrls} onChange={e => setBatchUrls(e.target.value)}
                placeholder={"https://example.com\nhttp://suspicious-domain.tk/login\n..."}
                rows={6}
                style={{ width: "100%", padding: "0.75rem", background: "#0f1117", border: "1px solid #2d3148", borderRadius: 6, color: "#e2e8f0", fontSize: "0.85rem", resize: "vertical", boxSizing: "border-box" }} />
              <button onClick={analyzeBatch} disabled={loading || !batchUrls.trim()}
                style={{ marginTop: "0.75rem", padding: "0.65rem 1.5rem", background: "#6366f1", border: "none", borderRadius: 6, color: "#fff", cursor: "pointer", fontWeight: 600 }}>
                {loading ? "Analysing..." : "Analyse Batch"}
              </button>
            </div>
          )}

          {error && (
            <div style={{ marginTop: "1rem", padding: "0.75rem 1rem", background: "#450a0a", border: "1px solid #991b1b", borderRadius: 8, color: "#fca5a5", fontSize: "0.85rem" }}>
              {error}
            </div>
          )}

          {/* Single result */}
          {result && !batchMode && colors && (
            <div style={{ marginTop: "1.5rem", background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 10, padding: "1.5rem" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
                  <div>
                    <strong style={{ fontSize: "1.1rem", color: colors.text }}>
                      {(result.prediction || "").toUpperCase()}
                    </strong>
                  </div>
                  <span style={{ background: colors.badge, color: "#fff", padding: "4px 12px", borderRadius: 20, fontSize: "0.75rem", fontWeight: 700 }}>
                    {result.risk_level} RISK
                  </span>
                </div>

              <p style={{ color: "#94a3b8", fontSize: "0.8rem", wordBreak: "break-all", margin: "0 0 1rem" }}>{result.url}</p>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.75rem", marginBottom: "1rem" }}>
                {[
                  { label: "Phishing Probability", value: `${(result.phishing_probability * 100).toFixed(1)}%` },
                  { label: "Confidence", value: `${(result.confidence * 100).toFixed(1)}%` },
                  { label: "Pipeline", value: result.pipeline },
                ].map(m => (
                  <div key={m.label} style={{ background: "rgba(0,0,0,0.3)", borderRadius: 6, padding: "0.6rem" }}>
                    <p style={{ margin: 0, fontSize: "0.7rem", color: "#64748b" }}>{m.label}</p>
                    <p style={{ margin: 0, fontSize: "1rem", fontWeight: 700, color: colors.text }}>{m.value}</p>
                  </div>
                ))}
              </div>

              {/* Probability bar */}
              <div style={{ marginBottom: "1rem" }}>
                <p style={{ fontSize: "0.75rem", color: "#64748b", margin: "0 0 4px" }}>Phishing Probability</p>
                <div style={{ background: "#0f1117", borderRadius: 4, height: 8 }}>
                  <div style={{ width: `${result.phishing_probability * 100}%`, height: "100%", background: result.risk_level === "HIGH" ? "#dc2626" : result.risk_level === "MEDIUM" ? "#ea580c" : "#16a34a", borderRadius: 4, transition: "width 0.5s" }} />
                </div>
              </div>

              {/* Key lexical features */}
              {result.features && (
                <details style={{ fontSize: "0.8rem" }}>
                  <summary style={{ cursor: "pointer", color: "#94a3b8", marginBottom: "0.5rem" }}>Feature Values</summary>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "4px", marginTop: "0.5rem" }}>
                    {Object.entries(result.features).map(([k, v]) => (
                      <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "3px 6px", background: "rgba(0,0,0,0.3)", borderRadius: 4 }}>
                        <span style={{ color: "#64748b" }}>{k}</span>
                        <span style={{ color: "#e2e8f0", fontWeight: 600 }}>{typeof v === "number" ? v.toFixed(3) : v}</span>
                      </div>
                    ))}
                  </div>
                </details>
              )}

              {result.warning && (
                <p style={{ fontSize: "0.75rem", color: "#f59e0b", marginTop: "0.75rem", padding: "0.5rem", background: "rgba(245,158,11,0.1)", borderRadius: 4 }}>
                  {result.warning}
                </p>
              )}
            </div>
          )}

          {/* Batch results */}
          {batchResults && batchMode && (
            <div style={{ marginTop: "1.5rem" }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: "1rem", marginBottom: "1rem" }}>
                {[
                  { label: "Total Analysed", value: batchResults.summary.total, color: "#a5b4fc" },
                  { label: "Phishing Detected", value: batchResults.summary.phishing, color: "#fca5a5" },
                  { label: "Legitimate", value: batchResults.summary.legitimate, color: "#86efac" },
                ].map(s => (
                  <div key={s.label} style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 8, padding: "1rem", textAlign: "center" }}>
                    <p style={{ margin: 0, fontSize: "0.75rem", color: "#64748b" }}>{s.label}</p>
                    <p style={{ margin: 0, fontSize: "1.8rem", fontWeight: 700, color: s.color }}>{s.value}</p>
                  </div>
                ))}
              </div>
              <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 8, overflow: "hidden" }}>
                {batchResults.results.map((r, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: "0.75rem", padding: "0.6rem 1rem", borderBottom: "1px solid #1e2235" }}>
                    <span style={{ fontSize: "0.8rem", color: r.prediction === "phishing" ? "#fca5a5" : "#86efac" }}>{r.prediction === "phishing" ? "Phishing" : "Legitimate"}</span>
                    <span style={{ flex: 1, fontSize: "0.8rem", color: "#94a3b8", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.url}</span>
                    <span style={{ fontSize: "0.75rem", fontWeight: 600, color: riskColors[r.risk_level]?.text }}>{r.risk_level}</span>
                    <span style={{ fontSize: "0.75rem", color: "#64748b", minWidth: 45, textAlign: "right" }}>{(r.phishing_probability * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right panel: condition selector */}
        <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.25rem" }}>
          <p style={{ fontSize: "0.8rem", color: "#94a3b8", margin: "0 0 0.75rem", fontWeight: 600 }}>EXPERIMENTAL CONDITION</p>
          {CONDITION_OPTIONS.map(opt => (
            <label key={opt.value} style={{ display: "flex", alignItems: "flex-start", gap: "0.5rem", marginBottom: "0.65rem", cursor: "pointer" }}>
              <input type="radio" name="condition" value={opt.value} checked={condition === opt.value} onChange={() => setCondition(opt.value)}
                style={{ marginTop: 3, accentColor: "#6366f1" }} />
              <span style={{ fontSize: "0.8rem", color: condition === opt.value ? "#a5b4fc" : "#64748b" }}>{opt.label}</span>
            </label>
          ))}
          <div style={{ marginTop: "1rem", padding: "0.75rem", background: "#0f1117", borderRadius: 6, fontSize: "0.75rem", color: "#475569", lineHeight: 1.5 }}>
            <strong style={{ color: "#64748b" }}>Pipeline reference:</strong><br />
            C1 C2: Lexical only (14 features)<br />
            C3 C4: Structural only (14 features)<br />
            C5 C6: Combined (25 features post reduction)
          </div>
        </div>
      </div>
    </div>
  );
}
