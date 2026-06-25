import { useState, useEffect } from "react";
import URLAnalyzer from "./components/URLAnalyzer";
import ResultsDashboard from "./components/ResultsDashboard";
import TrainingPanel from "./components/TrainingPanel";
import FeatureExplorer from "./components/FeatureExplorer";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:5000";

export { API_BASE };

export default function App() {
  const [activeTab, setActiveTab] = useState("analyze");
  const [modelStatus, setModelStatus] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('fishmark_results') || '[]');
    } catch { return []; }
  });

  useEffect(() => {
    fetch(`${API_BASE}/api/models/status`)
      .then((r) => r.json())
      .then(setModelStatus)
      .catch(() => setModelStatus({ any_trained: false }));
  }, []);

  function addAnalysisResult(result) {
    setAnalysisResults(prev => {
      const next = [result, ...prev].slice(0, 50);
      localStorage.setItem('fishmark_results', JSON.stringify(next));
      return next;
    });
  }

  function clearAnalysisResults() {
    setAnalysisResults([]);
    localStorage.removeItem('fishmark_results');
  }

  const tabs = [
    { id: "analyze", label: "URL Analyser" },
    { id: "results", label: "Experiment Results" },
    { id: "train", label: "Training Pipeline" },
    { id: "features", label: "Feature Definitions" },
  ];

  return (
    <div className="app-shell" style={{ minHeight: "100vh", background: "#0f1117", color: "#e2e8f0", fontFamily: "'Inter', sans-serif" }}>
      {/* Header */}
      <header className="app-header" style={{ background: "#1a1d2e", borderBottom: "1px solid #2d3148", padding: "0 2rem" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", alignItems: "center", gap: "1rem", padding: "1rem 0" }}>
          <img src="/logo.svg" alt="FishMark" style={{ height: 40 }} />
          <div>
            <h1 style={{ margin: 0, fontSize: "1.1rem", fontWeight: 700, color: "#a5b4fc" }}>
              FishMark
            </h1>
            <p style={{ margin: 0, fontSize: "0.75rem", color: "#64748b" }}>
              Phishing Domain Detection — Lexical & Structural Feature Extraction | RF vs XGBoost | FUT Minna Research Project
            </p>
          </div>
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{
              fontSize: "0.7rem", padding: "3px 8px", borderRadius: 20,
              background: modelStatus?.any_trained ? "#064e3b" : "#451a03",
              color: modelStatus?.any_trained ? "#6ee7b7" : "#fed7aa",
              border: `1px solid ${modelStatus?.any_trained ? "#065f46" : "#7c2d12"}`
            }}>
              {modelStatus?.any_trained ? "✓ Models Loaded" : "⚠ Models Not Trained"}
            </span>
          </div>
        </div>

        {/* Nav tabs */}
        <nav style={{ maxWidth: 1200, margin: "0 auto", display: "flex", gap: "0.25rem" }}>
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              style={{
                padding: "0.65rem 1.1rem",
                background: activeTab === t.id ? "#6366f1" : "transparent",
                border: "none",
                borderBottom: activeTab === t.id ? "2px solid #818cf8" : "2px solid transparent",
                color: activeTab === t.id ? "#fff" : "#94a3b8",
                cursor: "pointer",
                fontSize: "0.85rem",
                fontWeight: activeTab === t.id ? 600 : 400,
                borderRadius: "6px 6px 0 0",
                transition: "all 0.15s",
              }}
            >
                {t.label}
            </button>
          ))}
        </nav>
      </header>

      {/* Main content */}
      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "2rem" }}>
        {activeTab === "analyze" && <URLAnalyzer modelsReady={modelStatus?.any_trained} onResult={addAnalysisResult} />}
        {activeTab === "results" && <ResultsDashboard analysisResults={analysisResults} onClearResults={clearAnalysisResults} />}
        {activeTab === "train" && <TrainingPanel onTrainingComplete={() => setModelStatus({ any_trained: true })} />}
        {activeTab === "features" && <FeatureExplorer />}
      </main>

      <footer style={{ textAlign: "center", padding: "2rem", color: "#374151", fontSize: "0.75rem", borderTop: "1px solid #1e2235" }}>
        LEXICAL AND STRUCTURAL FEATURE EXTRACTION FRAMEWORK FOR COMPARATIVE ANALYSIS OF PHISHING DOMAIN DETECTION BY RANDOM FOREST AND XGBOOST — FishMark | FUT Minna
      </footer>
    </div>
  );
}
