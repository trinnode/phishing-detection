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
      return JSON.parse(localStorage.getItem('markup_results') || '[]');
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
      localStorage.setItem('markup_results', JSON.stringify(next));
      return next;
    });
  }

  function clearAnalysisResults() {
    setAnalysisResults([]);
    localStorage.removeItem('markup_results');
  }

  const tabs = [
    { id: "analyze", label: "URL Analysis" },
    { id: "results", label: "Experiment Results" },
    { id: "train", label: "Training Pipeline" },
    { id: "features", label: "Feature Definitions" },
  ];

  return (
    <div className="appShell">
      <header className="appHeader">
        <div className="headerInner">
          <div className="headerTop">
            <a className="brand" href="/">
              <img src="/logo.svg" alt="MARKup" className="logoImg" />
              <div className="titleBlock">
                <h1>MARKup</h1>
                <p>Phishing Domain Detection | Random Forest vs XGBoost</p>
              </div>
            </a>
            <div className="headerStatus">
              <span className={`badge ${modelStatus?.any_trained ? "badgeReady" : "badgeNotReady"}`}>
                {modelStatus?.any_trained ? "Models Loaded" : "Models Not Trained"}
              </span>
            </div>
          </div>
          <nav className="nav">
            {tabs.map((t) => (
              <button
                key={t.id}
                className={`navItem ${activeTab === t.id ? "navItemActive" : ""}`}
                onClick={() => setActiveTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="main">
        {activeTab === "analyze" && <URLAnalyzer modelsReady={modelStatus?.any_trained} onResult={addAnalysisResult} />}
        {activeTab === "results" && <ResultsDashboard analysisResults={analysisResults} onClearResults={clearAnalysisResults} />}
        {activeTab === "train" && <TrainingPanel onTrainingComplete={() => setModelStatus({ any_trained: true })} />}
        {activeTab === "features" && <FeatureExplorer />}
      </main>

      <footer className="footer">
        Lexical and Structural Feature Extraction Framework for Comparative Analysis of Phishing Domain Detection by Random Forest and XGBoost | MARKup | FUT Minna
      </footer>
    </div>
  );
}
