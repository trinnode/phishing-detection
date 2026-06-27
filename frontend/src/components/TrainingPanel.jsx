import { useState, useEffect, useRef } from "react";
import { API_BASE } from "../App";

const CONDITION_CONFIG = [
  { id: "C1", pipeline: "Lexical", classifier: "RF", description: "Lexical only with Random Forest (14 features)" },
  { id: "C2", pipeline: "Lexical", classifier: "XGB", description: "Lexical only with XGBoost (14 features)" },
  { id: "C3", pipeline: "Structural", classifier: "RF", description: "Structural only with Random Forest (14 features)" },
  { id: "C4", pipeline: "Structural", classifier: "XGB", description: "Structural only with XGBoost (14 features)" },
  { id: "C5", pipeline: "Combined", classifier: "RF", description: "Combined pipeline with Random Forest (25 features post reduction)" },
  { id: "C6", pipeline: "Combined", classifier: "XGB", description: "Combined pipeline with XGBoost (25 features post reduction)" },
];

export default function TrainingPanel({ onTrainingComplete }) {
  const [status, setStatus] = useState(null);
  const [polling, setPolling] = useState(false);
  const [fastMode, setFastMode] = useState(true);
  const [dataDir, setDataDir] = useState("");
  const [inputMode, setInputMode] = useState("files");
  const [batchUrls, setBatchUrls] = useState("");
  const [files, setFiles] = useState([]);
  const [uploadMessage, setUploadMessage] = useState("");
  const [uploadError, setUploadError] = useState("");
  const [currentPhase, setCurrentPhase] = useState("");
  const [conditionProgress, setConditionProgress] = useState({});
  const pollRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/models/status`).then(r => r.json()).then(setStatus).catch(() => {});
    return () => clearInterval(pollRef.current);
  }, []);

  function parseUrls(text) {
    return text.split("\n")
      .map(u => u.trim())
      .filter(u => u.length > 0 && (u.startsWith("http://") || u.startsWith("https://")));
  }

  async function handleUpload() {
    setUploadError("");
    setUploadMessage("");

    if (inputMode === "files") {
      if (files.length === 0) {
        setUploadError("Please select at least one CSV or TXT file.");
        return;
      }
      const formData = new FormData();
      let totalRecords = 0;
      for (const file of files) {
        formData.append("files", file);
        totalRecords++;
      }
      setUploadMessage(`Uploading ${totalRecords} file${totalRecords > 1 ? "s" : ""}...`);
      try {
        const resp = await fetch(`${API_BASE}/api/dataset/upload`, {
          method: "POST",
          body: formData,
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "Upload failed");
        setUploadMessage(data.message || "Files uploaded successfully.");
        setDataDir(data.data_dir || "");
      } catch (e) {
        setUploadError(e.message);
      }
    } else {
      const urls = parseUrls(batchUrls);
      if (urls.length === 0) {
        setUploadError("Please enter at least one valid URL (starting with http:// or https://).");
        return;
      }
      setUploadMessage(`Processing ${urls.length} URL${urls.length > 1 ? "s" : ""} individually...`);
      const successes = [];
      const failures = [];
      for (let i = 0; i < urls.length; i++) {
        try {
          const resp = await fetch(`${API_BASE}/api/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: urls[i], condition: "C6" }),
          });
          let d = {};
          try { d = await resp.json(); } catch {}
          if (resp.ok) successes.push({ url: urls[i], result: d });
          else failures.push({ url: urls[i], error: d?.error || `Server error (${resp.status})` });
        } catch (e) {
          failures.push({ url: urls[i], error: e.message });
        }
      }
      setUploadMessage(`Processed ${urls.length} URLs: ${successes.length} succeeded, ${failures.length} failed. ${failures.length > 0 ? "Failures were isolated and did not affect other analyses." : ""}`);
    }
  }

  async function startTraining() {
    const payload = { fast_mode: fastMode };
    if (dataDir.trim()) payload.data_dir = dataDir.trim();

    setStatus(prev => ({ ...prev, training: { running: true, done: false, error: null, log: ["⏳ Initiating training request..."] } }));
    setCurrentPhase("Initialising training pipeline...");
    setConditionProgress({});

    try {
      const resp = await fetch(`${API_BASE}/api/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      let errData = {};
      try { errData = await resp.json(); } catch {}
      if (!resp.ok) throw new Error(errData?.error || `Server responded with ${resp.status}`);
      setPolling(true);
      pollRef.current = setInterval(async () => {
        try {
          const r = await fetch(`${API_BASE}/api/train/status`);
          let d = {};
          try { d = await r.json(); } catch {}
          if (!r.ok) throw new Error(`Status poll failed (${r.status})`);
          setStatus(prev => ({ ...prev, training: d }));
          if (d.log && d.log.length > 0) {
            const lastEntry = d.log[d.log.length - 1];
            if (lastEntry.includes("C1") || lastEntry.includes("Condition 1")) setCurrentPhase("Training Condition C1: Lexical with Random Forest");
            else if (lastEntry.includes("C2") || lastEntry.includes("Condition 2")) setCurrentPhase("Training Condition C2: Lexical with XGBoost");
            else if (lastEntry.includes("C3") || lastEntry.includes("Condition 3")) setCurrentPhase("Training Condition C3: Structural with Random Forest");
            else if (lastEntry.includes("C4") || lastEntry.includes("Condition 4")) setCurrentPhase("Training Condition C4: Structural with XGBoost");
            else if (lastEntry.includes("C5") || lastEntry.includes("Condition 5")) setCurrentPhase("Training Condition C5: Combined with Random Forest");
            else if (lastEntry.includes("C6") || lastEntry.includes("Condition 6")) setCurrentPhase("Training Condition C6: Combined with XGBoost");
            else if (lastEntry.includes("extract") || lastEntry.includes("feature")) setCurrentPhase("Extracting features from dataset...");
            else if (lastEntry.includes("preprocess") || lastEntry.includes("dataset")) setCurrentPhase("Preprocessing dataset...");
            else if (lastEntry.includes("complete") || lastEntry.includes("done")) setCurrentPhase("Training complete!");
            else setCurrentPhase(lastEntry);
          }
          if (d.done || d.error) {
            clearInterval(pollRef.current);
            setPolling(false);
            setCurrentPhase(d.done ? "Training complete! All 6 conditions trained and evaluated." : `Training error: ${d.error}`);
            if (d.done) {
              onTrainingComplete?.();
              fetch(`${API_BASE}/api/models/status`).then(r => r.json()).then(s => setStatus(prev => ({ ...prev, ...s })));
            }
          }
        } catch {
          setCurrentPhase("Polling training status...");
        }
      }, 1500);
    } catch (e) {
      setCurrentPhase(`Error: ${e.message}`);
      setPolling(false);
      setStatus(prev => ({ ...prev, training: { running: false, error: e.message, log: [] } }));
    }
  }

  const trainingState = status?.training;
  const trainedModels = status?.models || {};

  return (
    <div style={{ maxWidth: 900 }}>
      <div style={{ marginBottom: "1.5rem" }}>
        <h2 style={{ color: "#e2e8f0", fontSize: "1.1rem", marginBottom: "0.25rem" }}>Training Pipeline</h2>
        <p style={{ color: "#64748b", fontSize: "0.85rem", marginBottom: "0" }}>
          Trains all 6 experimental conditions with 10 fold cross validation, SMOTE oversampling, grid search hyperparameter tuning, and McNemar statistical significance testing.
        </p>
      </div>

      {/* Dataset Input Section */}
      <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.5rem", marginBottom: "1.5rem" }}>
        <h3 style={{ margin: "0 0 1rem", fontSize: "0.9rem", color: "#94a3b8" }}>Dataset Input</h3>

        <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
          {[
            { id: "files", label: "Upload Files" },
            { id: "urls", label: "Batch URLs" },
            { id: "manual", label: "Manual Path" },
          ].map(m => (
            <button key={m.id} onClick={() => setInputMode(m.id)}
              style={{ padding: "0.45rem 1rem", background: inputMode === m.id ? "#6366f1" : "#1e2235", border: "1px solid #2d3148", borderRadius: 6, color: inputMode === m.id ? "#fff" : "#94a3b8", cursor: "pointer", fontSize: "0.8rem", fontWeight: inputMode === m.id ? 600 : 400 }}>
              {m.label}
            </button>
          ))}
        </div>

        {inputMode === "files" && (
          <div>
            <div style={{ border: "2px dashed #2d3148", borderRadius: 8, padding: "2rem", textAlign: "center", cursor: "pointer", transition: "border-color 0.2s", marginBottom: "0.75rem" }}
              onDragOver={e => { e.preventDefault(); e.currentTarget.style.borderColor = "#6366f1"; }}
              onDragLeave={e => { e.currentTarget.style.borderColor = "#2d3148"; }}
              onDrop={e => { e.preventDefault(); setFiles([...files, ...Array.from(e.dataTransfer.files).filter(f => f.name.endsWith(".csv") || f.name.endsWith(".txt"))]); }}>
              <p style={{ fontSize: "0.85rem", color: "#64748b", margin: "0 0 0.3rem" }}>Drag and drop CSV or TXT files here</p>
              <p style={{ fontSize: "0.72rem", color: "#475569", margin: 0 }}>or click to select files</p>
              <input type="file" multiple accept=".csv,.txt" style={{ display: "none" }}
                onChange={e => setFiles([...files, ...Array.from(e.target.files)])} />
              <button onClick={() => {
                const input = document.querySelector('input[type="file"]');
                if (input) input.click();
              }} style={{ marginTop: "0.75rem", padding: "0.4rem 1rem", background: "#0f1117", border: "1px solid #2d3148", borderRadius: 6, color: "#94a3b8", cursor: "pointer", fontSize: "0.78rem" }}>
                Browse Files
              </button>
            </div>
            {files.length > 0 && (
              <div style={{ marginBottom: "0.75rem" }}>
                <p style={{ fontSize: "0.75rem", color: "#64748b", marginBottom: "0.3rem" }}>{files.length} file{files.length > 1 ? "s" : ""} selected:</p>
                {files.map((f, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: "0.5rem", padding: "0.25rem 0", fontSize: "0.75rem", color: "#94a3b8" }}>
                    <span>{f.name} ({(f.size / 1024).toFixed(1)} KB)</span>
                    <button onClick={() => setFiles(files.filter((_, j) => j !== i))}
                      style={{ color: "#e0595b", cursor: "pointer", fontSize: "0.7rem", background: "none", border: "none", padding: 0 }}>Remove</button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {inputMode === "urls" && (
          <div>
            <textarea value={batchUrls} onChange={e => setBatchUrls(e.target.value)}
              placeholder={"https://example.com\nhttp://suspicious.domain.tk/login\nhttps://another-url.com/path"}
              rows={6}
              style={{ width: "100%", padding: "0.75rem", background: "#0f1117", border: "1px solid #2d3148", borderRadius: 6, color: "#e2e8f0", fontSize: "0.85rem", resize: "vertical", boxSizing: "border-box" }} />
            <p style={{ fontSize: "0.72rem", color: "#475569", marginTop: "0.3rem" }}>Enter one URL per line. Each URL is analysed independently so a single failure does not affect the rest.</p>
          </div>
        )}

        {inputMode === "manual" && (
          <div>
            <input value={dataDir} onChange={e => setDataDir(e.target.value)}
              placeholder="/path/to/data/directory  (leave empty for synthetic demo dataset)"
              style={{ width: "100%", padding: "0.65rem 0.9rem", background: "#0f1117", border: "1px solid #2d3148", borderRadius: 6, color: "#e2e8f0", fontSize: "0.85rem", boxSizing: "border-box" }} />
            <p style={{ fontSize: "0.72rem", color: "#475569", marginTop: "0.3rem" }}>
              Expects phishtank.csv, openphish.txt, tranco.csv in that directory. If absent, a synthetic 41,250 sample dataset is auto generated.
            </p>
          </div>
        )}

        {(inputMode === "files" || inputMode === "urls") && (
          <button onClick={handleUpload}
            style={{ padding: "0.45rem 1.2rem", background: "#0f1117", border: "1px solid #2d3148", borderRadius: 6, color: "#94a3b8", cursor: "pointer", fontSize: "0.8rem", marginTop: "0.5rem" }}>
            {inputMode === "files" ? "Upload Files" : "Analyse URLs"}
          </button>
        )}

        {uploadMessage && <p style={{ fontSize: "0.78rem", color: "#6ee7b7", marginTop: "0.5rem" }}>{uploadMessage}</p>}
        {uploadError && <p style={{ fontSize: "0.78rem", color: "#fca5a5", marginTop: "0.5rem" }}>{uploadError}</p>}
      </div>

      {/* Training Configuration */}
      <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.5rem", marginBottom: "1.5rem" }}>
        <h3 style={{ margin: "0 0 1rem", fontSize: "0.9rem", color: "#94a3b8" }}>Training Configuration</h3>
        <div style={{ marginBottom: "1.25rem" }}>
          <label style={{ display: "flex", alignItems: "center", gap: "0.75rem", cursor: "pointer" }}>
            <input type="checkbox" checked={fastMode} onChange={e => setFastMode(e.target.checked)} style={{ accentColor: "#6366f1", width: 16, height: 16 }} />
            <div>
              <span style={{ color: "#e2e8f0", fontWeight: 600, fontSize: "0.9rem" }}>Fast Mode</span>
              <p style={{ margin: 0, fontSize: "0.78rem", color: "#64748b" }}>Reduced hyperparameter grid completes in approximately 5 minutes. Uncheck for full GridSearch which may take hours.</p>
            </div>
          </label>
        </div>

        <div style={{ marginBottom: "1.25rem", display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.75rem" }}>
          {[
            { label: "Cross Validation", value: "10 Fold Stratified" },
            { label: "Class Balancing", value: "SMOTE Oversampling" },
            { label: "Hyperparameter Tuning", value: "GridSearchCV (Nested)" },
            { label: "Significance Test", value: "McNemar (alpha = 0.05)" },
            { label: "Feature Reduction", value: "Pearson Correlation (r > 0.90)" },
            { label: "Evaluation", value: "7 Metrics per Condition" },
          ].map(m => (
            <div key={m.label} style={{ padding: "0.5rem", background: "#0f1117", borderRadius: 6, textAlign: "center" }}>
              <p style={{ margin: 0, fontSize: "0.62rem", color: "#475569" }}>{m.label}</p>
              <p style={{ margin: "0.15rem 0 0", fontSize: "0.72rem", color: "#94a3b8", fontWeight: 600 }}>{m.value}</p>
            </div>
          ))}
        </div>

        <button onClick={startTraining} disabled={polling}
          style={{ padding: "0.75rem 2rem", background: polling ? "#374151" : "#6366f1", border: "none", borderRadius: 8, color: "#fff", cursor: polling ? "not-allowed" : "pointer", fontWeight: 600, fontSize: "0.9rem" }}>
          {polling ? "Training in Progress..." : "Start Training"}
        </button>
      </div>

      {/* Live Training Session Display */}
      {trainingState && (
        <div style={{ background: "#0a0c14", border: "1px solid #2d3148", borderRadius: 10, padding: "1.25rem", marginBottom: "1.5rem", fontFamily: "monospace" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.75rem" }}>
            <p style={{ margin: 0, fontSize: "0.8rem", color: "#64748b", fontFamily: "inherit" }}>
              Status: <span style={{ color: trainingState.done ? "#6ee7b7" : trainingState.error ? "#fca5a5" : "#fbbf24" }}>
                {trainingState.done ? "COMPLETE" : trainingState.error ? "ERROR" : "RUNNING"}
              </span>
            </p>
            {polling && (
              <span style={{ fontSize: "0.68rem", color: "#475569", fontFamily: "inherit" }}>
                Polling every 1.5s
              </span>
            )}
          </div>

          {/* Current Phase Indicator */}
          {currentPhase && !trainingState.done && !trainingState.error && (
            <div style={{ marginBottom: "0.75rem", padding: "0.5rem 0.75rem", background: "rgba(99,102,241,0.1)", border: "1px solid rgba(99,102,241,0.2)", borderRadius: 6 }}>
              <p style={{ margin: 0, fontSize: "0.75rem", color: "#a5b4fc", fontFamily: "inherit" }}>
                {currentPhase}
              </p>
            </div>
          )}

          {/* Condition Progress Grid */}
          {!trainingState.done && !trainingState.error && (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.5rem", marginBottom: "0.75rem" }}>
              {CONDITION_CONFIG.map(c => (
                <div key={c.id} style={{ padding: "0.5rem", background: "#0f1117", borderRadius: 6, opacity: currentPhase.includes(c.id) ? 1 : 0.5 }}>
                  <p style={{ margin: 0, fontSize: "0.65rem", color: "#64748b", fontFamily: "inherit" }}>{c.id}</p>
                  <p style={{ margin: "0.1rem 0", fontSize: "0.7rem", color: currentPhase.includes(c.id) ? "#a5b4fc" : "#475569", fontWeight: 600, fontFamily: "inherit" }}>{c.pipeline} {c.classifier}</p>
                  <div style={{ height: 3, background: "#1e2235", borderRadius: 2, overflow: "hidden" }}>
                    <div style={{ width: currentPhase.includes(c.id) ? "60%" : "0%", height: "100%", background: "#6366f1", borderRadius: 2, transition: "width 1s" }} />
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Condition Results (when complete) */}
          {trainingState.done && trainingState.results && (
            <div style={{ marginBottom: "0.75rem" }}>
              <p style={{ fontSize: "0.75rem", color: "#6ee7b7", marginBottom: "0.5rem", fontFamily: "inherit" }}>Training Results Summary:</p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.5rem" }}>
                {CONDITION_CONFIG.map(c => {
                  const condResult = trainingState.results?.conditions?.[c.id];
                  return (
                    <div key={c.id} style={{ padding: "0.5rem", background: "#0f1117", borderRadius: 6 }}>
                      <p style={{ margin: 0, fontSize: "0.65rem", color: "#a5b4fc", fontWeight: 600, fontFamily: "inherit" }}>{c.id}: {c.pipeline} {c.classifier}</p>
                      {condResult ? (
                        <p style={{ margin: "0.1rem 0 0", fontSize: "0.62rem", color: "#6ee7b7", fontFamily: "inherit" }}>
                          F1: {condResult.f1_score?.toFixed(3) || "N/A"} | AUC: {condResult.auc_roc?.toFixed(3) || "N/A"}
                        </p>
                      ) : (
                        <p style={{ margin: "0.1rem 0 0", fontSize: "0.62rem", color: "#475569", fontFamily: "inherit" }}>No data</p>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Full Log */}
          <div style={{ maxHeight: 300, overflowY: "auto", border: "1px solid #1e2235", borderRadius: 4 }}>
            {trainingState.log?.length > 0 ? (
              trainingState.log.map((line, i) => (
                <p key={i} style={{ margin: 0, padding: "2px 8px", fontSize: "0.72rem", color: "#94a3b8", fontFamily: "inherit", background: i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.01)" }}>
                  {line}
                </p>
              ))
            ) : (
              <p style={{ margin: 0, padding: "8px", fontSize: "0.72rem", color: "#475569", fontFamily: "inherit" }}>
                {trainingState.running ? "⏳ Preparing training pipeline..." : "No log entries."}
              </p>
            )}
            {polling && trainingState.running && (
              <p style={{ margin: 0, padding: "4px 8px", fontSize: "0.65rem", color: "#475569", fontFamily: "inherit", borderTop: "1px solid #1e2235", fontStyle: "italic" }}>
                Training in progress — logs update every 1.5s
              </p>
            )}
          </div>

          {trainingState.error && (
            <p style={{ color: "#fca5a5", fontSize: "0.78rem", marginTop: "0.5rem", fontFamily: "inherit" }}>{trainingState.error}</p>
          )}
        </div>
      )}

      {/* Model Status Grid */}
      <div style={{ background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.25rem", marginBottom: "1.5rem" }}>
        <h3 style={{ margin: "0 0 1rem", fontSize: "0.9rem", color: "#94a3b8" }}>Model Status</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: "0.75rem" }}>
          {CONDITION_CONFIG.map(c => {
            const trained = trainedModels[c.id];
            return (
              <div key={c.id} style={{ padding: "0.75rem", background: "#0f1117", borderRadius: 8, border: `1px solid ${trained ? "#065f46" : "#1e2235"}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontWeight: 700, color: "#a5b4fc" }}>{c.id}</span>
                  <span style={{ fontSize: "0.7rem", padding: "2px 6px", borderRadius: 10, background: trained ? "#052e16" : "#1e2235", color: trained ? "#6ee7b7" : "#475569" }}>
                    {trained ? "Ready" : "Not trained"}
                  </span>
                </div>
                <p style={{ margin: "4px 0 0", fontSize: "0.72rem", color: "#64748b" }}>{c.pipeline} {c.classifier}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Pipeline Reference */}
      <div style={{ marginTop: "1.5rem", background: "#1a1d2e", border: "1px solid #2d3148", borderRadius: 10, padding: "1.25rem" }}>
        <h3 style={{ margin: "0 0 0.75rem", fontSize: "0.9rem", color: "#94a3b8" }}>Training Pipeline Architecture</h3>
        {[
          ["Phase 1: Dataset Acquisition", "PhishTank, OpenPhish, and Tranco datasets are loaded. If unavailable, a synthetic 41,250 sample dataset is auto generated with class balanced distribution."],
          ["Phase 2: Feature Extraction", "14 lexical features are extracted from URL strings (length, entropy, digit ratio, special chars, etc.) and 14 structural features from registration/DNS/SSL metadata. Combined pipeline concatenates all 28 then applies Pearson correlation reduction to remove collinear pairs (threshold r > 0.90), producing 25 features."],
          ["Phase 3: Preprocessing", "80/20 stratified train test split is applied. Min Max scaling normalises all features to [0,1]. SMOTE oversampling is applied to the training partition only to address class imbalance without data leakage."],
          ["Phase 4: Nested Cross Validation", "Outer loop: Stratified 10 fold cross validation. Inner loop: 5 fold GridSearchCV for hyperparameter tuning. This nesting ensures validation fold information never influences parameter selection."],
          ["Phase 5: Condition Training", "Six conditions are trained sequentially: C1 (Lexical RF), C2 (Lexical XGB), C3 (Structural RF), C4 (Structural XGB), C5 (Combined RF), C6 (Combined XGB). Each produces accuracy, precision, recall, F1, AUC ROC, FPR, and MCC metrics."],
          ["Phase 6: Statistical Evaluation", "McNemars test is applied to three paired comparisons: C1 vs C2 (RF vs XGB on Lexical), C3 vs C4 (RF vs XGB on Structural), C5 vs C6 (RF vs XGB on Combined). Statistical significance is assessed at alpha = 0.05."],
          ["Phase 7: Persistence", "Trained models are serialised to backend/models/saved/ as pickle files. Results are saved to backend/results/experiment_results.json for the Results page to load."],
        ].map(([step, desc]) => (
          <div key={step} style={{ display: "flex", gap: "0.75rem", marginBottom: "0.75rem", padding: "0.6rem", background: "#0f1117", borderRadius: 6 }}>
            <span style={{ fontWeight: 600, color: "#6366f1", whiteSpace: "nowrap", fontSize: "0.78rem", minWidth: 160, flexShrink: 0 }}>{step}</span>
            <span style={{ color: "#64748b", fontSize: "0.78rem", lineHeight: 1.5 }}>{desc}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
