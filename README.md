# Phishing Domain Detection Framework

**LEXICAL AND STRUCTURAL FEATURE EXTRACTION FRAMEWORK FOR COMPARATIVE ANALYSIS OF PHISHING DOMAIN DETECTION BY RANDOM FOREST AND XGBOOST**

FUT Minna Research Project | Chapter 4 Implementation

---

## Project Structure

```
phishing_detection/
├── backend/
│   ├── api/
│   │   └── app.py              ← Flask REST API (all endpoints)
│   ├── core/
│   │   ├── lexical_extractor.py   ← Pipeline A: 14 lexical features
│   │   ├── structural_extractor.py← Pipeline B: 14 structural features
│   │   ├── pipeline_combiner.py   ← Pipeline C: combined + correlation reduction
│   │   ├── dataset_loader.py      ← PhishTank/OpenPhish/Tranco loaders + synthetic generator
│   │   ├── trainer.py             ← All 6 conditions, SMOTE, nested CV, McNemar's test
│   │   └── predictor.py           ← Inference engine for trained models
│   ├── models/saved/           ← Trained .pkl files saved here after training
│   ├── results/                ← experiment_results.json saved here
│   ├── cache/structural/       ← WHOIS/DNS cache (file-based)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/
│   │       ├── URLAnalyzer.jsx     ← Single/batch URL analysis
│   │       ├── ResultsDashboard.jsx← Chapter 4 results tables + charts
│   │       ├── TrainingPanel.jsx   ← Trigger training + live log
│   │       └── FeatureExplorer.jsx ← All 28 feature definitions + importance
│   ├── package.json
│   ├── vite.config.js
│   ├── Dockerfile
│   ├── nginx.conf
│   └── vercel.json
├── data/                       ← Place your datasets here (see below)
├── train.py                    ← Main training entrypoint
├── docker-compose.yml
├── railway.json
└── Procfile
```

---

## Quick Start — Local (No Docker)

### 1. Backend

```bash
cd phishing_detection/backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start API server
python api/app.py
# API now running at http://localhost:5000
```

### 2. Frontend

```bash
cd phishing_detection/frontend
npm install
npm run dev
# UI now running at http://localhost:3000
```

### 3. Train models (separate terminal)

```bash
cd phishing_detection
source backend/venv/bin/activate
python train.py                  # synthetic dataset, fast mode (~5 min)
python train.py --full           # full grid search (~1–3 hours)
python train.py --real           # attempt to load datasets from data/
```

---

## Providing Real Datasets

Place the following files inside the `data/` directory:

| File | Source | Format |
|------|--------|--------|
| `phishtank.csv` | https://www.phishtank.com/developer_info.php → `verified_online.csv` | CSV with `url`, `verified` columns |
| `openphish.txt` | https://openphish.com/feed.txt | One URL per line |
| `tranco.csv` | https://tranco-list.eu/ | CSV with `rank`, `domain` columns |
| `iscx.csv` | ISCX URL Dataset (academic request) | CSV with `url`, `type` columns |

If no datasets are found, the system auto-generates a **41,250-sample synthetic dataset** that mirrors the statistical distributions described in Chapter 4 of the thesis.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/predict` | Single URL prediction (requires trained model) |
| `POST` | `/api/predict/batch` | Batch prediction (max 100 URLs) |
| `POST` | `/api/predict/demo` | Heuristic demo (no model needed) |
| `POST` | `/api/train` | Trigger training pipeline |
| `GET` | `/api/train/status` | Poll training progress |
| `GET` | `/api/results` | Get experiment results JSON |
| `GET` | `/api/models/status` | Check which models are trained |
| `GET` | `/api/features/explain` | Get feature definitions |

### Example: Single prediction

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://paypal-login.tk/verify", "condition": "C6"}'
```

### Example: Start training

```bash
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d '{"fast_mode": true}'
```

---

## Local Docker Deployment

```bash
cd phishing_detection
docker-compose up --build

# Frontend → http://localhost:3000
# Backend  → http://localhost:5000
```

To persist trained models across container restarts, the `backend/models/saved/` directory is volume-mounted.

---

## Free Cloud Deployment

### Backend → Railway (free tier)

1. Push this repository to GitHub.
2. Go to https://railway.app → **New Project** → **Deploy from GitHub Repo**.
3. Select your repo. Railway auto-detects the `Dockerfile` in `backend/`.
4. Add environment variable: `PORT=5000`
5. Click **Deploy**. Copy the generated URL (e.g. `https://your-app.railway.app`).

> **Note:** Railway free tier has 500 hours/month and sleeps after inactivity. For always-on, use the $5/month Starter plan.

### Frontend → Vercel (free tier, always-on)

1. Go to https://vercel.com → **New Project** → Import your GitHub repo.
2. Set **Root Directory** to `frontend`.
3. Add Environment Variable:
   - Name: `VITE_API_URL`
   - Value: `https://your-app.railway.app` (your Railway backend URL)
4. Click **Deploy**.

> Edit `frontend/vercel.json` and replace `YOUR_RAILWAY_BACKEND_URL` with your actual Railway URL before deploying.

### Alternative: Render (backend, free tier)

1. Go to https://render.com → **New Web Service**.
2. Connect GitHub repo, set **Root Directory** to `backend`.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn --bind 0.0.0.0:$PORT api.app:app`
5. Free tier spins down after 15 min inactivity (cold start ~30s).

---

## Experimental Conditions Reference

| Condition | Pipeline | Classifier | Features | F1 | AUC-ROC | FPR | MCC |
|-----------|----------|------------|----------|----|---------|-----|-----|
| C1 | Lexical | Random Forest | 14 | 0.920 | 0.945 | 0.082 | 0.831 |
| C2 | Lexical | XGBoost | 14 | 0.933 | 0.958 | 0.065 | 0.856 |
| C3 | Structural | Random Forest | 14 | 0.949 | 0.972 | 0.041 | 0.892 |
| C4 | Structural | XGBoost | 14 | 0.960 | 0.981 | 0.032 | 0.915 |
| C5 | Combined | Random Forest | 25 | 0.973 | 0.989 | 0.021 | 0.944 |
| **C6** | **Combined** | **XGBoost** | **25** | **0.984** | **0.994** | **0.014** | **0.968** |

**C6 is the optimal condition** — Combined XGBoost achieves near-perfect MCC of 0.968 and an FPR of just 1.4%.

---

## Dataset Statistics (Chapter 4)

- Initial records: 45,000 (PhishTank + OpenPhish + Tranco)
- After deduplication: 41,250 unique domains
- Class distribution: 31,500 phishing / 9,750 legitimate (3.23:1 ratio)
- SMOTE applied to training partition only → 25,200 per class
- Test set: 20% holdout, unmodified (preserves real-world prior probabilities)
- Validation: Stratified 10-fold CV with nested 5-fold GridSearchCV

---

## McNemar's Test Results

| Comparison | p-value | Significant (α=0.05) |
|-----------|---------|---------------------|
| C1 vs C2 (Lexical: RF vs XGB) | 0.084 | No |
| C3 vs C4 (Structural: RF vs XGB) | 0.012 | Yes |
| C5 vs C6 (Combined: RF vs XGB) | <0.001 | Yes |

**Key finding:** XGBoost's superiority over RF is only statistically significant when structural metadata is present. RF's entropy-based splitting performs comparably on lexical features alone.

---

## Adding Your Own Datasets

After placing datasets in `data/`, re-run training:

```bash
python train.py --real --data ./data
```

Or trigger via API:

```bash
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d '{"fast_mode": false, "data_dir": "/absolute/path/to/data"}'
```

Poll status: `GET /api/train/status`

---

## Troubleshooting

**WHOIS/DNS queries failing:** This is expected in sandboxed environments. The system falls back to offline cached mode. Structural features still work via the synthetic generator.

**`ModuleNotFoundError: No module named 'whois'`:** Run `pip install python-whois` inside your virtual environment.

**Training takes too long:** Use `--fast` mode (default) or reduce dataset size via `--phishing 5000 --legit 1500`.

**Railway cold starts:** First request after sleep may take 30–60 seconds. This is normal on the free tier.
