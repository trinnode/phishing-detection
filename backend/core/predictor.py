"""
Inference Engine — loads trained models for real-time single-URL prediction.
Supports all three pipelines and returns probability scores + feature importance.
"""

import pickle
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .lexical_extractor import extract_lexical_features, LEXICAL_FEATURE_NAMES
from .structural_extractor import extract_structural_features, STRUCTURAL_FEATURE_NAMES
from .pipeline_combiner import assemble_combined_vector

warnings.filterwarnings('ignore')

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

# Condition → (pipeline, classifier)
CONDITION_MAP = {
    'C1': ('lexical', 'RF'),
    'C2': ('lexical', 'XGB'),
    'C3': ('structural', 'RF'),
    'C4': ('structural', 'XGB'),
    'C5': ('combined', 'RF'),
    'C6': ('combined', 'XGB'),
}

# Best condition from paper: C6 (Combined XGBoost)
DEFAULT_CONDITION = 'C6'


def _load_model(condition_id: str):
    path = MODELS_DIR / f"{condition_id}.pkl"
    scaler_path = MODELS_DIR / f"{condition_id}_scaler.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Model for {condition_id} not found at {path}. "
            f"Run the training pipeline first."
        )
    with open(path, 'rb') as f:
        model = pickle.load(f)
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    return model, scaler


def _extract_features(url: str, pipeline: str, live: bool = False) -> tuple:
    """Extract feature vector and return (feature_dict, feature_names, feature_values)."""
    if pipeline == 'lexical':
        feat_dict = extract_lexical_features(url)
        names = LEXICAL_FEATURE_NAMES
    elif pipeline == 'structural':
        feat_dict = extract_structural_features(url, live=live)
        names = STRUCTURAL_FEATURE_NAMES
    else:
        feat_dict = assemble_combined_vector(url, live=live)
        names = LEXICAL_FEATURE_NAMES + STRUCTURAL_FEATURE_NAMES

    values = np.array([feat_dict[n] for n in names], dtype=float).reshape(1, -1)
    return feat_dict, names, values


def predict(
    url: str,
    condition_id: str = DEFAULT_CONDITION,
    live_structural: bool = False,
) -> dict:
    """
    Predict whether a URL is phishing using a specified trained condition.

    Args:
        url:              Raw URL string to analyse.
        condition_id:     Which trained model to use (C1–C6).
        live_structural:  Enable live WHOIS/DNS queries for structural features.

    Returns:
        dict with prediction, confidence, features, and importance scores.
    """
    if condition_id not in CONDITION_MAP:
        raise ValueError(f"Invalid condition '{condition_id}'. Choose from {list(CONDITION_MAP.keys())}")

    pipeline, clf_name = CONDITION_MAP[condition_id]

    # Extract features
    feat_dict, feat_names, X = _extract_features(url, pipeline, live=live_structural)

    # Load model
    model, scaler = _load_model(condition_id)

    # Handle feature count mismatch for combined models (post-reduction)
    n_expected = model.n_features_in_
    if X.shape[1] > n_expected:
        X = X[:, :n_expected]
        feat_names = feat_names[:n_expected]
    elif X.shape[1] < n_expected:
        pad = np.zeros((1, n_expected - X.shape[1]))
        X = np.hstack([X, pad])

    # Scale
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    # Predict
    label = int(model.predict(X_scaled)[0])
    prob = float(model.predict_proba(X_scaled)[0][1])

    # Feature importance
    importances = []
    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        n = min(len(feat_names), len(fi))
        importances = sorted(
            [{'feature': feat_names[i], 'importance': round(float(fi[i]), 4)}
             for i in range(n)],
            key=lambda x: x['importance'],
            reverse=True,
        )[:10]

    return {
        'url': url,
        'prediction': 'phishing' if label == 1 else 'legitimate',
        'phishing_probability': round(prob, 4),
        'confidence': round(abs(prob - 0.5) * 2, 4),   # 0–1 scale
        'condition': condition_id,
        'pipeline': pipeline,
        'classifier': clf_name,
        'features': {k: round(float(v), 4) if isinstance(v, float) else v
                     for k, v in feat_dict.items()},
        'top_feature_importances': importances,
        'risk_level': (
            'HIGH' if prob >= 0.75 else
            'MEDIUM' if prob >= 0.45 else
            'LOW'
        ),
    }


def predict_batch(
    urls: list,
    condition_id: str = DEFAULT_CONDITION,
    live_structural: bool = False,
) -> list:
    """Predict a batch of URLs. Returns list of result dicts."""
    return [predict(url, condition_id, live_structural) for url in urls]


def load_experiment_results() -> Optional[dict]:
    """Load saved experiment results from training run."""
    path = RESULTS_DIR / 'experiment_results.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def models_available() -> dict:
    """Check which condition models have been trained."""
    return {
        cid: (MODELS_DIR / f"{cid}.pkl").exists()
        for cid in CONDITION_MAP
    }
