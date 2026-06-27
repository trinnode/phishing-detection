"""
Training Pipeline — Six Experimental Conditions (C1–C6)
Implements:
  - Stratified 80/20 split (holdout preserved without SMOTE)
  - SMOTE on training partition only
  - Nested Stratified 10-Fold CV with GridSearchCV for hyperparameter tuning
  - Random Forest (C1, C3, C5) and XGBoost (C2, C4, C6)
  - McNemar's test for pairwise significance
  - Full metric suite: Accuracy, Precision, Recall, F1, AUC-ROC, FPR, MCC
"""

import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    matthews_corrcoef
)
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from statsmodels.stats.contingency_tables import mcnemar

warnings.filterwarnings('ignore')

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Hyperparameter grids ───────────────────────────────────────────────────────

RF_PARAM_GRID = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2'],
}

XGB_PARAM_GRID = {
    'n_estimators': [100, 200, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 1.5],
}


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics = {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1_score': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'false_positive_rate': round(fpr, 4),
        'mcc': round(matthews_corrcoef(y_true, y_pred), 4),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }
    if y_prob is not None:
        metrics['auc_roc'] = round(roc_auc_score(y_true, y_prob), 4)
    return metrics


def mcnemar_test(y_true, preds_a, preds_b) -> dict:
    """McNemar's test comparing two classifiers on same test set."""
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))   # A correct, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B correct
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=True)
    return {
        'statistic': round(float(result.statistic), 4),
        'p_value': round(float(result.pvalue), 6),
        'significant_at_0.05': bool(result.pvalue < 0.05),
    }


# ── Core training function ─────────────────────────────────────────────────────

def train_condition(
    X: pd.DataFrame,
    y: pd.Series,
    classifier: str,          # 'rf' or 'xgb'
    condition_id: str,        # e.g. 'C1'
    random_state: int = 42,
    fast_mode: bool = False,  # True = reduced grid for quick testing
) -> Tuple[dict, object, np.ndarray, np.ndarray]:
    """
    Train one experimental condition using nested CV + holdout evaluation.

    Returns:
        (metrics_dict, best_model, y_test, y_pred_test)
    """
    print(f"\n{'='*60}")
    print(f"Training Condition {condition_id} | Classifier: {classifier.upper()} | Features: {X.shape[1]}")
    print(f"{'='*60}")

    # ── 80/20 stratified split ────────────────────────────────────────────
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # ── Min-Max scaling ───────────────────────────────────────────────────
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test)

    # ── SMOTE on training partition ONLY ─────────────────────────────────
    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train_raw)
    print(f"  Post-SMOTE training set: {len(X_train_bal)} samples "
          f"({sum(y_train_bal==0)} legit / {sum(y_train_bal==1)} phishing)")

    # ── Build classifier and param grid ──────────────────────────────────
    if classifier == 'rf':
        base_model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        param_grid = (
            {'n_estimators': [100, 200], 'max_depth': [None, 10]}
            if fast_mode else RF_PARAM_GRID
        )
    else:
        base_model = XGBClassifier(
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1,
        )
        param_grid = (
            {'n_estimators': [100], 'max_depth': [6], 'learning_rate': [0.1]}
            if fast_mode else XGB_PARAM_GRID
        )

    # ── Nested Stratified 10-Fold CV ──────────────────────────────────────
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='f1',
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    grid_search.fit(X_train_bal, y_train_bal)
    best_model = grid_search.best_estimator_
    print(f"  Best params: {grid_search.best_params_}")

    # ── Final evaluation on holdout test set ─────────────────────────────
    y_pred = best_model.predict(X_test_scaled)
    y_prob = (
        best_model.predict_proba(X_test_scaled)[:, 1]
        if hasattr(best_model, 'predict_proba') else None
    )

    metrics = compute_metrics(y_test.values, y_pred, y_prob)
    metrics['condition'] = condition_id
    metrics['classifier'] = classifier.upper()
    metrics['n_features'] = X.shape[1]
    metrics['best_params'] = grid_search.best_params_

    print(f"  Accuracy: {metrics['accuracy']} | F1: {metrics['f1_score']} | AUC-ROC: {metrics.get('auc_roc','N/A')}")
    print(f"  FPR: {metrics['false_positive_rate']} | MCC: {metrics['mcc']}")

    # ── Save model + scaler ───────────────────────────────────────────────
    model_path = MODELS_DIR / f"{condition_id}.pkl"
    scaler_path = MODELS_DIR / f"{condition_id}_scaler.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    return metrics, best_model, y_test.values, y_pred


def run_all_conditions(
    df_lexical: pd.DataFrame,
    df_structural: pd.DataFrame,
    df_combined: pd.DataFrame,
    y: pd.Series,
    fast_mode: bool = False,
    status: Optional[dict] = None,
) -> dict:
    """
    Run all 6 experimental conditions and save results.

    Args:
        df_lexical:    14-feature lexical matrix
        df_structural: 14-feature structural matrix
        df_combined:   25-feature combined matrix (post-reduction)
        y:             Binary label series (1=phishing, 0=legitimate)
        fast_mode:     Reduce grid search for quick testing
    """
    all_results = {}
    all_preds = {}

    conditions = [
        ('C1', df_lexical, 'rf'),
        ('C2', df_lexical, 'xgb'),
        ('C3', df_structural, 'rf'),
        ('C4', df_structural, 'xgb'),
        ('C5', df_combined, 'rf'),
        ('C6', df_combined, 'xgb'),
    ]

    for cid, X, clf in conditions:
        cond_desc = {'C1': 'Lexical RF', 'C2': 'Lexical XGB', 'C3': 'Structural RF',
                     'C4': 'Structural XGB', 'C5': 'Combined RF', 'C6': 'Combined XGB'}
        cls_name = 'Random Forest' if clf == 'rf' else 'XGBoost'
        if status is not None:
            status['log'].append(f'  Training {cid} ({cond_desc[cid]}) — {X.shape[1]} features, {cls_name} classifier...')
        metrics, model, y_test, y_pred = train_condition(
            X, y, clf, cid, fast_mode=fast_mode
        )
        all_results[cid] = metrics
        all_preds[cid] = {'y_test': y_test.tolist(), 'y_pred': y_pred.tolist()}
        if status is not None:
            status['log'].append(
                f'  ✓ {cid} complete — F1: {metrics["f1_score"]:.4f}, '
                f'AUC: {metrics.get("auc_roc", 0):.4f}, FPR: {metrics["false_positive_rate"]:.4f}'
            )

    # ── McNemar's significance tests ──────────────────────────────────────
    if status is not None:
        status['log'].append('[6/7] Phase 6: Statistical Evaluation — McNemar\'s test...')
    print("\n── Statistical Significance (McNemar's Test) ──")
    significance = {}

    # C1 vs C2: RF vs XGB on lexical
    mn_12 = mcnemar_test(
        np.array(all_preds['C1']['y_test']),
        np.array(all_preds['C1']['y_pred']),
        np.array(all_preds['C2']['y_pred']),
    )
    significance['C1_vs_C2_lexical'] = mn_12
    print(f"  C1 vs C2 (Lexical): p={mn_12['p_value']} | Significant: {mn_12['significant_at_0.05']}")

    # C3 vs C4: RF vs XGB on structural
    mn_34 = mcnemar_test(
        np.array(all_preds['C3']['y_test']),
        np.array(all_preds['C3']['y_pred']),
        np.array(all_preds['C4']['y_pred']),
    )
    significance['C3_vs_C4_structural'] = mn_34
    print(f"  C3 vs C4 (Structural): p={mn_34['p_value']} | Significant: {mn_34['significant_at_0.05']}")

    # C5 vs C6: RF vs XGB on combined
    mn_56 = mcnemar_test(
        np.array(all_preds['C5']['y_test']),
        np.array(all_preds['C5']['y_pred']),
        np.array(all_preds['C6']['y_pred']),
    )
    significance['C5_vs_C6_combined'] = mn_56
    print(f"  C5 vs C6 (Combined): p={mn_56['p_value']} | Significant: {mn_56['significant_at_0.05']}")

    # ── Persist results ───────────────────────────────────────────────────
    n_phishing = int(sum(y == 1))
    n_legitimate = int(sum(y == 0))
    final_output = {
        'conditions': all_results,
        'significance_tests': significance,
        'dataset_info': {
            'total_samples': len(y),
            'phishing': n_phishing,
            'legitimate': n_legitimate,
            'ratio': f'{n_phishing / max(n_legitimate, 1):.2f}:1',
            'n_features_lexical': df_lexical.shape[1],
            'n_features_structural': df_structural.shape[1],
            'n_features_combined': df_combined.shape[1],
        },
    }
    results_path = RESULTS_DIR / 'experiment_results.json'
    with open(results_path, 'w') as f:
        json.dump(final_output, f, indent=2, default=str)
    print(f"\n✓ Results saved to {results_path}")

    return final_output
