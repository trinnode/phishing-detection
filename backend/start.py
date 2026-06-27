"""
Startup script: auto-trains models if needed, then launches gunicorn.
"""
import os
import sys
import subprocess
from pathlib import Path

MODELS_DIR = Path(__file__).parent / 'models' / 'saved'
C6_PATH = MODELS_DIR / 'C6.pkl'

def needs_training():
    return not C6_PATH.exists()

def run_training():
    """Run training synchronously before starting the server."""
    sys.path.insert(0, str(Path(__file__).parent))
    from core.dataset_loader import prepare_datasets_from_real_data
    from core.trainer import run_all_conditions

    print("[startup] No trained models found. Auto-training with synthetic dataset...")
    print("[startup] Step 1/3: Preparing dataset...")
    df_lex, df_struct, df_combined, y = prepare_datasets_from_real_data()
    print(f"[startup] Dataset ready: {len(y)} samples")

    print("[startup] Step 2/3: Training all 6 conditions (C1–C6)...")
    results = run_all_conditions(df_lex, df_struct, df_combined, y, fast_mode=True)

    print("[startup] Step 3/3: Training complete!")
    c6 = results.get('conditions', {}).get('C6', {})
    print(f"[startup] C6 (Combined XGBoost) — F1: {c6.get('f1_score', 'N/A'):.4f}, AUC: {c6.get('auc_roc', 'N/A'):.4f}")
    print("[startup] Models saved to backend/models/saved/")

if __name__ == '__main__':
    if needs_training():
        run_training()
    else:
        print("[startup] Models already trained.")

    # Launch gunicorn
    port = os.environ.get('PORT', '5000')
    cmd = [
        'gunicorn',
        '--bind', f'0.0.0.0:{port}',
        '--workers', '2',
        '--timeout', '300',
        'api.app:app',
    ]
    os.execvp('gunicorn', cmd)
