#!/usr/bin/env python3
"""
Main Training Script
Usage:
    python train.py                   # synthetic dataset, fast mode
    python train.py --real            # attempt real dataset loading
    python train.py --full            # full grid search (slow, ~hours)
    python train.py --data ./data     # specify data directory
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from core.dataset_loader import prepare_datasets_from_real_data, generate_synthetic_dataset
from core.trainer import run_all_conditions


def main():
    parser = argparse.ArgumentParser(description='Train all 6 experimental conditions')
    parser.add_argument('--real', action='store_true', help='Use real datasets from data/')
    parser.add_argument('--full', action='store_true', help='Full grid search (not fast mode)')
    parser.add_argument('--data', type=str, default=None, help='Path to data directory')
    parser.add_argument('--phishing', type=int, default=31500, help='Synthetic phishing count')
    parser.add_argument('--legit', type=int, default=9750, help='Synthetic legitimate count')
    args = parser.parse_args()

    fast_mode = not args.full
    print(f"\n{'='*60}")
    print("PHISHING DETECTION TRAINING PIPELINE")
    print(f"Mode: {'Full GridSearch' if args.full else 'Fast Mode'}")
    print(f"{'='*60}\n")

    if args.real:
        df_lex, df_struct, df_combined, y = prepare_datasets_from_real_data(args.data)
    else:
        df_lex, df_struct, df_combined, y = generate_synthetic_dataset(
            n_phishing=args.phishing,
            n_legitimate=args.legit,
        )

    results = run_all_conditions(df_lex, df_struct, df_combined, y, fast_mode=fast_mode)

    print("\n\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for cid, metrics in results['conditions'].items():
        print(f"\n  {cid} ({metrics['classifier']}, {metrics['n_features']} features):")
        print(f"    F1={metrics['f1_score']} | AUC-ROC={metrics.get('auc_roc','N/A')} | FPR={metrics['false_positive_rate']} | MCC={metrics['mcc']}")

    print("\n\nSignificance Tests:")
    for test_name, test_result in results['significance_tests'].items():
        sig = "✓ SIGNIFICANT" if test_result['significant_at_0.05'] else "✗ NOT significant"
        print(f"  {test_name}: p={test_result['p_value']} | {sig}")

    print("\n✓ All models saved to backend/models/saved/")
    print("✓ Results saved to backend/results/experiment_results.json")
    print("\nNext step: python -m backend.api.app  (to start API server)")


if __name__ == '__main__':
    main()
