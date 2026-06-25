#!/usr/bin/env python3
"""
FishMark Feature Extraction Framework — Standalone CLI
Extracts lexical and structural features from domain name datasets in batch.
Can be used independently as a reusable feature extraction pipeline.

Usage:
    # Extract lexical features from a CSV file
    python scripts/extract_features.py --input data/urls.csv --output features.csv --pipeline lexical

    # Extract combined features from a TXT file (one URL per line)
    python scripts/extract_features.py --input data/urls.txt --output features.csv --pipeline combined

    # Extract structural features with live WHOIS/DNS queries
    python scripts/extract_features.py --input data/urls.csv --output features.csv --pipeline structural --live
"""

import sys
import csv
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

import pandas as pd
from core.lexical_extractor import extract_lexical_features, LEXICAL_FEATURE_NAMES
from core.structural_extractor import extract_structural_features, STRUCTURAL_FEATURE_NAMES
from core.pipeline_combiner import build_feature_matrix, apply_correlation_reduction


def load_urls(path: str) -> list:
    path = Path(path)
    if not path.exists():
        print(f'Error: File not found: {path}', file=sys.stderr)
        sys.exit(1)

    urls = []
    if path.suffix.lower() == '.csv':
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader, None)
            col_idx = 0
            if header:
                url_col = [i for i, c in enumerate(header) if c.lower().strip() in ('url', 'domain', 'website')]
                col_idx = url_col[0] if url_col else 0
            for row in reader:
                if row and row[col_idx].strip():
                    urls.append(row[col_idx].strip())
    else:
        with open(path) as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    return urls


def main():
    parser = argparse.ArgumentParser(
        description='FishMark Feature Extraction Framework — Extract lexical/structural features from domain datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--input', '-i', required=True, help='Input file path (CSV or TXT, one URL per line)')
    parser.add_argument('--output', '-o', default='features.csv', help='Output CSV path (default: features.csv)')
    parser.add_argument('--pipeline', '-p', choices=['lexical', 'structural', 'combined'], default='lexical',
                        help='Feature pipeline to use (default: lexical)')
    parser.add_argument('--live', action='store_true', help='Enable live WHOIS/DNS queries for structural features')
    parser.add_argument('--reduce', action='store_true', help='Apply Pearson correlation reduction (r > 0.90) on combined features')
    parser.add_argument('--stats', action='store_true', help='Print feature statistics after extraction')

    args = parser.parse_args()

    print(f'FishMark Feature Extraction Framework')
    print(f'Pipeline: {args.pipeline}')
    print(f'Input:    {args.input}')

    urls = load_urls(args.input)
    if not urls:
        print('Error: No valid URLs found in input file', file=sys.stderr)
        sys.exit(1)

    print(f'URLs loaded: {len(urls)}')

    print('Extracting features...')
    if args.pipeline == 'lexical':
        features = build_feature_matrix(urls, pipeline='lexical')
        n_feats = len(LEXICAL_FEATURE_NAMES)
    elif args.pipeline == 'structural':
        features = build_feature_matrix(urls, pipeline='structural', live=args.live)
        n_feats = len(STRUCTURAL_FEATURE_NAMES)
    else:
        features = build_feature_matrix(urls, pipeline='combined', live=args.live)
        n_feats = len(LEXICAL_FEATURE_NAMES) + len(STRUCTURAL_FEATURE_NAMES)
        if args.reduce:
            features, dropped = apply_correlation_reduction(features)
            print(f'  Correlation reduction removed {len(dropped)} features: {dropped}')

    print(f'Features extracted: {features.shape[1]} per URL')

    features.to_csv(args.output, index=False)
    print(f'Output saved: {args.output}')

    if args.stats:
        print()
        print('Feature statistics:')
        print(features.describe().to_string())
        print()


if __name__ == '__main__':
    main()
