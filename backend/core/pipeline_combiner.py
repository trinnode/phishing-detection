"""
Combined Feature Matrix Assembler (Pipeline C)
Horizontally concatenates lexical + structural vectors.
Applies Pearson correlation-based feature reduction (r > 0.90 threshold).
"""

import numpy as np
import pandas as pd
from typing import List, Optional

from .lexical_extractor import extract_lexical_features, LEXICAL_FEATURE_NAMES
from .structural_extractor import extract_structural_features, STRUCTURAL_FEATURE_NAMES


def assemble_combined_vector(url: str, live: bool = True, use_cache: bool = True) -> dict:
    """
    Build the full 28-feature (pre-reduction) combined vector for a single URL.
    """
    lex = extract_lexical_features(url)
    struct = extract_structural_features(url, use_cache=use_cache, live=live)
    return {**lex, **struct}


def build_feature_matrix(
    urls: List[str],
    pipeline: str = 'combined',
    live: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Build a feature DataFrame for a list of URLs.

    Args:
        urls:      List of URL strings.
        pipeline:  'lexical', 'structural', or 'combined'.
        live:      Enable live structural queries (requires network).
        use_cache: Use cached structural data when available.

    Returns:
        pd.DataFrame with one row per URL, columns = feature names.
    """
    records = []
    for url in urls:
        if pipeline == 'lexical':
            record = extract_lexical_features(url)
        elif pipeline == 'structural':
            record = extract_structural_features(url, use_cache=use_cache, live=live)
        else:  # combined
            record = assemble_combined_vector(url, live=live, use_cache=use_cache)
        records.append(record)
    return pd.DataFrame(records)


def apply_correlation_reduction(
    df: pd.DataFrame,
    threshold: float = 0.90,
    drop_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Remove highly collinear features from a feature matrix.
    Per methodology: Pearson r > 0.90 triggers removal of the second feature.

    Args:
        df:         Feature DataFrame (no label column).
        threshold:  Pearson correlation threshold.
        drop_cols:  Pre-specified columns to drop (if already determined from training).

    Returns:
        Reduced DataFrame with collinear features removed.
    """
    if drop_cols is not None:
        return df.drop(columns=[c for c in drop_cols if c in df.columns])

    corr_matrix = df.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop), to_drop


def get_pipeline_feature_names(pipeline: str) -> List[str]:
    if pipeline == 'lexical':
        return LEXICAL_FEATURE_NAMES
    elif pipeline == 'structural':
        return STRUCTURAL_FEATURE_NAMES
    else:
        return LEXICAL_FEATURE_NAMES + STRUCTURAL_FEATURE_NAMES
