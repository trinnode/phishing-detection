"""
Generate and save a small sample CSV for testing without real datasets.
Run: python scripts/generate_sample_data.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

from core.dataset_loader import generate_synthetic_dataset
import pandas as pd

df_lex, df_struct, df_combined, y = generate_synthetic_dataset(
    n_phishing=500, n_legitimate=150
)

out = pd.concat([df_lex, df_struct, y], axis=1)
out.to_csv('data/sample/sample_dataset.csv', index=False)
print(f"Saved {len(out)} rows to data/sample/sample_dataset.csv")
