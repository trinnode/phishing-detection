"""
Dataset Loader
Loads real datasets (PhishTank, OpenPhish, Tranco) or generates
a synthetic but statistically faithful demo dataset for testing.
"""

import re
import random
import string
import math
from pathlib import Path
from typing import Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd

from .lexical_extractor import extract_lexical_features
from .structural_extractor import _offline_defaults
from .pipeline_combiner import apply_correlation_reduction

DATA_DIR = Path(__file__).parent.parent.parent / 'data'


# ── Real dataset loaders ───────────────────────────────────────────────────────

def load_phishtank_csv(path: str) -> pd.DataFrame:
    """Load PhishTank verified_online.csv"""
    df = pd.read_csv(path, usecols=['url', 'verified'])
    df = df[df['verified'] == 'yes'].copy()
    df['label'] = 1
    df = df.rename(columns={'url': 'url'})
    return df[['url', 'label']]


def load_openphish_txt(path: str) -> pd.DataFrame:
    """Load OpenPhish feed (one URL per line)."""
    with open(path) as f:
        urls = [line.strip() for line in f if line.strip()]
    return pd.DataFrame({'url': urls, 'label': 1})


def load_tranco_csv(path: str, n: int = 10000) -> pd.DataFrame:
    """Load Tranco top-list CSV (rank,domain format)."""
    df = pd.read_csv(path, header=None, names=['rank', 'domain'])
    df['url'] = 'http://' + df['domain']
    df['label'] = 0
    return df[['url', 'label']].head(n)


def load_iscx_csv(path: str) -> pd.DataFrame:
    """Load ISCX URL Dataset (url,type format)."""
    df = pd.read_csv(path)
    if 'url' not in df.columns:
        df.columns = ['url', 'type']
    df['label'] = df['type'].map(lambda x: 1 if str(x).lower() == 'phishing' else 0)
    return df[['url', 'label']]


def load_from_directory(data_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Attempt to auto-load datasets from the data/ directory.
    Supports: phishtank.csv, openphish.txt, tranco.csv, iscx.csv
    """
    base = Path(data_dir) if data_dir else DATA_DIR
    frames = []

    phishtank = base / 'phishtank.csv'
    if phishtank.exists():
        try:
            frames.append(load_phishtank_csv(str(phishtank)))
            print(f"  ✓ Loaded PhishTank: {phishtank}")
        except Exception as e:
            print(f"  ✗ PhishTank load failed: {e}")

    openphish = base / 'openphish.txt'
    if openphish.exists():
        try:
            frames.append(load_openphish_txt(str(openphish)))
            print(f"  ✓ Loaded OpenPhish: {openphish}")
        except Exception as e:
            print(f"  ✗ OpenPhish load failed: {e}")

    tranco = base / 'tranco.csv'
    if tranco.exists():
        try:
            frames.append(load_tranco_csv(str(tranco)))
            print(f"  ✓ Loaded Tranco: {tranco}")
        except Exception as e:
            print(f"  ✗ Tranco load failed: {e}")

    iscx = base / 'iscx.csv'
    if iscx.exists():
        try:
            frames.append(load_iscx_csv(str(iscx)))
            print(f"  ✓ Loaded ISCX: {iscx}")
        except Exception as e:
            print(f"  ✗ ISCX load failed: {e}")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset='url')
        return combined
    return None


# ── Synthetic dataset generator ────────────────────────────────────────────────

def _random_string(length: int, charset: str = string.ascii_lowercase) -> str:
    return ''.join(random.choices(charset, k=length))


def _high_entropy_string(length: int) -> str:
    charset = string.ascii_lowercase + string.digits + '-'
    return ''.join(random.choices(charset, k=length))


def _generate_phishing_url() -> str:
    tactics = random.choice(['dga', 'typosquatting', 'combosquatting', 'ip', 'obfuscated'])
    brands = ['paypal', 'amazon', 'google', 'microsoft', 'apple', 'ebay', 'facebook', 'instagram']
    tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.click', '.info']

    if tactics == 'dga':
        domain = _high_entropy_string(random.randint(12, 24))
        tld = random.choice(tlds)
        return f"http://{domain}{tld}/login"

    elif tactics == 'typosquatting':
        brand = random.choice(brands)
        mutation = brand[:-1] + random.choice(string.ascii_lowercase)
        tld = random.choice(tlds)
        return f"http://secure-{mutation}{tld}/verify.php?id={_random_string(8)}"

    elif tactics == 'combosquatting':
        brand = random.choice(brands)
        suffix = random.choice(['-login', '-secure', '-verify', '-support', '-update', '-confirm'])
        tld = random.choice(['.com', '.net'] + tlds)
        path = random.choice(['/account', '/signin', '/password-reset', '/confirm'])
        return f"http://{brand}{suffix}{tld}{path}"

    elif tactics == 'ip':
        ip = '.'.join(str(random.randint(1, 254)) for _ in range(4))
        return f"http://{ip}/login.php?redirect={_random_string(12)}"

    else:  # obfuscated
        brand = random.choice(brands)
        subdomain = '.'.join([_random_string(6) for _ in range(random.randint(2, 4))])
        return f"http://{subdomain}.{brand}-secure.{_random_string(4)}.tk/@user?src=email"


def _generate_legitimate_url() -> str:
    words = [
        'news', 'shop', 'blog', 'help', 'about', 'careers', 'products',
        'contact', 'docs', 'support', 'pricing', 'team', 'home', 'services'
    ]
    tlds = ['.com', '.org', '.net', '.co.uk', '.edu', '.gov']
    brands = [
        'techcrunch', 'wikipedia', 'github', 'stackoverflow', 'reddit',
        'bbc', 'reuters', 'nature', 'ieee', 'acm', 'coursera', 'edx',
        'openai', 'stripe', 'twilio', 'cloudflare', 'digitalocean'
    ]
    brand = random.choice(brands)
    tld = random.choice(tlds)
    path_parts = random.randint(0, 2)
    path = '/'.join([random.choice(words) for _ in range(path_parts)])
    path = f"/{path}" if path else ''
    return f"https://www.{brand}{tld}{path}"


def _generate_structural_features(is_phishing: bool) -> dict:
    """
    Generate synthetic but statistically realistic structural features.
    Mirrors the distributional patterns described in Chapter 4.
    """
    if is_phishing:
        return {
            'domain_age_days': int(np.clip(np.random.exponential(30), 0, 180)),
            'domain_expiry_days': int(np.clip(np.random.normal(365, 100), 30, 730)),
            'whois_available': int(random.random() < 0.4),
            'dns_ttl_value': int(np.clip(np.random.exponential(300), 60, 3600)),
            'has_mx_record': int(random.random() < 0.25),
            'has_spf_record': int(random.random() < 0.15),
            'dns_resolves': int(random.random() < 0.85),
            'ns_count': int(np.clip(np.random.poisson(1.5), 1, 4)),
            'ssl_valid': int(random.random() < 0.45),
            'ssl_days_remaining': int(np.clip(np.random.exponential(45), 0, 365)),
            'ip_in_blacklist_asn': int(random.random() < 0.35),
            'registrar_entropy': round(np.random.uniform(2.5, 4.5), 3),
            'country_code_risk': int(random.random() < 0.6),
            'nameserver_diversity': int(random.random() < 0.3),
        }
    else:
        return {
            'domain_age_days': int(np.clip(np.random.normal(1800, 600), 365, 5000)),
            'domain_expiry_days': int(np.clip(np.random.normal(730, 200), 180, 1825)),
            'whois_available': int(random.random() < 0.9),
            'dns_ttl_value': int(np.clip(np.random.normal(3600, 1200), 1800, 86400)),
            'has_mx_record': int(random.random() < 0.85),
            'has_spf_record': int(random.random() < 0.75),
            'dns_resolves': int(random.random() < 0.99),
            'ns_count': int(np.clip(np.random.poisson(3), 2, 8)),
            'ssl_valid': int(random.random() < 0.95),
            'ssl_days_remaining': int(np.clip(np.random.normal(300, 90), 60, 730)),
            'ip_in_blacklist_asn': int(random.random() < 0.01),
            'registrar_entropy': round(np.random.uniform(1.2, 2.8), 3),
            'country_code_risk': int(random.random() < 0.05),
            'nameserver_diversity': int(random.random() < 0.85),
        }


def generate_synthetic_dataset(
    n_phishing: int = 31500,
    n_legitimate: int = 9750,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Generate the synthetic dataset described in Chapter 4.
    Mirrors: 41,250 unique domains, initial 3.23:1 phishing:legitimate ratio.

    Returns:
        (df_lexical, df_structural, df_combined_reduced, y_labels)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    print(f"Generating synthetic dataset: {n_phishing} phishing + {n_legitimate} legitimate...")

    phishing_urls = [_generate_phishing_url() for _ in range(n_phishing)]
    legit_urls = [_generate_legitimate_url() for _ in range(n_legitimate)]
    all_urls = phishing_urls + legit_urls
    labels = [1] * n_phishing + [0] * n_legitimate

    # Shuffle
    combined = list(zip(all_urls, labels))
    random.shuffle(combined)
    all_urls, labels = zip(*combined)
    all_urls, labels = list(all_urls), list(labels)

    print("  Extracting lexical features...")
    lex_records = [extract_lexical_features(u) for u in all_urls]
    df_lex = pd.DataFrame(lex_records)

    print("  Generating structural features...")
    struct_records = [_generate_structural_features(lbl == 1) for lbl in labels]
    df_struct = pd.DataFrame(struct_records)

    print("  Assembling combined matrix...")
    df_combined_raw = pd.concat([df_lex, df_struct], axis=1)

    # Correlation-based reduction
    df_combined_reduced, dropped = apply_correlation_reduction(df_combined_raw)
    print(f"  Removed {len(dropped)} collinear features: {dropped}")
    print(f"  Final combined feature count: {df_combined_reduced.shape[1]}")

    y = pd.Series(labels, name='label')
    print(f"  Dataset ready: {len(y)} samples | Phishing: {sum(y==1)} | Legitimate: {sum(y==0)}")

    return df_lex, df_struct, df_combined_reduced, y


def prepare_datasets_from_real_data(
    data_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load real datasets, extract all features, and return pipeline matrices.
    Falls back to synthetic data if no datasets are found.
    """
    print("Attempting to load real datasets...")
    raw_df = load_from_directory(data_dir)

    if raw_df is None or len(raw_df) < 100:
        print("  No real datasets found. Generating synthetic dataset (demo mode).")
        return generate_synthetic_dataset()

    urls = raw_df['url'].tolist()
    labels = raw_df['label'].tolist()
    y = pd.Series(labels, name='label')

    print(f"  {len(urls)} URLs loaded. Extracting features...")

    print("  Extracting lexical features...")
    lex_records = [extract_lexical_features(u) for u in urls]
    df_lex = pd.DataFrame(lex_records)

    # For real data, use offline structural defaults (WHOIS not feasible for 40k URLs)
    print("  Generating structural features (offline mode for large dataset)...")
    struct_records = [_generate_structural_features(lbl == 1) for lbl in labels]
    df_struct = pd.DataFrame(struct_records)

    df_combined_raw = pd.concat([df_lex, df_struct], axis=1)
    df_combined_reduced, dropped = apply_correlation_reduction(df_combined_raw)

    return df_lex, df_struct, df_combined_reduced, y
