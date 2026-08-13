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
from .paths import data_root

DATA_DIR = data_root() / 'data'


# ── Real dataset loaders ───────────────────────────────────────────────────────

def load_phishtank_csv(path: str) -> pd.DataFrame:
    """Load PhishTank verified_online.csv — handles both column schemas."""
    df = pd.read_csv(path, nrows=0)
    cols = df.columns.tolist()
    if 'url' not in cols:
        return pd.DataFrame(columns=['url', 'label'])

    if 'verified' in cols:
        df = pd.read_csv(path, usecols=['url', 'verified'])
        df = df[df['verified'] == 'yes'].copy()
    elif 'verification_status' in cols:
        df = pd.read_csv(path, usecols=['url', 'verification_status'])
        df = df[df['verification_status'] == 'verified'].copy()
    else:
        df = pd.read_csv(path, usecols=['url'])
    df['label'] = 1
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
    tactics = random.choice(['dga', 'typosquatting', 'combosquatting', 'ip', 'obfuscated', 'masked_legit'])
    brands = ['paypal', 'amazon', 'google', 'microsoft', 'apple', 'ebay', 'facebook', 'instagram']
    risky_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.click', '.info']
    safe_tlds = ['.com', '.org', '.net', '.co.uk']

    if tactics == 'dga':
        domain = _high_entropy_string(random.randint(12, 24))
        tld = random.choice(risky_tlds if random.random() < 0.7 else safe_tlds)
        return f"http://{domain}{tld}/login"

    elif tactics == 'typosquatting':
        brand = random.choice(brands)
        mutation = brand[:-1] + random.choice(string.ascii_lowercase)
        tld = random.choice(risky_tlds if random.random() < 0.6 else safe_tlds)
        return f"http://secure-{mutation}{tld}/verify.php?id={_random_string(8)}"

    elif tactics == 'combosquatting':
        brand = random.choice(brands)
        suffix = random.choice(['-login', '-secure', '-verify', '-support', '-update', '-confirm'])
        tld = random.choice(risky_tlds + safe_tlds)
        path = random.choice(['/account', '/signin', '/password-reset', '/confirm'])
        return f"http://{brand}{suffix}{tld}{path}"

    elif tactics == 'ip':
        ip = '.'.join(str(random.randint(1, 254)) for _ in range(4))
        return f"http://{ip}/login.php?redirect={_random_string(12)}"

    elif tactics == 'obfuscated':
        brand = random.choice(brands)
        subdomain = '.'.join([_random_string(6) for _ in range(random.randint(2, 4))])
        return f"http://{subdomain}.{brand}-secure.{_random_string(4)}.tk/@user?src=email"

    else:  # masked_legit — phishing URL that looks legitimate on surface
        brand = random.choice(brands)
        return f"https://www.{brand}.com/{_random_string(4)}/{_random_string(8)}"


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

    # ~10% of legitimate URLs use unusual structures (CDN, short URLs, hyphens)
    if random.random() < 0.10:
        return f"https://cdn-{random.choice(brands)}-static{random.choice(tlds)}/assets/{_random_string(6)}"

    return f"https://www.{brand}{tld}{path}"


def _generate_structural_features(is_phishing: bool) -> dict:
    """
    Generate synthetic but statistically realistic structural features.
    Distributions overlap between classes (no perfect separation).
    """
    if is_phishing:
        # Overlapping distributions: phishing sites skew young/risky but some overlap with legitimate
        age = int(np.clip(np.random.exponential(120), 0, 2000))
        expiry = int(np.clip(np.random.normal(300, 180), 1, 2000))
        dns_ttl = int(np.clip(np.random.lognormal(6, 1.5), 60, 86400))
        ssl_days = int(np.clip(np.random.exponential(90), 0, 730))
        registrar_ent = round(np.random.uniform(1.5, 4.5), 3)

        return {
            'domain_age_days': age,
            'domain_expiry_days': expiry,
            'whois_available': int(random.random() < 0.55),
            'dns_ttl_value': dns_ttl,
            'has_mx_record': int(random.random() < 0.45),
            'has_spf_record': int(random.random() < 0.35),
            'dns_resolves': int(random.random() < 0.88),
            'ns_count': int(np.clip(np.random.poisson(2.0), 1, 8)),
            'ssl_valid': int(random.random() < 0.55),
            'ssl_days_remaining': ssl_days,
            'ip_in_blacklist_asn': int(random.random() < 0.20),
            'registrar_entropy': registrar_ent,
            'country_code_risk': int(random.random() < 0.35),
            'nameserver_diversity': int(random.random() < 0.50),
        }
    else:
        # Legitimate domains skew older/stable but overlap with phishing tail
        age = int(np.clip(np.random.exponential(800), 30, 5000))
        expiry = int(np.clip(np.random.normal(500, 250), 30, 2500))
        dns_ttl = int(np.clip(np.random.lognormal(7.5, 1.2), 120, 86400))
        ssl_days = int(np.clip(np.random.exponential(200), 0, 1500))
        registrar_ent = round(np.random.uniform(1.0, 3.5), 3)

        return {
            'domain_age_days': age,
            'domain_expiry_days': expiry,
            'whois_available': int(random.random() < 0.85),
            'dns_ttl_value': dns_ttl,
            'has_mx_record': int(random.random() < 0.80),
            'has_spf_record': int(random.random() < 0.70),
            'dns_resolves': int(random.random() < 0.98),
            'ns_count': int(np.clip(np.random.poisson(3.0), 1, 10)),
            'ssl_valid': int(random.random() < 0.90),
            'ssl_days_remaining': ssl_days,
            'ip_in_blacklist_asn': int(random.random() < 0.03),
            'registrar_entropy': registrar_ent,
            'country_code_risk': int(random.random() < 0.08),
            'nameserver_diversity': int(random.random() < 0.80),
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

    # Add Gaussian noise to lexical features to prevent unrealistic perfect separation
    # Real-world lexical features have significant noise — same URL type can vary widely
    rng = np.random.RandomState(random_seed)
    for col in df_lex.columns:
        if col in ('has_ip_address', 'has_at_symbol', 'has_double_slash', 'tld_in_legitimate_list'):
            continue
        scale = df_lex[col].std() * 0.12 if df_lex[col].std() > 0 else 0.01
        df_lex[col] += rng.normal(0, scale, size=len(df_lex))
        df_lex[col] = df_lex[col].clip(lower=0)

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

    if len(set(labels)) < 2:
        print(f"  Dataset contains only one class ({set(labels)}). Falling back to synthetic dataset.")
        return generate_synthetic_dataset()
    y = pd.Series(labels, name='label')

    print(f"  {len(urls)} URLs loaded. Extracting features...")

    print("  Extracting lexical features...")
    lex_records = [extract_lexical_features(u) for u in urls]
    df_lex = pd.DataFrame(lex_records)

    # For real data, use offline structural defaults (WHOIS queries not feasible for 40k URLs)
    # Generate features from blended distribution to avoid label leakage
    print("  Generating structural features (offline mode for large dataset)...")
    blended_labels = [random.choice([0, 1]) for _ in labels]
    struct_records = [_generate_structural_features(bool(lbl)) for lbl in blended_labels]
    df_struct = pd.DataFrame(struct_records)

    df_combined_raw = pd.concat([df_lex, df_struct], axis=1)
    df_combined_reduced, dropped = apply_correlation_reduction(df_combined_raw)

    return df_lex, df_struct, df_combined_reduced, y
