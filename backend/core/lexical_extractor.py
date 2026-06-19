"""
Lexical Feature Extraction Pipeline (Pipeline A)
Derives 14 syntactic/statistical features from raw URL/domain strings.
No external network queries required.
"""

import re
import math
import string
from urllib.parse import urlparse
from collections import Counter


LEGITIMATE_TLDS = {
    '.com', '.org', '.net', '.edu', '.gov', '.co.uk', '.io',
    '.info', '.biz', '.us', '.ca', '.au', '.de', '.fr', '.jp'
}

SUSPICIOUS_KEYWORDS = [
    'login', 'secure', 'account', 'update', 'verify', 'banking',
    'paypal', 'ebay', 'amazon', 'google', 'microsoft', 'apple',
    'support', 'helpdesk', 'signin', 'confirm', 'password', 'free',
    'winner', 'lucky', 'prize', 'click', 'validate', 'authentication'
]


def _shannon_entropy(s: str) -> float:
    """Shannon entropy of a string: H(X) = -sum(p(xi) * log2(p(xi)))"""
    if not s:
        return 0.0
    freq = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _extract_domain_parts(url: str):
    """Parse URL into components safely."""
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    try:
        parsed = urlparse(url)
        full_domain = parsed.netloc or ''
        path = parsed.path or ''
        # strip port
        full_domain = full_domain.split(':')[0]
        # strip www
        domain_no_www = re.sub(r'^www\.', '', full_domain)
        # extract registrable domain (last two labels)
        parts = domain_no_www.split('.')
        subdomain = '.'.join(parts[:-2]) if len(parts) > 2 else ''
        sld = parts[-2] if len(parts) >= 2 else domain_no_www
        tld = '.' + parts[-1] if len(parts) >= 2 else ''
        return full_domain, domain_no_www, subdomain, sld, tld, path
    except Exception:
        return url, url, '', url, '', ''


def extract_lexical_features(url: str) -> dict:
    """
    Extract all 14 lexical features from a URL string.

    Returns:
        dict with keys matching the 14-feature lexical vector definition.
    """
    full_domain, domain_no_www, subdomain, sld, tld, path = _extract_domain_parts(url)
    full_url = url if url.startswith('http') else 'http://' + url

    # ── Feature 1: url_length ─────────────────────────────────────────────
    url_length = len(full_url)

    # ── Feature 2: domain_length ──────────────────────────────────────────
    domain_length = len(domain_no_www)

    # ── Feature 3: shannon_entropy ────────────────────────────────────────
    shannon_entropy = _shannon_entropy(sld)

    # ── Feature 4: digit_ratio ────────────────────────────────────────────
    digit_count = sum(c.isdigit() for c in domain_no_www)
    digit_ratio = digit_count / max(len(domain_no_www), 1)

    # ── Feature 5: hyphen_count ───────────────────────────────────────────
    hyphen_count = domain_no_www.count('-')

    # ── Feature 6: dot_count ──────────────────────────────────────────────
    dot_count = full_url.count('.')

    # ── Feature 7: subdomain_count ────────────────────────────────────────
    subdomain_count = len(subdomain.split('.')) if subdomain else 0

    # ── Feature 8: special_char_ratio ────────────────────────────────────
    special_chars = set(string.punctuation) - {'.', '-', '_', '/'}
    special_count = sum(c in special_chars for c in full_url)
    special_char_ratio = special_count / max(len(full_url), 1)

    # ── Feature 9: has_ip_address ─────────────────────────────────────────
    ip_pattern = re.compile(
        r'(\d{1,3}\.){3}\d{1,3}|'
        r'0x[0-9a-fA-F]+|'
        r'\d{5,10}'  # decimal IP
    )
    has_ip_address = int(bool(ip_pattern.search(full_domain)))

    # ── Feature 10: has_at_symbol ─────────────────────────────────────────
    has_at_symbol = int('@' in full_url)

    # ── Feature 11: has_double_slash ──────────────────────────────────────
    # double slash after protocol
    has_double_slash = int('//' in full_url.split('://', 1)[-1])

    # ── Feature 12: path_length ───────────────────────────────────────────
    path_length = len(path)

    # ── Feature 13: suspicious_keyword_count ─────────────────────────────
    url_lower = full_url.lower()
    suspicious_keyword_count = sum(kw in url_lower for kw in SUSPICIOUS_KEYWORDS)

    # ── Feature 14: tld_in_legitimate_list ───────────────────────────────
    tld_in_legitimate_list = int(tld in LEGITIMATE_TLDS)

    return {
        'url_length': url_length,
        'domain_length': domain_length,
        'shannon_entropy': shannon_entropy,
        'digit_ratio': digit_ratio,
        'hyphen_count': hyphen_count,
        'dot_count': dot_count,
        'subdomain_count': subdomain_count,
        'special_char_ratio': special_char_ratio,
        'has_ip_address': has_ip_address,
        'has_at_symbol': has_at_symbol,
        'has_double_slash': has_double_slash,
        'path_length': path_length,
        'suspicious_keyword_count': suspicious_keyword_count,
        'tld_in_legitimate_list': tld_in_legitimate_list,
    }


LEXICAL_FEATURE_NAMES = list(extract_lexical_features('example.com').keys())
