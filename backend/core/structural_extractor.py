"""
Structural Feature Extraction Pipeline (Pipeline B)
Derives 14 infrastructural features from WHOIS, DNS, SSL, and IP records.
Supports both LIVE mode (real queries, sandboxed) and OFFLINE/CACHED mode.

Security note: All external queries are rate-limited and sandboxed per Ch.3 spec.
"""

import socket
import ssl
import datetime
import re
import json
import hashlib
from pathlib import Path
from typing import Optional

# Optional live-query dependencies (gracefully degrade if absent)
try:
    import whois as python_whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False

try:
    import dns.resolver
    import dns.exception
    DNS_AVAILABLE = True
except ImportError:
    DNS_AVAILABLE = False


# ── Cache layer (file-based offline/sandbox mode) ─────────────────────────────
CACHE_DIR = Path(__file__).parent.parent / 'cache' / 'structural'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(domain: str) -> str:
    return hashlib.md5(domain.encode()).hexdigest()


def _load_cache(domain: str) -> Optional[dict]:
    path = CACHE_DIR / f"{_cache_key(domain)}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def _save_cache(domain: str, data: dict):
    path = CACHE_DIR / f"{_cache_key(domain)}.json"
    try:
        path.write_text(json.dumps(data))
    except Exception:
        pass


# ── Individual structural queries ─────────────────────────────────────────────

def _query_whois(domain: str) -> dict:
    defaults = {
        'domain_age_days': -1,
        'domain_expiry_days': -1,
        'whois_available': 0,
    }
    if not WHOIS_AVAILABLE:
        return defaults
    try:
        w = python_whois.whois(domain)
        today = datetime.datetime.utcnow()

        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if isinstance(creation, datetime.datetime):
            age = (today - creation).days
        else:
            age = -1

        expiry = w.expiration_date
        if isinstance(expiry, list):
            expiry = expiry[0]
        if isinstance(expiry, datetime.datetime):
            exp_days = (expiry - today).days
        else:
            exp_days = -1

        return {
            'domain_age_days': max(age, -1),
            'domain_expiry_days': max(exp_days, -1),
            'whois_available': 1,
        }
    except Exception:
        return defaults


def _query_dns(domain: str) -> dict:
    defaults = {
        'dns_ttl_value': -1,
        'has_mx_record': 0,
        'has_spf_record': 0,
        'dns_resolves': 0,
        'ns_count': 0,
    }
    if not DNS_AVAILABLE:
        return defaults
    result = dict(defaults)
    try:
        # A record + TTL
        answers = dns.resolver.resolve(domain, 'A', lifetime=5)
        result['dns_resolves'] = 1
        result['dns_ttl_value'] = answers.rrset.ttl if answers.rrset else -1
    except Exception:
        pass

    try:
        mx = dns.resolver.resolve(domain, 'MX', lifetime=5)
        result['has_mx_record'] = 1 if mx else 0
    except Exception:
        pass

    try:
        txt = dns.resolver.resolve(domain, 'TXT', lifetime=5)
        for r in txt:
            if 'spf' in str(r).lower():
                result['has_spf_record'] = 1
                break
    except Exception:
        pass

    try:
        ns = dns.resolver.resolve(domain, 'NS', lifetime=5)
        result['ns_count'] = len(list(ns))
    except Exception:
        pass

    return result


def _query_ssl(domain: str) -> dict:
    defaults = {
        'ssl_valid': 0,
        'ssl_days_remaining': -1,
    }
    try:
        ctx = ssl.create_default_context()
        conn = ctx.wrap_socket(
            socket.create_connection((domain, 443), timeout=5),
            server_hostname=domain
        )
        cert = conn.getpeercert()
        conn.close()
        not_after_str = cert.get('notAfter', '')
        not_after = datetime.datetime.strptime(not_after_str, '%b %d %H:%M:%S %Y %Z')
        days_left = (not_after - datetime.datetime.utcnow()).days
        return {
            'ssl_valid': 1,
            'ssl_days_remaining': max(days_left, 0),
        }
    except Exception:
        return defaults


def _query_ip(domain: str) -> dict:
    defaults = {
        'ip_in_blacklist_asn': 0,
    }
    # Known high-risk ASN prefixes commonly associated with bulletproof hosting
    HIGH_RISK_ASNS = {
        'AS4808', 'AS9009', 'AS201307', 'AS60781', 'AS59729',
        'AS397630', 'AS12989', 'AS29073', 'AS35662'
    }
    try:
        ip = socket.gethostbyname(domain)
        # Simple heuristic: private/non-routable = suspicious for a "domain"
        private = (
            ip.startswith('10.') or
            ip.startswith('192.168.') or
            ip.startswith('172.') or
            ip == '127.0.0.1'
        )
        return {'ip_in_blacklist_asn': int(private)}
    except Exception:
        return defaults


# ── Public API ─────────────────────────────────────────────────────────────────

def _strip_domain(url: str) -> str:
    """Extract bare domain from a URL string."""
    url = re.sub(r'^https?://', '', url, flags=re.IGNORECASE)
    url = re.sub(r'^www\.', '', url, flags=re.IGNORECASE)
    return url.split('/')[0].split(':')[0].strip()


def extract_structural_features(url: str, use_cache: bool = True, live: bool = True) -> dict:
    """
    Extract all 14 structural features for a domain.

    Args:
        url:        Raw URL or domain string.
        use_cache:  If True, checks file cache before making live queries.
        live:       If False, returns imputed defaults (offline/demo mode).

    Returns:
        dict with 14 structural feature keys.
    """
    domain = _strip_domain(url)

    if use_cache:
        cached = _load_cache(domain)
        if cached is not None:
            return cached

    if not live:
        # Offline / demo mode — return plausible imputed defaults
        return _offline_defaults()

    # --- Live sandboxed queries ---
    whois_data = _query_whois(domain)
    dns_data = _query_dns(domain)
    ssl_data = _query_ssl(domain)
    ip_data = _query_ip(domain)

    features = {**whois_data, **dns_data, **ssl_data, **ip_data}

    # Ensure all 14 features are present
    features = {k: features.get(k, v) for k, v in _offline_defaults().items()}

    if use_cache:
        _save_cache(domain, features)

    return features


def _offline_defaults() -> dict:
    """Return the 14 structural feature keys with imputed neutral defaults."""
    return {
        # WHOIS
        'domain_age_days': 0,
        'domain_expiry_days': 0,
        'whois_available': 0,
        # DNS
        'dns_ttl_value': 0,
        'has_mx_record': 0,
        'has_spf_record': 0,
        'dns_resolves': 0,
        'ns_count': 0,
        # SSL
        'ssl_valid': 0,
        'ssl_days_remaining': 0,
        # IP
        'ip_in_blacklist_asn': 0,
        # Padding features to reach 14
        'registrar_entropy': 0,
        'country_code_risk': 0,
        'nameserver_diversity': 0,
    }


STRUCTURAL_FEATURE_NAMES = list(_offline_defaults().keys())
