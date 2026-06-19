"""
Flask REST API
Endpoints:
  POST /api/predict          — single URL prediction
  POST /api/predict/batch    — batch URL prediction
  POST /api/train            — trigger training pipeline
  GET  /api/results          — retrieve experiment results
  GET  /api/models/status    — check which models are trained
  GET  /api/health           — health check
  GET  /api/features/explain — feature definitions
"""

import os
import json
import threading
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

# Adjust import path when running as module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.predictor import predict, predict_batch, load_experiment_results, models_available
from core.dataset_loader import prepare_datasets_from_real_data
from core.trainer import run_all_conditions

app = Flask(__name__)
CORS(app)

TRAINING_STATUS = {'running': False, 'done': False, 'error': None, 'log': []}


# ── Health ─────────────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'phishing-detection-api'})


# ── Prediction ─────────────────────────────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
def predict_url():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing "url" field'}), 400

    url = str(data['url']).strip()
    condition = str(data.get('condition', 'C6'))
    live = bool(data.get('live_structural', False))

    status = models_available()
    if not status.get(condition, False):
        return jsonify({
            'error': f"Model {condition} not trained yet. Run /api/train first or use demo mode.",
            'models_available': status,
        }), 503

    try:
        result = predict(url, condition_id=condition, live_structural=live)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch_urls():
    data = request.get_json()
    if not data or 'urls' not in data:
        return jsonify({'error': 'Missing "urls" field (list)'}), 400

    urls = data['urls']
    if not isinstance(urls, list) or len(urls) == 0:
        return jsonify({'error': '"urls" must be a non-empty list'}), 400
    if len(urls) > 100:
        return jsonify({'error': 'Batch limit is 100 URLs'}), 400

    condition = str(data.get('condition', 'C6'))
    live = bool(data.get('live_structural', False))

    status = models_available()
    if not status.get(condition, False):
        return jsonify({'error': f"Model {condition} not trained. Run /api/train first."}), 503

    try:
        results = predict_batch(urls, condition_id=condition, live_structural=live)
        summary = {
            'total': len(results),
            'phishing': sum(1 for r in results if r['prediction'] == 'phishing'),
            'legitimate': sum(1 for r in results if r['prediction'] == 'legitimate'),
        }
        return jsonify({'results': results, 'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Training ───────────────────────────────────────────────────────────────────

def _run_training_job(fast_mode: bool, data_dir: str = None):
    global TRAINING_STATUS
    try:
        TRAINING_STATUS['running'] = True
        TRAINING_STATUS['log'] = ['Starting feature extraction...']

        df_lex, df_struct, df_combined, y = prepare_datasets_from_real_data(data_dir)
        TRAINING_STATUS['log'].append(f'Dataset ready: {len(y)} samples')

        TRAINING_STATUS['log'].append('Running 6 experimental conditions...')
        results = run_all_conditions(df_lex, df_struct, df_combined, y, fast_mode=fast_mode)

        TRAINING_STATUS['done'] = True
        TRAINING_STATUS['running'] = False
        TRAINING_STATUS['results'] = results
        TRAINING_STATUS['log'].append('Training complete!')
    except Exception as e:
        TRAINING_STATUS['error'] = str(e)
        TRAINING_STATUS['running'] = False
        TRAINING_STATUS['log'].append(f'Error: {e}')


@app.route('/api/train', methods=['POST'])
def train():
    global TRAINING_STATUS
    if TRAINING_STATUS['running']:
        return jsonify({'message': 'Training already in progress', 'status': TRAINING_STATUS}), 409

    data = request.get_json() or {}
    fast_mode = bool(data.get('fast_mode', True))
    data_dir = data.get('data_dir', None)

    TRAINING_STATUS = {'running': False, 'done': False, 'error': None, 'log': []}

    thread = threading.Thread(target=_run_training_job, args=(fast_mode, data_dir))
    thread.daemon = True
    thread.start()

    return jsonify({
        'message': 'Training started',
        'fast_mode': fast_mode,
        'tip': 'Poll /api/train/status for progress'
    })


@app.route('/api/train/status', methods=['GET'])
def train_status():
    return jsonify(TRAINING_STATUS)


# ── Results ────────────────────────────────────────────────────────────────────

@app.route('/api/results', methods=['GET'])
def get_results():
    results = load_experiment_results()
    if results is None:
        return jsonify({'error': 'No results found. Train models first.'}), 404
    return jsonify(results)


@app.route('/api/models/status', methods=['GET'])
def model_status():
    status = models_available()
    any_trained = any(status.values())
    return jsonify({
        'models': status,
        'any_trained': any_trained,
        'recommended': 'C6',
        'note': 'C6 = Combined XGBoost (best F1 per Chapter 4 results)'
    })


# ── Feature definitions ────────────────────────────────────────────────────────

FEATURE_DEFINITIONS = {
    'lexical': [
        {'name': 'url_length', 'description': 'Total character length of the full URL', 'type': 'numeric'},
        {'name': 'domain_length', 'description': 'Character length of the registrable domain', 'type': 'numeric'},
        {'name': 'shannon_entropy', 'description': 'Shannon entropy of the second-level domain string — higher values indicate DGA usage', 'type': 'numeric'},
        {'name': 'digit_ratio', 'description': 'Proportion of digit characters in the domain', 'type': 'ratio'},
        {'name': 'hyphen_count', 'description': 'Number of hyphens in the domain', 'type': 'count'},
        {'name': 'dot_count', 'description': 'Number of dot characters in the full URL', 'type': 'count'},
        {'name': 'subdomain_count', 'description': 'Number of subdomain labels present', 'type': 'count'},
        {'name': 'special_char_ratio', 'description': 'Ratio of special punctuation characters in the URL', 'type': 'ratio'},
        {'name': 'has_ip_address', 'description': 'Binary: 1 if the URL contains an IP address instead of a domain name', 'type': 'binary'},
        {'name': 'has_at_symbol', 'description': 'Binary: 1 if the URL contains an @ symbol (credential redirect)', 'type': 'binary'},
        {'name': 'has_double_slash', 'description': 'Binary: 1 if a double slash occurs after the protocol', 'type': 'binary'},
        {'name': 'path_length', 'description': 'Character length of the URL path component', 'type': 'numeric'},
        {'name': 'suspicious_keyword_count', 'description': 'Count of known phishing keywords in the URL', 'type': 'count'},
        {'name': 'tld_in_legitimate_list', 'description': 'Binary: 1 if the TLD is in the set of common legitimate TLDs', 'type': 'binary'},
    ],
    'structural': [
        {'name': 'domain_age_days', 'description': 'Age of domain registration in days — very low values indicate newly registered phishing domains', 'type': 'numeric'},
        {'name': 'domain_expiry_days', 'description': 'Days until domain registration expires', 'type': 'numeric'},
        {'name': 'whois_available', 'description': 'Binary: 1 if WHOIS record is publicly accessible', 'type': 'binary'},
        {'name': 'dns_ttl_value', 'description': 'DNS Time-To-Live in seconds — very low values indicate fast-flux evasion techniques', 'type': 'numeric'},
        {'name': 'has_mx_record', 'description': 'Binary: 1 if a valid MX record exists for the domain', 'type': 'binary'},
        {'name': 'has_spf_record', 'description': 'Binary: 1 if a Sender Policy Framework TXT record is present', 'type': 'binary'},
        {'name': 'dns_resolves', 'description': 'Binary: 1 if the domain successfully resolves to an IP address', 'type': 'binary'},
        {'name': 'ns_count', 'description': 'Count of nameserver records — low counts indicate minimal infrastructure', 'type': 'count'},
        {'name': 'ssl_valid', 'description': 'Binary: 1 if a valid SSL/TLS certificate is present', 'type': 'binary'},
        {'name': 'ssl_days_remaining', 'description': 'Days until the SSL certificate expires', 'type': 'numeric'},
        {'name': 'ip_in_blacklist_asn', 'description': 'Binary: 1 if the IP resolves to a known high-risk ASN', 'type': 'binary'},
        {'name': 'registrar_entropy', 'description': 'Shannon entropy of the registrar name — high values suggest automated bulk registration', 'type': 'numeric'},
        {'name': 'country_code_risk', 'description': 'Binary: 1 if the hosting country is in a high-risk jurisdiction', 'type': 'binary'},
        {'name': 'nameserver_diversity', 'description': 'Binary: 1 if nameservers are hosted across diverse providers (legitimate indicator)', 'type': 'binary'},
    ],
}


@app.route('/api/features/explain', methods=['GET'])
def feature_definitions():
    return jsonify(FEATURE_DEFINITIONS)


# ── Demo predict (no trained model required — uses lexical heuristics) ─────────

@app.route('/api/predict/demo', methods=['POST'])
def predict_demo():
    """
    Heuristic-only prediction for demo/UI purposes when models aren't trained.
    Not academically valid — for UI demonstration only.
    """
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing "url" field'}), 400

    url = str(data['url']).strip()

    from core.lexical_extractor import extract_lexical_features
    feats = extract_lexical_features(url)

    # Simple scoring heuristic
    score = 0.0
    score += min(feats['shannon_entropy'] / 4.5, 1.0) * 0.30
    score += feats['digit_ratio'] * 0.15
    score += min(feats['suspicious_keyword_count'] / 3, 1.0) * 0.20
    score += feats['has_ip_address'] * 0.15
    score += (1 - feats['tld_in_legitimate_list']) * 0.10
    score += min(feats['hyphen_count'] / 3, 1.0) * 0.05
    score += feats['has_at_symbol'] * 0.05
    score = round(min(score, 0.99), 4)

    return jsonify({
        'url': url,
        'prediction': 'phishing' if score >= 0.5 else 'legitimate',
        'phishing_probability': score,
        'confidence': round(abs(score - 0.5) * 2, 4),
        'condition': 'DEMO-HEURISTIC',
        'pipeline': 'lexical',
        'classifier': 'Heuristic (no trained model)',
        'features': feats,
        'top_feature_importances': [],
        'risk_level': 'HIGH' if score >= 0.75 else 'MEDIUM' if score >= 0.45 else 'LOW',
        'warning': 'Demo mode: rule-based heuristic only, not a trained ML model.',
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
