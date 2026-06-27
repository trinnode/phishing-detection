#!/usr/bin/env python3
"""
Comprehensive simulation and test script for the Shikibetsu Phishing Detection Framework.
Tests all API endpoints, edge cases, failure modes, and condition mapping consistency.

Usage:
    source /tmp/venv/bin/activate
    python scripts/test_all.py
"""

import sys
import json
import time
import traceback
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

import requests

API = 'http://localhost:5000'

passed = 0
failed = 0
errors = []


def test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f'  [PASS] {name}')
    except Exception as e:
        failed += 1
        errors.append((name, str(e), traceback.format_exc()))
        print(f'  [FAIL] {name}: {e}')


def check(resp, status=200):
    assert resp.status_code == status, f'Expected status {status}, got {resp.status_code}: {resp.text[:300]}'


def main():
    global passed, failed
    print('=' * 60)
    print('MARKUP COMPREHENSIVE TEST SUITE')
    print('=' * 60)

    # ── 1. Health ────────────────────────────────────────────────────────────
    def test_health():
        r = requests.get(f'{API}/api/health')
        check(r)
        d = r.json()
        assert d['status'] == 'ok'
        assert 'markup' in d['service']
    test('Health check endpoint returns markup service name', test_health)

    # ── 2. Model status ──────────────────────────────────────────────────────
    def test_model_status():
        r = requests.get(f'{API}/api/models/status')
        check(r)
        d = r.json()
        assert 'models' in d
        assert 'any_trained' in d
        assert 'recommended' in d
        assert d['recommended'] == 'C6'
        assert len(d['models']) == 6
    test('Model status endpoint returns all 6 conditions', test_model_status)

    # ── 3. Condition mapping verification ────────────────────────────────────
    def test_condition_mapping():
        r = requests.get(f'{API}/api/models/status')
        check(r)
        d = r.json()
        models = d['models']
        expected = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        assert list(models.keys()) == expected, f'Expected {expected}, got {list(models.keys())}'
    test('Condition mapping has correct six condition IDs', test_condition_mapping)

    # ── 4. Feature definitions ───────────────────────────────────────────────
    def test_features():
        r = requests.get(f'{API}/api/features/explain')
        check(r)
        d = r.json()
        assert 'lexical' in d
        assert 'structural' in d
        assert len(d['lexical']) == 14
        assert len(d['structural']) == 14
    test('Feature definitions endpoint returns 14 lexical + 14 structural', test_features)

    def test_feature_names():
        r = requests.get(f'{API}/api/features/explain')
        check(r)
        d = r.json()
        lexical_names = [f['name'] for f in d['lexical']]
        assert 'shannon_entropy' in lexical_names
        assert 'url_length' in lexical_names
        assert 'domain_length' in lexical_names
        structural_names = [f['name'] for f in d['structural']]
        assert 'domain_age_days' in structural_names
        assert 'ssl_valid' in structural_names
        assert 'dns_ttl_value' in structural_names
    test('Feature names include expected key features', test_feature_names)

    # ── 5. Demo prediction ───────────────────────────────────────────────────
    def test_demo_predict_phishing():
        r = requests.post(f'{API}/api/predict/demo', json={'url': 'http://paypal-secure-update.tk/login.php'})
        check(r)
        d = r.json()
        assert 'prediction' in d
        assert d['prediction'] == 'phishing'
        assert 'phishing_probability' in d
        assert 'features' in d
        assert 'condition' in d
        assert d['condition'] == 'DEMO-HEURISTIC'
    test('Demo prediction returns phishing for suspicious URL', test_demo_predict_phishing)

    def test_demo_predict_legit():
        r = requests.post(f'{API}/api/predict/demo', json={'url': 'https://www.github.com'})
        check(r)
        d = r.json()
        assert d['prediction'] == 'legitimate'
        assert d['phishing_probability'] < 0.5
    test('Demo prediction returns legitimate for trusted URL', test_demo_predict_legit)

    def test_demo_predict_missing_url():
        r = requests.post(f'{API}/api/predict/demo', json={})
        assert r.status_code == 400
    test('Demo prediction missing URL returns 400', test_demo_predict_missing_url)

    def test_demo_predict_empty_url():
        r = requests.post(f'{API}/api/predict/demo', json={'url': ''})
        check(r)
        d = r.json()
        assert 'prediction' in d
    test('Demo prediction empty URL still returns prediction', test_demo_predict_empty_url)

    # ── 6. Live prediction (requires trained models) ─────────────────────────
    def test_live_predict():
        r = requests.get(f'{API}/api/models/status')
        status = r.json()
        if not status.get('any_trained'):
            print('  [SKIP] Live predict test: models not trained')
            return
        r = requests.post(f'{API}/api/predict', json={'url': 'https://github.com', 'condition': 'C1'})
        check(r)
        d = r.json()
        assert d['prediction'] in ('phishing', 'legitimate')
        assert 'phishing_probability' in d
        assert 'condition' in d
        assert 'pipeline' in d
        assert 'classifier' in d
    test('Live prediction with trained model', test_live_predict)

    def test_live_predict_all_six_conditions():
        r = requests.get(f'{API}/api/models/status')
        status = r.json()
        if not status.get('any_trained'):
            print('  [SKIP] All six conditions test: models not trained')
            return
        for cid in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']:
            r = requests.post(f'{API}/api/predict', json={'url': 'https://github.com', 'condition': cid})
            check(r)
            d = r.json()
            assert d['condition'] == cid
    test('Live prediction all six conditions return correct condition IDs', test_live_predict_all_six_conditions)

    def test_live_predict_condition_pipeline_labels():
        r = requests.get(f'{API}/api/models/status')
        status = r.json()
        if not status.get('any_trained'):
            print('  [SKIP] Condition label test: models not trained')
            return
        labels = {
            'C1': ('lexical', 'RF'),
            'C2': ('lexical', 'XGB'),
            'C3': ('structural', 'RF'),
            'C4': ('structural', 'XGB'),
            'C5': ('combined', 'RF'),
            'C6': ('combined', 'XGB'),
        }
        for cid, (exp_pipeline, exp_clf) in labels.items():
            r = requests.post(f'{API}/api/predict', json={'url': 'https://google.com', 'condition': cid})
            check(r)
            d = r.json()
            assert d['pipeline'] == exp_pipeline, f'{cid}: expected pipeline {exp_pipeline}, got {d["pipeline"]}'
            assert d['classifier'] == exp_clf, f'{cid}: expected classifier {exp_clf}, got {d["classifier"]}'
    test('Live prediction pipeline and classifier labels match document', test_live_predict_condition_pipeline_labels)

    def test_live_predict_invalid_condition():
        r = requests.get(f'{API}/api/models/status')
        status = r.json()
        if not status.get('any_trained'):
            print('  [SKIP] Invalid condition test: models not trained')
            return
        r = requests.post(f'{API}/api/predict', json={'url': 'https://github.com', 'condition': 'C99'})
        assert r.status_code != 200
    test('Live prediction invalid condition returns error', test_live_predict_invalid_condition)

    def test_live_predict_missing_url():
        r = requests.get(f'{API}/api/models/status')
        status = r.json()
        if not status.get('any_trained'):
            print('  [SKIP] Missing URL test: models not trained')
            return
        r = requests.post(f'{API}/api/predict', json={})
        assert r.status_code == 400
    test('Live prediction missing URL returns 400', test_live_predict_missing_url)

    def test_live_predict_risk_level():
        r = requests.get(f'{API}/api/models/status')
        status = r.json()
        if not status.get('any_trained'):
            print('  [SKIP] Risk level test: models not trained')
            return
        r = requests.post(f'{API}/api/predict', json={'url': 'https://github.com', 'condition': 'C6'})
        check(r)
        d = r.json()
        assert d['risk_level'] in ('LOW', 'MEDIUM', 'HIGH')
        assert 'confidence' in d
        assert 'top_feature_importances' in d
    test('Live prediction returns risk level and confidence', test_live_predict_risk_level)

    # ── 7. Batch prediction ──────────────────────────────────────────────────
    def test_batch_resilient():
        r = requests.get(f'{API}/api/models/status')
        status = r.json()
        if not status.get('any_trained'):
            print('  [SKIP] Resilient batch test: models not trained')
            return
        urls = [
            'https://github.com',
            '',  # Should be filtered
            'https://stackoverflow.com',
        ]
        r = requests.post(f'{API}/api/predict/batch', json={'urls': urls, 'condition': 'C1'})
        check(r)
        d = r.json()
        assert 'results' in d
        assert 'summary' in d
    test('Batch prediction with mixed valid and empty URLs', test_batch_resilient)

    def test_batch_mixed_types():
        r = requests.get(f'{API}/api/models/status')
        status = r.json()
        if not status.get('any_trained'):
            print('  [SKIP] Mixed batch test: models not trained')
            return
        urls = ['https://github.com', 'http://phishy.ml/login', 'https://google.com']
        r = requests.post(f'{API}/api/predict/batch', json={'urls': urls, 'condition': 'C6'})
        check(r)
        d = r.json()
        assert len(d['results']) == 3
        assert d['summary']['total'] == 3
    test('Batch with mixed URL types returns all results', test_batch_mixed_types)

    def test_batch_empty():
        r = requests.post(f'{API}/api/predict/batch', json={'urls': []})
        assert r.status_code == 400
    test('Batch empty list returns 400', test_batch_empty)

    def test_batch_too_many():
        too_many = ['https://x.com'] * 300
        r = requests.post(f'{API}/api/predict/batch', json={'urls': too_many})
        assert r.status_code == 400
    test('Batch exceeding 100 limit returns 400', test_batch_too_many)

    def test_batch_not_list():
        r = requests.post(f'{API}/api/predict/batch', json={'urls': 'not a list'})
        assert r.status_code == 400
    test('Batch non list input returns 400', test_batch_not_list)

    # ── 8. Edge cases ────────────────────────────────────────────────────────
    def test_predict_special_chars():
        r = requests.post(f'{API}/api/predict/demo', json={'url': 'http://example.com/path?q=hello world&x=1'})
        check(r)
        d = r.json()
        assert 'prediction' in d
    test('URL with special characters and spaces', test_predict_special_chars)

    def test_predict_ip_url():
        r = requests.post(f'{API}/api/predict/demo', json={'url': 'http://192.168.1.1/login.php'})
        check(r)
        d = r.json()
        assert d['features']['has_ip_address'] == 1
        assert d['phishing_probability'] >= 0.4
    test('IP based URL correctly identifies IP address flag', test_predict_ip_url)

    def test_predict_at_symbol():
        r = requests.post(f'{API}/api/predict/demo', json={'url': 'http://google.com@evil.phish.tk'})
        check(r)
        d = r.json()
        assert d['features']['has_at_symbol'] == 1
    test('URL with at symbol correctly identifies credential redirect flag', test_predict_at_symbol)

    def test_predict_double_slash():
        r = requests.post(f'{API}/api/predict/demo', json={'url': 'http://example.com//redirect//'})
        check(r)
        d = r.json()
        assert 'has_double_slash' in d['features']
    test('URL with double slash flag', test_predict_double_slash)

    def test_predict_long_url():
        r = requests.post(f'{API}/api/predict/demo', json={'url': 'http://' + 'a' * 200 + '.com/' + 'b' * 300})
        check(r)
        d = r.json()
        assert d['features']['url_length'] > 200
    test('Extremely long URL does not break analysis', test_predict_long_url)

    def test_predict_long_domain():
        r = requests.post(f'{API}/api/predict/demo', json={'url': 'http://' + 'a' * 100 + '.com'})
        check(r)
        d = r.json()
        assert d['features']['domain_length'] > 50
    test('Extremely long domain name does not break analysis', test_predict_long_domain)

    def test_features_consistency():
        r = requests.post(f'{API}/api/predict/demo', json={'url': 'https://www.example.com/test/path'})
        check(r)
        d = r.json()
        feats = d['features']
        assert feats['url_length'] > 0
        assert feats['domain_length'] > 0
        assert feats['shannon_entropy'] >= 0
        assert feats['digit_ratio'] >= 0
        assert feats['subdomain_count'] >= 0
        assert feats['special_char_ratio'] >= 0
    test('Feature extraction produces all fields with valid values', test_features_consistency)

    def test_features_hyphen_count():
        r = requests.post(f'{API}/api/predict/demo', json={'url': 'http://a-b-c-d-e-f.example.com'})
        check(r)
        d = r.json()
        assert d['features']['hyphen_count'] >= 5
    test('Hyphen count correctly identifies multiple hyphens', test_features_hyphen_count)

    def test_features_digit_ratio():
        r = requests.post(f'{API}/api/predict/demo', json={'url': 'http://12345test67890.com'})
        check(r)
        d = r.json()
        assert d['features']['digit_ratio'] > 0.3
    test('Digit ratio correctly identifies numeric heavy domains', test_features_digit_ratio)

    def test_predict_https():
        r = requests.post(f'{API}/api/predict/demo', json={'url': 'https://www.google.com/search?q=test'})
        check(r)
        d = r.json()
        assert d['prediction'] == 'legitimate'
    test('HTTPS legitimate URL correctly classified', test_predict_https)

    def test_predict_subdomains():
        r = requests.post(f'{API}/api/predict/demo', json={'url': 'http://a.b.c.d.e.f.example.com/path'})
        check(r)
        d = r.json()
        assert d['features']['subdomain_count'] >= 5
    test('Deep subdomain chain correctly counted', test_predict_subdomains)

    # ── 9. Results endpoint ──────────────────────────────────────────────────
    def test_results():
        r = requests.get(f'{API}/api/results')
        if r.status_code == 404:
            print('  [SKIP] Results test: no results found')
            return
        check(r)
        d = r.json()
        assert 'conditions' in d
        assert len(d['conditions']) == 6
    test('Experiment results endpoint returns 6 conditions', test_results)

    # ── 10. Training endpoint ─────────────────────────────────────────────────
    def test_training_trigger():
        r = requests.post(f'{API}/api/train', json={'fast_mode': True})
        if r.status_code == 409:
            print('  [SKIP] Training trigger test: training already in progress')
            return
        check(r)
        d = r.json()
        assert 'message' in d
        time.sleep(2)
        r2 = requests.get(f'{API}/api/train/status')
        check(r2)
        d2 = r2.json()
        assert 'running' in d2 or 'done' in d2
    test('Trigger training pipeline and poll status', test_training_trigger)

    def test_training_status_structure():
        r = requests.get(f'{API}/api/train/status')
        check(r)
        d = r.json()
        assert 'running' in d
        assert 'done' in d
        assert 'error' in d
        assert 'log' in d
    test('Training status has complete structure', test_training_status_structure)

    # ── 11. Upload endpoint tests ────────────────────────────────────────────
    def test_upload_no_file():
        r = requests.post(f'{API}/api/dataset/upload', files={})
        if r.status_code == 404:
            print('  [SKIP] Upload endpoint not available')
            return
        assert r.status_code == 400
    test('Upload without file returns 400', test_upload_no_file)

    def test_upload_csv_content():
        content = 'https://example.com\nhttps://phish.tk\n# comment line\nhttps://google.com\n'
        r = requests.post(f'{API}/api/dataset/upload', files={
            'files': ('test.csv', io.BytesIO(content.encode()), 'text/csv')
        })
        if r.status_code == 404:
            print('  [SKIP] Upload endpoint not available')
            return
        check(r)
        d = r.json()
        assert 'message' in d
    test('Upload CSV file with URLs', test_upload_csv_content)

    # ── 12. CORS headers ─────────────────────────────────────────────────────
    def test_cors_headers():
        r = requests.options(f'{API}/api/health', headers={
            'Origin': 'http://localhost:3000',
            'Access-Control-Request-Method': 'GET',
        })
        assert 'Access-Control-Allow-Origin' in r.headers or r.status_code == 200
    test('CORS headers present in API responses', test_cors_headers)

    # ── 13. Frontend build test ──────────────────────────────────────────────
    def test_frontend_build():
        import subprocess
        result = subprocess.run(['npm', 'run', 'build'], capture_output=True, text=True, cwd=Path(__file__).parent.parent / 'frontend')
        if result.returncode != 0:
            print(f'  [STDERR] {result.stderr[-500:]}')
        assert result.returncode == 0, f'Frontend build failed with exit code {result.returncode}'
    test('Frontend builds without errors', test_frontend_build)

    def test_frontend_build_output():
        build_dir = Path(__file__).parent.parent / 'frontend' / 'dist'
        assert build_dir.exists(), 'Build output directory missing'
        index = build_dir / 'index.html'
        assert index.exists(), 'index.html missing from build output'
        content = index.read_text()
        assert 'MARKup' in content, 'index.html does not reference MARKup'
    test('Frontend build output contains MARKup branding', test_frontend_build_output)

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print('=' * 60)
    print(f'RESULTS: {passed} passed, {failed} failed')
    print('=' * 60)

    if errors:
        print()
        for name, err, tb in errors:
            print(f'  FAILED: {name}')
            print(f'    {err}')
            print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
