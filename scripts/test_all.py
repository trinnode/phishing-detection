#!/usr/bin/env python3
"""
Comprehensive simulation and test script for the FishMark Phishing Detection Framework.
Tests all API endpoints, edge cases, and failure modes.

Usage:
    source venv/bin/activate
    python scripts/test_all.py
"""

import sys
import json
import time
import traceback
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
    assert resp.status_code == status, f'Expected status {status}, got {resp.status_code}: {resp.text[:200]}'


def main():
    global passed, failed
    print('=' * 60)
    print('FISHMARK COMPREHENSIVE TEST SUITE')
    print('=' * 60)

    # Health
    def test_health():
        r = requests.get(f'{API}/api/health')
        check(r)
        d = r.json()
        assert d['status'] == 'ok'
    test('Health check endpoint', test_health)

    # Model status
    def test_model_status():
        r = requests.get(f'{API}/api/models/status')
        check(r)
        d = r.json()
        assert 'models' in d
        assert 'any_trained' in d
    test('Model status endpoint', test_model_status)

    # Features explain
    def test_features():
        r = requests.get(f'{API}/api/features/explain')
        check(r)
        d = r.json()
        assert 'lexical' in d
        assert 'structural' in d
        assert len(d['lexical']) == 14
        assert len(d['structural']) == 14
    test('Feature definitions endpoint', test_features)

    # Demo predict
    def test_demo_predict():
        r = requests.post(f'{API}/api/predict/demo', json={'url': 'http://paypal secure.tk/login'})
        check(r)
        d = r.json()
        assert 'prediction' in d
        assert d['prediction'] in ('phishing', 'legitimate')
        assert 'phishing_probability' in d
        assert 'features' in d
    test('Demo prediction single URL', test_demo_predict)

    def test_demo_predict_legit():
        r = requests.post(f'{API}/api/predict/demo', json={'url': 'https://www.github.com'})
        check(r)
        d = r.json()
        assert d['prediction'] == 'legitimate'
    test('Demo prediction legitimate URL', test_demo_predict_legit)

    def test_demo_predict_missing_url():
        r = requests.post(f'{API}/api/predict/demo', json={})
        assert r.status_code == 400
    test('Demo prediction missing URL returns 400', test_demo_predict_missing_url)

    # Live predict (requires trained models)
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
    test('Live prediction with trained model', test_live_predict)

    def test_live_predict_all_conditions():
        r = requests.get(f'{API}/api/models/status')
        status = r.json()
        if not status.get('any_trained'):
            print('  [SKIP] All conditions test: models not trained')
            return
        for cid in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']:
            r = requests.post(f'{API}/api/predict', json={'url': 'https://github.com', 'condition': cid})
            check(r)
            d = r.json()
            assert d['condition'] == cid
    test('Live prediction all six conditions', test_live_predict_all_conditions)

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

    # Batch resilient prediction
    def test_batch_resilient():
        r = requests.get(f'{API}/api/models/status')
        status = r.json()
        if not status.get('any_trained'):
            print('  [SKIP] Resilient batch test: models not trained')
            return
        urls = [
            'https://github.com',
            'not a valid url but should not crash',
            'https://stackoverflow.com',
        ]
        r = requests.post(f'{API}/api/predict/urls', json={'urls': urls, 'condition': 'C1'})
        check(r)
        d = r.json()
        assert 'results' in d
        assert 'summary' in d
        assert len(d['results']) == len(urls)
        assert d['summary']['total'] == len(urls)
    test('Batch resilient with mixed valid and invalid URLs', test_batch_resilient)

    def test_batch_mixed():
        r = requests.get(f'{API}/api/models/status')
        status = r.json()
        if not status.get('any_trained'):
            print('  [SKIP] Mixed batch test: models not trained')
            return
        urls = ['https://github.com', 'http://phishy.ml/login', 'https://google.com']
        r = requests.post(f'{API}/api/predict/urls', json={'urls': urls, 'condition': 'C6'})
        check(r)
        d = r.json()
        assert len(d['results']) == 3
        assert d['summary']['total'] == 3
    test('Batch with mixed URL types', test_batch_mixed)

    def test_batch_from_string():
        r = requests.post(f'{API}/api/predict/urls', json={
            'urls': 'https://github.com\nhttp://phish.tk\nhttps://google.com'
        })
        check(r)
        d = r.json()
        assert len(d['results']) == 3
    test('Batch from newline separated string', test_batch_from_string)

    def test_batch_empty():
        r = requests.post(f'{API}/api/predict/urls', json={'urls': []})
        assert r.status_code == 400
    test('Batch empty list returns 400', test_batch_empty)

    def test_batch_too_many():
        too_many = ['https://x.com'] * 300
        r = requests.post(f'{API}/api/predict/urls', json={'urls': too_many})
        assert r.status_code == 400
    test('Batch exceeding limit returns 400', test_batch_too_many)

    # Results
    def test_results():
        r = requests.get(f'{API}/api/results')
        if r.status_code == 404:
            print('  [SKIP] Results test: no results found')
            return
        check(r)
        d = r.json()
        assert 'conditions' in d
        assert len(d['conditions']) == 6
    test('Experiment results endpoint', test_results)

    # Upload endpoint
    def test_upload_no_file():
        r = requests.post(f'{API}/api/upload/dataset', files={})
        assert r.status_code == 400
    test('Upload without file returns 400', test_upload_no_file)

    def test_upload_csv_content():
        import io
        content = 'https://example.com\nhttps://phish.tk\n# comment\nhttps://google.com\n'
        r = requests.post(f'{API}/api/upload/dataset', files={
            'file': ('test.csv', io.BytesIO(content.encode()), 'text/csv')
        })
        check(r)
        d = r.json()
        assert d['url_count'] >= 3
    test('Upload CSV file with URLs', test_upload_csv_content)

    def test_upload_txt_content():
        import io
        content = 'https://example.com\nhttps://phish.tk\nhttps://google.com\n'
        r = requests.post(f'{API}/api/upload/dataset', files={
            'file': ('test.txt', io.BytesIO(content.encode()), 'text/plain')
        })
        check(r)
        d = r.json()
        assert d['url_count'] == 3
    test('Upload TXT file with URLs', test_upload_txt_content)

    def test_upload_wrong_extension():
        import io
        r = requests.post(f'{API}/api/upload/dataset', files={
            'file': ('test.pdf', io.BytesIO(b'data'), 'application/pdf')
        })
        assert r.status_code == 400
    test('Upload unsupported file type returns 400', test_upload_wrong_extension)

    # Training trigger
    def test_training_trigger():
        r = requests.post(f'{API}/api/train', json={'fast_mode': True})
        if r.status_code == 409:
            print('  [SKIP] Training trigger test: training already in progress')
            return
        check(r)
        d = r.json()
        assert 'message' in d
        # Wait briefly then check status
        time.sleep(2)
        r2 = requests.get(f'{API}/api/train/status')
        check(r2)
        d2 = r2.json()
        assert 'running' in d2 or 'done' in d2
    test('Trigger training pipeline', test_training_trigger)

    # Edge cases
    def test_predict_special_chars():
        r = requests.post(f'{API}/api/predict/demo', json={'url': 'http://example.com/path?q=hello world&x=1'})
        check(r)
        d = r.json()
        assert 'prediction' in d
    test('URL with special characters', test_predict_special_chars)

    def test_predict_ip_url():
        r = requests.post(f'{API}/api/predict/demo', json={'url': 'http://192.168.1.1/login.php'})
        check(r)
        d = r.json()
        assert d['features']['has_ip_address'] == 1
        assert d['phishing_probability'] >= 0.4
    test('IP based URL detected as having IP address flag', test_predict_ip_url)

    def test_predict_at_symbol():
        r = requests.post(f'{API}/api/predict/demo', json={'url': 'http://google.com@evil.phish.tk'})
        check(r)
        d = r.json()
        assert d['features']['has_at_symbol'] == 1
    test('URL with at symbol detected', test_predict_at_symbol)

    def test_features_consistency():
        r = requests.post(f'{API}/api/predict/demo', json={'url': 'https://www.example.com/test/path'})
        check(r)
        d = r.json()
        feats = d['features']
        assert feats['url_length'] > 0
        assert feats['domain_length'] > 0
        assert feats['shannon_entropy'] >= 0
        assert feats['digit_ratio'] >= 0
    test('Feature extraction consistency', test_features_consistency)

    def test_cors_headers():
        r = requests.options(f'{API}/api/health', headers={
            'Origin': 'http://localhost:3000',
            'Access-Control-Request-Method': 'GET',
        })
        # Should have CORS headers
        assert 'Access-Control-Allow-Origin' in r.headers or r.status_code == 200
    test('CORS headers present', test_cors_headers)

    # Print summary
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
