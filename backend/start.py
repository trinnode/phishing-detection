"""
Startup script: launches gunicorn immediately, then triggers auto-training
in background if no models exist. This lets Railway healthchecks pass right away.
"""
import os
import sys
import time
import threading
from pathlib import Path

MODELS_DIR = Path(__file__).parent / 'models' / 'saved'
C6_PATH = MODELS_DIR / 'C6.pkl'

def needs_training():
    return not C6_PATH.exists()

def trigger_auto_train():
    """Wait for gunicorn to be ready, then POST /api/train."""
    time.sleep(3)
    import urllib.request
    port = os.environ.get('PORT', '5000')
    for attempt in range(30):
        try:
            req = urllib.request.Request(
                f'http://127.0.0.1:{port}/api/train',
                data=b'{"fast_mode": true}',
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            resp = urllib.request.urlopen(req, timeout=5)
            print(f"[startup] Auto-training triggered (status {resp.status})")
            return
        except Exception as e:
            if attempt == 0:
                print(f"[startup] Waiting for gunicorn to start... ({e})")
            time.sleep(2)
    print("[startup] Could not reach gunicorn to trigger training.")

if __name__ == '__main__':
    if needs_training():
        print("[startup] No trained models found. Will auto-train in background after gunicorn starts.")
        t = threading.Thread(target=trigger_auto_train, daemon=True)
        t.start()
    else:
        print("[startup] Models already trained.")

    # Launch gunicorn (this replaces the process)
    port = os.environ.get('PORT', '5000')
    cmd = [
        'gunicorn',
        '--bind', f'0.0.0.0:{port}',
        '--workers', '2',
        '--timeout', '300',
        'api.app:app',
    ]
    os.execvp('gunicorn', cmd)
