"""
Startup script: launches gunicorn as a child process, then triggers
auto-training in background if no models exist.
Using subprocess (not execvp) so the parent stays alive to manage training.
"""
import os
import sys
import time
import signal
import threading
import subprocess
from pathlib import Path

MODELS_DIR = Path(__file__).parent / 'models' / 'saved'
C6_PATH = MODELS_DIR / 'C6.pkl'

def needs_training():
    return not C6_PATH.exists()

def trigger_auto_train(port):
    """Wait for gunicorn to be ready, then POST /api/train."""
    import urllib.request
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
                print(f"[startup] Waiting for gunicorn... ({e})")
            time.sleep(2)
    print("[startup] Could not reach gunicorn to trigger training.")

if __name__ == '__main__':
    port = os.environ.get('PORT', '5000')

    # Start gunicorn as a subprocess (not exec, so parent stays alive)
    # Worker count is configurable to fit constrained machines (Fly/Render free tiers)
    # Fewer workers free RAM for the background auto-training.
    cmd = [
        'gunicorn',
        '--bind', f'0.0.0.0:{port}',
        '--workers', os.environ.get('GUNICORN_WORKERS', '1'),
        '--timeout', '300',
        'api.app:app',
    ]
    proc = subprocess.Popen(cmd)

    if needs_training():
        print("[startup] No trained models found. Auto-training in background after gunicorn starts.")
        t = threading.Thread(target=trigger_auto_train, args=(port,), daemon=True)
        t.start()
    else:
        print("[startup] Models already trained.")

    # Forward signals to gunicorn and wait
    def sig_handler(signum, frame):
        proc.send_signal(signum)
    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
