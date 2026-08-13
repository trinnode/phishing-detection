"""
Path resolution for persistent runtime data.
Single source of truth so models, results, uploads, and cache can all live
under one mountable directory (required for Fly.io, which allows only one
volume per machine). Override the base with the MARKUP_DATA_DIR env var.
"""
import os
from pathlib import Path


def data_root() -> Path:
    """Base directory for persistent runtime data (models, results, uploads, cache)."""
    return Path(os.environ.get('MARKUP_DATA_DIR', str(Path(__file__).parent.parent)))