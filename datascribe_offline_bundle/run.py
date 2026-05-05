#!/usr/bin/env python3
"""
Offline bundle launcher.

This lets users run from inside `datascribe_offline_bundle/` with:
    python run.py --target Survived --train
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    # Make sure parent repo root is importable
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Import and execute the existing offline runner
    from datascribe_offline_bundle.run_offline import main as offline_main

    return int(offline_main())


if __name__ == "__main__":
    raise SystemExit(main())
