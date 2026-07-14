#!/usr/bin/env python
"""Launch the optional local live-PyBullet showcase service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the local SFN live-PyBullet showcase service.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    args = parser.parse_args()
    try:
        import uvicorn
        from sfn.showcase.service import create_app
    except (ModuleNotFoundError, RuntimeError) as error:
        print("Optional showcase dependencies are required: pip install 'fastapi>=0.110' 'uvicorn[standard]>=0.27' 'pydantic>=2'", file=sys.stderr)
        print(f"Details: {error}", file=sys.stderr)
        return 2
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
