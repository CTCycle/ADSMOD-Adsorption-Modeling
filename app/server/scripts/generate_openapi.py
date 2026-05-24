from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / 'app' / 'scripts' / 'generate_openapi.py'
    runpy.run_path(str(script_path), run_name='__main__')


if __name__ == '__main__':
    main()
