from __future__ import annotations

import sys
from pathlib import Path


CORE_ROOT = Path(__file__).parents[1]
COMMON_ROOT = CORE_ROOT.parent / "common"
for source_root in (CORE_ROOT / "src", COMMON_ROOT / "src"):
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)