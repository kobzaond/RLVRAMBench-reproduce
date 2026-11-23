"""Publish complete phase records to a concurrently sampled marker file."""
from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
import time


def publish_phase(path, step, phase):
    if not isinstance(step, int) or step < 0 or not re.fullmatch(r"[A-Za-z0-9_]+", phase):
        raise ValueError("Invalid phase marker")
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix="."+path.name+".", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(f"{time.time_ns()},{step},{phase}\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
