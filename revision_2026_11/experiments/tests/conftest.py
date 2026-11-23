import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for path in (str(EXPERIMENT_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
