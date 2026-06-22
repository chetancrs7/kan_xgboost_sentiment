from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from src.data.split_dataset import main  # noqa: E402


if __name__ == "__main__":
    main(default_selected_split="test")
