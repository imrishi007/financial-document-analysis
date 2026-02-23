from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_rows_to_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return

    csv_path = Path(path)
    ensure_directory(csv_path.parent)
    frame = pd.DataFrame(rows)

    if csv_path.exists() and csv_path.stat().st_size > 0:
        frame.to_csv(csv_path, mode="a", index=False, header=False)
    else:
        frame.to_csv(csv_path, mode="w", index=False, header=True)
