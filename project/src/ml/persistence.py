from __future__ import annotations
from pathlib import Path
import json
import joblib
from typing import Any, Dict

# Сохраняет модель и метаданные
def save_model(obj: Any, meta: Dict[str, Any], path: Path, meta_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

# Загружает модель
def load_model(path: Path):
    return joblib.load(path)
