# llm_lab/utils/config_loader.py
from __future__ import annotations

import json
from dataclasses import asdict,fields,is_dataclass
from pathlib import Path
from typing import Type,TypeVar

T = TypeVar("T")

def load_json_config(path: str | Path, cls : Type[T]) -> T:
    """
    Load a JSON file and map its keys into the given dataclass type.
    Extra keys can either be ignored or cause an error (your choice).
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    field_names = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in field_names}

    return cls(**filtered)