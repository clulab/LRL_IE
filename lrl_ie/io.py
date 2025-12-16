import json
from pathlib import Path
from typing import Iterable, List, Any

import yaml


def load_jsonl(path: str) -> List[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            rows.append(json.loads(s))
    return rows


def save_jsonl(path: str, rows: Iterable[Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_yaml(path: str):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

