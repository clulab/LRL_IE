import json
from typing import List, Optional


def extract_first_json(s: str) -> Optional[str]:
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(s[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def normalize_json_obj(obj: dict, schema_keys: List[str]) -> dict:
    fixed = {}
    for k in schema_keys:
        vals = obj.get(k, [])
        if not isinstance(vals, list):
            vals = []
        norm = []
        seen = set()
        for v in vals:
            if isinstance(v, str):
                vv = " ".join(v.split()).strip()
                if vv and vv not in seen:
                    seen.add(vv)
                    norm.append(vv)
        fixed[k] = norm
    return fixed


def decode_to_json(text: str, schema_keys: List[str]) -> dict:
    js = extract_first_json(text)
    if not js:
        return {k: [] for k in schema_keys}
    try:
        obj = json.loads(js)
    except Exception:
        js2 = (js.replace("None", "null")
               .replace("True", "true")
               .replace("False", "false"))
        try:
            obj = json.loads(js2)
        except Exception:
            return {k: [] for k in schema_keys}
    return normalize_json_obj(obj, schema_keys)

