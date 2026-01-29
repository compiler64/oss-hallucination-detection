import csv
import os
import time
from typing import List, Dict, Optional


def prompts_dir() -> str:
    return os.path.join('data', 'prompts')


def hallucination_dir() -> str:
    return os.path.join('data', 'hallucination_frequencies')


def ensure_dirs() -> None:
    os.makedirs(prompts_dir(), exist_ok=True)
    os.makedirs(hallucination_dir(), exist_ok=True)


def find_prompts_file(filename: str) -> str:
    ensure_dirs()
    path = os.path.join(prompts_dir(), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(row)
    return out


def load_prompts(filename: str) -> List[Dict[str, str]]:
    path = find_prompts_file(filename)
    return load_csv_rows(path)


def _hallucination_path_for_language(language: str) -> str:
    ensure_dirs()
    return os.path.join(hallucination_dir(), f"{language}.csv")


def load_hallucination_frequencies(language: str) -> List[Dict[str, str]]:
    path = _hallucination_path_for_language(language)
    if not os.path.exists(path):
        return []
    return load_csv_rows(path)


def _row_key(row: Dict[str, str]) -> str:
    # Unique key for prompt+settings to detect duplicates
    keys = [
        row.get('prompt', '').strip(),
        str(row.get('max_new_tokens', '')).strip(),
        str(row.get('temperature', '')).strip(),
        str(row.get('reasoning_level', '')).strip(),
        str(row.get('num_beams', '')).strip(),
        str(row.get('system_prompt', '')).strip(),
    ]
    return '||'.join(keys)


def save_hallucination_frequencies(language: str, rows: List[Dict[str, str]]) -> None:
    path = _hallucination_path_for_language(language)
    # Ensure index column is present and is unique integer
    # If rows already have 'index', preserve it; otherwise assign sequentially
    existing = load_hallucination_frequencies(language)
    max_index = 0
    for r in existing:
        try:
            idx = int(r.get('index', 0))
            if idx > max_index:
                max_index = idx
        except Exception:
            pass

    # Build a map of existing keys to preserve old rows
    existing_map = { _row_key(r): r for r in existing }

    out_map: Dict[str, Dict[str, str]] = dict(existing_map)

    # Add/replace with provided rows
    next_index = max_index + 1
    for r in rows:
        key = _row_key(r)
        if 'index' in r and r['index']:
            out_map[key] = r
        else:
            # assign new index
            r_copy = dict(r)
            r_copy['index'] = str(next_index)
            next_index += 1
            out_map[key] = r_copy

    # Write CSV with consistent header order
    header = [
        'index', 'prompt', 'max_new_tokens', 'temperature', 'reasoning_level', 'num_beams', 'system_prompt',
        'trials', 'hits', 'hallucination_rate', 'last_updated'
    ]
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for v in out_map.values():
            row_out = {k: v.get(k, '') for k in header}
            # ensure numeric fields are strings
            writer.writerow(row_out)


def find_cached_for_row(language: str, row: Dict[str, str]) -> Optional[Dict[str, str]]:
    cached = load_hallucination_frequencies(language)
    key = _row_key(row)
    for r in cached:
        if _row_key(r) == key:
            return r
    return None
