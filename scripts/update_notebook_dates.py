#!/usr/bin/env python3

import json
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
DATE_LINE_RE = re.compile(r'^<div align="right"><sub>.*?</sub></div>\n$')


def get_last_commit_date(path: Path) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "log",
            "-1",
            "--date=format:%Y-%m-%d",
            "--format=%ad",
            "--",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def update_notebook(path: Path) -> bool:
    with path.open(encoding="utf-8") as f:
        notebook = json.load(f)

    first_cell = notebook["cells"][0]
    if first_cell.get("cell_type") != "markdown":
        raise ValueError(f"First cell is not markdown: {path}")

    source = first_cell.get("source", [])
    date_line = f'<div align="right"><sub>最終更新: {get_last_commit_date(path)}</sub></div>\n'

    new_source = [line for line in source if not DATE_LINE_RE.match(line)]
    insert_at = 1 if new_source and new_source[0].startswith("#") else 0
    new_source.insert(insert_at, date_line)

    if new_source == source:
        return False

    first_cell["source"] = new_source
    with path.open("w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return True


def main() -> None:
    changed = []
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        if update_notebook(path):
            changed.append(path.name)

    if changed:
        print("Updated notebooks:")
        for name in changed:
            print(f"- {name}")
    else:
        print("No notebook changes needed.")


if __name__ == "__main__":
    main()
