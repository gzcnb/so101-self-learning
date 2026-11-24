"""Check public source syntax, fixtures, motion defaults, and obvious secret leaks."""

import ast
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    python_files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or set(path.relative_to(ROOT).parts) & {".git", ".venv", "__pycache__", "runtime", "profiles"}:
            continue
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            python_files.append(path)
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        if path.suffix in {".py", ".json", ".toml", ".md", ".yml", ".cmd", ".ps1"}:
            text = path.read_text(encoding="utf-8")
            patterns = [r"gh[pousr]_[A-Za-z0-9]{20,}", r"github_pat_[A-Za-z0-9_]{20,}",
                        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
                        r"C:[/\\]Users[/\\][^/\\]+[/\\]", r"/home/[^\s'\"/]+"]
            if path.resolve() != Path(__file__).resolve() and any(re.search(pattern, text) for pattern in patterns):
                errors.append(f"Sensitive-looking content in {path.relative_to(ROOT)}")
    sys.path.insert(0, str(ROOT / "workbench"))
    from settings import require_hardware_enabled
    try:
        require_hardware_enabled()
    except RuntimeError:
        pass
    else:
        errors.append("Motion must be disabled in a fresh release")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validated {len(python_files)} Python files, JSON fixtures, and disabled-motion defaults.")


if __name__ == "__main__":
    main()
