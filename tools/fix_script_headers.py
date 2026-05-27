"""Исправить порядок imports: from __future__ до REPO = Path(...)."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def fix_file(path: Path) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    repo_start = future_idx = pathlib_idx = None
    for i, line in enumerate(lines):
        if line.startswith("REPO = Path(__file__)"):
            repo_start = i
        if "from __future__ import annotations" in line and future_idx is None:
            future_idx = i
        if line.strip() == "from pathlib import Path" and future_idx is not None and pathlib_idx is None:
            if i > future_idx:
                pathlib_idx = i

    if repo_start is None or future_idx is None or repo_start >= future_idx:
        return False

    repo_end = future_idx
    block = lines[repo_start:repo_end]
    new_lines = lines[:repo_start] + lines[repo_end:]

    # re-find pathlib after removal
    pathlib_idx = None
    future_idx2 = None
    for i, line in enumerate(new_lines):
        if "from __future__ import annotations" in line:
            future_idx2 = i
        if line.strip() == "from pathlib import Path" and future_idx2 is not None and i > future_idx2:
            pathlib_idx = i
            break

    if pathlib_idx is None:
        # insert after future + empty imports
        insert_at = future_idx2 + 1
        while insert_at < len(new_lines) and new_lines[insert_at].strip().startswith("import"):
            insert_at += 1
        new_lines[insert_at:insert_at] = ["from pathlib import Path\n", "\n"] + block
    else:
        new_lines[pathlib_idx + 1 : pathlib_idx + 1] = ["\n"] + block

    # dedupe duplicate sys import
    text = "".join(new_lines)
    if text.count("import sys\n") > 1:
        seen = False
        out = []
        for line in new_lines:
            if line == "import sys\n":
                if seen:
                    continue
                seen = True
            out.append(line)
        new_lines = out

    new_text = "".join(new_lines)
    if new_text != "".join(lines):
        path.write_text(new_text, encoding="utf-8")
        return True
    return False


def main() -> None:
    n = 0
    for base in [REPO / "scripts"]:
        for py in sorted(base.rglob("*.py")):
            if fix_file(py):
                print(py.relative_to(REPO))
                n += 1
    print(f"fixed {n}")


if __name__ == "__main__":
    main()
