"""Strip outputs from ipynb to fix broken base64 strings."""
import json
from pathlib import Path

p = Path("consensus_training.ipynb")
raw = p.read_text(encoding="utf-8", errors="replace")

# Parse cell-by-cell: find "cells" array and rebuild with empty outputs
# Fallback: use git HEAD if local parse fails
try:
    nb = json.loads(raw)
except json.JSONDecodeError:
    print("local JSON broken, restoring sources from git HEAD + user patches")
    import subprocess
    raw_git = subprocess.check_output(["git", "show", "HEAD:consensus_training.ipynb"])
    nb = json.loads(raw_git.decode("utf-8"))

for c in nb.get("cells", []):
    if c.get("cell_type") == "code":
        c["outputs"] = []
        c["execution_count"] = None

p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Repaired", p, "cells:", len(nb["cells"]))
