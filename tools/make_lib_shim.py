"""Generate root shim that re-exports full lib module."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODULES = [
    "consensus_kit",
    "lidar_depth_map",
    "lidar_density_mask",
    "layered_parallax",
    "mega_parallax",
    "static_mask",
    "ego_mask_extract",
    "ego_mask_policy",
]

TEMPLATE = '''"""Shim → lib.{name} (запуск из корня: import {name})."""
import lib.{name} as _impl
import sys as _sys

_m = _sys.modules[__name__]
for _n in dir(_impl):
    if _n.startswith("__") and _n not in ("__doc__", "__all__"):
        continue
    setattr(_m, _n, getattr(_impl, _n))
'''

for name in MODULES:
    (REPO / f"{name}.py").write_text(TEMPLATE.format(name=name), encoding="utf-8")
    print(f"shim {name}.py")
