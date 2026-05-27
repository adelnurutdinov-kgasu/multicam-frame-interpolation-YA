"""Shim → lib.lidar_density_mask (запуск из корня: import lidar_density_mask)."""
import lib.lidar_density_mask as _impl
import sys as _sys

_m = _sys.modules[__name__]
for _n in dir(_impl):
    if _n.startswith("__") and _n not in ("__doc__", "__all__"):
        continue
    setattr(_m, _n, getattr(_impl, _n))
