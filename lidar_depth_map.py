"""Shim → lib.lidar_depth_map (запуск из корня: import lidar_depth_map)."""
import lib.lidar_depth_map as _impl
import sys as _sys

_m = _sys.modules[__name__]
for _n in dir(_impl):
    if _n.startswith("__") and _n not in ("__doc__", "__all__"):
        continue
    setattr(_m, _n, getattr(_impl, _n))
