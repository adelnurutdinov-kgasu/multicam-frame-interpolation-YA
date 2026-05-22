from __future__ import annotations

from pathlib import Path


def resolve_path(path: str | Path, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()


def collect_images(
    root: Path,
    globs: list[str],
    exclude_dirs: set[str] | None = None,
) -> list[Path]:
    exclude = {d.lower() for d in (exclude_dirs or set())}
    root = root.resolve()
    if not root.is_dir():
        return []

    found: set[Path] = set()
    for pattern in globs:
        for p in root.glob(pattern):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
                continue
            parts_lower = {part.lower() for part in p.parts}
            if exclude and parts_lower & exclude:
                continue
            if any(ex in part.lower() for part in p.parts for ex in exclude):
                continue
            found.add(p.resolve())

    return sorted(found)


def filter_by_path(images: list[Path], path_filters) -> list[Path]:
    if path_filters == "all" or not path_filters:
        return images
    if isinstance(path_filters, str):
        path_filters = [path_filters]
    needles = [s.replace("\\", "/") for s in path_filters]
    out = []
    for p in images:
        s = str(p).replace("\\", "/")
        if any(n in s for n in needles):
            out.append(p)
    return out


def load_class_ids(class_names: list[str]) -> list[int]:
    """Имена классов COCO -> id для model.predict(classes=...)."""
    from yolo.coco_classes import NAME_TO_ID

    ids = []
    missing = []
    for name in class_names:
        if name in NAME_TO_ID:
            ids.append(NAME_TO_ID[name])
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"Неизвестные классы COCO: {missing}")
    return ids
