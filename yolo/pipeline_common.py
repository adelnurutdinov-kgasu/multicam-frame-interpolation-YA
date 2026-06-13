"""Общая логика для detect/segment пайплайнов."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from yolo.utils import collect_images, filter_by_path, resolve_path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_paths(cfg: dict, args: argparse.Namespace) -> dict:
    if getattr(args, "dataset", None):
        cfg["dataset_dir"] = args.dataset
    if getattr(args, "output", None):
        cfg["output_dir"] = args.output
    if getattr(args, "model", None):
        cfg["model"] = args.model
    if getattr(args, "conf", None) is not None:
        cfg["conf"] = args.conf
    if getattr(args, "device", None) is not None:
        cfg["device"] = args.device
    return cfg


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="yolo/config.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--flat-output", action="store_true", help="Писать прямо в output_dir без подпапки с датой")
    return p


def prepare_run(
    args: argparse.Namespace,
    default_config: str,
    default_output: str,
) -> tuple[dict, Path, Path, Path, list[Path], int]:
    """Возвращает (cfg, config_path, dataset_dir, run_dir, images, exit_code)."""
    config_path = resolve_path(args.config or default_config, REPO_ROOT)
    cfg = load_config(config_path)
    cfg = merge_paths(cfg, args)
    if getattr(args, "flat_output", False):
        cfg["flat_output"] = True
    if not cfg.get("output_dir"):
        cfg["output_dir"] = default_output

    dataset_dir = resolve_path(cfg["dataset_dir"], REPO_ROOT)
    output_root = resolve_path(cfg["output_dir"], REPO_ROOT)
    if cfg.get("flat_output"):
        run_dir = output_root
    else:
        run_name = cfg.get("run_name") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / run_name

    images = collect_images(
        dataset_dir,
        globs=cfg.get("image_globs") or ["**/*.jpg"],
        exclude_dirs=set(cfg.get("exclude_dirs") or []),
    )
    images = filter_by_path(images, cfg.get("path_filters", "all"))
    if args.limit:
        images = images[: args.limit]

    print(f"Датасет: {dataset_dir}")
    print(f"Найдено изображений: {len(images)}")
    if not images:
        print("Нет файлов. Положите датасет в data/dataset или укажите --dataset")
        return cfg, config_path, dataset_dir, run_dir, images, 1

    if args.dry_run:
        for p in images[:20]:
            try:
                print(p.relative_to(dataset_dir))
            except ValueError:
                print(p)
        if len(images) > 20:
            print(f"... и ещё {len(images) - 20}")
        return cfg, config_path, dataset_dir, run_dir, images, 0

    return cfg, config_path, dataset_dir, run_dir, images, -1


def save_config_copy(config_path: Path, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )


def write_run_meta(run_dir: Path, meta: dict) -> None:
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
