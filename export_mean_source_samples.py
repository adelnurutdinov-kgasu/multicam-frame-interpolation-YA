"""Export sample_ids and target frames used in vehicle/camera means for re-annotation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent
ROOT = REPO / "methods_gallery/_ego_manual_masks"
OUT = ROOT / "mean_source_frames"
JSON_OUT = ROOT / "mean_source_samples.json"
MD_OUT = ROOT / "MEAN_SOURCES_FOR_REANNOTATION.md"
DEFAULT_DATASET = Path(r"C:/Users/adel/Downloads/cv_dataset/final_dataset_v5_participants/train")


def load_index() -> dict[tuple[str, str], dict]:
    items = json.loads(INDEX_PATH.read_text(encoding="utf-8"))["items"]
    return {(x["vehicle"], x["camera"]): x for x in items}


INDEX_PATH = ROOT / "index.json"


def group_record(manifest_item: dict, index: dict[tuple[str, str], dict], dataset: Path) -> dict:
    key = (manifest_item["vehicle"], manifest_item["camera"])
    g = index.get(key, {})
    camera = manifest_item["camera"]
    sample_ids = g.get("sample_ids", [])
    frames = []
    for i, sid in enumerate(sample_ids, 1):
        src = dataset / sid / "target" / f"{camera}.jpg"
        frames.append(
            {
                "index": i,
                "sample_id": sid,
                "target_jpg": str(src),
                "exists": src.is_file(),
            }
        )
    return {
        "vehicle": manifest_item["vehicle"],
        "camera": camera,
        "status": manifest_item.get("status"),
        "n_available": g.get("n_available"),
        "n_used": g.get("n_used", len(sample_ids)),
        "mean_png": g.get("mean_png"),
        "sample_ids": sample_ids,
        "frames": frames,
    }


def export_frames(record: dict, subdir: Path, dataset: Path) -> None:
    subdir.mkdir(parents=True, exist_ok=True)
    camera = record["camera"]
    for fr in record["frames"]:
        src = Path(fr["target_jpg"])
        if not src.is_file():
            continue
        dst = subdir / f"{fr['index']:02d}_{fr['sample_id']}.jpg"
        if not dst.exists():
            shutil.copy2(src, dst)


def md_section(record: dict) -> str:
    lines = [
        f"### {record['vehicle']} / {record['camera']}",
        "",
        f"- **Статус:** `{record['status']}`",
        f"- **В mean:** {record['n_used']} из {record['n_available']} доступных сцен",
        "",
        "**Sample IDs в текущем mean:**",
        "",
    ]
    for fr in record["frames"]:
        mark = "" if fr["exists"] else " *(файл не найден)*"
        lines.append(f"{fr['index']}. `{fr['sample_id']}`{mark}")
        lines.append(f"   - `{fr['target_jpg']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    dataset = DEFAULT_DATASET
    index = load_index()
    manifest_items = json.loads((ROOT / "manual_masks_manifest.json").read_text(encoding="utf-8"))["items"]

    reject = [x for x in manifest_items if x.get("status") == "reject_black"]
    baland = [x for x in manifest_items if x.get("vehicle") == "baland"]
    no_artifact = [x for x in manifest_items if x.get("status") in ("needs_annotation", "bad_mean_few_scenes")]

    reject_records = [group_record(x, index, dataset) for x in reject]
    baland_records = [group_record(x, index, dataset) for x in baland]
    no_artifact_records = [group_record(x, index, dataset) for x in no_artifact]

    OUT.mkdir(parents=True, exist_ok=True)
    for rec in reject_records + baland_records:
        tag = f"{rec['vehicle']}_{rec['camera']}"
        export_frames(rec, OUT / "reject_black" / tag if rec in reject_records else OUT / "baland" / tag, dataset)

    payload = {
        "dataset": str(dataset),
        "note": "Группы без разметки = нет ego-артефактов (нулевая маска).",
        "reject_black": reject_records,
        "baland": baland_records,
        "no_artifact_unannotated": no_artifact_records,
    }
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Кадры, усреднённые в mean (для переразметки)",
        "",
        "Папка с копиями target JPG: `mean_source_frames/`",
        "",
        "Полный JSON: `mean_source_samples.json`",
        "",
        "## Чёрная метка (reject_black) — mean не совпал, нужны другие поездки",
        "",
    ]
    for rec in reject_records:
        md.append(md_section(rec))

    md.extend(["## baland (все камеры)", ""])
    for rec in baland_records:
        md.append(md_section(rec))

    md.extend(
        [
            "## Без разметки — ego-артефактов нет (маска = нули)",
            "",
            "Эти группы не размечались намеренно; в обучении artifact_mask = 0.",
            "",
        ]
    )
    for rec in no_artifact_records:
        md.append(md_section(rec))

    MD_OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {MD_OUT}")
    print(f"Frames under {OUT}")


if __name__ == "__main__":
    main()
