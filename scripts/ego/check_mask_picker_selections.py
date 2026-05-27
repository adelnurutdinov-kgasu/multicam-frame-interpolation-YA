"""Сводка mask_picker_selections.json + сверка с test split."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


from consensus_kit import _parse_vehicle
from ego_mask_policy import ZERO_MASK_GROUP_IDS, is_zero_mask_group

ROOT = Path(__file__).resolve().parent
SELECTIONS = ROOT / "methods_gallery/_ego_manual_masks/mask_picker_selections.json"
TEST_DIR = Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants\test")


def main() -> int:
  data = {}
  if SELECTIONS.is_file():
    raw = json.loads(SELECTIONS.read_text(encoding="utf-8"))
    data = raw.get("selections", raw)
  print(f"Файл: {SELECTIONS}")
  print(f"  размер: {SELECTIONS.stat().st_size if SELECTIONS.is_file() else 0} байт")
  print(f"  записей: {len(data)}")
  by_group = Counter()
  for k, row in data.items():
    g = row.get("group") or k.split("|")[0]
    by_group[g] += 1
  print(f"  уникальных групп: {len(by_group)}")
  for g, n in by_group.most_common(25):
    print(f"    {g}: {n} кадр(ов)")
  if len(by_group) > 25:
    print(f"    … и ещё {len(by_group) - 25}")

  test_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
  for sd in sorted(TEST_DIR.iterdir()):
    if not sd.is_dir():
      continue
    meta = json.loads((sd / "meta.json").read_text(encoding="utf-8"))
    sid = meta.get("sample_id", sd.name)
    cam = meta["target_camera"]
    v = _parse_vehicle(sid)
    if v:
      test_groups[(v, cam)].append(sid)

  picked_groups = set(by_group)
  missing_picker = []
  for key in sorted(test_groups):
    gid = f"{key[0]}_{key[1]}"
    if is_zero_mask_group(key[0], key[1]):
      continue
    if gid not in picked_groups:
      missing_picker.append(gid)
  print(f"\ntest: {len(test_groups)} групп, {sum(len(v) for v in test_groups.values())} кадров")
  print(f"  без записи в picker: {len(missing_picker)}")
  for g in missing_picker[:20]:
    print(f"    {g}")
  print(f"\nZERO (без маски): {sorted(ZERO_MASK_GROUP_IDS)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
