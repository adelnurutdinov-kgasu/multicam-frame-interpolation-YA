"""HTML-галерея всех test outputs (199 сэмплов)."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


import numpy as np
from PIL import Image



from consensus_kit import ConsensusConfig, _load_ego_mask, _parse_vehicle  # noqa: E402
from ego_mask_policy import is_zero_mask_group  # noqa: E402

OUT_ROOT = Path(r"C:\Users\adel\Downloads\cv_dataset\consensus_test_outputs")
GALLERY_HTML = OUT_ROOT / "index.html"
EGO_TODO_JSON = OUT_ROOT / "ego_masks_todo.json"
TEST_ROOT = Path(r"C:\Users\adel\Downloads\cv_dataset\final_dataset_v5_participants\test")
BAKED_ROOT = Path(r"C:\Users\adel\Downloads\cv_dataset\rife_refinement_baked\test")
EGO_MASKS_ROOT = REPO / "methods_gallery/_ego_manual_masks/masks_approved"
EGO_MASKS_DIR = REPO / "methods_gallery/_ego_manual_masks/masks"
EGO_SELECTED_DIR = REPO / "methods_gallery/_ego_manual_masks/masks_selected"
EGO_DONOR_DIR = REPO / "methods_gallery/_ego_manual_masks/masks_donor_suggested"

COLUMNS = (
    ("t0", "t0.jpg"),
    ("t1", "t1.jpg"),
    ("ref", "input_ref.jpg"),
    ("consensus", "consensus_model.jpg"),
    ("RIFE", "rife.jpg"),
    ("blend", "blend_lidar_rife.jpg"),
    ("blend blur50", "blend_blur50.jpg"),
    ("ego mask", "ego_art_mask.png"),
)

CAMERAS = ("front", "rear", "left_fwd", "right_fwd", "left_bwd", "right_bwd")


def source_root(sid: str) -> Path:
    baked_meta = BAKED_ROOT / sid / "meta.json"
    if baked_meta.is_file():
        meta = json.loads(baked_meta.read_text(encoding="utf-8"))
        return Path(meta.get("source_dir", TEST_ROOT / sid))
    return TEST_ROOT / sid


def ensure_source_frames(out_dir: Path, sid: str, cam: str) -> None:
    src_root = source_root(sid)
    for tag in ("t0", "t1"):
        src = src_root / "input" / tag / f"{cam}.jpg"
        dst = out_dir / f"{tag}.jpg"
        if not src.is_file() or dst.is_file():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def ego_cfg() -> ConsensusConfig:
    return ConsensusConfig(
        dataset_root=str(TEST_ROOT),
        consensus_root=str(TEST_ROOT),
        baked_root=str(BAKED_ROOT),
        rife_root=str(TEST_ROOT),
        ego_masks_root=str(EGO_MASKS_ROOT),
    )


def ensure_ego_mask_preview(out_dir: Path, sid: str, cam: str) -> np.ndarray | None:
    ref = out_dir / "consensus_model.jpg"
    if not ref.is_file() or not cam:
        return None
    hw = np.array(Image.open(ref).convert("RGB")).shape[:2]
    mask = _load_ego_mask(ego_cfg(), sid, cam, hw)
    Image.fromarray((mask * 255).astype(np.uint8)).save(out_dir / "ego_art_mask.png")
    return mask


def mask_file_info(group_id: str) -> dict:
    def nz(path: Path) -> bool:
        return path.is_file() and np.array(Image.open(path).convert("L")).max() > 0

    return {
        "approved_path": str(EGO_MASKS_ROOT / f"{group_id}.png"),
        "approved_exists": bool((EGO_MASKS_ROOT / f"{group_id}.png").is_file()),
        "approved_nonempty": bool(nz(EGO_MASKS_ROOT / f"{group_id}.png")),
        "masks_exists": bool((EGO_MASKS_DIR / f"{group_id}.png").is_file()),
        "masks_nonempty": bool(nz(EGO_MASKS_DIR / f"{group_id}.png")),
        "selected_exists": bool((EGO_SELECTED_DIR / f"{group_id}.png").is_file()),
        "selected_nonempty": bool(nz(EGO_SELECTED_DIR / f"{group_id}.png")),
        "donor_exists": bool((EGO_DONOR_DIR / f"{group_id}.png").is_file()),
        "donor_nonempty": bool(nz(EGO_DONOR_DIR / f"{group_id}.png")),
    }


def classify_ego(sid: str, cam: str, mask: np.ndarray) -> tuple[str, str, str]:
    vehicle = _parse_vehicle(sid) or "?"
    group_id = f"{vehicle}_{cam}"
    if mask.max() > 0:
        return "ok", group_id, ""
    if is_zero_mask_group(vehicle, cam):
        return "zero", group_id, "ZERO_MASK_GROUPS — намеренно без маски"
    info = mask_file_info(group_id)
    if info["approved_exists"] and not info["approved_nonempty"]:
        return "zero", group_id, "approved пустой файл"
    if not info["approved_exists"]:
        hints = []
        if info["selected_nonempty"]:
            hints.append("есть masks_selected")
        elif info["masks_nonempty"]:
            hints.append("есть masks/")
        elif info["donor_nonempty"]:
            hints.append("есть donor_suggested")
        else:
            hints.append("нет в train / не размечено")
        return "missing", group_id, "; ".join(hints)
    return "missing", group_id, "approved есть, но маска пустая"


def build_todo_summary(rows: list[dict]) -> dict:
    groups: dict[str, dict] = {}
    for r in rows:
        if r["ego_status"] == "ok":
            continue
        g = r["ego_group"]
        if g not in groups:
            info = mask_file_info(g)
            groups[g] = {
                "group_id": g,
                "status": r["ego_status"],
                "note": r["ego_note"],
                "samples": [],
                "sample_count": 0,
                "approved_path": info["approved_path"],
                **{k: info[k] for k in info if k != "approved_path"},
            }
        groups[g]["samples"].append(r["sid"])
        groups[g]["sample_count"] += 1
    missing = sorted([g for g in groups.values() if g["status"] == "missing"], key=lambda x: x["group_id"])
    zero = sorted([g for g in groups.values() if g["status"] == "zero"], key=lambda x: x["group_id"])
    return {"missing_groups": missing, "zero_groups": zero, "missing_samples": sum(g["sample_count"] for g in missing)}


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for d in sorted(OUT_ROOT.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "meta_infer.json"
        if not meta_path.is_file():
            continue
        if not (d / "consensus_model.jpg").is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sid = d.name
        cam = meta.get("camera", "")
        mask = np.zeros((1, 1), dtype=np.float32)
        if cam:
            ensure_source_frames(d, sid, cam)
            m = ensure_ego_mask_preview(d, sid, cam)
            if m is not None:
                mask = m
        ego_status, ego_group, ego_note = classify_ego(sid, cam, mask)
        images = []
        for label, fn in COLUMNS:
            p = d / fn
            if p.is_file():
                images.append({"label": label, "file": fn, "rel": f"{sid}/{fn}"})
        rows.append({
            "sid": sid,
            "camera": meta.get("camera", "?"),
            "mirrored": bool(meta.get("mirrored_input")),
            "ego_status": ego_status,
            "ego_group": ego_group,
            "ego_note": ego_note,
            "images": images,
        })
    return rows


def write_html(rows: list[dict], todo: dict) -> None:
    data_json = json.dumps(rows, ensure_ascii=False)
    todo_json = json.dumps(todo, ensure_ascii=False)
    n_missing = todo["missing_samples"]
    n_zero = sum(g["sample_count"] for g in todo["zero_groups"])
    by_cam = {c: sum(1 for r in rows if r["camera"] == c) for c in CAMERAS}
    html = f"""<!doctype html>
<html lang="ru"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Consensus test outputs — {len(rows)} samples</title>
<style>
:root {{
  --bg:#0f1115; --card:#171a21; --line:#2a3140; --txt:#e8ecf1; --muted:#9aa3b2; --acc:#6ea8fe;
  --warn:#e8a838; --bad:#e85d5d;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Segoe UI,Arial,sans-serif; background:var(--bg); color:var(--txt); }}
header {{ position:sticky; top:0; z-index:10; background:#12151bcc; backdrop-filter:blur(8px);
  border-bottom:1px solid var(--line); padding:12px 16px; }}
h1 {{ margin:0 0 8px; font-size:18px; }}
.meta {{ color:var(--muted); font-size:13px; margin-bottom:10px; }}
.controls {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
.controls label {{ font-size:13px; color:var(--muted); }}
select, input {{
  background:var(--card); color:var(--txt); border:1px solid var(--line);
  border-radius:6px; padding:6px 10px; font-size:13px;
}}
#count {{ font-size:13px; color:var(--muted); margin-left:auto; }}
.todo-panel {{
  margin:12px 16px 0; padding:12px 14px; background:#1a1520; border:1px solid #4a3540;
  border-radius:8px; font-size:13px;
}}
.todo-panel h2 {{ margin:0 0 8px; font-size:14px; color:#f0c674; }}
.todo-panel p {{ margin:0 0 10px; color:var(--muted); line-height:1.45; }}
.todo-groups {{ display:flex; flex-wrap:wrap; gap:8px; }}
.todo-chip {{
  background:#252029; border:1px solid #5a4555; color:var(--txt); border-radius:999px;
  padding:5px 12px; font-size:12px; cursor:pointer;
}}
.todo-chip:hover {{ border-color:var(--warn); }}
.todo-chip.zero {{ border-color:#444; color:var(--muted); }}
.todo-chip .n {{ color:var(--warn); font-weight:700; }}
main {{ padding:12px 16px 40px; }}
.grid-head, .row {{
  display:grid;
  grid-template-columns: 220px 72px repeat({len(COLUMNS)}, minmax(110px, 1fr));
  gap:8px; align-items:start;
}}
.grid-head {{
  position:sticky; top:92px; z-index:5; background:var(--bg);
  padding:8px 0; border-bottom:1px solid var(--line); font-size:11px; color:var(--muted);
}}
.row {{
  background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:8px; margin-bottom:8px;
}}
.row.mirror {{ outline:1px solid #5a4a20; }}
.row.ego-missing {{ outline:1px solid var(--bad); background:#1c1518; }}
.row.ego-zero {{ outline:1px dashed #555; }}
.sid {{ font-size:11px; word-break:break-all; line-height:1.35; }}
.sid a {{ color:var(--acc); text-decoration:none; }}
.sid .ego-tag {{ display:block; margin-top:4px; font-size:10px; color:var(--warn); }}
.sid .ego-tag.zero {{ color:var(--muted); }}
.cam {{ font-size:12px; font-weight:600; }}
.cell img {{
  width:100%; aspect-ratio:16/9; object-fit:cover; border-radius:4px;
  border:1px solid #333; background:#000; display:block;
}}
.cell .lbl {{ font-size:10px; color:var(--muted); margin-bottom:4px; text-align:center; }}
.empty {{ color:#555; font-size:11px; text-align:center; padding:18px 0; }}
@media (max-width: 1200px) {{
  .grid-head {{ display:none; }}
  .row {{ grid-template-columns:1fr; }}
}}
</style>
</head>
<body>
<header>
  <h1>Consensus test outputs</h1>
  <div class="meta">
    {len(rows)} сэмплов ·
    front {by_cam.get('front',0)} · rear {by_cam.get('rear',0)} ·
    left_fwd {by_cam.get('left_fwd',0)} · right_fwd {by_cam.get('right_fwd',0)} ·
    left_bwd {by_cam.get('left_bwd',0)} · right_bwd {by_cam.get('right_bwd',0)} ·
    ego ok {sum(1 for r in rows if r['ego_status']=='ok')} ·
    <span style="color:var(--bad)">нужна маска {n_missing}</span> ·
    zero {n_zero}
  </div>
  <div class="controls">
    <label>Камера</label>
    <select id="camFilter">
      <option value="">все</option>
      {''.join(f'<option value="{c}">{c}</option>' for c in CAMERAS)}
    </select>
    <label>Ego</label>
    <select id="egoFilter">
      <option value="">все</option>
      <option value="missing">нужна маска ({n_missing})</option>
      <option value="zero">zero по политике ({n_zero})</option>
      <option value="no_ego">без ego-маски ({n_missing + n_zero})</option>
      <option value="ok">с маской</option>
    </select>
    <label>Поиск</label>
    <input id="q" type="search" placeholder="sample_id / vehicle / group…" size="28">
    <label><input id="mirrorOnly" type="checkbox"> только mirror (left_fwd, right_bwd)</label>
    <span id="count"></span>
  </div>
</header>
<div class="todo-panel" id="todoPanel">
  <h2>Ego-маски: нужно добавить</h2>
  <p>
    Положить маску в <code>methods_gallery/_ego_manual_masks/masks_approved/&lt;vehicle&gt;_&lt;camera&gt;.png</code>,
    затем пересобрать blend и галерею. Подробности: <code>ego_masks_todo.json</code>
  </p>
  <div class="todo-groups" id="todoGroups"></div>
</div>
<main>
  <div class="grid-head">
    <div>sample</div><div>cam</div>
    {''.join(f'<div>{lbl}</div>' for lbl, _ in COLUMNS)}
  </div>
  <div id="rows"></div>
</main>
<script>
const DATA = {data_json};
const TODO = {todo_json};
const COLS = {json.dumps([c[0] for c in COLUMNS])};

function renderTodoChips() {{
  const root = document.getElementById('todoGroups');
  const chips = [];
  TODO.missing_groups.forEach(g => {{
    chips.push(`<button type="button" class="todo-chip" data-group="${{g.group_id}}" data-ego="missing">
      <span class="n">${{g.sample_count}}</span> ${{g.group_id}} <span style="opacity:.7">— ${{g.note}}</span>
    </button>`);
  }});
  TODO.zero_groups.forEach(g => {{
    chips.push(`<button type="button" class="todo-chip zero" data-group="${{g.group_id}}" data-ego="zero">
      <span class="n">${{g.sample_count}}</span> ${{g.group_id}} <span style="opacity:.7">— ${{g.note}}</span>
    </button>`);
  }});
  root.innerHTML = chips.join('');
  root.querySelectorAll('.todo-chip').forEach(btn => {{
    btn.onclick = () => {{
      document.getElementById('egoFilter').value = btn.dataset.ego;
      document.getElementById('q').value = btn.dataset.group;
      apply();
    }};
  }});
}}

function render(list) {{
  const root = document.getElementById('rows');
  root.innerHTML = list.map(r => {{
    const imgs = Object.fromEntries(r.images.map(x => [x.label, x]));
    const cells = COLS.map(lbl => {{
      const x = imgs[lbl];
      if (!x) return '<div class="cell"><div class="empty">—</div></div>';
      return `<div class="cell"><div class="lbl">${{lbl}}</div><a href="${{x.rel}}" target="_blank"><img src="${{x.rel}}" loading="lazy" alt="${{lbl}}"></a></div>`;
    }}).join('');
    let cls = 'row';
    if (r.mirrored) cls += ' mirror';
    if (r.ego_status === 'missing') cls += ' ego-missing';
    if (r.ego_status === 'zero') cls += ' ego-zero';
    const tag = r.ego_status !== 'ok'
      ? `<span class="ego-tag ${{r.ego_status === 'zero' ? 'zero' : ''}}">${{r.ego_group}} — ${{r.ego_note}}</span>`
      : '';
    return `<div class="${{cls}}" data-cam="${{r.camera}}" data-sid="${{r.sid}}" data-ego="${{r.ego_status}}" data-group="${{r.ego_group}}">
      <div class="sid"><a href="${{r.sid}}/" target="_blank">${{r.sid}}</a>${{tag}}</div>
      <div class="cam">${{r.camera}}${{r.mirrored ? ' ↔' : ''}}</div>${{cells}}
    </div>`;
  }}).join('');
  document.getElementById('count').textContent = 'показано: ' + list.length;
}}

function apply() {{
  const cam = document.getElementById('camFilter').value;
  const ego = document.getElementById('egoFilter').value;
  const q = document.getElementById('q').value.trim().toLowerCase();
  const mirrorOnly = document.getElementById('mirrorOnly').checked;
  let list = DATA;
  if (cam) list = list.filter(r => r.camera === cam);
  if (ego === 'missing') list = list.filter(r => r.ego_status === 'missing');
  else if (ego === 'zero') list = list.filter(r => r.ego_status === 'zero');
  else if (ego === 'no_ego') list = list.filter(r => r.ego_status !== 'ok');
  else if (ego === 'ok') list = list.filter(r => r.ego_status === 'ok');
  if (mirrorOnly) list = list.filter(r => r.mirrored);
  if (q) list = list.filter(r =>
    r.sid.toLowerCase().includes(q) || r.ego_group.toLowerCase().includes(q));
  render(list);
}}

document.getElementById('camFilter').onchange = apply;
document.getElementById('egoFilter').onchange = apply;
document.getElementById('q').oninput = apply;
document.getElementById('mirrorOnly').onchange = apply;
renderTodoChips();
apply();
</script>
</body></html>"""
    GALLERY_HTML.write_text(html, encoding="utf-8")


def main() -> None:
    rows = collect_rows()
    todo = build_todo_summary(rows)
    EGO_TODO_JSON.write_text(json.dumps(todo, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(rows, todo)
    print(f"Gallery: {GALLERY_HTML}")
    print(f"Todo:    {EGO_TODO_JSON}")
    print(f"Samples: {len(rows)}  missing={todo['missing_samples']}  zero_groups={len(todo['zero_groups'])}")


if __name__ == "__main__":
    main()
