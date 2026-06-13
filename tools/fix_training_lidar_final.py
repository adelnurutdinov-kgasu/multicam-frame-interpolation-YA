"""Fix lidar config, dataset _load, move probe cell after dataset."""
import json
from pathlib import Path

NB = Path("consensus_training.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]

# --- CONFIG lidar fields ---
for c in cells:
    if c.get("id") != "04_code":
        continue
    src = "".join(c["source"])
    if "lidar_zone_sigma" not in src:
        src = src.replace(
            "    max_samples: int = 0                        # 0 = все подходящие front/rear\n\n",
            "    max_samples: int = 0                        # 0 = все подходящие front/rear\n\n"
            "    use_lidar_trust: bool = True\n"
            "    lidar_timestep: str = \"target\"\n"
            "    lidar_splat_radius: int = 2\n"
            "    lidar_zone_sigma: float = 14.0\n"
            "    lidar_zone_min: float = 0.12\n"
            "    lidar_min_hits_pixel: int = 1\n\n",
        )
    if "ego_masks:" not in src and "print(f\"  ego_masks" not in src:
        src = src.replace(
            'print(f"  static_far:',
            'print(f"  ego_masks: {CFG.ego_masks_root}")\n'
            'print(f"  cameras:   {CFG.allowed_cameras}")\n'
            'print(f"  static_far:',
        )
    c["source"] = [ln + "\n" for ln in src.splitlines()]
    break

# --- dataset _load lidar ---
for c in cells:
    if c.get("id") != "06_code":
        continue
    src = "".join(c["source"])
    needle = "        art_mask = _load_ego_mask(self.cfg, sid, cam, target_hw)\n\n        rife_rgb"
    if needle in src and "lidar_trust = np.zeros" not in src:
        src = src.replace(
            needle,
            "        art_mask = _load_ego_mask(self.cfg, sid, cam, target_hw)\n\n"
            "        lidar_trust = np.zeros((H, W), dtype=np.float32)\n"
            "        if getattr(self.cfg, \"use_lidar_trust\", True):\n"
            "            _lm = load_lidar_trust_maps(self.cfg, src, cam, target_hw)\n"
            "            lidar_trust = _lm[\"lidar_trust\"]\n\n"
            "        rife_rgb",
        )
    if '"lidar_trust": lidar_trust' not in src:
        src = src.replace(
            '"artifact_mask": art_mask,\n',
            '"artifact_mask": art_mask,\n            "lidar_trust": lidar_trust,\n',
        )
    if "lidar_trust = lidar_trust[sl]" not in src:
        src = src.replace(
            "art_mask = art_mask[sl]; mean_t0t1 = mean_t0t1[sl]",
            "art_mask = art_mask[sl]; lidar_trust = lidar_trust[sl]\n            mean_t0t1 = mean_t0t1[sl]",
        )
    if "def load_lidar_trust_maps" not in src:
        src = src.replace(
            "from lidar_density_mask import build_lidar_density\n\n",
            "from lidar_density_mask import build_lidar_density\n\n\n"
            "def load_lidar_trust_maps(cfg, sample_dir: Path, camera: str, target_hw) -> dict:\n"
            "    return build_lidar_density(\n"
            "        sample_dir, camera, timestep=cfg.lidar_timestep,\n"
            "        splat_radius=cfg.lidar_splat_radius, zone_sigma=cfg.lidar_zone_sigma,\n"
            "        zone_min=cfg.lidar_zone_min, min_hits_pixel=cfg.lidar_min_hits_pixel,\n"
            "        target_hw=target_hw,\n"
            "    )\n\n",
        )
    c["source"] = [ln + "\n" for ln in src.splitlines()]
    break

# --- move probe cells after dataset (06_code) ---
probe_md = probe_code = None
rest = []
for c in cells:
    if c.get("id") in ("04b_md", "04b_code"):
        if c.get("id") == "04b_md":
            probe_md = c
        else:
            probe_code = c
        continue
    rest.append(c)

if probe_code and probe_md:
  # fix probe: no _load_ego_mask before definition
    src = "".join(probe_code["source"])
    src = src.replace(
        "_art = _load_ego_mask(CFG, _pm[\"sample_id\"], _cam, _hw)\n",
        "_art = None\n"
        "if CFG.use_artifact_mask:\n"
        "    _art = _load_ego_mask(CFG, _pm[\"sample_id\"], _cam, _hw)\n",
    )
    probe_code["source"] = [ln + "\n" for ln in src.splitlines()]

    new_cells = []
    inserted = False
    for c in rest:
        new_cells.append(c)
        if not inserted and c.get("id") == "06_code":
            new_cells.append(probe_md)
            new_cells.append(probe_code)
            inserted = True
    cells = new_cells if inserted else rest + [probe_md, probe_code]
else:
    cells = rest

nb["cells"] = cells
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("fixed", NB, "n_cells", len(cells))
