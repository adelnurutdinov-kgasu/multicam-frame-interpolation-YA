"""Insert LiDAR density cells into consensus_training.ipynb."""
import json
from pathlib import Path

NB = Path(__file__).parent / "consensus_training.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]

LIDAR_MD = {
    "cell_type": "markdown",
    "id": "04b_md",
    "metadata": {},
    "source": [
        "## 2.5 LiDAR: плотность точек → зоны (не depth-маска)\n",
        "\n",
        "Проецируем `input/lidar.npz` в **камеру сэмпла** (`meta.camera`, timestep=`target`).\n",
        "Сырые hits → Gaussian blur → `lidar_trust` [0..1]. Это маска доверия для blend,\n",
        "а `d1.npy` остаётся только геометрией на вход U-Net.\n",
        "\n",
        "Параметры — в `Config` (`lidar_zone_sigma`, `lidar_splat_radius`, …).",
    ],
}

LIDAR_CODE = {
    "cell_type": "code",
    "id": "04b_code",
    "metadata": {},
    "outputs": [],
    "source": [
        "from lidar_density_mask import build_lidar_density, plot_lidar_density_debug\n",
        "\n",
        "\n",
        "def _pick_probe_baked(cfg, camera: str | None = None) -> Path:\n",
        "    root = Path(cfg.baked_root)\n",
        "    for baked in sorted(root.iterdir()):\n",
        "        if not (baked / \"meta.json\").is_file():\n",
        "            continue\n",
        "        meta = json.loads((baked / \"meta.json\").read_text(encoding=\"utf-8\"))\n",
        "        cam = meta.get(\"camera\")\n",
        "        if cfg.allowed_cameras and cam not in cfg.allowed_cameras:\n",
        "            continue\n",
        "        if camera and cam != camera:\n",
        "            continue\n",
        "        src = Path(meta.get(\"source_dir\", Path(cfg.dataset_root) / meta[\"sample_id\"]))\n",
        "        if (src / \"input\" / \"lidar.npz\").is_file():\n",
        "            return baked\n",
        "    raise FileNotFoundError(\"Нет baked-сэмпла с lidar.npz для allowed_cameras\")\n",
        "\n",
        "\n",
        "def load_lidar_trust_maps(cfg, sample_dir: Path, camera: str, target_hw) -> dict:\n",
        "    return build_lidar_density(\n",
        "        sample_dir,\n",
        "        camera,\n",
        "        timestep=cfg.lidar_timestep,\n",
        "        splat_radius=cfg.lidar_splat_radius,\n",
        "        zone_sigma=cfg.lidar_zone_sigma,\n",
        "        zone_min=cfg.lidar_zone_min,\n",
        "        min_hits_pixel=cfg.lidar_min_hits_pixel,\n",
        "        target_hw=target_hw,\n",
        "    )\n",
        "\n",
        "\n",
        "PROBE_BAKED = _pick_probe_baked(CFG)\n",
        "_pm = json.loads((PROBE_BAKED / \"meta.json\").read_text(encoding=\"utf-8\"))\n",
        "_src = Path(_pm.get(\"source_dir\", Path(CFG.dataset_root) / _pm[\"sample_id\"]))\n",
        "_cam = _pm[\"camera\"]\n",
        "_hw = (CFG.image_h, CFG.image_w)\n",
        "_lm = load_lidar_trust_maps(CFG, _src, _cam, _hw)\n",
        "_tgt = np.array(Image.open(_src / \"target\" / f\"{_cam}.jpg\").convert(\"RGB\")).astype(np.float32) / 255.0\n",
        "if _tgt.shape[:2] != _hw:\n",
        "    _tgt = _resize_hw(_tgt, _hw, cv2.INTER_LINEAR)\n",
        "_art = _load_ego_mask(CFG, _pm[\"sample_id\"], _cam, _hw)\n",
        "_depth_b = np.load(PROBE_BAKED / \"d1.npy\").astype(np.float32)\n",
        "_depth_b = _resize_hw(np.where(np.isfinite(_depth_b), _depth_b, 0), _hw, cv2.INTER_NEAREST)\n",
        "\n",
        "plot_lidar_density_debug(_lm, rgb=_tgt, art=_art, depth_baked=_depth_b,\n",
        "    title=f\"probe {_pm['sample_id']}  cam={_cam}\")\n",
        "plt.show()\n",
        "print(f\"projected points: {_lm['n_projected']}  |  sigma={CFG.lidar_zone_sigma}  splat={CFG.lidar_splat_radius}\")\n",
    ],
}

# insert after CONFIG code cell (id 04_code)
if not any(c.get("id") == "04b_code" for c in cells):
    for i, c in enumerate(cells):
        if c.get("id") == "04_code":
            cells.insert(i + 1, LIDAR_MD)
            cells.insert(i + 2, LIDAR_CODE)
            break

# patch Config dataclass in 04_code
for c in cells:
    if c.get("id") != "04_code":
        continue
    src = "".join(c["source"])
    if "lidar_zone_sigma" in src:
        break
    src = src.replace(
        "    max_samples: int = 0                        # 0 = все подходящие front/rear\n",
        "    max_samples: int = 0                        # 0 = все подходящие front/rear\n"
        "\n"
        "    # LiDAR density → зоны доверия (blend / отладка; depth на входе U-Net — отдельно)\n"
        "    use_lidar_trust: bool = True\n"
        "    lidar_timestep: str = \"target\"\n"
        "    lidar_splat_radius: int = 2      # радиус splat при растеризации точек\n"
        "    lidar_zone_sigma: float = 14.0   # blur hits → зоны\n"
        "    lidar_zone_min: float = 0.12     # порог trust относительно peak зоны\n"
        "    lidar_min_hits_pixel: int = 1\n",
    )
    src = src.replace(
        'print(f"  num_workers: {CFG.num_workers}  (0 на Windows/Jupyter — стабильная загрузка)")',
        'print(f"  lidar:     splat={CFG.lidar_splat_radius} sigma={CFG.lidar_zone_sigma} min={CFG.lidar_zone_min}")\n'
        'print(f"  num_workers: {CFG.num_workers}  (0 на Windows/Jupyter — стабильная загрузка)")',
    )
    c["source"] = [line + "\n" for line in src.splitlines()]
    if c["source"] and not c["source"][-1].endswith("\n"):
        c["source"][-1] += "\n"

# patch dataset cell 06_code
for c in cells:
    if c.get("id") != "06_code":
        continue
    src = "".join(c["source"])
    if "lidar_trust" in src and "load_lidar_trust_maps" in src:
        break
    if "from lidar_density_mask" not in src:
        src = "from lidar_density_mask import build_lidar_density\n\n" + src
    # in _load after art_mask
    needle = "        art_mask = _load_ego_mask(self.cfg, sid, cam, target_hw)\n\n        rife_rgb"
    if needle in src and "lidar_trust" not in src.split(needle)[1][:400]:
        insert = (
            "        art_mask = _load_ego_mask(self.cfg, sid, cam, target_hw)\n\n"
            "        lidar_trust = np.zeros((H, W), dtype=np.float32)\n"
            "        if getattr(self.cfg, \"use_lidar_trust\", True):\n"
            "            _lm = load_lidar_trust_maps(self.cfg, src, cam, target_hw)\n"
            "            lidar_trust = _lm[\"lidar_trust\"]\n\n"
            "        rife_rgb"
        )
        src = src.replace(needle, insert)
    if '"artifact_mask": art_mask,' in src and '"lidar_trust"' not in src:
        src = src.replace(
            '            "artifact_mask": art_mask,\n',
            '            "artifact_mask": art_mask,\n'
            '            "lidar_trust": lidar_trust,\n',
        )
    # augment crop
    if "art_mask = art_mask[sl]" in src and "lidar_trust = lidar_trust[sl]" not in src:
        src = src.replace(
            "            art_mask = art_mask[sl]; mean_t0t1 = mean_t0t1[sl]; rife = rife[sl]\n",
            "            art_mask = art_mask[sl]; lidar_trust = lidar_trust[sl]\n"
            "            mean_t0t1 = mean_t0t1[sl]; rife = rife[sl]\n",
        )
    for flip in ("[:, ::-1]", "[::-1]"):
        if f"art_mask = art_mask{flip}" in src and f"lidar_trust = lidar_trust{flip}" not in src:
            src = src.replace(
                f"                art_mask = art_mask{flip}.copy()\n",
                f"                art_mask = art_mask{flip}.copy(); lidar_trust = lidar_trust{flip}.copy()\n",
            )
    if "art_mask = rot(art_mask)" in src and "lidar_trust = rot(lidar_trust)" not in src:
        src = src.replace(
            "                sf_rgb = rot(sf_rgb); sf_mask = rot(sf_mask); art_mask = rot(art_mask)\n",
            "                sf_rgb = rot(sf_rgb); sf_mask = rot(sf_mask); art_mask = rot(art_mask); lidar_trust = rot(lidar_trust)\n",
        )
    c["source"] = [line + "\n" for line in src.splitlines()]

nb["cells"] = cells
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Patched", NB)
