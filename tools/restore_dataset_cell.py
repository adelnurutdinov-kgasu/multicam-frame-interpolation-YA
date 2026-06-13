"""Restore full consensus_training dataset cell (ego + rife + lidar)."""
import json
from pathlib import Path

CELL = r'''from lidar_density_mask import build_lidar_density

EPS = 1e-6
VEHICLE_RE = re.compile(r"_([a-zA-Z]+)_\d+__\d{3}$")

try:
    import cv2
except ImportError:
    cv2 = None


def _parse_vehicle(sample_id: str) -> str | None:
    m = VEHICLE_RE.search(sample_id)
    return m.group(1) if m else None


def _load_ego_mask(cfg, sample_id: str, camera: str, target_hw) -> np.ndarray:
    H, W = target_hw
    if not cfg.use_artifact_mask:
        return np.zeros((H, W), dtype=np.float32)
    vehicle = _parse_vehicle(sample_id)
    if not vehicle:
        return np.zeros((H, W), dtype=np.float32)
    path = Path(cfg.ego_masks_root) / f"{vehicle}_{camera}.png"
    if not path.is_file():
        return np.zeros((H, W), dtype=np.float32)
    m = np.array(Image.open(path).convert("L"))
    m = _resize_hw(m, target_hw, cv2.INTER_NEAREST)
    return (m > 127).astype(np.float32)


def load_lidar_trust_maps(cfg, sample_dir: Path, camera: str, target_hw) -> dict:
    return build_lidar_density(
        sample_dir,
        camera,
        timestep=cfg.lidar_timestep,
        splat_radius=cfg.lidar_splat_radius,
        zone_sigma=cfg.lidar_zone_sigma,
        zone_min=cfg.lidar_zone_min,
        min_hits_pixel=cfg.lidar_min_hits_pixel,
        target_hw=target_hw,
    )


def build_base_init(warp, cov, sf_rgb, sf_mask, art_mask, mean_t0t1, use_anchor: bool):
    warp_w = cov[..., None]
    sf_w = sf_mask[..., None] * (1.0 - warp_w)
    base_init = warp * warp_w + sf_rgb * sf_w
    base_mask = np.clip(cov + sf_mask * (1.0 - cov), 0.0, 1.0)
    base_init = base_init * (1.0 - art_mask[..., None])
    base_mask = base_mask * (1.0 - art_mask)
    if use_anchor:
        hole = (1.0 - base_mask)[..., None]
        mean_scene = mean_t0t1 * (1.0 - art_mask[..., None])
        mean_art = mean_t0t1 * art_mask[..., None]
        base_init = base_init + mean_scene * hole + mean_art * art_mask[..., None]
    return base_init, base_mask


def _norm_depth(d: np.ndarray) -> np.ndarray:
    d = d.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() == 0:
        return np.zeros_like(d, dtype=np.float32)
    dmax = float(d[valid].max())
    out = np.zeros_like(d, dtype=np.float32)
    out[valid] = d[valid] / (dmax + EPS)
    return out


def _chw_npy_to_hwc_u8(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.clip(arr.transpose(1, 2, 0), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _resize_hw(img: np.ndarray, target_hw, interp) -> np.ndarray:
    H, W = target_hw
    if img.shape[:2] == (H, W):
        return img
    if cv2 is None:
        raise ImportError("opencv-python нужен для resize карт")
    return cv2.resize(img, (W, H), interpolation=interp)


def _resize_maps(warp_rgb, coverage, depth, target_hw):
    warp_rgb = _resize_hw(warp_rgb, target_hw, cv2.INTER_LINEAR)
    coverage = _resize_hw(coverage, target_hw, cv2.INTER_LINEAR)
    depth = _resize_hw(depth, target_hw, cv2.INTER_NEAREST)
    return warp_rgb, coverage, depth


class ConsensusDataset(Dataset):
    def __init__(self, sample_dirs, cfg, patch_size=256, augment=True):
        self.dirs = [Path(p) for p in sample_dirs]
        self.cfg = cfg
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.dirs)

    def _load(self, baked_dir: Path) -> dict:
        meta = json.loads((baked_dir / "meta.json").read_text(encoding="utf-8"))
        sid = meta["sample_id"]
        cam = meta["camera"]
        src = Path(meta.get("source_dir", Path(self.cfg.dataset_root) / sid))
        warp_dir = Path(self.cfg.consensus_root) / sid

        warp_rgb = _chw_npy_to_hwc_u8(warp_dir / self.cfg.consensus_file)
        coverage = np.load(warp_dir / "coverage.npy").astype(np.float32)
        depth = np.load(baked_dir / "d1.npy").astype(np.float32)
        depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
        target_rgb = np.array(Image.open(src / "target" / f"{cam}.jpg").convert("RGB"))

        target_hw = (self.cfg.image_h, self.cfg.image_w)
        warp_rgb, coverage, depth = _resize_maps(warp_rgb, coverage, depth, target_hw)
        target_rgb = _resize_hw(target_rgb, target_hw, cv2.INTER_LINEAR)

        t0_rgb = np.array(Image.open(src / "input" / "t0" / f"{cam}.jpg").convert("RGB"))
        t1_rgb = np.array(Image.open(src / "input" / "t1" / f"{cam}.jpg").convert("RGB"))
        t0_rgb = _resize_hw(t0_rgb, target_hw, cv2.INTER_LINEAR)
        t1_rgb = _resize_hw(t1_rgb, target_hw, cv2.INTER_LINEAR)
        mean_t0t1_rgb = ((t0_rgb.astype(np.float32) + t1_rgb.astype(np.float32)) * 0.5).astype(np.uint8)

        H, W = target_hw
        sf_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        sf_mask = np.zeros((H, W), dtype=np.float32)
        if self.cfg.use_static_far:
            sf_path = Path(self.cfg.static_far_root) / sid / "static_far.npz"
            if sf_path.is_file():
                z = np.load(sf_path)
                sf_mask = z["static"].astype(np.float32)
                sf_rgb = z["mean_pred"].astype(np.uint8)
                if sf_rgb.shape[:2] != (H, W):
                    sf_rgb, sf_mask, _ = _resize_maps(sf_rgb, sf_mask, sf_mask, (H, W))

        art_mask = _load_ego_mask(self.cfg, sid, cam, target_hw)

        lidar_trust = np.zeros((H, W), dtype=np.float32)
        if getattr(self.cfg, "use_lidar_trust", True):
            lidar_trust = load_lidar_trust_maps(self.cfg, src, cam, target_hw)["lidar_trust"]

        rife_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rife_path = Path(self.cfg.rife_root) / f"{sid}.jpg"
        if rife_path.is_file():
            rife_rgb = np.array(Image.open(rife_path).convert("RGB"))
            rife_rgb = _resize_hw(rife_rgb, target_hw, cv2.INTER_LINEAR)

        return {
            "warp_rgb": warp_rgb,
            "coverage": coverage,
            "depth": depth,
            "static_far_rgb": sf_rgb,
            "static_far_mask": sf_mask,
            "artifact_mask": art_mask,
            "lidar_trust": lidar_trust,
            "mean_t0t1_rgb": mean_t0t1_rgb,
            "rife_rgb": rife_rgb,
            "target_rgb": target_rgb,
        }

    def __getitem__(self, idx):
        s = self._load(self.dirs[idx])

        warp = s["warp_rgb"].astype(np.float32) / 255.0
        cov = np.clip(s["coverage"], 0.0, 1.0)
        depth_n = _norm_depth(s["depth"])
        sf_rgb = s["static_far_rgb"].astype(np.float32) / 255.0
        sf_mask = np.clip(s["static_far_mask"], 0.0, 1.0)
        art_mask = np.clip(s["artifact_mask"], 0.0, 1.0)
        lidar_trust = np.clip(s["lidar_trust"], 0.0, 1.0)
        mean_t0t1 = s["mean_t0t1_rgb"].astype(np.float32) / 255.0
        rife = s["rife_rgb"].astype(np.float32) / 255.0
        target = s["target_rgb"].astype(np.float32) / 255.0

        H, W = warp.shape[:2]

        if self.patch_size and (H > self.patch_size or W > self.patch_size):
            ph = pw = self.patch_size
            y0 = random.randint(0, H - ph)
            x0 = random.randint(0, W - pw)
            sl = (slice(y0, y0 + ph), slice(x0, x0 + pw))
            warp = warp[sl]; cov = cov[sl]; depth_n = depth_n[sl]
            sf_rgb = sf_rgb[sl]; sf_mask = sf_mask[sl]; art_mask = art_mask[sl]
            lidar_trust = lidar_trust[sl]
            mean_t0t1 = mean_t0t1[sl]; rife = rife[sl]
            target = target[sl]

        if self.augment:
            if random.random() < 0.5:
                warp = warp[:, ::-1].copy(); cov = cov[:, ::-1].copy()
                depth_n = depth_n[:, ::-1].copy()
                sf_rgb = sf_rgb[:, ::-1].copy(); sf_mask = sf_mask[:, ::-1].copy()
                art_mask = art_mask[:, ::-1].copy(); lidar_trust = lidar_trust[:, ::-1].copy()
                mean_t0t1 = mean_t0t1[:, ::-1].copy(); rife = rife[:, ::-1].copy()
                target = target[:, ::-1].copy()
            if random.random() < 0.5:
                warp = warp[::-1].copy(); cov = cov[::-1].copy()
                depth_n = depth_n[::-1].copy()
                sf_rgb = sf_rgb[::-1].copy(); sf_mask = sf_mask[::-1].copy()
                art_mask = art_mask[::-1].copy(); lidar_trust = lidar_trust[::-1].copy()
                mean_t0t1 = mean_t0t1[::-1].copy(); rife = rife[::-1].copy()
                target = target[::-1].copy()
            k = random.randint(0, 3)
            if k:
                rot = lambda a: np.rot90(a, k).copy()
                warp = rot(warp); cov = rot(cov); depth_n = rot(depth_n)
                sf_rgb = rot(sf_rgb); sf_mask = rot(sf_mask); art_mask = rot(art_mask)
                lidar_trust = rot(lidar_trust)
                mean_t0t1 = rot(mean_t0t1); rife = rot(rife)
                target = rot(target)

        base_init, base_mask = build_base_init(
            warp, cov, sf_rgb, sf_mask, art_mask, mean_t0t1, self.cfg.use_anchor_frames
        )
        mean_in = mean_t0t1 * (1.0 - art_mask[..., None])
        eff_mask = np.ones_like(base_mask) if self.cfg.use_anchor_frames else base_mask

        to_chw_3 = lambda a: torch.from_numpy(a.transpose(2, 0, 1)).contiguous().float()
        to_chw_1 = lambda a: torch.from_numpy(a)[None].contiguous().float()
        parts = [
            to_chw_3(warp), to_chw_1(cov), to_chw_1(depth_n),
            to_chw_3(sf_rgb), to_chw_1(sf_mask), to_chw_1(art_mask),
        ]
        if self.cfg.use_anchor_frames:
            parts.append(to_chw_3(mean_in))
        inputs = torch.cat(parts, dim=0)

        return {
            "inputs": inputs,
            "effective_mask": to_chw_1(eff_mask),
            "base_init": to_chw_3(base_init),
            "base_mask": to_chw_1(base_mask),
            "target": to_chw_3(target),
            "mean_raw": to_chw_3(mean_t0t1),
            "rife_rgb": to_chw_3(rife),
            "lidar_trust": to_chw_1(lidar_trust),
        }


def discover_samples(cfg) -> list[Path]:
    baked_root = Path(cfg.baked_root)
    consensus_root = Path(cfg.consensus_root)
    dataset_root = Path(cfg.dataset_root)
    out: list[Path] = []
    for baked_dir in sorted(baked_root.iterdir()):
        if not baked_dir.is_dir() or not (baked_dir / "meta.json").is_file():
            continue
        meta = json.loads((baked_dir / "meta.json").read_text(encoding="utf-8"))
        cam = meta["camera"]
        if getattr(cfg, "allowed_cameras", None) and cam not in cfg.allowed_cameras:
            continue
        sid = meta["sample_id"]
        src = Path(meta.get("source_dir", dataset_root / sid))
        warp_dir = consensus_root / sid
        ok = (
            (warp_dir / cfg.consensus_file).is_file()
            and (warp_dir / "coverage.npy").is_file()
            and (baked_dir / "d1.npy").is_file()
            and (src / "target" / f"{cam}.jpg").is_file()
            and (src / "input" / "lidar.npz").is_file()
        )
        if ok:
            out.append(baked_dir)
    if cfg.max_samples > 0 and len(out) > cfg.max_samples:
        rng = np.random.RandomState(cfg.seed)
        idx = rng.permutation(len(out))[: cfg.max_samples]
        out = [out[i] for i in idx]
    return out


ALL_SAMPLES = discover_samples(CFG)
print(f"Готово к обучению: {len(ALL_SAMPLES)} сэмплов (cameras={CFG.allowed_cameras})")
if ALL_SAMPLES:
    ds = ConsensusDataset(ALL_SAMPLES[:4], CFG, patch_size=CFG.patch_size, augment=True)
    b = ds[0]
    art_ch = b["inputs"][9:10]
    n_art_px = int((art_ch > 0.5).sum().item())
    print(f"  ego mask px (sample0): {n_art_px}")
    for k, v in b.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype} [{v.min():.3f}..{v.max():.3f}]")
'''

nb = json.loads(Path("consensus_training.ipynb").read_text(encoding="utf-8"))
for c in nb["cells"]:
    if c.get("id") == "06_code":
        c["source"] = [ln + "\n" for ln in CELL.splitlines()]
        c["outputs"] = []
        break

# fix config lidar fields
for c in nb["cells"]:
    if c.get("id") != "04_code":
        continue
    src = "".join(c["source"])
    if "use_lidar_trust" not in src:
        src = src.replace(
            "    max_samples: int = 0",
            "    max_samples: int = 0",
        )
        src = src.replace(
            "    allowed_cameras: tuple = (\"front\", \"rear\")  # пока только перед/зад\n",
            "    allowed_cameras: tuple = (\"front\", \"rear\")  # пока только перед/зад\n\n"
            "    use_lidar_trust: bool = True\n"
            "    lidar_timestep: str = \"target\"\n"
            "    lidar_splat_radius: int = 2\n"
            "    lidar_zone_sigma: float = 14.0\n"
            "    lidar_zone_min: float = 0.12\n"
            "    lidar_min_hits_pixel: int = 1\n",
        )
    c["source"] = [ln + "\n" for ln in src.splitlines()]
    break

Path("consensus_training.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("restored 06_code", len(CELL))
