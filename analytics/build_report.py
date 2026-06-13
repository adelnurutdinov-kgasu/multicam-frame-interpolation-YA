"""HTML-отчёт из stratified_results.json."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def heat_style(mae: float | None, vmin: float, vmax: float) -> str:
    if mae is None:
        return "background:#333;color:#888"
    t = (mae - vmin) / max(vmax - vmin, 1e-6)
    t = max(0.0, min(1.0, t))
    # зелёный=лучше (низкий MAE), красный=хуже
    r = int(255 * t)
    g = int(255 * (1 - t))
    return f"background:rgb({r},{g},80);color:#111"


def table_block(title: str, data: dict, methods: list[str], row_keys: list[str]) -> str:
    rows_html = ""
    all_mae = []
    for rk in row_keys:
        for m in methods:
            v = data.get(m, {}).get(rk, {}).get("mae")
            if v is not None:
                all_mae.append(v)
    vmin = min(all_mae) if all_mae else 0
    vmax = max(all_mae) if all_mae else 1

    for rk in row_keys:
        cells = f"<td><b>{html.escape(rk)}</b></td>"
        for m in methods:
            st = data.get(m, {}).get(rk, {})
            mae = st.get("mae")
            psnr = st.get("psnr_db")
            px = st.get("pixels", 0)
            if mae is None:
                cells += "<td>—</td>"
            else:
                cells += (
                    f'<td style="{heat_style(mae, vmin, vmax)}">'
                    f"MAE {mae:.2f}<br>PSNR {psnr:.1f} dB<br><small>{px:,} px</small></td>"
                )
        rows_html += f"<tr>{cells}</tr>"

    hdr = "".join(f"<th>{html.escape(m)}</th>" for m in methods)
    return f"""
    <section><h2>{html.escape(title)}</h2>
    <table><tr><th>слой</th>{hdr}</tr>{rows_html}</table>
    <p class="hint">Цвет: зелёный = меньше ошибка (MAE), красный = больше.</p></section>"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="analytics/out/stratified_results.json")
    p.add_argument("--output", default="analytics/out/report.html")
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.is_absolute():
        inp = ROOT / inp
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out

    data = json.loads(inp.read_text(encoding="utf-8"))
    cfg_path = ROOT / "analytics" / "config.yaml"
    config_methods = []
    if cfg_path.is_file():
        import yaml
        config_methods = yaml.safe_load(cfg_path.read_text(encoding="utf-8")).get("methods") or []

    methods = list(data.get("config", {}).get("methods") or [])
    for m in config_methods:
        if m not in methods:
            methods.append(m)
    if not methods:
        methods = sorted(set(data.get("overall", {}).keys()))
    by_depth = data.get("by_depth", {})
    by_sem = data.get("by_semantic", {})
    overall = data.get("overall", {})

    depth_rows = sorted({k for m in methods for k in by_depth.get(m, {})})
    sem_rows = sorted({k for m in methods for k in by_sem.get(m, {})})

    overall_rows = ""
    for m in methods:
        st = overall.get(m, {})
        mae = st.get("mae")
        if mae is None:
            overall_rows += (
                f"<tr><td>{html.escape(m)}</td>"
                f"<td colspan='3' style='color:#888'>нет данных — "
                f"<code>python analytics/run_stratified_eval.py --backfill-new</code></td></tr>"
            )
            continue
        overall_rows += (
            f"<tr><td>{html.escape(m)}</td>"
            f"<td>{mae}</td>"
            f"<td>{st.get('psnr_db', '—')}</td>"
            f"<td>{st.get('pixels', 0):,}</td></tr>"
        )

    cfg = data.get("config", {})
    page = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>Stratified analytics</title>
<style>
  body {{ font-family: system-ui,sans-serif; background:#121212; color:#eee; margin:0; padding:20px; }}
  h1 {{ text-align:center; }}
  table {{ border-collapse:collapse; margin:12px 0; width:100%; max-width:1200px; }}
  td,th {{ border:1px solid #444; padding:8px; text-align:center; font-size:13px; }}
  th {{ background:#2a2a2a; }}
  section {{ max-width:1200px; margin:24px auto; background:#1a1a1a; padding:16px; border-radius:8px; }}
  .hint {{ color:#888; font-size:12px; }}
  .meta {{ color:#aaa; text-align:center; }}
</style></head><body>
<h1>Ошибки по глубине и семантике</h1>
<p class="meta">samples ok={data.get('samples_ok')} fail={data.get('samples_fail')} |
device={cfg.get('device')} elapsed={cfg.get('elapsed_sec')}s</p>
<section><h2>Overall</h2>
<table><tr><th>метод</th><th>MAE</th><th>PSNR</th><th>pixels</th></tr>{overall_rows}</table></section>
{table_block("По глубине LiDAR", by_depth, methods, depth_rows)}
{table_block("По семантическим классам", by_sem, methods, sem_rows)}
</body></html>"""

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
