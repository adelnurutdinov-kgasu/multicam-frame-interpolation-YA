"""Rebuild composer.html with embedded JSON (works via file://)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "methods_gallery/_ego_manual_masks"
DATA_JSON = ROOT / "browser_gallery/composer_data.json"
HTML_PATH = ROOT / "browser_gallery/composer.html"

JS = r"""
let selectedCandId = null;
let savedSelections = {};
let cv = null;
let ctx = null;
let renderSeq = 0;
const cache = new Map();
const overlayCache = new Map();

async function loadImg(rel) {
  if (cache.has(rel)) return cache.get(rel);
  const p = new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => { cache.set(rel, im); res(im); };
    im.onerror = () => rej(new Error('не загрузилось: ' + rel));
    im.src = rel;
  });
  return p;
}

async function loadSelections() {
  try {
    const r = await fetch('/api/selections');
    if (!r.ok) return;
    const d = await r.json();
    savedSelections = d.selections || d || {};
  } catch (_) {}
}

function setSaveStatus(msg, ok) {
  const el = document.getElementById('saveStatus');
  if (!el) return;
  el.textContent = msg || '';
  el.style.color = ok ? '#8d8' : '#f88';
}

function getCurrentFrame(g) {
  if (!g) return null;
  const rel = document.getElementById('frameSel').value;
  return visibleFrames(g).find(fr => fr.rel === rel) || g.frames.find(fr => fr.rel === rel) || null;
}

function selectionKey(g, frame) {
  return g.id + '|' + (frame ? frame.id : '__unknown__');
}

function getSavedSelection(g, frame) {
  if (!g || !frame) return null;
  const key = selectionKey(g, frame);
  if (savedSelections[key]) return savedSelections[key];
  const legacy = savedSelections[g.id];
  if (legacy && !legacy.frame_id) return legacy;
  return null;
}

function countSavedInGroup(g) {
  const prefix = g.id + '|';
  return Object.keys(savedSelections).filter(k => k.startsWith(prefix)).length;
}

function visibleFrames(g) {
  const onlyMean = document.getElementById('onlyMean').checked;
  return g.frames.filter(fr => !onlyMean || fr.in_mean);
}

function getGroup() {
  return DATA.groups.find(g => g.id === document.getElementById('groupSel').value);
}

function matchedGroups() {
  const q = (document.getElementById('groupFilter').value || '').toLowerCase();
  return DATA.groups.filter(g => {
    if (!q) return true;
    return `${g.id} ${g.vehicle} ${g.camera} ${g.status}`.toLowerCase().includes(q);
  });
}

function fillGroups() {
  const sel = document.getElementById('groupSel');
  const prev = sel.value;
  sel.innerHTML = '';
  const matched = matchedGroups();
  matched.forEach(g => {
    const o = document.createElement('option');
    o.value = g.id;
    const nSaved = countSavedInGroup(g);
    const legacy = savedSelections[g.id] && !savedSelections[g.id].frame_id;
    const mark = (nSaved || legacy) ? ` ✓${nSaved || 1}` : '';
    o.textContent = `${g.vehicle} / ${g.camera}  [${g.status}]  (${g.candidates.length})${mark}`;
    sel.appendChild(o);
  });
  if (!matched.length) return;
  sel.value = (matched.find(g => g.id === prev) || matched[0]).id;
}

function fillFrames(g, keepRel) {
  const sel = document.getElementById('frameSel');
  const frames = visibleFrames(g);
  const prev = keepRel || sel.value;
  sel.innerHTML = '';
  frames.forEach(fr => {
    const o = document.createElement('option');
    o.value = fr.rel;
    const saved = getSavedSelection(g, fr);
    o.textContent = fr.label + (fr.in_mean ? ' · mean' : '') + (saved ? ' ✓' : '');
    sel.appendChild(o);
  });
  if (!frames.length) return;
  sel.value = (frames.find(fr => fr.rel === prev) || frames[0]).rel;
}

function defaultCandidateId(g, items, frame) {
  const saved = getSavedSelection(g, frame);
  if (saved && saved.candidate_id && items.find(c => c.id === saved.candidate_id)) {
    return saved.candidate_id;
  }
  return (items.find(c => c.tag === 'own') || items[0]).id;
}

function fillCandidates(g, resetSelection) {
  const list = document.getElementById('candList');
  const q = document.getElementById('candFilter').value.toLowerCase();
  const frame = getCurrentFrame(g);
  list.innerHTML = '';
  const items = g.candidates.filter(c => !q || c.label.toLowerCase().includes(q) || c.id.includes(q));
  if (!items.length) { list.innerHTML = '<p class="sub">Нет кандидатов</p>'; return; }
  if (resetSelection || !selectedCandId || !items.find(c => c.id === selectedCandId)) {
    selectedCandId = defaultCandidateId(g, items, frame);
  }
  items.forEach(c => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'cand' + (c.id === selectedCandId ? ' active' : '');
    btn.innerHTML = `<img src="${c.rel}" alt=""><div class="txt"><b>${c.label}</b><div class="pct">${(c.mask_pct*100).toFixed(1)}% px</div></div>`;
    btn.onclick = () => {
      selectedCandId = c.id;
      fillCandidates(g, false);
      render().catch(e => showErr(e));
    };
    list.appendChild(btn);
  });
}

function showErr(e) {
  document.getElementById('info').textContent = 'Ошибка: ' + (e && e.message ? e.message : e);
}

async function drawMaskRedOverlay(maskRel, fw, fh, alpha) {
  const key = maskRel + '|' + fw + 'x' + fh + '|' + alpha;
  if (overlayCache.has(key)) {
    ctx.drawImage(overlayCache.get(key), 0, 0);
    return;
  }
  const maskIm = await loadImg(maskRel);
  const off = document.createElement('canvas');
  off.width = fw;
  off.height = fh;
  const octx = off.getContext('2d');
  octx.drawImage(maskIm, 0, 0, fw, fh);
  let src;
  try {
    src = octx.getImageData(0, 0, fw, fh);
  } catch (e) {
    throw new Error('маска: откройте через python serve_mask_picker.py (не file://)');
  }
  const imgData = octx.createImageData(fw, fh);
  const s = src.data;
  const d = imgData.data;
  const a = Math.round(Math.max(0, Math.min(1, alpha)) * 255);
  for (let i = 0; i < fw * fh; i++) {
    const p = i * 4;
    const lum = Math.max(s[p], s[p + 1], s[p + 2]);
    if (lum <= 127) continue;
    d[p] = 255;
    d[p + 1] = 48;
    d[p + 2] = 48;
    d[p + 3] = a;
  }
  octx.clearRect(0, 0, fw, fh);
  octx.putImageData(imgData, 0, 0);
  overlayCache.set(key, off);
  ctx.drawImage(off, 0, 0);
}

async function render() {
  const seq = ++renderSeq;
  const g = getGroup();
  if (!g || !g.frames.length) {
    showErr('нет кадров для группы');
    return;
  }
  const frame = getCurrentFrame(g);
  const cand = g.candidates.find(c => c.id === selectedCandId);
  const frameRel = document.getElementById('frameSel').value;
  if (!frameRel) {
    showErr('кадр не выбран');
    return;
  }
  if (!frame) {
    showErr('кадр не найден');
    return;
  }
  if (!cand) {
    showErr('маска не выбрана');
    return;
  }

  const frameIm = await loadImg(frameRel);
  if (seq !== renderSeq) return;
  const fw = frameIm.naturalWidth || g.w;
  const fh = frameIm.naturalHeight || g.h;
  cv.width = fw;
  cv.height = fh;
  ctx.clearRect(0, 0, fw, fh);
  ctx.drawImage(frameIm, 0, 0, fw, fh);
  await drawMaskRedOverlay(cand.rel, fw, fh, 0.52);
  if (seq !== renderSeq) return;

  const saved = getSavedSelection(g, frame);
  const savedMark = saved && saved.candidate_id === cand.id ? '\n✓ выбор записан для этого кадра' : '';
  document.getElementById('info').textContent =
    `Группа: ${g.id}\nКадр: ${frame.label}\nМаска: ${cand.label}\nПокрытие: ${(cand.mask_pct * 100).toFixed(1)}%\nКадров: ${visibleFrames(g).length}${savedMark}\n${g.note || ''}`;
}

async function saveSelection() {
  const g = getGroup();
  const frame = getCurrentFrame(g);
  const cand = g && g.candidates.find(c => c.id === selectedCandId);
  if (!g || !frame || !cand) {
    setSaveStatus('Сначала выберите группу, кадр и маску', false);
    return;
  }
  const key = selectionKey(g, frame);
  const row = {
    group: g.id,
    frame_id: frame.id,
    frame_rel: frame.rel,
    frame_label: frame.label,
    candidate_id: cand.id,
    label: cand.label,
    mask_rel: cand.rel,
    source: cand.source,
    tag: cand.tag,
    mask_pct: cand.mask_pct,
  };
  try {
    let selections = { ...savedSelections };
    try {
      const r = await fetch('/api/selections');
      if (r.ok) {
        const d = await r.json();
        selections = d.selections || d || {};
      }
    } catch (_) {}
    selections[key] = row;
    const resp = await fetch('/api/selections', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ selections }),
    });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    savedSelections = selections;
    fillGroups();
    fillFrames(g, frame.rel);
    setSaveStatus('Записано: ' + frame.label + ' → ' + cand.label, true);
    render().catch(e => showErr(e));
  } catch (e) {
    setSaveStatus('Запустите: python serve_mask_picker.py', false);
    showErr('Запись только через сервер: python serve_mask_picker.py\n' + (e.message || e));
  }
}

function stepFrame(delta) {
  const g = getGroup();
  const sel = document.getElementById('frameSel');
  const frames = visibleFrames(g);
  const idx = frames.findIndex(f => f.rel === sel.value);
  const next = frames[(idx + delta + frames.length) % frames.length];
  if (next) { sel.value = next.rel; onFrameChange(); }
}

function onFrameChange() {
  selectedCandId = null;
  const g = getGroup();
  if (!g) return;
  fillCandidates(g, true);
  render().catch(e => showErr(e));
}

function stepGroup(delta) {
  const sel = document.getElementById('groupSel');
  const groups = matchedGroups();
  if (!groups.length) return;
  const idx = groups.findIndex(g => g.id === sel.value);
  const next = groups[(idx + delta + groups.length) % groups.length];
  if (next) {
    sel.value = next.id;
    onGroupChange();
  }
}

function onGroupChange() {
  selectedCandId = null;
  const g = getGroup();
  if (!g) return;
  fillFrames(g);
  fillCandidates(g, true);
  render().catch(e => showErr(e));
}

async function init() {
  cv = document.getElementById('cv');
  ctx = cv.getContext('2d');
  fillGroups();
  document.getElementById('groupFilter').addEventListener('input', () => { fillGroups(); onGroupChange(); });
  document.getElementById('groupSel').addEventListener('change', onGroupChange);
  document.getElementById('frameSel').addEventListener('change', onFrameChange);
  document.getElementById('onlyMean').addEventListener('change', () => {
    const g = getGroup();
    fillFrames(g, document.getElementById('frameSel').value);
    onFrameChange();
  });
  document.getElementById('candFilter').addEventListener('input', () => fillCandidates(getGroup(), false));
  document.getElementById('prevGroup').addEventListener('click', () => stepGroup(-1));
  document.getElementById('nextGroup').addEventListener('click', () => stepGroup(1));
  document.getElementById('prevFrame').addEventListener('click', () => stepFrame(-1));
  document.getElementById('nextFrame').addEventListener('click', () => stepFrame(1));
  document.getElementById('btnSave').addEventListener('click', () => saveSelection().catch(e => showErr(e)));
  await loadSelections();
  onGroupChange();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
"""


def build_html(payload: dict) -> str:
    n = payload.get("n_groups", len(payload.get("groups", [])))
    data_blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    css = """
:root { --bg:#0f0f0f; --panel:#1a1a1a; --border:#333; --text:#eee; --muted:#888; --accent:#5a9; --sel:#264; }
* { box-sizing:border-box; }
body { margin:0; font-family:system-ui,sans-serif; background:var(--bg); color:var(--text); }
.layout { display:grid; grid-template-columns:300px 1fr 340px; min-height:100vh; }
aside, .candidates { background:var(--panel); padding:12px; overflow-y:auto; max-height:100vh; }
aside { border-right:1px solid var(--border); position:sticky; top:0; }
.candidates { border-left:1px solid var(--border); position:sticky; top:0; }
main { padding:12px; min-width:0; }
h1 { font-size:1.05rem; margin:0 0 6px; }
.sub { font-size:0.8rem; color:var(--muted); margin:0 0 10px; line-height:1.4; }
label { display:block; font-size:0.78rem; color:var(--muted); margin-top:8px; }
select, input[type=text] { width:100%; padding:6px; border-radius:6px; border:1px solid var(--border); background:#222; color:var(--text); }
select[size] { min-height:120px; }
.frame-nav { display:flex; gap:6px; margin-top:6px; }
.frame-nav button { flex:1; padding:6px; border-radius:6px; border:1px solid var(--border); background:#333; color:#fff; cursor:pointer; }
.canvas-wrap { background:#000; border-radius:8px; overflow:hidden; border:1px solid var(--border); }
canvas { width:100%; height:auto; display:block; }
#info { font-size:0.8rem; color:var(--muted); margin-top:8px; white-space:pre-wrap; line-height:1.45; }
.cand-search { margin-bottom:8px; }
.cand-list { display:flex; flex-direction:column; gap:6px; max-height:calc(100vh - 200px); overflow-y:auto; }
.cand { display:flex; gap:8px; align-items:center; padding:8px; border:1px solid var(--border); border-radius:8px; cursor:pointer; background:#222; text-align:left; width:100%; color:var(--text); }
.cand:hover { border-color:#555; }
.cand.active { border-color:var(--accent); background:var(--sel); }
.cand img { width:72px; height:40px; object-fit:contain; background:#000; border-radius:4px; flex-shrink:0; }
.cand .txt { font-size:0.78rem; line-height:1.35; }
.cand .pct { color:var(--muted); font-size:0.72rem; }
.save-bar { position:sticky; bottom:0; background:var(--panel); padding:10px 0 0; border-top:1px solid var(--border); margin-top:10px; }
.save-bar button { width:100%; padding:10px; border-radius:8px; border:none; background:#2d6a4f; color:#fff; font-size:0.9rem; cursor:pointer; }
#saveStatus { font-size:0.78rem; margin-top:8px; min-height:1.2em; line-height:1.35; }
.toplink a { color:var(--accent); font-size:0.82rem; }
.server-hint { font-size:0.75rem; color:#b88; margin-top:6px; line-height:1.35; }
"""
    body = f"""
<div class="layout">
<aside>
<div class="toplink"><a href="index.html">← Галерея оверлеев</a></div>
<h1>Выбор маски</h1>
<p class="sub">«Записать выбор» сохраняет маску для <b>текущего кадра</b> (группа + фото) в mask_picker_selections.json</p>
<p class="server-hint">Открывайте через <b>python serve_mask_picker.py</b> → http://127.0.0.1:8765/composer.html</p>
<label>Поиск группы<input type="text" id="groupFilter" placeholder="crozby, left_fwd…"></label>
<label>Группа ({n})<select id="groupSel" size="8"></select></label>
<div class="frame-nav"><button type="button" id="prevGroup" title="Предыдущая группа">← группа</button><button type="button" id="nextGroup" title="Следующая группа">группа →</button></div>
<label>Кадр<select id="frameSel" size="6"></select></label>
<div class="frame-nav"><button type="button" id="prevFrame" title="Предыдущий кадр">← кадр</button><button type="button" id="nextFrame" title="Следующий кадр">кадр →</button></div>
<label><input type="checkbox" id="onlyMean"> только кадры из mean</label>
<div id="info"></div>
</aside>
<main><div class="canvas-wrap"><canvas id="cv"></canvas></div></main>
<div class="candidates">
<h1 style="font-size:1rem">Кандидаты (та же camera)</h1>
<input type="text" id="candFilter" class="cand-search" placeholder="badat, v2…">
<div class="cand-list" id="candList"></div>
<div class="save-bar">
<button type="button" id="btnSave">Записать выбор</button>
<div id="saveStatus"></div>
</div>
</div>
</div>
"""
    return (
        "<!DOCTYPE html>\n<html lang=\"ru\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>Выбор маски — ego masks</title>"
        f"<style>{css}</style></head><body>{body}"
        f"<script>\nconst DATA = {data_blob};\n{JS}\n</script></body></html>"
    )


def main() -> int:
    payload = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    HTML_PATH.write_text(build_html(payload), encoding="utf-8")
    print(f"Rebuilt {HTML_PATH}  groups={payload.get('n_groups')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
