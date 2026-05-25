"""
Local server for mask picker UI (fixes file:// fetch/canvas issues).

  python serve_mask_picker.py

Open: http://127.0.0.1:8765/composer.html

POST /api/selections  — save mask_picker_selections.json
GET  /api/selections  — load current selections
"""

from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent
GALLERY = REPO / "methods_gallery/_ego_manual_masks/browser_gallery"
SELECTIONS = REPO / "methods_gallery/_ego_manual_masks/mask_picker_selections.json"
PORT = 8765


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(GALLERY), **kwargs)

    def log_message(self, fmt, *args) -> None:
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def do_GET(self) -> None:
        if self.path == "/api/selections":
            data = {}
            if SELECTIONS.is_file():
                data = json.loads(SELECTIONS.read_text(encoding="utf-8"))
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/selections":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "invalid json")
            return
        SELECTIONS.parent.mkdir(parents=True, exist_ok=True)
        SELECTIONS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    if not GALLERY.is_dir():
        print(f"Missing gallery dir: {GALLERY}")
        return 1
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Mask picker: http://127.0.0.1:{PORT}/composer.html")
    print(f"Selections:  {SELECTIONS}")
    print("Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
