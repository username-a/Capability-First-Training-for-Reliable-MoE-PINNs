"""Serve the local APINN training dashboard without extra dependencies."""

from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
STATIC_DIR = PACKAGE_ROOT / "dashboard" / "apinn"


class DashboardHandler(SimpleHTTPRequestHandler):
    result_dir: Path

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/status":
            status_file = self.result_dir / "status.json"
            if not status_file.exists():
                self._json({"state": "pending", "message": "waiting for status.json"}, 404)
                return
            try:
                self._json(json.loads(status_file.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                self._json({"state": "pending", "message": str(exc)}, 503)
            return
        if path == "/api/history":
            history_file = self.result_dir / "history.jsonl"
            rows = []
            if history_file.exists():
                for line in history_file.read_text(encoding="utf-8").splitlines():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            self._json(rows)
            return
        if path == "/api/summary":
            summary_file = self.result_dir.parent / "apinn_multiseed_summary.json"
            if not summary_file.exists():
                self._json({"runs": []})
                return
            try:
                payload = json.loads(summary_file.read_text(encoding="utf-8"))
                official_file = self.result_dir.parent / "apinn_official2_multiseed_summary.json"
                if official_file.exists():
                    payload["official2"] = json.loads(official_file.read_text(encoding="utf-8"))
                matched2_file = self.result_dir.parent / "apinn_matched2_multiseed_summary.json"
                if matched2_file.exists():
                    payload["matched2"] = json.loads(matched2_file.read_text(encoding="utf-8"))
                self._json(payload)
            except (OSError, json.JSONDecodeError) as exc:
                self._json({"runs": [], "message": str(exc)}, 503)
            return
        if path == "/api/health":
            self._json({"ok": True, "result_dir": str(self.result_dir)})
            return
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default=str(PACKAGE_ROOT / "results" / "apinn_reproduction" / "latest"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    DashboardHandler.result_dir = Path(args.result_dir).resolve()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"APINN dashboard: http://{args.host}:{args.port}")
    print(f"Reading results from: {DashboardHandler.result_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
