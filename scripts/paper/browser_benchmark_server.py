"""Serve local test assets and collect ORT Web browser benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.collect_environment import sha256

MODELS = {
    "fp32": "model_space/yolo_v4_signs_fp32.onnx",
    "fp16": "model_space/yolo_v4_signs_fp16.onnx",
    "full_qdq": "model_space/yolo_v4_signs_int8_static.onnx",
    "head_excluded_qdq": "model_space/yolo_v4_signs_int8_head_excluded.onnx",
}
HTML = Path(__file__).with_name("browser_benchmark.html")


def handler_factory(artifact_root: Path, frame: Path, output: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlparse(self.path)
            if request.path == "/browser_benchmark.html":
                file, mime = HTML, "text/html; charset=utf-8"
            elif request.path == "/frame.jpg":
                file, mime = frame, "image/jpeg"
            elif request.path == "/model":
                name = parse_qs(request.query).get("name", [None])[0]
                if name not in MODELS:
                    self.send_error(404, "unknown model")
                    return
                file, mime = artifact_root / MODELS[name], "application/octet-stream"
            else:
                self.send_error(404)
                return
            if not file.is_file():
                self.send_error(404, "asset missing")
                return
            content = file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            if request.path == "/model":
                self.send_header("X-Model-SHA256", sha256(file))
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            if urlparse(self.path).path != "/result":
                self.send_error(404)
                return
            if int(self.headers.get("Content-Length", "0")) > 5_000_000:
                self.send_error(413)
                return
            try:
                result = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                name, ep = result["model"], result["ep"]
                if name not in MODELS or ep not in ("wasm", "webgpu"):
                    raise ValueError("invalid benchmark identity")
                target = output / f"{ep}_{name}.json"
                if target.exists():
                    raise FileExistsError(f"benchmark result already exists: {target}")
                target.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                trace = result.get("trace", [])
                if trace:
                    with (output / f"{ep}_{name}_trace.csv").open(
                        "w", newline="", encoding="utf-8"
                    ) as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
                        writer.writeheader()
                        writer.writerows(trace)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            except (ValueError, KeyError, FileExistsError, json.JSONDecodeError) as exc:
                self.send_error(400, str(exc))

        def log_message(self, format, *args):
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("paper_evidence/splits/test_manifest.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("paper_evidence/runtime/browser"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    root = args.artifact_root.resolve(strict=True)
    first = json.loads(args.manifest.read_text(encoding="utf-8").splitlines()[0])
    frame = root / first["image_rel"]
    if not frame.is_file():
        raise FileNotFoundError(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), handler_factory(root, frame, args.output)
    )
    print(f"browser benchmark server on http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
