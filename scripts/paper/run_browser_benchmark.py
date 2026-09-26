"""Run the local ORT Web benchmark page in an isolated headless Chrome profile.

Start browser_benchmark_server.py first. Each model/EP combination uses a new
profile to measure a cold model download; profile files are removed afterward.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import psutil

MODELS = ("fp32", "fp16", "full_qdq", "head_excluded_qdq")
EPS = ("wasm", "webgpu")


def _terminate_tree(pid: int) -> None:
    try:
        process = psutil.Process(pid)
        children = process.children(recursive=True)
        for child in children:
            child.terminate()
        process.terminate()
        _, alive = psutil.wait_procs(children + [process], timeout=5)
        for remaining in alive:
            remaining.kill()
    except psutil.NoSuchProcess:
        pass


def run(
    browser: Path,
    server_url: str,
    output: Path,
    *,
    models: tuple[str, ...],
    eps: tuple[str, ...],
    warmup: int,
    iterations: int,
    timeout: int,
) -> None:
    if not browser.is_file():
        raise FileNotFoundError(browser)
    if warmup < 1 or iterations < 2 or timeout < 1:
        raise ValueError("warmup>=1, iterations>=2, timeout>=1 required")
    with urlopen(f"{server_url}/browser_benchmark.html", timeout=5) as response:
        if response.status != 200:
            raise RuntimeError("benchmark server is unavailable")
    output.mkdir(parents=True, exist_ok=True)
    temp_base = Path(tempfile.gettempdir()).resolve()
    for ep in eps:
        for model in models:
            target = output / f"{ep}_{model}.json"
            if target.exists():
                raise FileExistsError(f"refusing to overwrite browser result: {target}")
            profile = Path(tempfile.mkdtemp(prefix="edge_sign_tiis_chrome_")).resolve()
            if not profile.is_relative_to(temp_base):
                raise RuntimeError(f"temporary browser profile is outside temp: {profile}")
            query = urlencode(
                {"model": model, "ep": ep, "warmup": warmup, "iterations": iterations}
            )
            url = f"{server_url}/browser_benchmark.html?{query}"
            command = [
                str(browser),
                "--headless=new",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-networking",
                "--enable-unsafe-webgpu",
                f"--user-data-dir={profile}",
                url,
            ]
            process = None
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline and not target.is_file():
                    time.sleep(0.25)
                if not target.is_file():
                    target.write_text(
                        json.dumps(
                            {
                                "status": "unsupported",
                                "model": model,
                                "ep": ep,
                                "reason": f"browser page did not submit a result within {timeout}s",
                                "browser": str(browser),
                                "headless": True,
                            },
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                status = json.loads(target.read_text(encoding="utf-8"))["status"]
                print(f"{ep}_{model}: {status}", flush=True)
            finally:
                if process is not None:
                    _terminate_tree(process.pid)
                if profile.is_relative_to(temp_base) and profile.name.startswith(
                    "edge_sign_tiis_chrome_"
                ):
                    try:
                        shutil.rmtree(profile)
                    except OSError:
                        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, default=Path("paper_evidence/runtime/browser"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--eps", nargs="+", choices=EPS, default=list(EPS))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    run(
        args.browser,
        args.server_url,
        args.output,
        models=tuple(args.models),
        eps=tuple(args.eps),
        warmup=args.warmup,
        iterations=args.iterations,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
