"""Measure the deployed Space's client-frame WebSocket pipeline.

The measured path is video decode and JPEG encode on this client, WebSocket
transfer, server JPEG decode, detector, ByteTrack, recognizer, and JSON return.
It does not include browser canvas rendering or the Space's /ws/session route.
Each result retains its Space commit, model lineage, and per-frame trace.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import platform
import statistics
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import cv2
import requests
import websockets
from huggingface_hub import HfApi


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def quantile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lo = int(position)
        return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (
            position - lo
        )

    return {
        "n": len(values),
        "mean_ms": statistics.fmean(values),
        "std_ms": statistics.pstdev(values),
        "p50_ms": quantile(0.5),
        "p95_ms": quantile(0.95),
    }


def _download_video(url: str, path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with requests.get(url, stream=True, timeout=45) as response:
        response.raise_for_status()
        with path.open("wb") as target:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    target.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
    return {"url": url, "size_bytes": size, "sha256": digest.hexdigest()}


def _space_info(base_url: str, repo_id: str, status_path: str) -> tuple[dict, dict]:
    response = requests.get(f"{base_url}{status_path}", timeout=30)
    response.raise_for_status()
    app_status = response.json()
    if not repo_id:
        return app_status, {"repo_id": None, "local_smoke_test": True}
    repo = HfApi().repo_info(repo_id, repo_type="space", files_metadata=True)
    runtime = HfApi().get_space_runtime(repo_id)
    remote_models = {}
    for sibling in repo.siblings:
        if sibling.rfilename.startswith("model_space/") and sibling.rfilename.endswith(".onnx"):
            lfs = getattr(sibling, "lfs", None)
            remote_models[sibling.rfilename] = {
                "size_bytes": sibling.size,
                "sha256": getattr(lfs, "sha256", None),
            }
    return app_status, {
        "repo_id": repo_id,
        "commit_sha": repo.sha,
        "runtime_stage": runtime.stage,
        "hardware": str(runtime.hardware),
        "models": remote_models,
    }


async def _reset_tracker(socket: websockets.ClientConnection) -> None:
    await socket.send(json.dumps({"type": "reset"}))
    reply = json.loads(await asyncio.wait_for(socket.recv(), timeout=30))
    if reply.get("type") != "ack" or reply.get("message") != "reset":
        raise RuntimeError(f"Space tracker reset was not acknowledged: {str(reply)[:300]}")


async def _run_frames(
    ws_url: str,
    cap: cv2.VideoCapture,
    variant: str,
    warmup: int,
    iterations: int,
    send_fps: float,
) -> tuple[list[dict], dict]:
    records: list[dict] = []
    measure_start = None
    measure_end = None
    frame_index = 0
    async with websockets.connect(
        ws_url, open_timeout=30, max_size=16 * 1024 * 1024, ping_interval=20
    ) as socket:
        await _reset_tracker(socket)
        start = time.perf_counter()
        for iteration in range(warmup + iterations):
            if send_fps:
                due = start + iteration / send_fps
                await asyncio.sleep(max(0.0, due - time.perf_counter()))
            decode_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_index = 0
                ok, frame = cap.read()
            if not ok:
                raise RuntimeError("sample video has no decodable frames")
            decode_end = time.perf_counter()
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise RuntimeError(f"JPEG encode failed at iteration {iteration}")
            payload = json.dumps(
                {
                    "type": "frame",
                    "data": base64.b64encode(encoded).decode("ascii"),
                    "variant": variant,
                }
            )
            send_start = time.perf_counter()
            await socket.send(payload)
            received = await asyncio.wait_for(socket.recv(), timeout=60)
            recv_end = time.perf_counter()
            message = json.loads(received)
            if message.get("type") != "result":
                raise RuntimeError(f"Space frame failed: {str(message)[:500]}")
            result = message["data"]
            if result.get("variant") != variant:
                raise RuntimeError(
                    f"requested {variant}, Space ran {result.get('variant')}; refusing mixed result"
                )
            measured = iteration >= warmup
            if measured and measure_start is None:
                measure_start = send_start
            if measured:
                measure_end = recv_end
            records.append(
                {
                    "iteration": iteration,
                    "warmup": not measured,
                    "utc": _now(),
                    "source_frame_index": frame_index,
                    "source_width": int(frame.shape[1]),
                    "source_height": int(frame.shape[0]),
                    "jpeg_bytes": len(encoded),
                    "wire_bytes": len(payload.encode("utf-8")),
                    "response_bytes": len(received.encode("utf-8")),
                    "client_decode_ms": (decode_end - decode_start) * 1000,
                    "client_jpeg_encode_ms": (send_start - decode_end) * 1000,
                    "round_trip_ms": (recv_end - send_start) * 1000,
                    "server_inference_ms": result.get("inference_ms"),
                    "server_stage_ms": result.get("stage_ms"),
                    "track_count": len(result.get("tracks", [])),
                    "frame_id": result.get("frame_id"),
                    "variant": result.get("variant"),
                }
            )
            frame_index += 1
    if measure_start is None or measure_end is None:
        raise RuntimeError("no measured frames")
    elapsed = measure_end - measure_start
    measured_records = [record for record in records if not record["warmup"]]
    keys = ("client_decode_ms", "client_jpeg_encode_ms", "round_trip_ms", "server_inference_ms")
    stats = {
        key: _summary(
            [float(record[key]) for record in measured_records if record[key] is not None]
        )
        for key in keys
    }
    stage_names = sorted(
        {name for record in measured_records for name in (record["server_stage_ms"] or {})}
    )
    stats["server_stage_ms"] = {
        name: _summary(
            [
                float(record["server_stage_ms"][name])
                for record in measured_records
                if record["server_stage_ms"] and name in record["server_stage_ms"]
            ]
        )
        for name in stage_names
    }
    return records, {
        "status": "completed",
        "traffic_mode": "closed_loop",
        "warmup_count": warmup,
        "measured_count": iterations,
        "elapsed_measured_s": elapsed,
        "observed_throughput_fps": iterations / elapsed,
        "send_fps_cap": send_fps,
        "metrics": stats,
        "mean_track_count": statistics.fmean(record["track_count"] for record in measured_records),
    }


async def _run_frames_open_loop(
    ws_url: str,
    cap: cv2.VideoCapture,
    variant: str,
    warmup: int,
    iterations: int,
    send_fps: float,
) -> tuple[list[dict], dict]:
    """Send on the browser's timer while collecting responses concurrently."""
    if send_fps <= 0:
        raise ValueError("open-loop mode requires send-fps > 0")
    total = warmup + iterations
    sent: list[dict] = []
    records: list[dict] = []
    async with websockets.connect(
        ws_url, open_timeout=30, max_size=16 * 1024 * 1024, ping_interval=20
    ) as socket:
        await _reset_tracker(socket)
        # The browser's timer starts after its WebSocket opens. Starting the
        # clock before the handshake would burst overdue frames at the server.
        begin = time.perf_counter()
        outgoing: asyncio.Queue[tuple[int, str]] = asyncio.Queue()

        async def send_frames() -> None:
            source_index = 0
            for iteration in range(total):
                await asyncio.sleep(max(0.0, begin + iteration / send_fps - time.perf_counter()))
                decode_start = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    source_index = 0
                    ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("sample video has no decodable frames")
                decode_end = time.perf_counter()
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise RuntimeError(f"JPEG encode failed at iteration {iteration}")
                payload = json.dumps(
                    {
                        "type": "frame",
                        "data": base64.b64encode(encoded).decode("ascii"),
                        "variant": variant,
                    }
                )
                send_at = time.perf_counter()
                sent.append(
                    {
                        "iteration": iteration,
                        "warmup": iteration < warmup,
                        "source_frame_index": source_index,
                        "source_width": int(frame.shape[1]),
                        "source_height": int(frame.shape[0]),
                        "jpeg_bytes": len(encoded),
                        "wire_bytes": len(payload.encode("utf-8")),
                        "client_decode_ms": (decode_end - decode_start) * 1000,
                        "client_jpeg_encode_ms": (send_at - decode_end) * 1000,
                        "send_at_ms": (send_at - begin) * 1000,
                    }
                )
                # Browser WebSocket.send() queues payloads and returns. Keep
                # the timer independent of network backpressure likewise.
                outgoing.put_nowait((iteration, payload))
                source_index += 1

        async def transmit_frames() -> None:
            for _ in range(total):
                iteration, payload = await outgoing.get()
                wire_start = time.perf_counter()
                await socket.send(payload)
                sent[iteration]["wire_send_start_ms"] = (wire_start - begin) * 1000
                sent[iteration]["wire_send_done_ms"] = (time.perf_counter() - begin) * 1000

        async def receive_frames() -> None:
            for iteration in range(total):
                received = await asyncio.wait_for(socket.recv(), timeout=90)
                recv_at = time.perf_counter()
                message = json.loads(received)
                if message.get("type") != "result":
                    raise RuntimeError(f"Space frame failed: {str(message)[:500]}")
                result = message["data"]
                if result.get("variant") != variant:
                    raise RuntimeError(
                        f"requested {variant}, Space ran {result.get('variant')}; refusing mixed result"
                    )
                outgoing = sent[iteration]
                records.append(
                    {
                        **outgoing,
                        "utc": _now(),
                        "received_at_ms": (recv_at - begin) * 1000,
                        "response_bytes": len(received.encode("utf-8")),
                        "round_trip_ms": (recv_at - begin) * 1000 - outgoing["send_at_ms"],
                        "server_inference_ms": result.get("inference_ms"),
                        "server_stage_ms": result.get("stage_ms"),
                        "track_count": len(result.get("tracks", [])),
                        "frame_id": result.get("frame_id"),
                        "variant": result.get("variant"),
                    }
                )

        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(send_frames())
            tasks.create_task(transmit_frames())
            tasks.create_task(receive_frames())
    measured = records[warmup:]
    first_send = measured[0]["send_at_ms"]
    first_recv = measured[0]["received_at_ms"]
    last_recv = measured[-1]["received_at_ms"]
    keys = ("client_decode_ms", "client_jpeg_encode_ms", "round_trip_ms", "server_inference_ms")
    stats = {
        key: _summary([float(record[key]) for record in measured if record[key] is not None])
        for key in keys
    }
    stage_names = sorted(
        {name for record in measured for name in (record["server_stage_ms"] or {})}
    )
    stats["server_stage_ms"] = {
        name: _summary(
            [
                float(record["server_stage_ms"][name])
                for record in measured
                if record["server_stage_ms"] and name in record["server_stage_ms"]
            ]
        )
        for name in stage_names
    }
    return records, {
        "status": "completed",
        "traffic_mode": "open_loop",
        "warmup_count": warmup,
        "measured_count": iterations,
        "send_fps_cap": send_fps,
        "observed_send_interval_ms": _summary(
            [
                measured[index]["send_at_ms"] - measured[index - 1]["send_at_ms"]
                for index in range(1, len(measured))
            ]
        ),
        "elapsed_first_send_to_last_result_s": (last_recv - first_send) / 1000,
        "end_to_end_completion_fps": iterations * 1000 / (last_recv - first_send),
        "result_delivery_fps": (iterations - 1) * 1000 / (last_recv - first_recv),
        "metrics": stats,
        "mean_track_count": statistics.fmean(record["track_count"] for record in measured),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://gyann-edge-sign.hf.space")
    parser.add_argument("--repo-id", default="gyann/edge-sign")
    parser.add_argument("--sample-video-url")
    parser.add_argument("--variant", choices=("fp32", "int8", "head_excluded_qdq"), required=True)
    parser.add_argument("--status-path", default="/api/status")
    parser.add_argument("--ws-path", default="/ws/stream")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--send-fps", type=float, default=10.0)
    parser.add_argument("--traffic-mode", choices=("open_loop", "closed_loop"), default="open_loop")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 2 or args.send_fps < 0:
        parser.error("warmup >= 0, iterations >= 2 and send-fps >= 0 are required")
    if args.traffic_mode == "open_loop" and args.send_fps == 0:
        parser.error("open-loop mode requires send-fps > 0")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    base_url = args.base_url.rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        parser.error("base-url must be an HTTP(S) URL")
    if not args.status_path.startswith("/api/") or not args.ws_path.startswith("/ws/"):
        parser.error("status-path must start /api/ and ws-path must start /ws/")
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed.netloc}{args.ws_path}"
    video_url = args.sample_video_url or f"{base_url}/detection/sample/seoul_daylight.mp4"
    app_status, remote = _space_info(base_url, args.repo_id, args.status_path)
    if args.status_path == "/api/status":
        ready = bool(app_status.get("pipeline"))
        variants = {item["name"] for item in app_status.get("variants", [])}
    else:
        ready = app_status.get("status") == "ready"
        variants = set(app_status.get("models", {})) - {"recognizer"}
    if not ready:
        raise RuntimeError(f"Space reports measurement route unavailable: {app_status}")
    if args.variant not in variants:
        raise RuntimeError(f"variant {args.variant} not deployed")
    with tempfile.TemporaryDirectory(prefix="edge_sign_hf_benchmark_") as tmp:
        video_path = Path(tmp) / "sample.mp4"
        sample = _download_video(video_url, video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("sample video cannot be opened")
        sample["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample["source_fps"] = float(cap.get(cv2.CAP_PROP_FPS))
        try:
            runner = _run_frames_open_loop if args.traffic_mode == "open_loop" else _run_frames
            records, metrics = asyncio.run(
                runner(ws_url, cap, args.variant, args.warmup, args.iterations, args.send_fps)
            )
        finally:
            cap.release()
    args.output.mkdir(parents=True)
    config = {
        "run_utc": _now(),
        "base_url": base_url,
        "websocket_url": ws_url,
        "route": args.ws_path,
        "status_route": args.status_path,
        "variant": args.variant,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "send_fps": args.send_fps,
        "traffic_mode": args.traffic_mode,
        "tracker_reset_at_start": True,
        "scope": "client video decode/JPEG encode + network round trip + server JPEG decode, "
        "detector, ByteTrack, recognizer + JSON response; browser rendering excluded",
        "sample": sample,
        "app_status_before": app_status,
        "remote": remote,
        "client": {
            "platform": platform.platform(),
            "python": sys.version,
            "opencv": cv2.__version__,
            "websockets": websockets.__version__,
        },
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "trace.jsonl").open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output / "command.txt").write_text(
        "python scripts/paper/benchmark_hf_space.py "
        f"--variant {args.variant} --warmup {args.warmup} --iterations {args.iterations} "
        f"--status-path {args.status_path} --ws-path {args.ws_path} "
        f"--send-fps {args.send_fps} --traffic-mode {args.traffic_mode} "
        f"--output {args.output.as_posix()}\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), **metrics}, indent=2))


if __name__ == "__main__":
    main()
