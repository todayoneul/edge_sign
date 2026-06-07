"""
Edge-Sign v2 FastAPI 백엔드 서버

엔드포인트:
  GET  /                  → /detection/ 리다이렉트
  GET  /detection/{file}  → web_modern/dist/ 정적 파일 (React 빌드)
  GET  /detection/ocr/    → Phase 1 한글 OCR 캔버스 데모 (public/ocr → dist/ocr)
  WS   /ws/stream         → 프레임 수신 → 파이프라인 → JSON 전송
  POST /api/qa            → context + question → Groq 스트리밍 답변 (SSE)
  GET  /api/status        → 파이프라인 상태

실행:
  uvicorn src.pipeline.app:app --reload --port 8000
  브라우저 → http://localhost:8000/detection/
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# onnxruntime-gpu가 torch cu128 동봉 CUDA/cuDNN DLL을 찾도록 등록 (GPU 추론).
# 반드시 onnxruntime(=e2e_pipeline) import 전에 실행돼야 CUDAExecutionProvider가 활성화된다.
import os as _os  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

try:
    import torch as _torch

    _tlib = _Path(_torch.__file__).parent / "lib"
    if _tlib.exists():
        _os.add_dll_directory(str(_tlib))
except Exception:
    pass  # torch 없거나 CPU 환경이면 CPU 폴백

from src.pipeline.e2e_pipeline import EdgeSignPipeline  # noqa: E402
from src.pipeline.logging_config import configure_logging  # noqa: E402
from src.pipeline.qa_bridge import ask_stream, build_context  # noqa: E402
from src.pipeline.session import SessionManager, save_upload  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────


# v3 (2026-05-30) — 신호등 분리 검출기 + 한국 표지판/신호등 분류기
#   검출기: 0=traffic_sign, 1=traffic_light (data/yolo_signs_v2 학습)
#   분류기: korean_sign_net 14클래스 (속도제한/규제/지시/주의 + 신호등 색상)
# 검출기 v3 ONNX(fp32/w8a8)가 있으면 우선 사용, 없으면 v2(w8a8)로 폴백.
# 택소노미는 로드된 검출기에 맞춰야 라우팅이 정확하다 (v2 간판↔v3 신호등 혼동 방지).
# 검출기 variant — 양자화 A/B 토글(FP32⇄INT8 static)용으로 여러 개 로드.
#   v3(신호등 분리)가 있으면 우선. int8_static(QDQ)은 CPU EP 실행 가능(동적 INT8은 미지원).
#   int8 파일이 아직 없으면 fp32 단일 variant로 동작(토글 옵션 1개).
def _resolve_variants() -> tuple[dict[str, str], str]:
    ms = ROOT / "model_space"
    v3_fp32 = ms / "yolov8s_signs_v3_fp32.onnx"
    if v3_fp32.exists():
        variants = {"fp32": str(v3_fp32)}
        v3_int8 = ms / "yolov8s_signs_v3_int8_static.onnx"
        if v3_int8.exists():
            variants["int8"] = str(v3_int8)
        return variants, "v3"  # 0=traffic_sign, 1=traffic_light (data/yolo_signs_v2, nc=2)
    # v2 폴백 (신호등 미분리)
    return {"w8a8": str(ms / "yolov8s_signs_w8a8.onnx")}, "v2"


YOLO_VARIANTS, DET_TAXONOMY = _resolve_variants()
YOLO_ONNX = next(iter(YOLO_VARIANTS.values()), "")  # status 표시용 대표 경로
OCR_ONNX = str(ROOT / "model_space" / "korean_ocr_net_w8a8.onnx")
# 분류기는 FP32 사용 (114KB로 작음 + 동적 INT8은 CPU EP ConvInteger 미지원)
TSIGN_ONNX = str(ROOT / "model_space" / "korean_sign_net_fp32.onnx")

# 서버 스트림(코덱 폴백 경로) 튜닝 — CPU 전용(HF Space)에서 "볼 만한" 재생을 위한 노브.
#   비호환 코덱(MPEG-4/HEVC)·URL·이미지는 서버 디코딩이 유일한 길이고, HF는 CPU 전용이라
#   fp32 풀해상도로는 끊긴다. 아래 3개로 완화: ① INT8 기본 ② 송출 다운스케일 ③ 실시간 스킵.
_CPU_ONLY = _os.environ.get("EDGE_SIGN_CPU_ONLY", "") not in ("", "0", "false", "False")
# 송출 목표 FPS(상한) — 실시간 추종 프레임 스킵의 기준.
STREAM_FPS = float(_os.environ.get("EDGE_SIGN_STREAM_FPS", "30") or "30")
# 송출 프레임 폭 상한(px) — CPU JPEG 인코딩·대역폭 절감. 검출은 내부 640 리사이즈라 영향 미미.
#   CPU 전용이면 기본 960, GPU/로컬이면 0(비활성).
STREAM_MAX_W = int(_os.environ.get("EDGE_SIGN_STREAM_MAX_W", "960" if _CPU_ONLY else "0") or "0")
# 프론트 정적 서빙 — React 빌드(web_modern/dist)만 서빙.
# dist는 `cd web_modern && npm run build`(또는 Docker 빌더 스테이지)로 생성한다.
# Phase 1 OCR 캔버스 데모는 web_modern/public/ocr → dist/ocr 로 함께 번들되어
# /detection/ocr/ 에서 체험 가능. (구 web/ 디렉토리는 web_modern으로 통합·제거됨)
WEB_DIST = ROOT / "web_modern" / "dist"

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI 앱 + 파이프라인 초기화
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Edge-Sign v2 데모", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 캐시 정책 — SPA 진입 HTML은 항상 재검증(no-cache)해 배포 즉시 반영되게 한다.
#   (index.html에 캐시가 걸리면 옛 번들 해시를 계속 가리켜 "배포해도 옛 화면"이 됨.
#    HF Space 래퍼 iframe에서 특히 두드러짐.) 반대로 해시 박힌 정적 자산은 영구 캐시 —
#   파일명이 빌드마다 바뀌므로 immutable 캐시가 안전하고 재방문 로딩도 빨라진다.
@app.middleware("http")
async def _cache_control(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/detection/assets/"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    elif path == "/" or path.startswith("/detection"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# 정적 파일 — /detection/ 은 React 빌드(web_modern/dist)를 서빙.
# dist 부재 시(빌드 전) 안내 로그만 남기고 마운트 생략 → API/WS는 정상 동작.
if WEB_DIST.exists():
    app.mount("/detection", StaticFiles(directory=str(WEB_DIST), html=True), name="detection")
else:
    logger.warning(
        "web_modern/dist 없음 — 프론트 미서빙. `cd web_modern && npm run build` 후 재기동하세요."
    )

# ONNX 모델 정적 서빙(읽기전용) — 브라우저 온디바이스 추론 타당성 스파이크(/detection/spike/)가
# ORT-Web으로 model_space/*.onnx 를 직접 fetch하기 위함. 서버 추론 경로와 무관.
_MODEL_DIR = ROOT / "model_space"
if _MODEL_DIR.exists():
    app.mount("/models", StaticFiles(directory=str(_MODEL_DIR)), name="models")

# 파이프라인 (전역 단일 인스턴스)
pipeline: EdgeSignPipeline | None = None


@app.on_event("startup")
async def startup() -> None:
    global pipeline
    configure_logging()
    pipeline = EdgeSignPipeline(
        yolo_variants=YOLO_VARIANTS,
        ocr_onnx=OCR_ONNX,
        tsign_onnx=TSIGN_ONNX,
        conf_thres=0.15,
        det_taxonomy=DET_TAXONOMY,
    )
    logger.info(f"파이프라인 초기화 완료 (택소노미={DET_TAXONOMY}, variant={list(YOLO_VARIANTS)})")
    # CPU 전용(HF Space) 기본 검출기는 INT8 — CPU에서 ~2.4× 빠르고 near-lossless(헤드 제외 static).
    # 변형 목록엔 fp32가 먼저라 웹 A/B 토글의 'FP32→INT8' 서사는 유지되고, 기본 활성만 INT8.
    if _CPU_ONLY and "int8" in YOLO_VARIANTS:
        pipeline.set_variant("int8")
        logger.info("CPU 전용 — 서버 기본 검출기 INT8 설정 (CPU ~2.4×, near-lossless)")


@app.on_event("shutdown")
async def _shutdown() -> None:
    session_mgr.close()


# ─────────────────────────────────────────────────────────────────────────────
# 정적 UI 서빙
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """루트 → detection UI로 리다이렉트."""
    return HTMLResponse(
        '<meta http-equiv="refresh" content="0; url=/detection/">',
        status_code=200,
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/ingest — 파일/URL/이미지 → 서버 스트림 세션 발급
# ─────────────────────────────────────────────────────────────────────────────

session_mgr = SessionManager()


@app.post("/api/ingest", response_model=None)
async def ingest(
    kind: str = Form(...),
    url: str = Form(None),
    file: UploadFile = File(None),  # noqa: B008
) -> dict | JSONResponse:
    """파일/URL/이미지 → 서버 스트림 세션 발급. 실패 시 400 + error JSON."""
    try:
        if kind == "url":
            if not url:
                return JSONResponse({"error": "url 누락"}, status_code=400)
            sid = session_mgr.from_url(url)
        elif kind in ("video", "image"):
            if file is None:
                return JSONResponse({"error": "file 누락"}, status_code=400)
            data = await file.read()
            suffix = Path(file.filename or "").suffix or (".jpg" if kind == "image" else ".mp4")
            path = save_upload(data, suffix)
            try:
                sid = (
                    session_mgr.from_image(path)
                    if kind == "image"
                    else session_mgr.from_video(path)
                )
            except Exception:
                # 디코딩 실패 등으로 세션 생성이 실패하면 임시파일 누수 방지
                path.unlink(missing_ok=True)
                raise
        else:
            return JSONResponse({"error": f"알 수 없는 kind: {kind}"}, status_code=400)
        return {"session_id": sid}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket: 프레임 스트림 처리
# ─────────────────────────────────────────────────────────────────────────────


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    """
    클라이언트에서 base64 JPEG 프레임을 수신, 파이프라인 처리 후 JSON 결과를 반환.

    메시지 프로토콜:
      수신: {"type": "frame", "data": "<base64 JPEG>"}
            {"type": "reset"}
      송신: {"type": "result", "data": <process_frame() 결과>}
            {"type": "error", "message": "..."}
    """
    await websocket.accept()
    logger.info("WS stream 연결")

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "reset":
                if pipeline:
                    pipeline.reset()
                await websocket.send_json({"type": "ack", "message": "reset"})
                continue

            if msg.get("type") != "frame":
                continue

            # base64 JPEG → numpy BGR
            data_b64 = msg.get("data", "")
            if "," in data_b64:
                data_b64 = data_b64.split(",", 1)[1]

            try:
                img_bytes = base64.b64decode(data_b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})
                continue

            if frame is None:
                await websocket.send_json({"type": "error", "message": "이미지 디코딩 실패"})
                continue

            # 양자화 A/B 토글 — 프레임마다 variant 지정(없거나 미로드면 active 사용).
            variant = msg.get("variant")
            if pipeline and variant not in pipeline.yolo_sessions:
                variant = None

            # 파이프라인 처리
            result = (
                pipeline.process_frame(frame, variant=variant)
                if pipeline
                else {"frame_id": 0, "tracks": [], "inference_ms": 0}
            )
            await websocket.send_json({"type": "result", "data": result})

    except WebSocketDisconnect:
        logger.info("WS stream 연결 해제")
    except Exception:
        logger.exception("WS stream 처리 중 오류")


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket: 서버 스트림 (세션 디코딩 → 파이프라인 → 주석 JPEG + JSON 푸시)
# ─────────────────────────────────────────────────────────────────────────────


@app.websocket("/ws/session")
async def ws_session(websocket: WebSocket) -> None:
    """서버 스트림: 세션 소스 디코딩 → 파이프라인 → 주석 JPEG + JSON 푸시.
    수신: {type:"control", action:"play|pause|seek|speed|stop", value:?}"""
    await websocket.accept()
    sess = session_mgr.get()
    if sess is None or pipeline is None:
        await websocket.send_json({"type": "error", "message": "세션 없음"})
        await websocket.close()
        return

    async def handle_controls() -> None:
        try:
            while True:
                msg = await websocket.receive_json()
                if msg.get("type") == "control":
                    act = msg.get("action")
                    if act == "stop":
                        break
                    if act == "variant":  # 양자화 A/B 토글
                        v = msg.get("value")
                        if pipeline and v in pipeline.yolo_sessions:
                            pipeline.set_variant(v)
                        continue
                    sess.control(act, msg.get("value"))
        except Exception:
            pass

    ctrl_task = asyncio.create_task(handle_controls())
    target_dt = 1.0 / max(STREAM_FPS, 1.0)  # 송출 목표 간격(FPS 캡)
    miss = 0  # 연속 read 실패 카운트 (라이브 글리치 흡수)
    try:
        while not ctrl_task.done():
            t0 = time.perf_counter()
            if not sess.playing:
                await asyncio.sleep(0.03)
                continue
            frame = sess.source.read()
            if frame is None:
                if sess.source.is_seekable:  # 영상 끝 → 정지
                    sess.playing = False
                    await websocket.send_json({"type": "ended"})
                    continue
                else:  # 라이브 스트림: 일시적 글리치 재시도
                    miss += 1
                    if miss >= 30:  # 약 1초 연속 실패 → 종료
                        await websocket.send_json({"type": "ended"})
                        break
                    await asyncio.sleep(0.03)
                    continue
            miss = 0
            # ② 송출 다운스케일 — 폭 상한 초과 시 축소(CPU 인코딩·대역폭 절감).
            #    검출 입력은 내부 640 리사이즈라 품질 영향 미미. 좌표는 축소 프레임 기준으로
            #    일관(전송 w/h도 축소값) → 클라 오버레이 레터박스 계산과 일치.
            if STREAM_MAX_W and frame.shape[1] > STREAM_MAX_W:
                _sc = STREAM_MAX_W / frame.shape[1]
                frame = cv2.resize(frame, (STREAM_MAX_W, max(1, round(frame.shape[0] * _sc))))
            result = pipeline.process_frame(frame)
            # 원본 프레임 + 좌표 JSON만 전송 → 박스/라벨은 클라이언트가 그림(클라 모드와 동일,
            # 한글 라벨·둥근 박스·pill 렌더링 일치). cv2 putText는 한글 미지원이라 서버 draw 미사용.
            h, w = frame.shape[:2]
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            await websocket.send_json(
                {
                    "type": "frame",
                    "frame_id": result["frame_id"],
                    "inference_ms": result["inference_ms"],
                    "tracks": result["tracks"],
                    "variant": result.get("variant"),
                    "model_mb": result.get("model_mb"),
                    "stage_ms": result.get("stage_ms"),
                    "w": w,
                    "h": h,
                    # 통합 seek 바용: 소스 실제 위치 + 총 프레임 + fps + seek 가능 여부
                    "pos": sess.source.position(),
                    "total": getattr(sess.source, "frame_count", 0),
                    "fps": getattr(sess.source, "fps", 30.0),
                    "seekable": getattr(sess.source, "is_seekable", False),
                }
            )
            await websocket.send_bytes(buf.tobytes())
            elapsed = time.perf_counter() - t0
            # ③ 실시간 추종 — 추론이 목표 간격보다 느리면(CPU 병목) 뒤처진 만큼 디코드-only
            #    프레임을 버려 영상이 slow-motion 대신 실시간으로 흐르게 한다. seekable + 정상속도
            #    에서만, 폭주 방지 위해 한 번에 최대 4프레임 스킵.
            behind = elapsed - target_dt
            if behind > 0 and sess.source.is_seekable and sess.speed >= 1.0:
                for _ in range(min(int(behind / target_dt), 4)):
                    if sess.source.read() is None:
                        break
            await asyncio.sleep(max(0, target_dt / max(sess.speed, 0.1) - elapsed))
    except Exception:
        # disconnect(WebSocketDisconnect) 외에 죽은 소켓 send 예외(RuntimeError 등)도 흡수
        pass
    finally:
        ctrl_task.cancel()
        # 연결 종료 시 세션 정리 — 단, 그 사이 새 ingest로 교체됐다면 그 새 세션은 닫지 않음
        if session_mgr.get() is sess:
            session_mgr.close()


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/qa — Groq 스트리밍 Q&A (SSE)
# ─────────────────────────────────────────────────────────────────────────────


class QARequest(BaseModel):
    tracks: list[dict]  # process_frame()["tracks"]
    question: str
    api_key: str | None = None  # BYOK — 방문자 본인 Groq 키 (없으면 서버 env 폴백)


@app.post("/api/qa")
async def qa_endpoint(req: QARequest) -> StreamingResponse:
    """
    인식된 tracks + 사용자 질문 → Groq LLM 스트리밍 답변 (SSE).

    클라이언트:
      const evtSrc = new EventSource(URL) — fetch + SSE 방식 사용
    """
    context = build_context(req.tracks)

    async def event_generator() -> AsyncIterator[str]:
        yield f"data: {json.dumps({'type': 'context', 'text': context}, ensure_ascii=False)}\n\n"
        async for token in ask_stream(context, req.question, api_key=req.api_key):
            payload = json.dumps({"type": "token", "text": token}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/status — 파이프라인 상태
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/api/status")
async def status() -> dict[str, object]:
    return {
        "pipeline": pipeline is not None,
        "yolo": pipeline.yolo_session is not None if pipeline else False,
        "ocr": pipeline.ocr_session is not None if pipeline else False,
        "tsign": pipeline.tsign_session is not None if pipeline else False,
        "yolo_path": YOLO_ONNX,
        "ocr_path": OCR_ONNX,
        "tsign_path": TSIGN_ONNX,
        "variants": pipeline.variant_info() if pipeline else [],
        "active_variant": pipeline.active_variant if pipeline else None,
        "taxonomy": DET_TAXONOMY,
        "version": (
            "v3 (신호등 분리 + 한국 분류기 14클래스)"
            if DET_TAXONOMY == "v3"
            else "v2 검출기 + 한국 분류기 (신호등 미분리 — v3 학습 대기)"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/labels — 온디바이스 인식기용 라벨 메타 (분류기 names/서브셋 + OCR idx→char)
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/api/labels")
async def labels() -> dict[str, object]:
    """브라우저 온디바이스 인식기가 분류기 출력을 한글 라벨로 디코딩하기 위한 메타.
    data/roi_cls/classes.json(분류기) + data/idx_to_char.json(OCR)을 그대로 노출."""
    cls_path = ROOT / "data" / "roi_cls" / "classes.json"
    idx_path = ROOT / "data" / "idx_to_char.json"
    out: dict[str, object] = {"names": [], "sign_ids": [], "light_ids": [], "idx_to_char": {}}
    if cls_path.exists():
        out.update(json.loads(cls_path.read_text(encoding="utf-8")))
    if idx_path.exists():
        out["idx_to_char"] = json.loads(idx_path.read_text(encoding="utf-8"))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 직접 실행
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.pipeline.app:app", host="0.0.0.0", port=8000, reload=True)
