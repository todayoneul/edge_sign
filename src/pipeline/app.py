"""
Edge-Sign v2 FastAPI 백엔드 서버

엔드포인트:
  GET  /                  → web/detection/index.html 서빙
  GET  /detection/{file}  → web/detection/ 정적 파일
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
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# onnxruntime-gpu가 torch cu128 동봉 CUDA/cuDNN DLL을 찾도록 등록 (GPU 추론)
import os as _os
from pathlib import Path as _Path
try:
    import torch as _torch
    _tlib = _Path(_torch.__file__).parent / "lib"
    if _tlib.exists():
        _os.add_dll_directory(str(_tlib))
except Exception:
    pass  # torch 없거나 CPU 환경이면 CPU 폴백

from src.pipeline.e2e_pipeline import EdgeSignPipeline
from src.pipeline.qa_bridge import build_context, ask_stream
from src.pipeline.session import SessionManager, save_upload

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

# v3 (2026-05-30) — 신호등 분리 검출기 + 한국 표지판/신호등 분류기
#   검출기: 0=traffic_sign, 1=traffic_light (data/yolo_signs_v2 학습)
#   분류기: korean_sign_net 14클래스 (속도제한/규제/지시/주의 + 신호등 색상)
# 검출기 v3 ONNX(fp32/w8a8)가 있으면 우선 사용, 없으면 v2(w8a8)로 폴백.
# 택소노미는 로드된 검출기에 맞춰야 라우팅이 정확하다 (v2 간판↔v3 신호등 혼동 방지).
_YOLO_V3 = next((p for p in [
    ROOT / "model_space" / "yolov8s_signs_v3_w8a8.onnx",
    ROOT / "model_space" / "yolov8s_signs_v3_fp32.onnx",
] if p.exists()), None)
if _YOLO_V3 is not None:
    YOLO_ONNX = str(_YOLO_V3)
    DET_TAXONOMY = "v3"   # 0=sign, 1=light, 2=signboard
else:
    YOLO_ONNX = str(ROOT / "model_space" / "yolov8s_signs_w8a8.onnx")
    DET_TAXONOMY = "v2"   # 0=sign, 1=signboard (신호등 미분리)
OCR_ONNX   = str(ROOT / "model_space" / "korean_ocr_net_w8a8.onnx")
# 분류기는 FP32 사용 (114KB로 작음 + 동적 INT8은 CPU EP ConvInteger 미지원)
TSIGN_ONNX = str(ROOT / "model_space" / "korean_sign_net_fp32.onnx")
WEB_DIR    = ROOT / "web" / "detection"

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

# 정적 파일 (web/detection/ 디렉토리가 존재할 때만)
if WEB_DIR.exists():
    app.mount("/detection", StaticFiles(directory=str(WEB_DIR), html=True), name="detection")

# 파이프라인 (전역 단일 인스턴스)
pipeline: EdgeSignPipeline | None = None

@app.on_event("startup")
async def startup():
    global pipeline
    pipeline = EdgeSignPipeline(
        yolo_onnx=YOLO_ONNX,
        ocr_onnx=OCR_ONNX,
        tsign_onnx=TSIGN_ONNX,
        conf_thres=0.15,
        det_taxonomy=DET_TAXONOMY,
    )
    print(f"[Server] 파이프라인 초기화 완료 (검출기 택소노미={DET_TAXONOMY})")


@app.on_event("shutdown")
async def _shutdown():
    session_mgr.close()


# ─────────────────────────────────────────────────────────────────────────────
# 정적 UI 서빙
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """루트 → detection UI로 리다이렉트."""
    return HTMLResponse(
        '<meta http-equiv="refresh" content="0; url=/detection/">',
        status_code=200,
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/ingest — 파일/URL/이미지 → 서버 스트림 세션 발급
# ─────────────────────────────────────────────────────────────────────────────

session_mgr = SessionManager()


@app.post("/api/ingest")
async def ingest(kind: str = Form(...),
                 url: str = Form(None),
                 file: UploadFile = File(None)):
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
                sid = (session_mgr.from_image(path) if kind == "image"
                       else session_mgr.from_video(path))
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
async def ws_stream(websocket: WebSocket):
    """
    클라이언트에서 base64 JPEG 프레임을 수신, 파이프라인 처리 후 JSON 결과를 반환.

    메시지 프로토콜:
      수신: {"type": "frame", "data": "<base64 JPEG>"}
            {"type": "reset"}
      송신: {"type": "result", "data": <process_frame() 결과>}
            {"type": "error", "message": "..."}
    """
    await websocket.accept()
    print("[WS] 클라이언트 연결")

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

            # 파이프라인 처리
            result = pipeline.process_frame(frame) if pipeline else {
                "frame_id": 0, "tracks": [], "inference_ms": 0
            }
            await websocket.send_json({"type": "result", "data": result})

    except WebSocketDisconnect:
        print("[WS] 클라이언트 연결 해제")
    except Exception as e:
        print(f"[WS] 오류: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket: 서버 스트림 (세션 디코딩 → 파이프라인 → 주석 JPEG + JSON 푸시)
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/session")
async def ws_session(websocket: WebSocket):
    """서버 스트림: 세션 소스 디코딩 → 파이프라인 → 주석 JPEG + JSON 푸시.
    수신: {type:"control", action:"play|pause|seek|speed|stop", value:?}"""
    await websocket.accept()
    sess = session_mgr.get()
    if sess is None or pipeline is None:
        await websocket.send_json({"type": "error", "message": "세션 없음"})
        await websocket.close()
        return

    async def handle_controls():
        try:
            while True:
                msg = await websocket.receive_json()
                if msg.get("type") == "control":
                    act = msg.get("action")
                    if act == "stop":
                        break
                    sess.control(act, msg.get("value"))
        except Exception:
            pass

    ctrl_task = asyncio.create_task(handle_controls())
    target_dt = 1.0 / 30.0
    miss = 0                                   # 연속 read 실패 카운트 (라이브 글리치 흡수)
    try:
        while not ctrl_task.done():
            t0 = time.perf_counter()
            if not sess.playing:
                await asyncio.sleep(0.03)
                continue
            frame = sess.source.read()
            if frame is None:
                if sess.source.is_seekable:        # 영상 끝 → 정지
                    sess.playing = False
                    await websocket.send_json({"type": "ended"})
                    continue
                else:                              # 라이브 스트림: 일시적 글리치 재시도
                    miss += 1
                    if miss >= 30:                 # 약 1초 연속 실패 → 종료
                        await websocket.send_json({"type": "ended"})
                        break
                    await asyncio.sleep(0.03)
                    continue
            miss = 0
            result = pipeline.process_frame(frame)
            # 원본 프레임 + 좌표 JSON만 전송 → 박스/라벨은 클라이언트가 그림(클라 모드와 동일,
            # 한글 라벨·둥근 박스·pill 렌더링 일치). cv2 putText는 한글 미지원이라 서버 draw 미사용.
            h, w = frame.shape[:2]
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            await websocket.send_json({
                "type": "frame", "frame_id": result["frame_id"],
                "inference_ms": result["inference_ms"], "tracks": result["tracks"],
                "w": w, "h": h,
            })
            await websocket.send_bytes(buf.tobytes())
            elapsed = time.perf_counter() - t0
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
    tracks: list[dict]       # process_frame()["tracks"]
    question: str


@app.post("/api/qa")
async def qa_endpoint(req: QARequest):
    """
    인식된 tracks + 사용자 질문 → Groq LLM 스트리밍 답변 (SSE).

    클라이언트:
      const evtSrc = new EventSource(URL) — fetch + SSE 방식 사용
    """
    context = build_context(req.tracks)

    async def event_generator():
        yield f"data: {json.dumps({'type': 'context', 'text': context}, ensure_ascii=False)}\n\n"
        async for token in ask_stream(context, req.question):
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
async def status():
    return {
        "pipeline": pipeline is not None,
        "yolo":     pipeline.yolo_session   is not None if pipeline else False,
        "ocr":      pipeline.ocr_session    is not None if pipeline else False,
        "tsign":    pipeline.tsign_session  is not None if pipeline else False,
        "yolo_path":  YOLO_ONNX,
        "ocr_path":   OCR_ONNX,
        "tsign_path": TSIGN_ONNX,
        "taxonomy": DET_TAXONOMY,
        "version":  ("v3 (신호등 분리 + 한국 분류기 14클래스)" if DET_TAXONOMY == "v3"
                     else "v2 검출기 + 한국 분류기 (신호등 미분리 — v3 학습 대기)"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 직접 실행
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.pipeline.app:app", host="0.0.0.0", port=8000, reload=True)
