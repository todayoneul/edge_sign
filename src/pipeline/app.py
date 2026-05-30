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

import base64
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.pipeline.e2e_pipeline import EdgeSignPipeline
from src.pipeline.qa_bridge import build_context, ask_stream

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
