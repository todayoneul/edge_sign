# Edge-Sign Web Demos

브라우저용 데모 두 가지를 제공한다.

| 경로 | 데모 | 추론 방식 |
| :--- | :--- | :--- |
| `web/detection/` | 검출 · 추적 · 인식 + 주행 Q&A (메인 시연) | FastAPI 서버 추론 (WebSocket/SSE) |
| `web/index.html` | 한글 OCR 캔버스 데모 (Phase 1) | ONNX Runtime Web (WASM/WebGPU) 클라이언트 추론 |

---

## 1. 검출·추적·인식 데모 (`web/detection/`)

메인 시연 시스템. 웹캠·이미지·URL·모든 코덱 영상을 입력으로 받아 실시간 검출/추적/인식하고,
인식 결과(JSON)를 Groq LLM에 전달해 주행 질문에 한국어로 답한다.

```powershell
# 서버는 convnext_env 에서 실행 (groq · onnxruntime-gpu 설치 필요)
copy .env.example .env          # GROQ_API_KEY 입력
uvicorn src.pipeline.app:app --port 8000
# 브라우저 → http://localhost:8000/detection/
```

- **입력 모드**는 클라이언트가 자동 선택한다. 브라우저 호환 영상·웹캠은 클라이언트가 디코딩해
  좌표만 받아 렌더하고(`WS /ws/stream`), 비호환 코덱·URL·이미지는 서버가 디코딩한다
  (`POST /api/ingest` → `WS /ws/session`). 두 모드의 박스/라벨 렌더는 동일하다.
- 라이트/다크 모드 토글을 지원한다.
- 아키텍처 상세는 저장소 루트 `README.md`의 "9. 실시간 시연 시스템 및 웹 배포 아키텍처" 참조.

## 2. 한글 OCR 캔버스 데모 (`web/index.html`)

`onnxruntime-web`으로 브라우저에서 직접 한글 OCR 모델(`korean_ocr_quant.onnx`)을 실행한다.

```powershell
python -m http.server 8000 --directory web
# 브라우저 → http://localhost:8000
```

- WebGPU 지원 브라우저에서는 WebGPU 백엔드로, 미지원 시 WASM으로 자동 대체된다.
- 모델 경로/설정 변경 시 페이지에서 모델을 다시 로드해야 반영된다.
