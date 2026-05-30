# 범용 실시간 입력 파이프라인 (SP1) — 설계 문서

- **작성일**: 2026-05-30
- **상태**: 설계 합의 완료 (구현 계획 대기)
- **범위**: SP1 (범용 실시간 입력). SP2(인식 품질·v3 검출기, 진행 중)·SP3(결론 데모 스토리)는 별도.

---

## 1. 목표 / 동기

> "어떤 형식의 입력이 들어와도 즉시 실시간으로 인식한다."

현재 데모는 브라우저가 영상을 디코딩하므로 브라우저 미지원 코덱(예: MPEG-4 Part2 블랙박스)은 **검은 화면**이 되고, URL·이미지 입력 경로가 없다. 또한 추론이 CPU로 폴백되어 실시간성이 제한된다.

SP1은 다음을 보장한다.
1. **4개 입력 소스**: 업로드 영상(모든 코덱) · 웹캠 · 정지 이미지(JPG/PNG) · URL/외부 스트림(직접 URL·RTSP, YouTube는 옵션)
2. **GPU 추론**으로 30+ FPS 목표(실패 시 CPU 자동 폴백)
3. 로컬(발표자 PC) 단일 사용자 데모 환경

### 성공 기준
- MPEG-4 Part2 등 브라우저 비호환 영상이 검은 화면 없이 재생·인식된다.
- 정지 이미지 업로드 시 즉시 검출/분류 결과 + Q&A가 가능하다.
- 직접 영상 URL/RTSP가 인식된다.
- `onnxruntime`이 `CUDAExecutionProvider`를 실제 사용한다(검증 가능).
- 기존 웹캠/H.264 경로가 회귀 없이 동작한다.

---

## 2. 아키텍처 — 하이브리드 (접근법 C)

입력 종류에 따라 **표시 모드 2종**을 클라이언트가 자동 선택한다.

### 모드 ① 클라이언트 캡처 (웹캠 + 브라우저 디코딩 가능 영상)
- `<video>`/웹캠 → 프레임 캡처 → `WS /ws/stream`로 전송 → 서버 GPU 파이프라인 → **좌표 JSON** 반환 → 클라가 canvas에 bbox 오버레이.
- 재생 컨트롤: 네이티브 `<video>`(속도/seek). **현행 동작 유지**.
- 진입: 업로드 영상이 `<video>`에 정상 로드되면 이 모드.

### 모드 ② 서버 스트림 (비호환 코덱 · URL · 이미지)
- `POST /api/ingest` → 서버가 `FrameSource` 생성·세션 발급 → `WS /ws/session`에서 서버가 디코딩→GPU 파이프라인→**주석을 그린 JPEG + 트랙 JSON**을 바이너리로 푸시 → 클라는 `<img>`/canvas에 표시.
- 진입: 영상 `<video>` 로드 실패(`NotSupportedError`)→자동 폴백 / URL·이미지→항상 이 모드.
- **주석 렌더링은 서버가 수행**(프레임+결과를 서버가 모두 보유 → 좌표 동기화 문제 원천 차단). 트랙목록·Q&A는 JSON 사용.
- 재생 컨트롤: WS 제어 메시지(`play/pause/seek/speed/stop`)→서버가 `cv2.set(CAP_PROP_POS_FRAMES)`·읽기 레이트 조절. 이미지=정지(컨트롤 숨김), URL 라이브=재생/정지만(seek 불가).

### 실시간성
서버 세션 루프가 목표 30FPS로 `grab→GPU추론→JPEG 인코딩→푸시`. 파이프라인이 느리면 **프레임 드롭**으로 라이브 유지(지연 누적 방지).

### Q&A
두 모드 모두 최신 트랙 JSON을 컨텍스트로 `POST /api/qa`(Groq SSE). **변경 없음**.

```
┌─ 브라우저 ──────────────┐         ┌─ FastAPI 서버 (GPU) ───────────────┐
│ 소스 선택 UI            │ ─POST──▶│ /api/ingest (파일/URL/이미지)       │
│  영상/웹캠/이미지/URL    │  파일/URL │   → FrameSource 생성·세션 발급      │
│ 모드① 캡처: 프레임 전송 │ ─WS───▶ │ /ws/stream  (모드① 좌표 반환)       │
│ 모드② 표시: 주석 수신   │ ◀─WS─── │ /ws/session (모드② 주석 JPEG+JSON) │
│ Q&A                     │ ─SSE──▶ │ /api/qa (Groq) — 기존              │
└─────────────────────────┘         └────────────────────────────────────┘
```

---

## 3. 컴포넌트 / 파일

### 신규
- **`src/pipeline/sources.py`** — `FrameSource` 추상 베이스 + 구현:
  - `VideoFileSource(path)` — cv2.VideoCapture(필요 시 ffmpeg 디코딩), 모든 코덱. `read()/seek(frame_idx)/release()/meta(fps,frames)`.
  - `UrlStreamSource(url)` — cv2 직접 URL·RTSP. YouTube 등은 `yt-dlp`로 스트림 URL 추출(옵션 의존성). 라이브는 seek 불가.
  - `ImageSource(path)` — 단일 프레임 반복 산출(정지).
  - 공통 인터페이스: `read() -> Optional[np.ndarray]`(BGR), `release()`, 속성 `is_seekable`, `fps`, `frame_count`.
  - 웹캠은 클라이언트 캡처이므로 서버 클래스 없음.

### 변경
- **`src/pipeline/app.py`**
  - 기동 시 `os.add_dll_directory(<torch/lib>)`로 onnxruntime-gpu가 CUDA/cuDNN DLL을 찾게 함.
  - `POST /api/ingest` — multipart 파일 / `{"url": ...}` / 이미지 수신 → temp 저장 → `FrameSource` 생성 → `session_id` 반환. 디코드 불가 시 에러 JSON.
  - `WS /ws/session?id=...` — 세션 루프: 소스 read→파이프라인→`pipeline.draw()`로 주석 JPEG 인코딩→`{type:"frame", jpeg:<binary>, tracks:[...], frame_id, ms}` 푸시. 클라 제어 메시지(`play/pause/seek/speed/stop`) 처리.
  - 기존 `/ws/stream`(모드① 좌표 반환)·`/api/qa` 유지. 단일 세션 매니저(이전 세션은 새 ingest 시 정리).
- **`web/detection/app.js`**
  - 소스 선택 + 모드 자동 판별(`<video>` 로드 실패→서버 인제스트 폴백; URL·이미지→서버).
  - 모드② 표시: 바이너리 WS JPEG→Blob URL→`<img>`(고정), 트랙목록·Q&A는 JSON.
  - 모드② 재생 컨트롤을 서버 명령(WS)으로. 모드① 경로 유지.
- **`web/detection/index.html`** — URL 입력 필드 + 이미지 업로드 + 소스 표시 UI 소폭. 기존 디자인/ID 보존.
- **`requirements.txt`** — `onnxruntime` → `onnxruntime-gpu`(CUDA12/cuDNN9), `yt-dlp`(URL 옵션).

---

## 4. 에러 처리

- **디코드 실패**(손상 파일/미지원 URL): `/api/ingest`가 명확한 에러 JSON → UI 토스트.
- **세션 수명**: WS 종료 시 `FrameSource.release()`·temp 삭제. 동시 세션 1개로 단순화.
- **업로드 크기 제한**(기본 500MB) + temp 정리. **URL 타임아웃**(10초)·**재시도 3회**. **GPU OOM → CPU 폴백**.
- **GPU 초기화 실패**: `CUDAExecutionProvider` 미가용 시 CPU로 자동 폴백, 데모 무중단.

---

## 5. 테스트

- **단위**: 각 `FrameSource.read()` — 합성 mp4(여러 코덱)·이미지·잘못된 URL.
- **통합**: `/api/ingest → /ws/session` E2E를 headless로 — 서버 디코딩→주석 JPEG 프레임 수신 카운트 검증.
- **코덱 매트릭스**: H.264 / MPEG-4 Part2(이전 검은화면) / HEVC 샘플로 서버 디코딩 성공 확인.
- **GPU**: `get_available_providers()`에 CUDA 포함 + 1프레임 GPU 추론 확인.
- **회귀**: 클라이언트 캡처 모드(웹캠·H.264)도 여전히 동작.

---

## 6. 범위 밖 (YAGNI)

다중 사용자/동시 세션, 인증/보안, 업로드 영구 저장, 주석 결과 녹화/내보내기, 적응형 비트레이트, 모바일/WASM 클라이언트 추론(후속 SP).

---

## 7. 의존 관계 / 비고

- SP2(v3 검출기·한국 분류기)와 독립적으로 진행 가능. 파이프라인 인터페이스(`EdgeSignPipeline.process_frame`, `draw`)는 그대로 사용.
- 서버는 `convnext_env` conda 환경에서 실행(런타임 의존성은 해당 env에 설치). GPU 라이브러리는 torch cu128 동봉분 재활용.
