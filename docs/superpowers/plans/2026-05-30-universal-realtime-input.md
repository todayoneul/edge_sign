# 범용 실시간 입력 파이프라인 (SP1) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 어떤 입력(영상 모든 코덱·웹캠·이미지·URL)이 들어와도 즉시 실시간으로 검출·추적·분류·Q&A 하도록 하이브리드 입력 파이프라인 + GPU 추론을 구축한다.

**Architecture:** 하이브리드 2모드. 모드①(웹캠·브라우저 호환 영상)은 클라이언트가 프레임을 캡처해 `/ws/stream`으로 보내고 좌표 JSON을 받아 직접 그린다(현행). 모드②(비호환 코덱·URL·이미지)는 서버가 `FrameSource`로 디코딩→GPU 파이프라인→주석 그린 JPEG+JSON을 `/ws/session`으로 푸시한다. GPU는 onnxruntime-gpu + torch cu128 동봉 CUDA DLL 재활용.

**Tech Stack:** Python 3.11(convnext_env), FastAPI, OpenCV, onnxruntime-gpu, ByteTrack, Groq, pytest(신규), 바닐라 JS + ONNX 없는 클라이언트(서버 추론).

> **실행 환경 주의:** 서버/스크립트는 `convnext_env` conda 환경에서 실행한다 (`C:\Users\leegy\miniconda3\envs\convnext_env\python.exe`). YOLO/일부 작업은 `$env:KMP_DUPLICATE_LIB_OK='TRUE'` 필요. OpenCV는 비ASCII 경로 read/write 실패 — 테스트 경로는 ASCII로.

---

## 파일 구조

| 파일 | 책임 | 신규/변경 |
|------|------|-----------|
| `src/pipeline/sources.py` | `FrameSource` 추상 + Image/VideoFile/UrlStream 구현 (디코딩 추상화) | 신규 |
| `tests/__init__.py` | 테스트 패키지 | 신규 |
| `tests/conftest.py` | 합성 미디어(mp4/jpg) fixture | 신규 |
| `tests/test_sources.py` | FrameSource 단위 테스트 | 신규 |
| `tests/test_ingest_api.py` | `/api/ingest`·`/ws/session` 통합 테스트 (TestClient) | 신규 |
| `src/pipeline/session.py` | 단일 세션 매니저 (FrameSource 수명·제어 상태) | 신규 |
| `src/pipeline/app.py` | GPU DLL 등록 + `/api/ingest` + `/ws/session` | 변경 |
| `web/detection/app.js` | 소스 선택·모드 자동판별·서버스트림 표시·서버 재생컨트롤 | 변경 |
| `web/detection/index.html` | URL 입력·이미지 소스 UI | 변경 |
| `requirements.txt` | onnxruntime-gpu, yt-dlp, pytest, httpx | 변경 |

---

## Task 1: GPU 추론 복구 (onnxruntime-gpu)

**Files:**
- Modify: `requirements.txt`
- Modify: `src/pipeline/app.py` (상단 import 영역, startup 이전)
- Create: `scripts/check_gpu_ort.py`

- [ ] **Step 1: convnext_env에 onnxruntime-gpu 설치 (CPU 빌드 제거 후)**

Run:
```bash
PY="C:/Users/leegy/miniconda3/envs/convnext_env/python.exe"
"$PY" -m pip uninstall -y onnxruntime
"$PY" -m pip install onnxruntime-gpu==1.23.2
```
Expected: `Successfully installed onnxruntime-gpu-1.23.2`

- [ ] **Step 2: GPU 검증 스크립트 작성 (torch lib을 DLL 경로에 추가)**

Create `scripts/check_gpu_ort.py`:
```python
"""onnxruntime-gpu가 CUDAExecutionProvider를 사용 가능한지 검증."""
import os
from pathlib import Path

# torch cu128가 동봉한 CUDA/cuDNN DLL을 onnxruntime-gpu가 찾도록 등록
import torch
_torch_lib = Path(torch.__file__).parent / "lib"
if _torch_lib.exists():
    os.add_dll_directory(str(_torch_lib))

import onnxruntime as ort
print("providers:", ort.get_available_providers())
assert "CUDAExecutionProvider" in ort.get_available_providers(), "CUDA EP 미가용"

# 실제 1프레임 추론 (검출기 ONNX) — GPU 세션 생성 확인
import numpy as np
ROOT = Path(__file__).parent.parent
yolo = ROOT / "model_space" / "yolov8s_signs_w8a8.onnx"
sess = ort.InferenceSession(str(yolo), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("session providers:", sess.get_providers())
assert sess.get_providers()[0] == "CUDAExecutionProvider", "GPU 세션 생성 실패"
x = np.zeros((1, 3, 640, 640), np.float32)
sess.run(None, {sess.get_inputs()[0].name: x})
print("OK: GPU 추론 1프레임 성공")
```

- [ ] **Step 3: 검증 스크립트 실행**

Run:
```bash
$env:KMP_DUPLICATE_LIB_OK='TRUE'; & "C:/Users/leegy/miniconda3/envs/convnext_env/python.exe" scripts/check_gpu_ort.py
```
Expected: `CUDAExecutionProvider`가 providers·session 양쪽에 포함, `OK: GPU 추론 1프레임 성공`.
실패(미가용) 시: `nvidia-cudnn-cu12`, `nvidia-cublas-cu12` 휠 설치 후 재시도. 그래도 실패하면 Task는 "CPU 폴백 유지"로 종료하고 다음 Task 진행(데모 무중단 보장).

- [ ] **Step 4: app.py 상단에 동일 DLL 등록 추가**

`src/pipeline/app.py`의 `import` 블록 맨 위(`from src.pipeline...` 이전)에 추가:
```python
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
```

- [ ] **Step 5: requirements.txt 갱신**

`requirements.txt`에서 `onnxruntime==1.23.2` 줄을 다음으로 교체:
```
onnxruntime-gpu==1.23.2
```

- [ ] **Step 6: 커밋**

```bash
git add requirements.txt src/pipeline/app.py scripts/check_gpu_ort.py
git commit -m "feat: onnxruntime-gpu로 GPU 추론 복구 (torch CUDA DLL 재활용)"
```

---

## Task 2: pytest 셋업 + FrameSource 베이스 + ImageSource

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`, `tests/conftest.py`, `tests/test_sources.py`
- Create: `src/pipeline/sources.py`

- [ ] **Step 1: pytest·httpx 설치 + requirements 반영**

Run:
```bash
"C:/Users/leegy/miniconda3/envs/convnext_env/python.exe" -m pip install pytest httpx
```
`requirements.txt` 끝에 추가:
```
pytest
httpx
```

- [ ] **Step 2: 실패하는 테스트 작성 (ImageSource)**

Create `tests/__init__.py` (빈 파일).

Create `tests/conftest.py`:
```python
import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_image(tmp_path):
    """64x48 단색 JPG (ASCII 경로)."""
    p = tmp_path / "img.jpg"
    img = np.full((48, 64, 3), (0, 128, 255), np.uint8)
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture
def sample_mp4(tmp_path):
    """10프레임 mp4 (mp4v 코덱, ASCII 경로)."""
    p = tmp_path / "clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(p), fourcc, 10.0, (64, 48))
    for i in range(10):
        frame = np.full((48, 64, 3), i * 20 % 255, np.uint8)
        w.write(frame)
    w.release()
    return p
```

Create `tests/test_sources.py`:
```python
import numpy as np
from src.pipeline.sources import ImageSource


def test_image_source_reads_same_frame(sample_image):
    src = ImageSource(str(sample_image))
    f1 = src.read()
    f2 = src.read()
    assert f1 is not None and f1.shape == (48, 64, 3)
    assert np.array_equal(f1, f2)        # 정지: 항상 동일 프레임
    assert src.is_seekable is False
    src.release()
```

- [ ] **Step 3: 실패 확인**

Run: `"C:/Users/leegy/miniconda3/envs/convnext_env/python.exe" -m pytest tests/test_sources.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.pipeline.sources'`

- [ ] **Step 4: FrameSource 베이스 + ImageSource 구현**

Create `src/pipeline/sources.py`:
```python
"""범용 프레임 소스 추상화 — 입력 종류와 무관하게 BGR 프레임을 산출.

ImageSource    : 정지 이미지 (동일 프레임 반복)
VideoFileSource: 모든 코덱 동영상 (cv2/ffmpeg)
UrlStreamSource: 직접 URL·RTSP (+ YouTube는 yt-dlp 옵션)
웹캠은 클라이언트 캡처이므로 여기 없음.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np


class FrameSource(ABC):
    is_seekable: bool = False
    fps: float = 0.0
    frame_count: int = 0

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """다음 BGR 프레임 또는 None(끝/실패)."""

    def seek(self, frame_idx: int) -> None:
        """seekable 소스만 의미 있음 (기본 no-op)."""

    def release(self) -> None:
        pass


class ImageSource(FrameSource):
    """정지 이미지: read()가 항상 같은 프레임을 반환."""

    def __init__(self, path: str):
        # 한글 경로 대비: np.fromfile + imdecode
        data = np.fromfile(path, dtype=np.uint8)
        self._frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if self._frame is None:
            raise ValueError(f"이미지 디코딩 실패: {path}")
        self.is_seekable = False
        self.fps = 1.0
        self.frame_count = 1

    def read(self) -> Optional[np.ndarray]:
        return self._frame.copy()
```

- [ ] **Step 5: 통과 확인**

Run: `"C:/Users/leegy/miniconda3/envs/convnext_env/python.exe" -m pytest tests/test_sources.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: 커밋**

```bash
git add requirements.txt tests/ src/pipeline/sources.py
git commit -m "feat: FrameSource 추상화 + ImageSource (pytest 도입)"
```

---

## Task 3: VideoFileSource (모든 코덱)

**Files:**
- Modify: `src/pipeline/sources.py`
- Modify: `tests/test_sources.py`

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_sources.py` 끝에 추가:
```python
from src.pipeline.sources import VideoFileSource


def test_video_file_source_reads_frames(sample_mp4):
    src = VideoFileSource(str(sample_mp4))
    assert src.is_seekable is True
    assert src.frame_count >= 9          # 인코더가 마지막 1프레임 누락할 수 있음
    n = 0
    while True:
        f = src.read()
        if f is None:
            break
        assert f.shape[2] == 3
        n += 1
    assert n >= 9
    src.release()


def test_video_file_source_seek(sample_mp4):
    src = VideoFileSource(str(sample_mp4))
    src.seek(5)
    f = src.read()
    assert f is not None
    src.release()
```

- [ ] **Step 2: 실패 확인**

Run: `"C:/Users/leegy/miniconda3/envs/convnext_env/python.exe" -m pytest tests/test_sources.py::test_video_file_source_reads_frames -v`
Expected: FAIL — `ImportError: cannot import name 'VideoFileSource'`

- [ ] **Step 3: VideoFileSource 구현**

`src/pipeline/sources.py` 끝에 추가:
```python
class VideoFileSource(FrameSource):
    """동영상 파일 — OpenCV(ffmpeg 백엔드)로 디코딩, 모든 코덱."""

    def __init__(self, path: str):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise ValueError(f"동영상 열기 실패: {path}")
        self.is_seekable = True
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None

    def seek(self, frame_idx: int) -> None:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))

    def release(self) -> None:
        self._cap.release()
```

- [ ] **Step 4: 통과 확인**

Run: `"C:/Users/leegy/miniconda3/envs/convnext_env/python.exe" -m pytest tests/test_sources.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/pipeline/sources.py tests/test_sources.py
git commit -m "feat: VideoFileSource (OpenCV 모든 코덱 디코딩 + seek)"
```

---

## Task 4: UrlStreamSource

**Files:**
- Modify: `src/pipeline/sources.py`
- Modify: `tests/test_sources.py`
- Modify: `requirements.txt`

- [ ] **Step 1: 실패하는 테스트 추가 (로컬 파일을 URL처럼 취급 — 네트워크 불요)**

`tests/test_sources.py` 끝에 추가:
```python
from src.pipeline.sources import UrlStreamSource


def test_url_source_opens_direct_path(sample_mp4):
    # cv2.VideoCapture는 로컬 경로/직접 URL 모두 동일 처리 → http(s) 아니면 yt-dlp 미사용
    src = UrlStreamSource(str(sample_mp4))
    f = src.read()
    assert f is not None and f.shape[2] == 3
    src.release()


def test_url_source_youtube_requires_ytdlp(monkeypatch):
    # http(s) youtube URL이면 _resolve_stream_url가 호출되는지만 확인 (네트워크 미접속)
    called = {}
    def fake_resolve(url):
        called["url"] = url
        raise RuntimeError("stop-before-network")
    import src.pipeline.sources as S
    monkeypatch.setattr(S, "_resolve_stream_url", fake_resolve)
    import pytest
    with pytest.raises(RuntimeError, match="stop-before-network"):
        UrlStreamSource("https://www.youtube.com/watch?v=abc")
    assert "youtube.com" in called["url"]
```

- [ ] **Step 2: 실패 확인**

Run: `"C:/Users/leegy/miniconda3/envs/convnext_env/python.exe" -m pytest tests/test_sources.py::test_url_source_opens_direct_path -v`
Expected: FAIL — `ImportError: cannot import name 'UrlStreamSource'`

- [ ] **Step 3: UrlStreamSource 구현**

`src/pipeline/sources.py` 끝에 추가:
```python
def _resolve_stream_url(url: str) -> str:
    """YouTube 등 페이지 URL → 직접 스트림 URL (yt-dlp). 미설치 시 안내."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError as e:
        raise RuntimeError("yt-dlp 미설치 — pip install yt-dlp") from e
    with YoutubeDL({"quiet": True, "format": "best[ext=mp4]/best"}) as ydl:
        info = ydl.extract_info(url, download=False)
        return info["url"]


_PAGE_HOSTS = ("youtube.com", "youtu.be")


class UrlStreamSource(FrameSource):
    """직접 영상 URL·RTSP. YouTube 등 페이지 URL은 yt-dlp로 해석."""

    def __init__(self, url: str):
        open_url = url
        if url.startswith(("http://", "https://")) and any(h in url for h in _PAGE_HOSTS):
            open_url = _resolve_stream_url(url)
        self._cap = cv2.VideoCapture(open_url)
        if not self._cap.isOpened():
            raise ValueError(f"스트림 열기 실패: {url}")
        self.is_seekable = False           # 라이브/스트림은 seek 불가로 단순화
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = 0

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()
```

- [ ] **Step 4: 통과 확인 + requirements 반영**

Run: `"C:/Users/leegy/miniconda3/envs/convnext_env/python.exe" -m pytest tests/test_sources.py -v`
Expected: PASS (5 passed)

`requirements.txt` 끝에 추가:
```
yt-dlp
```

- [ ] **Step 5: 커밋**

```bash
git add src/pipeline/sources.py tests/test_sources.py requirements.txt
git commit -m "feat: UrlStreamSource (직접 URL/RTSP + yt-dlp 페이지 해석)"
```

---

## Task 5: 세션 매니저 + POST /api/ingest

**Files:**
- Create: `src/pipeline/session.py`
- Modify: `src/pipeline/app.py`
- Create: `tests/test_ingest_api.py`

- [ ] **Step 1: 세션 매니저 구현 (단일 세션)**

Create `src/pipeline/session.py`:
```python
"""단일 서버 스트림 세션 — FrameSource + 재생 제어 상태."""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Optional

from src.pipeline.sources import (
    FrameSource, ImageSource, VideoFileSource, UrlStreamSource,
)

MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500MB


class Session:
    def __init__(self, source: FrameSource, temp_path: Optional[Path] = None):
        self.source = source
        self.temp_path = temp_path
        self.playing = True
        self.speed = 1.0
        self.lock = threading.Lock()

    def control(self, action: str, value=None):
        if action == "play":
            self.playing = True
        elif action == "pause":
            self.playing = False
        elif action == "speed" and value:
            self.speed = float(value)
        elif action == "seek" and value is not None and self.source.is_seekable:
            self.source.seek(int(value))

    def close(self):
        try:
            self.source.release()
        finally:
            if self.temp_path and self.temp_path.exists():
                self.temp_path.unlink(missing_ok=True)


class SessionManager:
    """동시 세션 1개. 새 세션 생성 시 이전 세션 정리."""

    def __init__(self):
        self._current: Optional[Session] = None

    def _replace(self, sess: Session) -> str:
        if self._current is not None:
            self._current.close()
        self._current = sess
        return "session"

    def get(self) -> Optional[Session]:
        return self._current

    def from_image(self, path: Path) -> str:
        return self._replace(Session(ImageSource(str(path)), temp_path=path))

    def from_video(self, path: Path) -> str:
        return self._replace(Session(VideoFileSource(str(path)), temp_path=path))

    def from_url(self, url: str) -> str:
        return self._replace(Session(UrlStreamSource(url)))

    def close(self):
        if self._current is not None:
            self._current.close()
            self._current = None


def save_upload(data: bytes, suffix: str) -> Path:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"업로드 크기 초과 (> {MAX_UPLOAD_BYTES // (1024*1024)}MB)")
    fd = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    fd.write(data)
    fd.close()
    return Path(fd.name)
```

- [ ] **Step 2: 실패하는 통합 테스트 작성**

Create `tests/test_ingest_api.py`:
```python
import io
from fastapi.testclient import TestClient
from src.pipeline.app import app


def test_ingest_image_creates_session(sample_image):
    client = TestClient(app)
    with open(sample_image, "rb") as f:
        resp = client.post("/api/ingest",
                           files={"file": ("img.jpg", f.read(), "image/jpeg")},
                           data={"kind": "image"})
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "session"


def test_ingest_bad_url_returns_error():
    client = TestClient(app)
    resp = client.post("/api/ingest", data={"kind": "url", "url": "rtsp://0.0.0.0:1/x"})
    assert resp.status_code == 400
    assert "error" in resp.json()
```

- [ ] **Step 3: 실패 확인**

Run: `"C:/Users/leegy/miniconda3/envs/convnext_env/python.exe" -m pytest tests/test_ingest_api.py -v`
Expected: FAIL — `/api/ingest` 404 (엔드포인트 없음)

- [ ] **Step 4: /api/ingest 엔드포인트 구현**

`src/pipeline/app.py`에 추가 (import에 `UploadFile, File, Form` 포함, 전역 매니저 생성):
```python
from fastapi import UploadFile, File, Form
from src.pipeline.session import SessionManager, save_upload

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
            sid = session_mgr.from_image(path) if kind == "image" else session_mgr.from_video(path)
        else:
            return JSONResponse({"error": f"알 수 없는 kind: {kind}"}, status_code=400)
        return {"session_id": sid}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
```
`JSONResponse` import 확인 (`from fastapi.responses import ... JSONResponse`). 없으면 추가.

- [ ] **Step 5: 통과 확인**

Run: `"C:/Users/leegy/miniconda3/envs/convnext_env/python.exe" -m pytest tests/test_ingest_api.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: 커밋**

```bash
git add src/pipeline/session.py src/pipeline/app.py tests/test_ingest_api.py
git commit -m "feat: 세션 매니저 + POST /api/ingest (파일/URL/이미지)"
```

---

## Task 6: WS /ws/session 서버 스트림 루프

**Files:**
- Modify: `src/pipeline/app.py`
- Modify: `tests/test_ingest_api.py`

- [ ] **Step 1: 실패하는 통합 테스트 추가 (WS로 주석 프레임 수신)**

`tests/test_ingest_api.py` 끝에 추가:
```python
import json

def test_ws_session_streams_annotated_frames(sample_mp4):
    client = TestClient(app)
    with open(sample_mp4, "rb") as f:
        client.post("/api/ingest",
                    files={"file": ("clip.mp4", f.read(), "video/mp4")},
                    data={"kind": "video"})
    with client.websocket_connect("/ws/session") as ws:
        msg = ws.receive_json()          # 첫 결과 메타(JSON)
        assert msg["type"] in ("frame_meta", "frame")
        assert "tracks" in msg
        # 바이너리 JPEG 프레임 수신
        jpeg = ws.receive_bytes()
        assert jpeg[:2] == b"\xff\xd8"   # JPEG SOI 마커
```

- [ ] **Step 2: 실패 확인**

Run: `"C:/Users/leegy/miniconda3/envs/convnext_env/python.exe" -m pytest tests/test_ingest_api.py::test_ws_session_streams_annotated_frames -v`
Expected: FAIL — `/ws/session` 미존재로 연결 거부.

- [ ] **Step 3: /ws/session 루프 구현**

`src/pipeline/app.py`에 추가:
```python
import asyncio, time
import cv2

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
                else:                              # 스트림 끊김
                    break
            result = pipeline.process_frame(frame)
            vis = pipeline.draw(frame, result)
            ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                continue
            await websocket.send_json({
                "type": "frame", "frame_id": result["frame_id"],
                "inference_ms": result["inference_ms"], "tracks": result["tracks"],
            })
            await websocket.send_bytes(buf.tobytes())
            # 프레임 드롭으로 라이브 유지 (speed 반영)
            elapsed = time.perf_counter() - t0
            await asyncio.sleep(max(0, target_dt / max(sess.speed, 0.1) - elapsed))
    except WebSocketDisconnect:
        pass
    finally:
        ctrl_task.cancel()
```

- [ ] **Step 4: 통과 확인**

Run: `"C:/Users/leegy/miniconda3/envs/convnext_env/python.exe" -m pytest tests/test_ingest_api.py -v`
Expected: PASS (3 passed). pipeline 미초기화로 실패하면 conftest에서 startup 트리거 필요 — TestClient를 `with TestClient(app) as client:` 형태(컨텍스트)로 바꿔 startup 이벤트 실행.

- [ ] **Step 5: 커밋**

```bash
git add src/pipeline/app.py tests/test_ingest_api.py
git commit -m "feat: WS /ws/session 서버 스트림 루프 (주석 JPEG + 제어)"
```

---

## Task 7: 프론트엔드 — 소스 선택 + 모드 자동판별 + 서버스트림 표시

**Files:**
- Modify: `web/detection/index.html`
- Modify: `web/detection/app.js`

- [ ] **Step 1: index.html에 URL 입력 + 이미지 버튼 추가**

`web/detection/index.html`의 `.controls` div 안 `#file-btn` 다음에 추가:
```html
<button class="btn btn-secondary" id="image-btn">🖼 이미지 열기</button>
<input type="file" id="image-input" accept="image/*" />
<input type="text" id="url-input" placeholder="영상 URL / RTSP" />
<button class="btn btn-secondary" id="url-btn">URL 열기</button>
```
그리고 비디오 래퍼 안 `#video-el` 다음에 서버 스트림 표시용 `<img>` 추가:
```html
<img id="stream-img" alt="" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:none;" />
```

- [ ] **Step 2: app.js에 서버 스트림 모드 구현**

`web/detection/app.js`의 DOM 참조 영역에 추가:
```javascript
const imageBtn  = document.getElementById('image-btn');
const imageInput= document.getElementById('image-input');
const urlInput  = document.getElementById('url-input');
const urlBtn    = document.getElementById('url-btn');
const streamImg = document.getElementById('stream-img');
let sessionWS = null;
```
그리고 서버 인제스트 + 스트림 표시 함수 추가:
```javascript
async function ingest(kind, fileOrUrl) {
  stopServerStream();
  const fd = new FormData();
  fd.append('kind', kind);
  if (kind === 'url') fd.append('url', fileOrUrl);
  else fd.append('file', fileOrUrl);
  const resp = await fetch('/api/ingest', { method: 'POST', body: fd });
  if (!resp.ok) { alert('입력 열기 실패: ' + (await resp.json()).error); return; }
  startServerStream();
}

function startServerStream() {
  stopMedia();                          // 클라 캡처 모드 정리
  videoEl.style.display = 'none';
  streamImg.style.display = 'block';
  noVideoMsg.style.display = 'none';
  sessionWS = new WebSocket(`ws://${location.host}/ws/session`);
  sessionWS.binaryType = 'arraybuffer';
  let lastUrl = null;
  sessionWS.onmessage = (ev) => {
    if (typeof ev.data === 'string') {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'frame') {
        state.lastResult = { tracks: msg.tracks };
        updateTrackList(msg.tracks);
        frameInfo.textContent = `frame: ${msg.frame_id}`;
        timeInfo.textContent  = `추론: ${msg.inference_ms} ms`;
        trackCount.textContent= `tracks: ${msg.tracks.length}`;
      }
    } else {                            // 바이너리 JPEG
      const blob = new Blob([ev.data], { type: 'image/jpeg' });
      if (lastUrl) URL.revokeObjectURL(lastUrl);
      lastUrl = URL.createObjectURL(blob);
      streamImg.src = lastUrl;
    }
  };
}

function stopServerStream() {
  if (sessionWS) { try { sessionWS.send(JSON.stringify({type:'control',action:'stop'})); } catch(e){} sessionWS.close(); sessionWS = null; }
  streamImg.style.display = 'none';
}
```

- [ ] **Step 3: 모드 자동판별 — 비호환 영상은 서버 폴백**

`web/detection/app.js`의 `loadVideoFile(file)` 안에서, `videoEl.play().catch(...)`를 다음으로 교체:
```javascript
  videoEl.play().then(() => {
    // 모드①: 브라우저 디코딩 성공 — 클라 캡처 유지
  }).catch(() => {
    // 모드②: 디코딩 불가 → 서버 인제스트 폴백
    console.warn('[video] 브라우저 디코딩 불가 → 서버 인제스트');
    ingest('video', file);
  });
  // codec 미지원은 error 이벤트로도 옴
  videoEl.onerror = () => ingest('video', file);
```

- [ ] **Step 4: 버튼 핸들러 연결**

`web/detection/app.js` 끝(초기화 직전)에 추가:
```javascript
imageBtn.addEventListener('click', () => imageInput.click());
imageInput.addEventListener('change', () => { if (imageInput.files[0]) ingest('image', imageInput.files[0]); imageInput.value=''; });
urlBtn.addEventListener('click', () => { const u = urlInput.value.trim(); if (u) ingest('url', u); });
```

- [ ] **Step 5: 수동 검증 (서버 재시작 후 브라우저)**

Run:
```bash
$env:KMP_DUPLICATE_LIB_OK='TRUE'; & "C:/Users/leegy/miniconda3/envs/convnext_env/python.exe" -m uvicorn src.pipeline.app:app --port 8000
```
브라우저 `http://127.0.0.1:8000/detection/` → "🖼 이미지 열기"로 JPG 업로드 → 주석 프레임 표시 + 트랙목록 갱신 확인. 이전 MPEG-4 블랙박스 mp4 업로드 → 서버 폴백으로 재생 확인.

- [ ] **Step 6: 커밋**

```bash
git add web/detection/index.html web/detection/app.js
git commit -m "feat: 프론트 소스선택 + 모드 자동판별 + 서버스트림 표시"
```

---

## Task 8: 프론트엔드 — 서버 스트림 재생 컨트롤

**Files:**
- Modify: `web/detection/app.js`

- [ ] **Step 1: 서버 스트림 모드에서 속도/일시정지/seek를 WS 제어로 전송**

`web/detection/app.js`의 재생 컨트롤 핸들러를 모드 분기 처리. speed 슬라이더 핸들러에 추가:
```javascript
function sendSessionControl(action, value) {
  if (sessionWS && sessionWS.readyState === WebSocket.OPEN)
    sessionWS.send(JSON.stringify({ type: 'control', action, value }));
}
```
speedRange `input` 리스너 끝에 추가: `sendSessionControl('speed', parseFloat(speedRange.value));`
stopBtn 클릭에 추가: `stopServerStream();`
step 버튼: 서버 스트림 활성 시 `sendSessionControl('seek', <현재추정 frame ± fps*5>)` — 단순화를 위해 프레임 카운터 기반. seekable 아니면 무시됨.

- [ ] **Step 2: 수동 검증**

서버 스트림(비호환 mp4) 재생 중 속도 슬라이더 → 재생 속도 변화, 정지 버튼 → 스트림 종료 확인.

- [ ] **Step 3: 커밋**

```bash
git add web/detection/app.js
git commit -m "feat: 서버 스트림 재생 컨트롤(WS 제어 메시지)"
```

---

## Task 9: 문서 + E2E 코덱 매트릭스 검증

**Files:**
- Modify: `CLAUDE.md`, `docs/ROADMAP.md`
- Create: `scripts/check_codec_matrix.py`

- [ ] **Step 1: 코덱 매트릭스 검증 스크립트**

Create `scripts/check_codec_matrix.py`:
```python
"""H.264/MPEG-4 Part2/HEVC 샘플을 VideoFileSource로 디코딩 성공하는지 확인."""
import subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipeline.sources import VideoFileSource

base = next(Path("data/demo_videos").glob("**/*.mp4"), None)
assert base, "기준 mp4 없음 — build_demo_video.py 먼저 실행"
tmp = Path(tempfile.mkdtemp())
for codec, args in [("h264", ["-c:v","libx264"]),
                    ("mpeg4", ["-c:v","mpeg4"]),
                    ("hevc", ["-c:v","libx265"])]:
    out = tmp / f"{codec}.mp4"
    subprocess.run(["ffmpeg","-y","-i",str(base),*args,"-t","2",str(out)],
                   capture_output=True)
    src = VideoFileSource(str(out))
    n = sum(1 for _ in iter(lambda: src.read(), None))
    src.release()
    print(f"{codec}: {n} frames 디코딩 {'OK' if n>0 else 'FAIL'}")
    assert n > 0, f"{codec} 디코딩 실패"
print("코덱 매트릭스 전부 통과")
```

- [ ] **Step 2: 실행**

Run: `& "C:/Users/leegy/miniconda3/envs/convnext_env/python.exe" scripts/check_codec_matrix.py`
Expected: `h264: ... OK`, `mpeg4: ... OK`, `hevc: ... OK`, `코덱 매트릭스 전부 통과`

- [ ] **Step 3: 전체 테스트 스위트 실행**

Run: `& "C:/Users/leegy/miniconda3/envs/convnext_env/python.exe" -m pytest tests/ -v`
Expected: 모든 테스트 PASS

- [ ] **Step 4: 문서 갱신**

`CLAUDE.md` 디렉토리 구조에 `src/pipeline/sources.py`, `src/pipeline/session.py`, `tests/` 추가. 명령어에 ingest 사용법 추가. `docs/ROADMAP.md`에 SP1 완료 항목 추가.

- [ ] **Step 5: 커밋**

```bash
git add CLAUDE.md docs/ROADMAP.md scripts/check_codec_matrix.py
git commit -m "docs+test: SP1 코덱 매트릭스 검증 + 문서 갱신"
```

---

## 자체 검토 메모

- 스펙 §2 모드①(클라 캡처) 회귀: Task 7 Step 3에서 `<video>` 정상 재생 시 기존 경로 유지 — 보존됨.
- 스펙 §2 모드②(서버 스트림) 4종 입력: 이미지(Task5/7)·비호환영상(Task6/7)·URL(Task4/5)·웹캠(모드① 유지) 전부 커버.
- 스펙 §3 GPU: Task 1. §4 에러처리: 세션 close·temp 삭제(Task5), 디코드 실패 400(Task5), ended/끊김(Task6). §5 테스트: Task2-6 단위/통합 + Task9 코덱매트릭스.
- 타입 일관성: `SessionManager.from_image/from_video/from_url`·`Session.control(action,value)`·`session_mgr.get()` 전 Task에서 동일 시그니처 사용 확인.
