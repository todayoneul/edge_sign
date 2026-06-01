# 코드 품질 기반 (SP-A) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 연구 백엔드(`src/pipeline/`)에 ruff(린트·포맷)·mypy(엄격 타입)·loguru(구조화 로깅)를 추론/양자화 로직 무변경으로 도입한다.

**Architecture:** `pyproject.toml`로 도구 설정을 중앙화하고, 5단계 커밋(설정→포맷→린트수정→loguru→타입)으로 점진 적용한다. 각 단계는 기존 `pytest tests/` 16개 통과로 회귀를 게이트한다. 신규 `src/pipeline/logging_config.py`만 TDD로 작성하고, 나머지는 기계적 변환 + 검증 명령으로 진행한다.

**Tech Stack:** ruff · mypy · loguru · pytest · FastAPI · onnxruntime · Python 3.10 (conda `convnext_env`)

---

## 사전 점검 (Pre-flight)

- **⚠️ 인터프리터(중요):** 런타임 의존성은 conda `convnext_env`(**Python 3.10.19**)에만 있다. 셸 기본 `python`은 base(3.13)이며 deps가 없으니 **절대 사용 금지**. 이 계획의 모든 `python`/`ruff`/`mypy`/`pytest` 명령은 다음 인터프리터로 실행한다:
  ```
  C:\Users\leegy\miniconda3\envs\convnext_env\python.exe
  ```
  도구는 `<위 경로> -m ruff …`, `-m mypy …`, `-m pytest …`, `-m pip …` 형태로 호출한다(PowerShell: `& "C:\Users\leegy\miniconda3\envs\convnext_env\python.exe" -m pytest tests/ -q`). ruff·mypy도 이 env에 설치한다.
- 현재 브랜치 `feat/code-quality-foundation` 확인: `git branch --show-current`
- **베이스라인 확인** — 시작 전 반드시 그린이어야 한다:
  Run: `& "C:\Users\leegy\miniconda3\envs\convnext_env\python.exe" -m pytest tests/ -q`
  Expected: `16 passed` (이미 확인됨 2026-06-02)

## 적용 범위 (스코프 결정 — 합의됨)

- **loguru 전환 대상(운영 로그)**: `app.py`(4곳) + `e2e_pipeline.py` `__init__`(7곳)뿐.
  - `qa_bridge._test()`·`e2e_pipeline.main()`(`__main__` CLI)·`eval_e2e.py`의 `print`는 **CLI 사용자 출력 → 유지**.
  - `session.py`·`sources.py`에는 `print` 없음.
- **mypy 엄격 대상**: `src.pipeline.*` (단, `eval_e2e.py`는 CLI 평가 스크립트라 완화 — `scripts/` CLI와 동일 취급).
- **ruff 대상 트리**: `src/`·`tests/`·활성 `scripts/`. `scripts/archive/`·데이터·산출물 디렉터리 제외.

## 파일 구조 (생성/수정)

| 파일 | 책임 | 상태 |
| :--- | :--- | :--- |
| `pyproject.toml` | ruff·mypy·pytest 설정 중앙화 | 생성 (Task 1) |
| `requirements-dev.txt` | dev 도구 선언(ruff·mypy) | 생성 (Task 1) |
| `src/pipeline/logging_config.py` | loguru 싱크 + stdlib 인터셉트 + `configure_logging()` | 생성 (Task 4) |
| `tests/test_logging_config.py` | `configure_logging` 동작 검증 | 생성 (Task 4) |
| `src/pipeline/app.py` | 임포트 재정렬·print→logger·configure_logging 호출·타입 | 수정 (Task 3·5·6) |
| `src/pipeline/e2e_pipeline.py` | print→logger(init)·타입 | 수정 (Task 5·6) |
| `src/pipeline/{qa_bridge,sources,session}.py` | 포맷·타입 | 수정 (Task 2·6) |
| (전 대상 트리) | `ruff format` + `ruff check --fix` 결과 | 수정 (Task 2·3) |

---

## Task 1: 도구 설정 + dev 의존성 (C1)

**Files:**
- Create: `pyproject.toml`
- Create: `requirements-dev.txt`

- [ ] **Step 1: `pyproject.toml` 작성**

```toml
# Edge-Sign — 코드 품질 도구 설정 (SP-A)
# ruff(린트+포맷) · mypy(타입) · pytest. 런타임 동작에 영향 없음.

[tool.ruff]
target-version = "py310"  # convnext_env = Python 3.10.19
line-length = 100
extend-exclude = [
    "scripts/archive",
    "runs",
    "data",
    "model_space",
    "models",
    "checkpoints",
    "assets",
    "web",
    "AIhub",
    ".pytest_cache",
]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "W", "C4"]
ignore = ["E501"]  # 줄 길이는 formatter가 처리, 긴 한글 주석 보호

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # 재노출(re-export) 허용

[tool.mypy]
python_version = "3.10"
mypy_path = "."
explicit_package_bases = true
namespace_packages = true
ignore_missing_imports = true   # cv2·onnxruntime·torch·groq·ultralytics 스텁 부재
warn_unused_ignores = true

# 서버/추론 오케스트레이션 경계 — 엄격 타입
[[tool.mypy.overrides]]
module = "src.pipeline.*"
disallow_untyped_defs = true
disallow_incomplete_defs = true
no_implicit_optional = true
warn_return_any = true
check_untyped_defs = true

# eval_e2e.py 는 CLI 평가 스크립트(서빙 경로 아님) — 완화
[[tool.mypy.overrides]]
module = "src.pipeline.eval_e2e"
disallow_untyped_defs = false
disallow_incomplete_defs = false
warn_return_any = false

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: `requirements-dev.txt` 작성**

```
# 개발 도구 (런타임 의존성 아님) — SP-A 코드 품질 + SP-B CI에서 사용
ruff>=0.6.0
mypy>=1.11.0
```

- [ ] **Step 3: dev 도구 설치**

Run: `pip install -r requirements-dev.txt`
Expected: ruff·mypy 설치 완료 (이미 있으면 "already satisfied")

- [ ] **Step 4: 설정이 유효한지 확인 (게이트 아님)**

Run: `ruff check . --statistics`
Expected: 위반 통계가 출력된다(아직 포맷 전이라 위반이 많은 것이 정상). **TOML/스키마 오류가 없으면 성공** — exit code가 0이 아니어도 무방.

Run: `mypy --version`
Expected: `mypy 1.x.x` 출력.

- [ ] **Step 5: 커밋**

```bash
git add -f pyproject.toml requirements-dev.txt
git commit -m "chore(quality): ruff·mypy·pytest 설정 + dev 의존성 추가

- pyproject.toml: ruff(E,F,I,UP,B,W,C4 / line 100 / E501 무시), mypy(src.pipeline 엄격), pytest
- requirements-dev.txt: ruff·mypy (런타임과 분리)"
```
> `pyproject.toml`/`requirements-dev.txt`는 `*.md`가 아니라 `-f` 불필요하지만, 확실히 하려면 위처럼 명시 add. 무시되지 않으면 일반 add도 동일.

---

## Task 2: ruff format 적용 (C2)

**Files:** 전 대상 트리(`src/`·`tests/`·활성 `scripts/`)의 Python 파일 — 기계적 포맷.

- [ ] **Step 1: 포맷 적용**

Run: `ruff format .`
Expected: `N files reformatted, M files left unchanged` (N>0).

- [ ] **Step 2: 변경 범위 확인 (의도치 않은 파일 없는지)**

Run: `git diff --stat`
Expected: 변경이 `src/`·`tests/`·`scripts/`(archive 제외)에 한정. `web/`·`runs/`·`model_space/` 등은 포함되지 않아야 함.

- [ ] **Step 3: 회귀 게이트 — 테스트**

Run: `python -m pytest tests/ -q`
Expected: `16 passed`

- [ ] **Step 4: 포맷 안정성 확인**

Run: `ruff format --check .`
Expected: `M files already formatted` (재포맷 대상 0).

- [ ] **Step 5: 커밋**

```bash
git add -u
git commit -m "style(quality): ruff format 전체 적용 (기계적 포맷, 로직 무변경)"
```

---

## Task 3: ruff 자동수정 + import 순서 지뢰 처리 (C3)

**Files:**
- Modify: `src/pipeline/app.py` (임포트 블록 재구성 + `# noqa: E402`)
- Modify: 기타 대상 트리 (자동수정 결과)

> ⚠️ **핵심 위험:** `app.py`는 onnxruntime(=`e2e_pipeline`) import **전에** `os.add_dll_directory(torch/lib)`가 실행돼야 GPU가 활성화된다. `sys.path.insert`도 `src.*` import 전에 와야 한다. 아래는 이 순서를 보존하면서 E402를 정리한다.

- [ ] **Step 1: 자동 수정 1차 적용**

Run: `ruff check --fix .`
Expected: import 정렬(I)·미사용 제거(F401)·pyupgrade(UP)·플레이스홀더 없는 f-string(F541) 등 자동 수정. 남은 위반이 보고될 수 있음(E402 등).

- [ ] **Step 2: 남은 위반 확인**

Run: `ruff check .`
Expected: 주로 `src/pipeline/app.py`의 `E402 module-level-import-not-at-top-of-file`가 남는다(`ROOT=`/`sys.path` 뒤 import들).

- [ ] **Step 3: `app.py` 임포트 블록 재구성 (수동, 정확히 이 형태로)**

`app.py` 최상단 import 영역(현재 `from __future__ ...` 부터 `from src.pipeline.session import ...` 까지)을 아래로 교체한다. 핵심: **서드파티 import를 `ROOT=` 위로 올려 E402를 제거**하고, `sys.path`/DLL 뒤에 와야만 하는 `src.*` import에만 `# noqa: E402`를 붙인다.

```python
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
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
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
from src.pipeline.qa_bridge import ask_stream, build_context  # noqa: E402
from src.pipeline.session import SessionManager, save_upload  # noqa: E402
```
> 참고: `import os as _os`·`from pathlib import Path as _Path`의 중복은 스펙대로 **그대로 유지**(SP-A 정리 대상 아님). `logging_config` import는 **여기서 추가하지 않는다** — 해당 파일은 Task 4에서 생성되며, Task 5에서 import를 추가한다.

- [ ] **Step 4: 재정렬 후 정렬 확정 + 위반 0 확인**

Run: `ruff check --fix .`  (상단 서드파티 블록 정렬 확정; 장벽 뒤 `# noqa` import는 이동되지 않음)
Run: `ruff check .`
Expected: `All checks passed!` (위반 0). 남은 위반이 있으면 해당 라인을 검토해 수정.
> 이 `--fix`가 미사용 import(예: 평문 `os`, `FileResponse`)를 제거하고 `from fastapi.responses import (...)`를 한 줄로 정리한다. **Step 3의 블록은 순서·noqa가 핵심인 안전한 상위집합**이며, 최종 기준은 `ruff check`가 0이 되는 것이다. 단, `# noqa: E402`가 붙은 import와 DLL 블록의 상대 순서는 보존돼야 한다.

- [ ] **Step 5: 회귀 게이트 — 테스트**

Run: `python -m pytest tests/ -q`
Expected: `16 passed`

- [ ] **Step 6: GPU import 순서 보존 실증**

Run: `python scripts/check_gpu_ort.py`
Expected: `CUDAExecutionProvider` 활성 + 1프레임 GPU 추론 OK.
> CPU 전용 환경에서 실행하는 경우: 위 스크립트가 CUDA 미가용을 보고할 수 있음. 그때는 대신 `python -c "import src.pipeline.app"`가 예외 없이 import되는지로 순서 무결성을 확인하고, 결과를 보고에 명시한다.

- [ ] **Step 7: 커밋**

```bash
git add -u
git commit -m "fix(quality): ruff --fix 자동수정 + app.py 임포트 재정렬

- import 정렬·미사용 제거·pyupgrade 자동 적용
- app.py: 서드파티 import를 sys.path 위로 이동, DLL 등록→onnxruntime 순서 보존
- 지연 src.* import에 # noqa: E402, GPU EP 활성 실증(check_gpu_ort)"
```

---

## Task 4: `logging_config.py` 작성 (TDD)

**Files:**
- Create: `src/pipeline/logging_config.py`
- Create: `tests/test_logging_config.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_logging_config.py`:
```python
"""configure_logging() — 멱등성·레벨·stdlib 인터셉트 검증."""
import io
import logging

from loguru import logger

import src.pipeline.logging_config as lc


def _reset() -> None:
    # 테스트 간 격리: 모듈 1회-구성 플래그 리셋
    lc._CONFIGURED = False


def test_configure_logging_is_idempotent():
    _reset()
    lc.configure_logging()
    lc.configure_logging()  # 두 번째 호출은 no-op, 예외 없어야 함
    assert lc._CONFIGURED is True


def test_logger_emits_at_info_level():
    _reset()
    lc.configure_logging(level="INFO")
    sink = io.StringIO()
    sink_id = logger.add(sink, level="INFO", format="{message}")
    try:
        logger.info("hello-edge-sign")
    finally:
        logger.remove(sink_id)
    assert "hello-edge-sign" in sink.getvalue()


def test_stdlib_logging_is_intercepted():
    _reset()
    lc.configure_logging(level="DEBUG")
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG", format="{message}")
    std = logging.getLogger("uvicorn.error")
    std.setLevel(logging.DEBUG)
    try:
        std.info("via-stdlib")
    finally:
        logger.remove(sink_id)
    assert "via-stdlib" in sink.getvalue()
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_logging_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.pipeline.logging_config'` (또는 `_CONFIGURED`/`configure_logging` 부재).

- [ ] **Step 3: `logging_config.py` 구현**

`src/pipeline/logging_config.py`:
```python
"""중앙 로깅 설정 — loguru 싱크 + 표준 logging 인터셉트.

서버 startup에서 configure_logging()을 1회 호출한다.
라이브러리 모듈은 `from loguru import logger`만 사용(설정은 여기서 일원화).
"""
from __future__ import annotations

import logging
import os
import sys

from loguru import logger

_CONFIGURED = False


class InterceptHandler(logging.Handler):
    """표준 logging 레코드를 loguru로 전달 (uvicorn/fastapi 로그 통합)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(level: str | None = None) -> None:
    """loguru 싱크를 stderr로 설정하고 표준 logging을 인터셉트한다.

    멱등(idempotent) — 여러 번 호출해도 싱크는 1회만 구성된다.

    Args:
        level: 로그 레벨. None이면 EDGE_SIGN_LOG_LEVEL 환경변수(기본 "INFO").
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    log_level = (level or os.environ.get("EDGE_SIGN_LOG_LEVEL", "INFO")).upper()

    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        backtrace=False,
        diagnose=False,
    )
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        lg = logging.getLogger(name)
        lg.handlers = [InterceptHandler()]
        lg.propagate = False

    _CONFIGURED = True
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_logging_config.py -q`
Expected: `3 passed`

- [ ] **Step 5: 전체 회귀 + 정적 검사(신규 파일)**

Run: `python -m pytest tests/ -q`
Expected: `19 passed` (기존 16 + 신규 3)
Run: `ruff check src/pipeline/logging_config.py tests/test_logging_config.py`
Expected: `All checks passed!`

- [ ] **Step 6: 커밋**

```bash
git add -f src/pipeline/logging_config.py tests/test_logging_config.py
git commit -m "feat(quality): 중앙 loguru 로깅 설정 + stdlib 인터셉트 (TDD)

- configure_logging(): 멱등 stderr 싱크, EDGE_SIGN_LOG_LEVEL 레벨
- InterceptHandler: uvicorn/fastapi 표준 로깅 → loguru 라우팅
- tests/test_logging_config.py 3건"
```

---

## Task 5: loguru 전환 — 운영 print → logger (C4)

**Files:**
- Modify: `src/pipeline/app.py` (logger·configure_logging import 추가, print 4곳 전환, startup에서 configure_logging 호출)
- Modify: `src/pipeline/e2e_pipeline.py` (logger import, `__init__` print 7곳 전환)

- [ ] **Step 1: `e2e_pipeline.py`에 logger import 추가**

기존 import 블록(상단)에 third-party로 추가:
```python
from loguru import logger
```
> 위치는 다른 서드파티 import 근처. 이어서 Step 4의 `ruff check --fix`가 정렬을 확정한다.

- [ ] **Step 2: `e2e_pipeline.py` `__init__` print 7곳 → logger**

`EdgeSignPipeline.__init__` 내부의 아래 7개 `print`를 각각 교체한다(현재 줄번호 기준; Task 3의 F541 자동수정으로 플레이스홀더 없는 항목은 `f` 접두어가 제거됐을 수 있음 — 내용/위치로 매칭).

```python
# (구) print(f"[Pipeline] YOLOv8s variant '{name}' loaded: " f"{Path(path).name} ({self._yolo_meta[name]['mb']} MB)")
logger.info(f"YOLOv8s variant '{name}' 로드: {Path(path).name} ({self._yolo_meta[name]['mb']} MB)")

# (구) print(f"[Pipeline] WARNING: variant '{name}' ONNX not found: {path}")
logger.warning(f"variant '{name}' ONNX 없음: {path}")

# (구) print("[Pipeline] WARNING: YOLOv8s ONNX not found -- detection disabled")
logger.warning("YOLOv8s ONNX 없음 — 검출 비활성")

# (구) print(f"[Pipeline] KoreanOCRNet loaded: {Path(ocr_onnx).name}")
logger.info(f"KoreanOCRNet 로드: {Path(ocr_onnx).name}")

# (구) print("[Pipeline] WARNING: OCR ONNX not found -- signboard OCR disabled")
logger.warning("OCR ONNX 없음 — signboard OCR 비활성")

# (구) print(f"[Pipeline] Korean classifier loaded: {Path(tsign_onnx).name} " f"({len(self._kcls['names'])} classes)")
logger.info(f"한국 분류기 로드: {Path(tsign_onnx).name} ({len(self._kcls['names'])} classes)")

# (구) print("[Pipeline] INFO: 분류기 ONNX 없음 -- 라벨 = class_name")
logger.info("분류기 ONNX 없음 — 라벨 = class_name")
```
> `main()`(`__main__` CLI)의 `print`(DryRun/json/Info/Error)는 **건드리지 않는다** — CLI 사용자 출력.

- [ ] **Step 3: `app.py` — import 추가 + configure_logging 호출 + print 4곳 전환**

(a) Task 3에서 만든 import 블록의 서드파티 영역에 추가:
```python
from loguru import logger
```
(b) `src.*` `# noqa: E402` 묶음에 한 줄 추가(이제 파일이 존재함):
```python
from src.pipeline.logging_config import configure_logging  # noqa: E402
```
(c) `startup()` 본문 — `configure_logging()`를 **맨 앞**에 호출, 완료 로그를 logger로:
```python
@app.on_event("startup")
async def startup():
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
```
(d) WS 핸들러의 `print` 3곳 교체:
```python
# (구) print("[WS] 클라이언트 연결")
logger.info("WS stream 연결")

# (구) print("[WS] 클라이언트 연결 해제")
logger.info("WS stream 연결 해제")

# (구) print(f"[WS] 오류: {e}")
logger.exception("WS stream 처리 중 오류")
```
> `logger.exception`은 `except Exception as e:` 블록 안에서 트레이스백을 자동 포함한다(기존 `f"{e}"`보다 진단성↑). 핫루프(프레임 처리)에는 로그를 추가하지 않는다.

- [ ] **Step 4: 정렬·린트 확정**

Run: `ruff check --fix . && ruff check .`
Expected: `All checks passed!`

- [ ] **Step 5: 회귀 게이트 — 테스트**

Run: `python -m pytest tests/ -q`
Expected: `19 passed`

- [ ] **Step 6: 서버 스모크 — 부팅 + 상태 + loguru 출력**

Run (PowerShell·bash 공통 — 한 줄):
```
python -c "from fastapi.testclient import TestClient; from src.pipeline.app import app; c=TestClient(app); c.__enter__(); r=c.get('/api/status'); print('STATUS', r.status_code, r.json().get('taxonomy'))"
```
Expected: stderr에 loguru 포맷 시작 로그(`... | INFO | src.pipeline... - 파이프라인 초기화 완료 ...`)가 보이고, stdout에 `STATUS 200 v3`.
> `c.__enter__()`가 startup 이벤트(모델 로드 + configure_logging)를 실행한다(컨텍스트 종료는 생략 — 스모크용). 모델 파일은 `model_space/`에 존재. bash heredoc은 Windows PowerShell에서 동작하지 않으므로 위 단일 명령을 사용한다.

- [ ] **Step 7: 커밋**

```bash
git add -u
git commit -m "feat(quality): 운영 로그 print→loguru 전환 (app·e2e_pipeline)

- app.py: startup에서 configure_logging() 호출, 서버/WS 로그 logger 전환
- e2e_pipeline.py __init__ 모델 로드 로그 logger 전환
- CLI(__main__·_test·eval_e2e) 사용자 출력 print는 유지, 핫루프 로깅 없음"
```

---

## Task 6: 엄격 타입 + mypy 게이트 (C5)

**Files:**
- Modify: `src/pipeline/app.py` (엔드포인트 반환 타입)
- Modify: `src/pipeline/e2e_pipeline.py` (미주석 함수 반환 타입)
- Modify: 필요 시 `src/pipeline/{qa_bridge,sources,session}.py` (mypy 보고분)

- [ ] **Step 1: 현재 mypy 위반 파악**

Run: `mypy src/pipeline`
Expected: `disallow_untyped_defs`/`warn_return_any` 등으로 다수 오류 보고. 모듈이 `pipeline.app`처럼 잡혀 override가 안 먹으면, `pyproject.toml`의 `explicit_package_bases=true`·`mypy_path="."` 확인(필요 시 `mypy -p src.pipeline`로 실행).

- [ ] **Step 2: 알려진 미주석 함수에 반환 타입 추가**

`app.py` 엔드포인트/이벤트 핸들러:
```python
async def startup() -> None: ...
async def _shutdown() -> None: ...
async def root() -> HTMLResponse: ...
async def ingest(...) -> JSONResponse | dict: ...        # 현재 dict 또는 JSONResponse 반환
async def ws_stream(websocket: WebSocket) -> None: ...
async def ws_session(websocket: WebSocket) -> None: ...
async def qa_endpoint(req: QARequest) -> StreamingResponse: ...
async def status() -> dict: ...
```
`e2e_pipeline.py` 미주석 함수:
```python
def _preprocess_yolo(self, frame: np.ndarray, w: int = 640, h: int = 640) -> np.ndarray: ...  # 실제 반환값 확인 후 정확히
def reset(self) -> None: ...
def main() -> None: ...
```
> 실제 반환형이 불확실하면 본문을 읽고 정확히 단다(추측 금지). `__init__`은 인자가 이미 주석돼 있어 반환 주석 불필요(mypy 특례).

- [ ] **Step 3: 반복 — mypy 오류를 0까지 수정**

Run: `mypy src/pipeline`
남는 오류 유형별 처리:
- **`Returning Any`(warn_return_any)** — onnxruntime/cv2 호출 반환을 그대로 반환하는 경우: 지역 변수에 정확한 타입을 주석하거나, 불가피하면 해당 라인에 `# type: ignore[no-any-return]`(사유 주석). `warn_unused_ignores=true`이므로 **실제 필요한 곳에만** 추가.
- **`has no attribute`** — `EdgeSignPipeline | None` 접근 시 `if pipeline:` 가드로 좁힌 뒤 접근(이미 대부분 가드됨).
- **dict 구체화** — 반환 `dict`가 모호하면 `dict[str, Any]` 등으로 명시.

각 수정 후 재실행하여 수렴시킨다.
Expected (최종): `Success: no issues found in N source files`

- [ ] **Step 4: 정적·회귀 동시 그린 확인**

Run: `ruff check . && ruff format --check .`
Expected: `All checks passed!` + 재포맷 0.
Run: `python -m pytest tests/ -q`
Expected: `19 passed`
Run: `mypy src/pipeline`
Expected: `Success: no issues found`

- [ ] **Step 5: 커밋**

```bash
git add -u
git commit -m "feat(quality): src/pipeline 엄격 타입 + mypy 게이트 통과

- 엔드포인트/공개 메서드 반환 타입 주석, ML 경계 최소 type: ignore
- mypy src/pipeline 0 오류 (eval_e2e CLI 스크립트 완화)"
```

---

## 완료 기준 (스펙 §5 대조)

- [ ] `ruff check .` → `All checks passed!`
- [ ] `ruff format --check .` → 재포맷 0
- [ ] `mypy src/pipeline` → `Success: no issues found`
- [ ] `python -m pytest tests/ -q` → `19 passed` (기존 16 + 로깅 3, 회귀 없음)
- [ ] 서버 스모크 `GET /api/status` → `200`, `taxonomy=v3`, loguru 출력 확인
- [ ] `scripts/check_gpu_ort.py` → `CUDAExecutionProvider` 활성 (import 순서 보존)

## 참고 / 다음 단계 메모

- **SP-B(CI) 주의:** `app.py`의 `os.add_dll_directory`는 Windows 전용 API다. Linux 러너에서 mypy 실행 시 `attr-defined` 오류가 날 수 있으니, SP-B에서 CI를 Windows 러너로 두거나 해당 라인에 플랫폼 가드/ignore를 추가한다(SP-A는 Windows `convnext_env` 실행이라 불필요 — 지금 ignore를 넣으면 `warn_unused_ignores`에 걸림).
- 커밋 메시지는 한국어 conventional-commit 스타일, **Co-Authored-By 줄 없음**.
