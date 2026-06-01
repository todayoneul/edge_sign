# 코드 품질 기반 (SP-A) — 설계 문서

- **작성일**: 2026-06-02
- **상태**: 설계 합의 완료 (구현 계획 대기)
- **범위**: SP-A (ruff · mypy · loguru · pyproject 도입). 상용화 고도화 4대 작업 중 1단계.
- **후속**: SP-B(CI/CD, GitHub Actions) → SP-C(React+Vite 프론트) → SP-D(docker-compose 컨테이너화). 각 별도 spec.

---

## 1. 목표 / 동기

> "연구 코드를 추론 동작 손상 없이, 일관된 포맷·강한 타입·구조화 로깅을 갖춘 상용 수준 백엔드로 끌어올린다."

현재 백엔드(`src/pipeline/`)는 동작은 견고하나 다음이 부족하다.
- 로깅이 전부 `print()` — 레벨·구조·싱크 제어 불가. (`loguru`는 이미 `requirements.txt`에 설치돼 있으나 미사용)
- 코드 포맷·import 정렬이 비일관적이고, 정적 검사 도구(린터/타입체커)가 없다.
- 타입 힌트가 부분적이라 서버/파이프라인 경계의 계약이 불명확하다.

SP-A는 이를 **추론·양자화 로직을 1줄도 바꾸지 않고** 해소한다.

### 성공 기준
- `ruff check .` 0 위반 · `ruff format --check .` clean (적용 범위 내)
- `mypy src/pipeline` 0 에러 (엄격 설정)
- `pytest tests/` **16 passed** (기존과 동일 — 회귀 없음)
- `uvicorn src.pipeline.app:app` 부팅 + `GET /api/status` 200 + WS 프레임 왕복 정상
- `scripts/check_gpu_ort.py` — `CUDAExecutionProvider` 여전히 활성 (import 순서 보존 증명)

---

## 2. 결정 사항 (합의됨)

| 항목 | 결정 | 비고 |
| :--- | :--- | :--- |
| **mypy 엄격 범위** | `src/pipeline/` 만 엄격 | 서버/추론 오케스트레이션 경계 = 타입 가치 최대. 수치·양자화 코드(`src/base_*` 등)는 제외 |
| **ML 라이브러리 경계** | 전역 `ignore_missing_imports=true` | cv2·onnxruntime·torch·groq·ultralytics 스텁 부재 완화 |
| **loguru 적용 범위** | 런타임 서비스(`src/pipeline/`) | 서버·파이프라인 `print()`→`logger.*`. CLI 스크립트(`scripts/`) 사용자 stdout은 유지 |
| **핫루프 로깅** | 미삽입 | WS `/ws/stream`·`/ws/session` 30fps 루프엔 프레임당 로그 금지(성능) |
| **ruff line-length** | 100 | 88은 디프 폭증, 120은 과다. `E501`은 ignore(긴 한글 주석 보호) |
| **ruff 규칙 세트** | `E,F,I,UP,B,W,C4` | 대부분 auto-fix 가능한 실용 베이스라인 |
| **적용 트리** | `src/` + `tests/` + 활성 `scripts/` | `scripts/archive/` 제외. JS/HTML은 SP-C(eslint/prettier) |

---

## 3. 컴포넌트 / 파일

### 신규
- **`pyproject.toml`** — 도구 설정 중앙화 (ruff · mypy · pytest).
  - `[tool.ruff]`: `target-version="py311"`, `line-length=100`,
    `extend-exclude=["scripts/archive","runs","data","model_space","assets","web","checkpoints","models","AIhub",".pytest_cache"]`
  - `[tool.ruff.lint]`: `select=["E","F","I","UP","B","W","C4"]`, `ignore=["E501"]`
  - `[tool.ruff.lint.per-file-ignores]`: `"__init__.py"=["F401"]` (계획된 유일 예외). 그 외는 C3에서 구체적 오탐이 나올 때만 라인 단위 `# noqa`로 처리
  - `[tool.mypy]`: `python_version="3.11"`, `ignore_missing_imports=true`, `warn_unused_ignores=true`
  - `[[tool.mypy.overrides]]` (`module="src.pipeline.*"`): `disallow_untyped_defs=true`,
    `disallow_incomplete_defs=true`, `no_implicit_optional=true`, `warn_return_any=true`, `check_untyped_defs=true`
  - `[tool.pytest.ini_options]`: `testpaths=["tests"]` (기존 동작 명문화)
- **`requirements-dev.txt`** — dev 도구 선언: `ruff`, `mypy` (런타임 `requirements.txt`와 분리; SP-B CI가 동일 파일 사용). `loguru`는 이미 런타임 의존성에 존재.
- **`src/pipeline/logging_config.py`** — `configure_logging()`:
  - loguru stderr 싱크, 레벨 `EDGE_SIGN_LOG_LEVEL`(기본 `INFO`), 타임스탬프·레벨·모듈 포맷.
  - uvicorn/stdlib `logging` 인터셉트 핸들러(표준 로깅을 loguru로 라우팅).
  - **서버 startup에서 1회 호출** — import 시점 부작용 없음(라이브러리 모듈은 `from loguru import logger`만).

### 변경 (로직 무변경 — 포맷/임포트/로깅/타입만)
- **`src/pipeline/app.py`** — `print()`→`logger`; startup에서 `configure_logging()` 호출; 공개 핸들러 타입 주석 보강.
  `@app.on_event` 유지(현행 동작 보존 — lifespan 전환은 SP-A 범위 밖, 회귀 위험 회피).
- **`src/pipeline/e2e_pipeline.py`** — 초기화/경고성 `print()`→`logger`. `process_frame()` 등 핫패스엔 로그 미추가. 타입 주석 보강.
- **`src/pipeline/{session,sources,qa_bridge}.py`** — `print()`→`logger`, 타입 주석 보강.
- 전 대상 트리 — `ruff format` + `ruff check --fix` 적용 결과 반영.

---

## 4. ⚠️ Import-order 지뢰 (핵심 위험)

`app.py:39-48`의 `os.add_dll_directory(<torch/lib>)` 블록은 **onnxruntime import보다 먼저** 실행돼야 GPU(`CUDAExecutionProvider`)가 활성화된다. 두 가지를 보장한다.

1. **순서 보존**: ruff isort는 import 사이의 실행문(`sys.path.insert`, DLL `try/except`)을 **장벽(barrier)**으로 취급해 그 경계를 넘어 import를 재배치하지 않는다. 따라서 현 구조에서 자동 재정렬 위험은 낮으나, **C3 직후 `scripts/check_gpu_ort.py`로 실증**한다.
2. **E402 처리**: `sys.path.insert`·DLL 블록 뒤에 오는 의도적 지연 import(`from src.pipeline... import ...`)는 `E402`(module-import-not-at-top)를 유발한다 → 해당 라인에 `# noqa: E402` 부여(패턴이 의도적임을 명시). 전역 E402 비활성화는 하지 않는다.

> DLL 블록 내부의 `import os as _os` 등 중복 import는 **로직 보존을 위해 그대로 둔다**(SP-A는 정리 대상 아님).

---

## 5. 롤아웃 — 5단계 커밋 (각 단계 후 `pytest tests/` 그린 확인)

| 커밋 | 내용 | 검증 |
| :--- | :--- | :--- |
| **C1** | `pyproject.toml` + `requirements-dev.txt` (설정·dev 의존성, 코드 무변경) | `ruff`/`mypy` 실행 가능 확인 |
| **C2** | `ruff format` 전 대상 트리 (기계적 포맷) | `pytest` 16 passed |
| **C3** | `ruff check --fix` (import 정렬·미사용 제거·pyupgrade) + E402 `# noqa` + 지뢰 수동검토 | `pytest` + `check_gpu_ort` |
| **C4** | loguru 전환 + `logging_config.py` | `pytest` + 서버 스모크(`/api/status`, WS 왕복) |
| **C5** | 타입 힌트 보강 + `mypy src/pipeline` clean | `pytest` + `mypy` |

각 커밋은 한국어 conventional-commit 스타일(예: `chore(quality): ruff 포맷 적용`), **Co-Authored-By 줄 없음**.

---

## 6. 🔒 불변 보장 (사용자 핵심 제약)

- **추론·양자화 로직 0줄 변경** — 변경은 포맷/임포트정렬/`print→logger`/타입주석으로 한정.
- 양자화 ONNX·모델 파일·파이프라인 수치 동작 불변 — A/B(FP32⇄INT8) 토글·stage_ms·검출/추적/OCR 결과 동일.
- DLL 등록 순서 명시 보호 + GPU EP 실증.
- WS 핫루프 프레임당 로깅 금지(지연 누적 방지).
- **매 커밋 16개 테스트 게이트** — 하나라도 깨지면 진행 중단.

---

## 7. 테스트 / 검증

- **회귀**: 기존 `pytest tests/`(FrameSource·세션·ingest·variant·BYOK) 16개 전부 통과 유지.
- **정적**: `ruff check .` 0 · `ruff format --check .` clean · `mypy src/pipeline` 0.
- **GPU**: `scripts/check_gpu_ort.py` — `CUDAExecutionProvider` 활성(import 순서 보존 증명).
- **서버 스모크**: `uvicorn` 부팅 → `GET /api/status` 200(yolo/ocr/tsign true) → `/ws/stream` 프레임 1장 왕복 → loguru 출력 확인.

---

## 8. 범위 밖 (YAGNI)

- **CI 자동화**(ruff/mypy/pytest 워크플로) → SP-B.
- **프론트 린팅**(eslint/prettier/JS 타입) → SP-C.
- `src/` 전역 strict mypy, 수치/양자화 코드 타입 전수 작업.
- `scripts/` CLI stdout의 loguru 전환.
- FastAPI `@app.on_event`→lifespan 마이그레이션(별도 회귀 검증 필요 — 추후).
- 런타임 동작/성능 최적화, 의존성 정리(`requirements.txt` 분할).

---

## 9. 의존 관계 / 비고

- SP-B(CI)는 본 SP-A의 `pyproject.toml` 게이트(ruff/mypy/pytest)를 그대로 워크플로화한다.
- 서버는 `convnext_env` conda 환경에서 실행 — `ruff`·`mypy`는 해당 env(또는 dev 도구)로 실행. `loguru`는 이미 설치됨.
- 본 작업은 `feat/code-quality-foundation` 브랜치에서 진행 후 리뷰/병합.
