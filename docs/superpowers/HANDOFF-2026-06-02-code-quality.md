# SP-A 코드 품질 — 진행 핸드오프 (2026-06-02)

> 세션 한도 도달로 작성. 다음 세션에서 이 문서로 **무손실 재개**한다.
> 실행 방식: **superpowers:subagent-driven-development** (태스크별 새 서브에이전트 + 2단계 리뷰: 스펙→품질).

## ⚠️ 환경 (필수)
- 인터프리터: `C:\Users\leegy\miniconda3\envs\convnext_env\python.exe` (Python 3.10.19). 셸 기본 `python`은 base 3.13(deps 없음) — 사용 금지.
- 도구 호출: `& "...\convnext_env\python.exe" -m ruff|mypy|pytest ...`
- 브랜치: `feat/code-quality-foundation` (main 아님).
- 베이스라인: pytest 16 passed (확인됨). ruff 0.15.15 / mypy 2.1.0 설치됨.

## 문서
- 스펙: `docs/superpowers/specs/2026-06-02-code-quality-foundation-design.md`
- 계획(6 태스크 전체 코드/명령): `docs/superpowers/plans/2026-06-02-code-quality-foundation.md`

## 커밋 이력 (이 브랜치)
```
be68384  fix(quality): Task 3 — ruff --fix + app.py 임포트 재정렬   ← 구현됨, 리뷰 미완
7ea3cb5  style(quality): Task 2 — ruff format 전체 적용             ← 완료
0de90b3  chore(quality): Task 1 — ruff·mypy·pytest 설정 + dev 의존성 ← 완료
1b84792  docs(plan): SP-A 환경 정정 (Python 3.10)
34748f0  docs(plan): SP-A 구현 계획
14fac8a  docs(spec): SP-A 설계 문서
```

## 태스크 상태
- **Task 1 (설정)**: ✅ DONE. 스펙+품질 리뷰 통과. 품질 리뷰 지적 3건 반영(amend): `.pytest_cache` 제외 제거, `requirements-dev.txt`에 `pytest` 추가, mypy override로 `src.track/detect/quant` ignore_errors(=`mypy src/pipeline` 게이트 격리). 커밋 `0de90b3`.
- **Task 2 (ruff format)**: ✅ DONE. 63파일 포맷, 16 passed, 스코프 정상(excluded dirs 무변경). 순수 포맷이라 스펙 리뷰가 품질까지 커버. 커밋 `7ea3cb5`.
- **Task 3 (ruff --fix + import 지뢰)**: ⏳ **IN PROGRESS** — 구현 완료(`be68384`)·컨트롤러 직접 조사로 안전 확인, 그러나 **형식 2단계 리뷰 미완**(스펙 리뷰 서브에이전트가 세션 한도로 출력 미반환, 품질 리뷰 미착수). **아직 완료 처리 안 함.**
- **Task 4 (logging_config.py, TDD)**: ⬜ 대기.
- **Task 5 (loguru 전환)**: ⬜ 대기.
- **Task 6 (엄격 타입 + mypy 게이트)**: ⬜ 대기.

## Task 3 상세 — 재개 시 먼저 처리할 것

### 구현 결과(`be68384`)와 컨트롤러 검증
- app.py import 블록 **정확**: 서드파티 상단 / `sys.path.insert` 후 / DLL `try: import torch; os.add_dll_directory` 블록이 `from src.pipeline.e2e_pipeline import` **앞** / 지연 `src.*` import에 `# noqa: E402` / `logging_config` import 없음(Task 5에서 추가 예정). `import os`·`FileResponse`는 미사용으로 ruff가 제거(정상).
- 게이트: `ruff check .` == 0, `ruff format --check` clean, **pytest 16 passed**, `scripts/check_gpu_ort.py` → **CUDAExecutionProvider 활성 + GPU 추론 OK**, `import src.pipeline.app` 무예외.
- 전 변경(59파일, +320/−330)은 **동작 보존**: I001 import정렬, F541(f접두 제거, 플레이스홀더 없는 것만), UP008(`super()`), UP007(`Optional[X]`→`X|None`), UP037(따옴표 주석 해제), B905(`zip(...,strict=False)` — **no-op**, 기본이 비엄격), B007(미사용 변수 `_`), UP015(`open(p,"r")`→`open(p)`), W293(빈 줄 공백 제거), `# noqa: B008`(FastAPI File/Form 기본인자 패턴).

### ⚠️ 두 가지 미해결 노트 (사용자 확인 권장)
1. **`--unsafe-fixes` 계획 이탈**: 구현자가 계획에 없던 `ruff check --fix --unsafe-fixes`로 W293(및 zip strict) 정리. 결과는 무해(빈 줄 공백 + `zip strict=False` no-op)하고 최종 상태는 수동 처리와 동일. 단 자율적 unsafe 사용은 규율 이탈 → 향후 태스크에선 안전 수정만.
2. **스펙 긴장 — UP/B가 양자화/연구 파일 현대화**: `base_train_w4a16_qat.py`·`multimodal_*`·`quant/*`·`quantize*.py`의 코드가 super()/zip/Optional→| 등으로 수정됨. 스펙 §6 "양자화 로직 0줄 변경"의 *문자적* 범위(포맷/임포트/타입)는 초과하나 *동작*은 불변(pytest+GPU 확인). UP/B는 승인된 룰셋(§2)이라 전 src/ 적용이 디폴트.
   - **사용자 결정 필요**: (A) 그대로 유지(안전·현대화, 권장) / (B) 비-파이프라인 연구·양자화 파일은 UP·B 미적용으로 되돌리고 `pyproject.toml`에서 해당 파일군 per-file-ignore 추가(연구 파일 원형 보존). 사용자 답에 따라 Task 3 재작업 여부 결정.

### Task 3 재개 절차
1. (사용자가 위 2번 B안을 택하면) `pyproject.toml`에 연구/양자화 파일군 `[tool.ruff.lint.per-file-ignores]`로 UP·B 완화 후 Task 3 재실행. A안이면 현 `be68384` 유지.
2. **스펙 준수 리뷰** 재디스패치(템플릿: `subagent-driven-development/spec-reviewer-prompt.md`). 검증: app.py import 순서·DLL→onnxruntime·noqa E402·logging_config 부재 / GPU EP 활성 / ruff 0·format clean·pytest 16 / **전 diff 동작보존 감사**(특히 양자화·serving 파일, zip no-op 확인). Base `7ea3cb5` → Head `be68384`.
3. **코드 품질 리뷰**(스펙 통과 후, 템플릿: `requesting-code-review/code-reviewer.md`). Base `7ea3cb5` Head `be68384`.
4. 리뷰 통과 시 Task 3 완료 → tracker Task #12 completed.

## Task 4~6 재개 (계획 문서의 해당 Task 전문 사용 — 서브에이전트에 전문 붙여넣기, 파일 읽게 하지 말 것)
- **Task 4**: `tests/test_logging_config.py`(실패 테스트 3건) → `src/pipeline/logging_config.py`(`configure_logging()`+`InterceptHandler`, 멱등, `EDGE_SIGN_LOG_LEVEL`). 19 passed. 커밋. (TDD, 전체 코드 계획에 있음.)
- **Task 5**: app.py 운영 print 4곳 + e2e_pipeline.py `__init__` 7곳 → `logger`. app.py에 `from loguru import logger` + `from src.pipeline.logging_config import configure_logging  # noqa: E402` 추가, `startup()` 첫 줄 `configure_logging()`. e2e_pipeline `__main__`·qa_bridge `_test`·eval_e2e의 CLI print는 **유지**. 핫루프 로깅 금지. 스모크: `python -c "...TestClient(app)...status..."`(계획에 단일 명령 있음). 커밋.
- **Task 6**: 엔드포인트/미주석 함수 반환 타입(app.py 핸들러, e2e_pipeline `_preprocess_yolo`/`reset`/`main`). `mypy src/pipeline` 0까지 반복 수정(ML 경계 최소 `# type: ignore`). ruff+format+pytest+mypy 동시 그린. 커밋.
  - 주의: `mypy src/pipeline`는 30개 파이프라인 에러(타입 전, 정상)에서 0으로 줄여야 함. track/detect/quant는 override로 이미 격리됨.
  - 주의(SP-B): `os.add_dll_directory`는 Windows 전용 — Linux CI mypy에서 `attr-defined` 가능. SP-A(Windows)에선 무관.

## 전체 완료 기준 (스펙 §5)
`ruff check .`==0 · `ruff format --check .` clean · `mypy src/pipeline` 0 · `pytest tests/` 19 passed · 서버 스모크 200/v3 + loguru 출력 · `check_gpu_ort` CUDA 활성.

## 이후 (SP-A 완료 후)
- `superpowers:finishing-a-development-branch`로 마무리(PR/머지 옵션).
- 그 다음 상용화 로드맵: **SP-B**(CI/CD GitHub Actions) → **SP-C**(React+Vite 프론트) → **SP-D**(docker-compose). 각 별도 brainstorming→spec→plan→실행 사이클.
- 커밋 스타일: 한국어 conventional-commit, **Co-Authored-By 줄 없음**.
