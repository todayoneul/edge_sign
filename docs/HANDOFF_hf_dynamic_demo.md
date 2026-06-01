# 작업 이어하기 노트 — HF Space 원격 데모 + 다이나믹 시연

> 이 문서는 한도/세션 중단 시 **다음 세션이 이어서 작업**하기 위한 상태 기록이다.
> 작업 브랜치: `feat/hf-space-dynamic-demo` · 계획 전문: `~/.claude/plans/superpowers-skills-wondrous-owl.md`
> 마지막 갱신: 2026-06-01

## 목표 (확정)
1. HF Spaces **Docker** 원격 데모 (CPU + INT8, 현 커스텀 UI 유지)
2. 다이나믹 시연: **양자화 A/B 토글**(단일 스트림+토글), **파이프라인 단계 시각화**(stage_ms), **Q&A 서사 강화**
3. UI는 `ui-ux-pro-max` 스킬로 신규 컴포넌트 설계
4. Q&A 키 = **BYOK**(방문자 본인 Groq 키), 데모 영상 소수 동봉 + 시연자 직접 입력

## 진행 상태 (체크리스트)

### ✅ 완료
- [x] **백엔드 variant 다중 로드 + stage_ms** (`src/pipeline/e2e_pipeline.py`)
  - `EdgeSignPipeline(yolo_variants={"fp32":..,"int8":..})`, `set_variant()`, `variant_info()`, `active_variant`
  - `process_frame(frame, variant=None)` → 반환에 `variant`, `model_mb`, `stage_ms{detect,track,recognize}` 추가
  - `_yolo_providers()` — `EDGE_SIGN_CPU_ONLY=1`이면 CPU만 (HF/테스트)
  - 하위호환: 단일 `yolo_onnx` → `{"default": ...}`, `self.yolo_session` 별칭 유지
- [x] **TDD 테스트** `tests/test_pipeline_variants.py` — 6 passed (fp32 + int8_static, CPU EP)
  - ⚠️ int8_**dyn**은 ConvInteger로 CPU EP 미지원 → A/B는 반드시 int8_**static**(QDQ) 사용
- [x] **app.py 연결**: `_resolve_variants()`(v3 fp32 + v3 int8_static 있으면 추가), startup `yolo_variants=`,
      `/ws/stream` 프레임별 `variant`, `/ws/session` `control action:"variant"`, 프레임 payload에 variant/model_mb/stage_ms,
      `/api/status`에 `variants`/`active_variant`
- [x] 전체 회귀 `pytest tests/` → 14 passed (기존 FastAPI on_event DeprecationWarning만)

### ✅ 완료 (추가)
- [x] **#2 Q&A BYOK** — `ask_stream(api_key=)`, `QARequest.api_key`, no-key 안내. tests/test_qa_byok.py 2 passed.
- [x] **#3 v3 INT8 static export** — `scripts/quantize_v3_detector.py` 생성·실행 →
      `model_space/yolov8s_signs_v3_int8_static.onnx` (**18.0 MB, 2.49x**).
      ⚠️ **핵심 교훈:** 헤드(`/model.22/*` cv2/cv3/dfl)까지 INT8화하면 **검출 0개로 붕괴**(CosSim 0.9995여도!).
      `nodes_to_exclude`로 헤드 FP32 유지 → 검출 수 11/12 일치, conf~0.40 동일. **CosSim 말고 실검출로 검증할 것.**
      재생성: `KMP_DUPLICATE_LIB_OK=TRUE python scripts/quantize_v3_detector.py --bench`
      (모델은 model_space/ gitignore — HF Space엔 LFS로 동봉 필요.)

### ✅ 완료 (추가 2)
- [x] **#4 프론트** (`web/detection/`) — FP32⇄INT8 토글(크기·FPS Δ), 검출→추적→인식 stage_ms 플로우(병목 강조),
      BYOK 키(password+표시토글·localStorage), 샘플 영상 버튼(번들 클립 루프), WS https→wss 자동.
      라이브 /ws/stream variant 라우팅(fp32/int8/폴백) 실측 OK.
- [x] **#5 HF Docker Space** — `Dockerfile`(CPU, EDGE_SIGN_CPU_ONLY=1, 7860, 비root), `requirements-hf.txt`(슬림),
      `.dockerignore`(ONNX 4개만), `spaces/README.md`(YAML+LFS), 샘플 클립 2종 `web/detection/samples/`.
- [x] **#6 문서** — ROADMAP/EXPERIMENTS Phase 11, CLAUDE.md 구조·명령. `pytest` 16 passed, CPU 부팅·샘플 서빙 OK.

### ⏳ 남은 작업 (사용자 액션)
- [ ] **docker build 실측** — Docker Desktop **데몬 꺼짐 → 보류**. 실행 후
      `docker build -t edge-sign . && docker run --rm -p 7860:7860 edge-sign` → http://localhost:7860/detection/
- [ ] **HF Space 푸시** — Space 리포 생성, `spaces/README.md`를 루트 README로, 대용량 onnx·mp4는 **git LFS**
      (model_space는 main 리포 gitignore → Space 리포에 별도 LFS 동봉).

## 핵심 계약 (프론트/배포가 참조)
- `process_frame()`/`/ws/stream` result 및 `/ws/session` frame: `variant`(str), `model_mb`(float), `stage_ms`{detect,track,recognize}
- `/api/status`: `variants`=[{name,mb}], `active_variant`
- 클라 모드 variant 전송: 프레임 msg에 `"variant": "<name>"` (미로드 name이면 서버가 active로 폴백)
- 서버 모드 variant 전송: `{type:"control", action:"variant", value:"<name>"}`
- 데모용 권장 variant: `fp32`(정확도/크기 큼) ⇄ `int8`(int8_static QDQ, ~3.6× 작음)

## 환경/실행 메모
- 서버/테스트 env: **convnext_env** (`conda run -n convnext_env ...`)
- 테스트: `set KMP_DUPLICATE_LIB_OK=TRUE & set EDGE_SIGN_CPU_ONLY=1 & python -m pytest tests/ -q`
- 데모 클립: `data/demo_videos/d_validation_1920_1080_daylight_2_clips/` (도메인 일치)
- 모델: `model_space/` (gitignore) — v3_fp32 44MB, **v3_int8_static 18MB(생성됨)**, OCR w8a8 2.8MB, 분류기 fp32 116KB

## 세션 운영 지침 (사용자 요청 2026-06-01)
- 한도로 중단 우려 시 모델 자동 교체하되 **sonnet high 미만으로 내려가지 말 것**.
- 끊길 것 같으면 이 문서에 진행사항 저장 후 중단 → 다음 세션이 이어받음.
