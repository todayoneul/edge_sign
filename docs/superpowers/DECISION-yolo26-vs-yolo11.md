# DECISION — 검출기 모델 채택 (Phase 12, P1 디리스킹 결과)

> 작성일: 2026-06-06 · Plan A Task 1 산출물 · **결론: YOLO26-n 채택**

## 채택 결정

**YOLO26-n 채택.** 폴백(YOLO11-n)은 불필요 — YOLO26이 모든 게이트를 통과했다.

## 근거 (실측)

### 1. 환경 — 업그레이드 불필요
- `convnext_env`(학습/서빙 실제 환경): **ultralytics 8.4.56** → YOLO26 지원(8.4.0+, 2026-01-14 출시).
  `cfg/models/26/`에 `yolo26.yaml` 등 10개 cfg 존재.
- base env(기본 셸 python): ultralytics 8.3.241 — YOLO26 미지원이나 **학습에 사용 안 함**. 무관.
- 따라서 사용자 환경 변경/업그레이드 리스크 **없음**.

### 2. ONNX export — 성공
- `YOLO("yolo26n.pt").export(format="onnx", imgsz=640, opset=14, simplify=True)` → 9.5 MB, 1.5s.
- YOLO26n: 122 layers, 2,408,932 params, 5.4 GFLOPs.

### 3. **출력 형식 — NMS-free end-to-end (v8과 근본적으로 다름)** ⚠️ Plan B 핵심
- **입력:** `images` `[1, 3, 640, 640]` float32.
- **출력:** `output0` **`[1, 300, 6]`** float32. (`--nms` 플래그 없이 기본이 이미 end-to-end)
  - 300 = 고정 top-k 후보 (저신뢰 행은 패딩).
  - 6 = `[x1, y1, x2, y2, conf, cls]` (xyxy + 신뢰도 + 클래스 인덱스).
- 대조: YOLOv8/v3는 `(1, 4+nc, 8400)` → **클라이언트가 직접 NMS 수행**해야 함.
- **Plan B 디코딩 메모:** `clientPipeline.ts` 후처리를 다음으로 변경한다.
  - 기존(v8): transpose → score=max(cls) → conf 필터 → **NMS** → 박스.
  - YOLO26: `output0[0]`(300×6) → `conf=row[4] > THRESH` 필터 → 박스 `row[0:4]`, 클래스 `row[5]`.
    **NMS 불필요.** 코드가 단순해짐.
  - **확인 필요(Task 4):** 박스 좌표가 입력 픽셀 스케일(0~640)인지 정규화(0~1)인지 —
    재학습 2클래스 모델로 실프레임에서 확인 후 렌더 스케일링 확정.

### 4. ORT-Web(WebGPU 경로) 호환 — 확인
- **Op 인벤토리(384 노드):** Conv 102 · Mul 90 · Sigmoid 88 · ... · **TopK 2** · GatherElements 2.
  - WebGPU 적대 op(`NonMaxSuppression`, `NonZero`, `RoiAlign`) **전무**. NMS-free라 NMS op 자체가 없음.
  - TopK(2개, 300 후보 대상)는 ORT-Web 지원. 필요 시 wasm 폴백이어도 비용 무시 가능.
- **실로드 검증:** onnxruntime-web 1.22.0(웹 콘솔과 동일 버전)으로 node에서 로드·1회 추론 →
  `out output0 [1,300,6] float32` 정상 반환. (브라우저 WebGPU 최종 FPS는 Plan B에서 스파이크 실측.)

## 서사에 미치는 영향 (긍정)

- YOLO26의 NMS-free + 단순화된 헤드는 **"엣지·양자화 친화"** 서사를 강화한다(CPU 추론 최대 43%↑).
- Phase 11 발견("v8 DFL 헤드는 INT8에서 붕괴 → 헤드 FP32 강제")과 정확히 대비된다:
  YOLO26은 그 헤드 구조를 제거 → **헤드 포함 풀-INT8 양자화 실험 가능**(Plan A Task 3/4에서 검증).
  이 대조가 데스크톱 "DFL 패널"의 실측 근거가 된다.

## 산출 스크립트
- `scripts/spike_yolo26_ortweb.py` — YOLO26 로드 + ONNX export + shape 리포트.
- `scripts/spike_ortweb_check.mjs` — onnxruntime-web 로드·추론 검증(Windows: `pathToFileURL` wasmPaths,
  dist 디렉토리에서 실행 시 glue 상대해석 OK).
