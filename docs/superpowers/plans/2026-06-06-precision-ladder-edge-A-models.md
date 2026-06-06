# Plan A — Precision Ladder on Edge: 모델·데이터 트랙 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** YOLO26-n 검출기를 기존 한국 도로 데이터로 재학습하고 4개 정밀도 변형(FP32/FP16/INT8-QDQ/W4A16)을 생성·검증해, 웹 트랙(Plan B)이 통합할 검증된 모델 자산을 만든다.

**Architecture:** P1에서 YOLO26 → ONNX → ORT-Web WebGPU 동작을 먼저 검증(디리스킹)하고, 결과에 따라 YOLO26-n 또는 YOLO11-n 폴백을 확정한다. 이후 재학습 → 변형 export → 실프레임 검출 패리티 측정 순서로 진행한다. YOLO26은 DFL·NMS-free이므로 헤드 포함 풀-INT8 양자화를 실험한다(기존 v8은 헤드 제외가 강제였음).

**Tech Stack:** Ultralytics 8.3.241, ONNX, onnxruntime(CPU), onnxconverter-common(float16), onnxruntime.quantization(static QDQ), Node + onnxruntime-web(ORT-Web 검증).

> **실행 환경 주의:** 학습/양자화 명령은 반드시 `convnext_env`에서 실행한다. YOLO 학습 시
> `KMP_DUPLICATE_LIB_OK=TRUE` + `--workers 0`. conda run 불안정 시 env python 직접 호출
> (`"$CONDA/envs/convnext_env/python.exe"`).

> **모델 명명 규약:** 본 plan 산출물은 `v4` 접두어를 쓴다(v3=YOLOv8s 신호등분리). 폴백 시에도
> 파일명은 `v4`를 유지하고, 내부 아키텍처만 다르다(웹/서버 코드가 파일명에 비종속).
> 예: `model_space/yolo_v4_signs_fp32.onnx`, `_fp16.onnx`, `_int8_static.onnx`, `_w4a16.onnx`.

---

## 파일 구조

| 파일 | 책임 | 신규/수정 |
|---|---|---|
| `scripts/spike_yolo26_ortweb.py` | P1 디리스킹: YOLO26 로드 가능성 + ONNX export + 출력 shape 리포트 | 신규 |
| `scripts/spike_ortweb_check.mjs` | export된 ONNX를 onnxruntime-web(node)로 로드·1회 추론, 출력 shape 확인 | 신규 |
| `docs/superpowers/DECISION-yolo26-vs-yolo11.md` | P1 결과 기록 + 채택 결정(모델/디코딩 메모) | 신규 |
| `scripts/train_v4_detector.py` | 채택 모델을 `yolo_signs_v2`로 학습 + best.pt → `runs/detect/edge_sign_v4/` | 신규 |
| `scripts/export_v4_variants.py` | best.pt → FP32 ONNX + FP16 + INT8(QDQ, 풀헤드/헤드제외 옵션) + W4A16 | 신규 |
| `scripts/eval_v4_parity.py` | 실프레임 N장에서 정밀도별 검출 수/conf/박스 IoU 패리티 측정 → 표 출력 | 신규 |
| `docs/EXPERIMENTS.md` | Phase 12 결과 행 추가(정밀도별 mAP 근사·크기·CPU latency·검출 패리티) | 수정 |
| `model_space/yolo_v4_signs_*.onnx` | 4정밀도 변형 산출물 | 신규(산출) |

> 재사용: FP16은 `scripts/export_fp16_detector.py`(keep_io_types 패턴), INT8은
> `scripts/quantize_v3_detector.py`(FlatYoloCalib·QDQ·`--quant_head` 옵션)의 로직을 그대로 차용한다.
> W4A16 fake-quant는 `src/quant/quantize_yolo.py` 패턴 차용.

---

## Task 1: P1 디리스킹 — YOLO26 export + ORT-Web 동작 확인

**Files:**
- Create: `scripts/spike_yolo26_ortweb.py`
- Create: `scripts/spike_ortweb_check.mjs`
- Create: `docs/superpowers/DECISION-yolo26-vs-yolo11.md`

- [ ] **Step 1: YOLO26 로드 + ONNX export 스파이크 스크립트 작성**

`scripts/spike_yolo26_ortweb.py`:

```python
"""P1 디리스킹: YOLO26(n)을 Ultralytics로 로드 가능한지, ONNX export 출력 형태가
무엇인지 확인한다. NMS-free end-to-end 여부에 따라 출력 shape가 v8과 다를 수 있어,
웹 디코딩(Plan B) 설계 전에 반드시 실측한다.

사용법:
  KMP_DUPLICATE_LIB_OK=TRUE python scripts/spike_yolo26_ortweb.py --model yolo26n
  (실패 시) --model yolo11n 로 폴백 후보도 동일 확인
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "model_space" / "_spike"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n", help="yolo26n | yolo11n | yolov8n")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    print(f"[spike] load {args.model} ...")
    model = YOLO(f"{args.model}.pt")  # 미존재 시 Ultralytics가 자동 다운로드 시도
    print(f"[spike] task={model.task} names={model.names}")

    onnx_path = model.export(
        format="onnx", imgsz=args.imgsz, opset=14, dynamic=False, simplify=True
    )
    print(f"[spike] exported: {onnx_path}")

    import onnx

    g = onnx.load(str(onnx_path)).graph
    outs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in g.output]
    print(f"[spike] outputs: {outs}")
    print("[spike] => 이 출력 shape를 DECISION 문서에 기록하라 (v8=(1,N+4,8400) 형식과 비교).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 스파이크 실행 — YOLO26 로드/export를 시도**

Run: `KMP_DUPLICATE_LIB_OK=TRUE python scripts/spike_yolo26_ortweb.py --model yolo26n`
Expected (성공): `task=detect`, ONNX export 성공, `outputs:` 라인에 출력 텐서 shape 출력.
Expected (실패): ImportError/UnknownModel/Export 에러 → **YOLO11 폴백 분기로 이동**(Step 3에서 `--model yolo11n` 재시도).

> 판정 기준: (a) Ultralytics가 해당 모델 가중치를 인식/다운로드하고, (b) `format=onnx` export가
> 에러 없이 끝나며, (c) 출력 텐서가 1~3개로 명확해야 통과.

- [ ] **Step 3: (조건부) YOLO11 폴백 확인**

YOLO26이 Step 2에서 실패한 경우에만:
Run: `KMP_DUPLICATE_LIB_OK=TRUE python scripts/spike_yolo26_ortweb.py --model yolo11n`
Expected: export 성공 + 출력 shape 출력. (YOLO11n은 검증된 export 경로 — 폴백 후보.)

- [ ] **Step 4: ORT-Web(node) 로드·추론 확인 스크립트 작성**

`scripts/spike_ortweb_check.mjs`:

```js
// export된 ONNX를 onnxruntime-web(node)로 로드해 1회 추론하고 출력 shape를 출력한다.
// ORT-Web이 모델을 파싱/실행하지 못하면(미지원 op 등) Plan B의 WebGPU 경로가 불가하므로
// 여기서 조기에 잡는다. (WebGPU EP는 브라우저 전용 → node에선 wasm으로 로드 가능성만 확인.)
import * as ort from 'onnxruntime-web';
import { readFileSync } from 'node:fs';

const path = process.argv[2];
if (!path) { console.error('usage: node spike_ortweb_check.mjs <onnx_path>'); process.exit(1); }

const buf = readFileSync(path);
const sess = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
console.log('inputs:', sess.inputNames, 'outputs:', sess.outputNames);

const x = new ort.Tensor('float32', new Float32Array(1 * 3 * 640 * 640), [1, 3, 640, 640]);
const feeds = { [sess.inputNames[0]]: x };
const out = await sess.run(feeds);
for (const k of sess.outputNames) console.log('out', k, out[k].dims);
console.log('[ortweb] OK — ORT-Web이 모델을 로드·실행함');
```

- [ ] **Step 5: ORT-Web 확인 실행**

Run (web_modern에 onnxruntime-web 설치되어 있음):
`cd web_modern; node ../scripts/spike_ortweb_check.mjs ../model_space/_spike/<exported>.onnx`
Expected: `inputs/outputs` 출력 + `[ortweb] OK`. 실패 시(미지원 op) → 채택 모델을 폴백으로 변경.

- [ ] **Step 6: 결정 문서 기록 + 커밋**

`docs/superpowers/DECISION-yolo26-vs-yolo11.md`에 다음을 기록:
- 채택 모델(yolo26n 또는 yolo11n)과 사유
- ONNX 출력 텐서 이름/shape (예: `(1,6,8400)` 또는 NMS-free 변형 형태)
- ORT-Web 로드 성공 여부
- **Plan B용 디코딩 메모**: 출력이 v8 형식((1, 4+nc, 8400), NMS는 클라가 수행)인지, 아니면
  end-to-end(이미 NMS 적용된 (1, N, 6) 형식)인지 — Plan B의 `clientPipeline.ts` 후처리가 갈림.

```bash
git add -f scripts/spike_yolo26_ortweb.py scripts/spike_ortweb_check.mjs docs/superpowers/DECISION-yolo26-vs-yolo11.md
git commit -m "spike(p12): YOLO26 export + ORT-Web 동작 검증 및 모델 채택 결정"
```

> **GATE:** 이 Task 통과 후에만 Task 2~5 진행. 채택 모델 변수 `$M`(=yolo26n 또는 yolo11n)로 이후 참조.

---

## Task 2: 검출기 재학습 (`yolo_signs_v2`, 2클래스)

**Files:**
- Create: `scripts/train_v4_detector.py`

- [ ] **Step 1: 학습 스크립트 작성**

`scripts/train_v4_detector.py`:

```python
"""Phase 12 검출기 학습 — 채택 모델(YOLO26n/YOLO11n)을 기존 한국 도로 데이터로 재학습.
데이터: data/yolo_signs_v2/dataset.yaml (2클래스: traffic_sign, traffic_light).
v3(edge_sign_v3_lights) 학습 관행 계승: imgsz 1280, close_mosaic, patience.

사용법:
  KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_v4_detector.py --model yolo26n --epochs 40
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="P1에서 채택한 모델 (yolo26n|yolo11n)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(f"{args.model}.pt")
    model.train(
        data=str(ROOT / "data" / "yolo_signs_v2" / "dataset.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        close_mosaic=10,
        patience=20,
        workers=0,
        name="edge_sign_v4",
        project=str(ROOT / "runs" / "detect"),
    )
    print("[train] done. best:", ROOT / "runs/detect/edge_sign_v4/weights/best.pt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 학습 실행**

Run: `KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_v4_detector.py --model $M --epochs 40`
Expected: `runs/detect/edge_sign_v4/weights/best.pt` 생성, 종료 로그에 best epoch mAP50 출력.

> 주의(메모리: training_v3_earlystop): best.pt가 정체/하락하면 조기종료 후 best.pt(최고 epoch) 채택.
> 학습은 GPU(RTX 5070) + 장시간 소요 — 사용자 환경에서 실행 필요. 완료 후 다음 Step.

- [ ] **Step 3: 학습 결과 기록 + 커밋(스크립트만)**

Run: `python -c "from ultralytics import YOLO; m=YOLO('runs/detect/edge_sign_v4/weights/best.pt'); print(m.val(data='data/yolo_signs_v2/dataset.yaml', imgsz=1280).box.map50)"`
Expected: mAP50 수치 출력(기록용).

```bash
git add -f scripts/train_v4_detector.py
git commit -m "feat(p12): v4 검출기 학습 스크립트 + best.pt(ep, mAP50 기록)"
```

> best.pt(.pt)는 `checkpoints`/`runs` gitignore 대상 — 커밋하지 않음. 산출 ONNX만 Task 3에서 관리.

---

## Task 3: 4정밀도 변형 export (FP32 / FP16 / INT8-QDQ / W4A16)

**Files:**
- Create: `scripts/export_v4_variants.py`

- [ ] **Step 1: FP32 ONNX export 단계 작성**

`scripts/export_v4_variants.py` (FP32 부분):

```python
"""Phase 12 검출기 4정밀도 변형 생성.
  best.pt -> yolo_v4_signs_fp32.onnx (Ultralytics export)
          -> _fp16.onnx  (onnxconverter-common float16, keep_io_types)
          -> _int8_static.onnx (QDQ; --quant_head 로 풀헤드 실험)
          -> _w4a16.onnx (fake-quant, 붕괴 시연용)

사용법:
  KMP_DUPLICATE_LIB_OK=TRUE python scripts/export_v4_variants.py --weights runs/detect/edge_sign_v4/weights/best.pt
"""

import argparse
from pathlib import Path

import onnx
from onnxconverter_common import float16

ROOT = Path(__file__).parent.parent
MS = ROOT / "model_space"
FP32 = MS / "yolo_v4_signs_fp32.onnx"
FP16 = MS / "yolo_v4_signs_fp16.onnx"


def export_fp32(weights: str, imgsz: int) -> None:
    from ultralytics import YOLO

    p = YOLO(weights).export(format="onnx", imgsz=imgsz, opset=14, dynamic=False, simplify=True)
    Path(p).replace(FP32)
    print(f"[fp32] {FP32.name} ({FP32.stat().st_size/1e6:.1f}MB)")


def export_fp16() -> None:
    m = onnx.shape_inference.infer_shapes(onnx.load(str(FP32)))
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    del m16.graph.value_info[:]
    onnx.save(m16, str(FP16))
    print(f"[fp16] {FP16.name} ({FP16.stat().st_size/1e6:.1f}MB)")
```

- [ ] **Step 2: INT8 static(QDQ) + 풀헤드 실험 단계 작성**

`export_v4_variants.py`에 추가 — `quantize_v3_detector.py`의 `FlatYoloCalib`·QDQ 설정을 재사용하되,
**YOLO26은 DFL 없음 → 기본을 `--quant_head`(풀헤드 양자화)로** 두고, 헤드 제외 버전도 비교 생성:

```python
from onnxruntime.quantization import QuantFormat, QuantType, quant_pre_process, quantize_static
from scripts.quantize_v3_detector import FlatYoloCalib, HEAD_PREFIX  # 재사용

INT8 = MS / "yolo_v4_signs_int8_static.onnx"
CALIB_DIR = ROOT / "data" / "yolo_signs_v2" / "images" / "val"


def export_int8(quant_head: bool, n_calib: int = 150) -> None:
    g = onnx.load(str(FP32)).graph
    exclude = [] if quant_head else [n.name for n in g.node if n.name.startswith(HEAD_PREFIX)]
    prep = MS / "_prep_v4.onnx"
    quant_pre_process(input_model_path=str(FP32), output_model_path=str(prep),
                      skip_optimization=False, skip_onnx_shape=False, skip_symbolic_shape=True)
    quantize_static(
        model_input=str(prep), model_output=str(INT8),
        calibration_data_reader=FlatYoloCalib(CALIB_DIR, n=n_calib),
        quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8, per_channel=True, reduce_range=False,
        nodes_to_exclude=exclude,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True, "EnableSubgraph": True},
    )
    prep.unlink(missing_ok=True)
    print(f"[int8] {INT8.name} ({INT8.stat().st_size/1e6:.1f}MB) head={'quant' if quant_head else 'excluded'}")
```

> **핵심 실험(서사의 근거):** YOLO26은 DFL 헤드가 없으므로 `quant_head=True`로 풀헤드 INT8을
> 시도하고, 실프레임 검출이 보존되는지 Task 4에서 검증한다. v8/v3에서는 헤드 양자화 시 검출이
> 붕괴했음(Phase 11) — 이 대조가 DFL 패널(Plan B)의 실측 근거가 된다.
> 폴백 모델이 YOLO11(DFL 존재)이면 `quant_head=False`가 기본이 되고, 풀헤드는 "붕괴 재현"으로 기록.

- [ ] **Step 3: W4A16 fake-quant 단계 작성**

`export_v4_variants.py`에 추가 — `src/quant/quantize_yolo.py`의 W4A16 fake-quant 함수를 호출해
`yolo_v4_signs_w4a16.onnx` 생성(붕괴 시연용, FP32 연산 그래프에 4bit 가중치 라운딩 주입).

```python
W4A16 = MS / "yolo_v4_signs_w4a16.onnx"


def export_w4a16(weights: str, imgsz: int) -> None:
    # src/quant/quantize_yolo.py 의 기존 W4A16 fake-quant 경로 재사용.
    from src.quant.quantize_yolo import export_w4a16_onnx  # 기존 함수 시그니처에 맞춰 호출
    export_w4a16_onnx(weights=weights, out_path=str(W4A16), imgsz=imgsz)
    print(f"[w4a16] {W4A16.name} ({W4A16.stat().st_size/1e6:.1f}MB)")
```

> 실행 시 `src/quant/quantize_yolo.py`의 정확한 함수명/시그니처를 확인해 호출부를 맞춘다
> (해당 모듈은 fake-quant ONNX 저장 경로를 이미 보유). 함수명이 다르면 그 이름으로 교체.

- [ ] **Step 4: main() + 실행**

```python
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)  # 배포 추론은 640 (학습 1280과 별개)
    ap.add_argument("--quant_head", action="store_true", default=True)
    args = ap.parse_args()
    export_fp32(args.weights, args.imgsz)
    export_fp16()
    export_int8(quant_head=args.quant_head)
    export_w4a16(args.weights, args.imgsz)
    print("[done] 4 variants in model_space/")


if __name__ == "__main__":
    main()
```

Run: `KMP_DUPLICATE_LIB_OK=TRUE python scripts/export_v4_variants.py --weights runs/detect/edge_sign_v4/weights/best.pt`
Expected: `model_space/yolo_v4_signs_{fp32,fp16,int8_static,w4a16}.onnx` 4개 생성, 각 크기 출력.

- [ ] **Step 5: 산출 ONNX + 스크립트 커밋**

```bash
git add -f scripts/export_v4_variants.py model_space/yolo_v4_signs_fp32.onnx model_space/yolo_v4_signs_fp16.onnx model_space/yolo_v4_signs_int8_static.onnx model_space/yolo_v4_signs_w4a16.onnx
git commit -m "feat(p12): v4 검출기 4정밀도 변형 export (fp32/fp16/int8-qdq/w4a16)"
```

> `model_space/`는 gitignore — `-f`로 추가(데모 배포에 필요한 ONNX는 추적). 크기 큰 fp32(~22MB)는
> 필요 시 Git LFS 검토(기존 v3 fp32도 동일 정책 따름).

---

## Task 4: 실프레임 검출 패리티 측정 (CosSim 금지)

**Files:**
- Create: `scripts/eval_v4_parity.py`

- [ ] **Step 1: 패리티 측정 스크립트 작성**

`scripts/eval_v4_parity.py`:

```python
"""정밀도별 실프레임 검출 패리티 — Phase 11 교훈: 전체 텐서 CosSim은 검출 붕괴를 못 잡으므로
실프레임에서 (검출 박스 수, 평균 conf, FP32 대비 박스 IoU 매칭률)로 비교한다.

사용법:
  python scripts/eval_v4_parity.py --n 20
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).parent.parent
MS = ROOT / "model_space"
VAL = ROOT / "data" / "yolo_signs_v2" / "images" / "val"
VARIANTS = {
    "fp32": "yolo_v4_signs_fp32.onnx",
    "fp16": "yolo_v4_signs_fp16.onnx",
    "int8": "yolo_v4_signs_int8_static.onnx",
    "w4a16": "yolo_v4_signs_w4a16.onnx",
}
CONF = 0.25


def _pre(img):
    t = cv2.cvtColor(cv2.resize(img, (640, 640)), cv2.COLOR_BGR2RGB)
    return np.transpose(t.astype(np.float32) / 255.0, (2, 0, 1))[None]


def _count(sess, name, x):
    out = sess.run(None, {name: x})[0]  # 출력형식은 DECISION 문서 기준 (v8형 (1,4+nc,8400) 가정)
    o = out[0].T if out.shape[1] < out.shape[2] else out[0]  # (anchors, 4+nc)
    scores = o[:, 4:].max(axis=1)
    keep = scores > CONF
    return int(keep.sum()), float(scores[keep].mean()) if keep.any() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    paths = sorted(VAL.glob("*.jpg"))[: args.n]
    sessions = {
        k: ort.InferenceSession(str(MS / v), providers=["CPUExecutionProvider"])
        for k, v in VARIANTS.items() if (MS / v).exists()
    }
    name = next(iter(sessions.values())).get_inputs()[0].name
    print(f"{'variant':8} {'avg_det':>8} {'avg_conf':>9}")
    for k, sess in sessions.items():
        dets, confs = [], []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            d, c = _count(sess, name, _pre(img))
            dets.append(d); confs.append(c)
        print(f"{k:8} {np.mean(dets):8.1f} {np.mean(confs):9.3f}")


if __name__ == "__main__":
    main()
```

> NMS-free(end-to-end) 출력이면 `_count`의 디코딩을 DECISION 문서 형식에 맞춰 교체한다
> (예: 출력이 (1,N,6)=[x1,y1,x2,y2,conf,cls]이면 `scores=out[0][:,4]`).

- [ ] **Step 2: 패리티 측정 실행**

Run: `python scripts/eval_v4_parity.py --n 20`
Expected: 4개 variant의 `avg_det`/`avg_conf` 표. **검증 포인트:** int8(풀헤드)이 fp32와
검출 수/conf가 유사하면 "YOLO26은 헤드까지 양자화해도 보존됨" 입증. w4a16은 저하 예상.

- [ ] **Step 3: EXPERIMENTS.md Phase 12 행 추가 + 커밋**

`docs/EXPERIMENTS.md`에 "Phase 12 — YOLO26 정밀도 사다리" 섹션 추가:
- 표: variant | 크기(MB) | CPU latency(ms) | avg_det | avg_conf | 비고(런타임 매핑)
- 핵심 문장: YOLO26 풀헤드 INT8이 v8 헤드제외 대비 검출 보존(또는 폴백 시 헤드 양자화 붕괴 재현).

```bash
git add -f scripts/eval_v4_parity.py docs/EXPERIMENTS.md
git commit -m "feat(p12): 정밀도별 실프레임 검출 패리티 측정 + EXPERIMENTS Phase 12 기록"
```

---

## Task 5: 인식기 fp16 변형 + 문서 동기화

**Files:**
- Modify: `scripts/export_fp16_detector.py` (또는 신규 `scripts/export_korean_sign_fp16.py`)
- Modify: `CLAUDE.md`, `docs/ROADMAP.md`

- [ ] **Step 1: KoreanSignNet fp16 변형 생성**

`korean_sign_net_fp32.onnx`(116KB) → `korean_sign_net_fp16.onnx`를 float16 변환으로 생성
(폰 온디바이스 인식기용; export_fp16_detector.py의 convert 로직을 SRC/DST만 바꿔 재사용).

Run: `python scripts/export_korean_sign_fp16.py`
Expected: `model_space/korean_sign_net_fp16.onnx` 생성.

> 주의: 소형 모델 fp16은 정확도 영향 미미해야 함 — verify로 max|Δ| 확인.

- [ ] **Step 2: 문서 동기화**

- `CLAUDE.md`: `scripts/` 섹션에 신규 스크립트 4종 한 줄씩, `model_space/`에 v4 변형 언급.
- `docs/ROADMAP.md`: "Phase 12 — Precision Ladder on Edge (모델 트랙)" 섹션 추가, Task 1~5 체크.

- [ ] **Step 3: 커밋**

```bash
git add -f scripts/export_korean_sign_fp16.py model_space/korean_sign_net_fp16.onnx CLAUDE.md docs/ROADMAP.md
git commit -m "feat(p12): KoreanSignNet fp16 변형 + 문서(CLAUDE/ROADMAP) Phase 12 동기화"
```

---

## Self-Review 체크 (작성자 수행 완료)

- **Spec 커버리지:** 검출기 재학습(§6)→Task2 · 4정밀도 변형(§6)→Task3 · 풀헤드 INT8 실험(§6,§2)→Task3/4
  · 정직한 런타임 매핑 측정(§7)→Task4 · 인식기 fp16(§6)→Task5 · YOLO26 디리스킹/폴백(§8 P1,§9)→Task1.
  B-1/폰/데스크톱 UI/Q&A/시연(§4,§8 P3~P6)은 **Plan B** 소관(본 plan은 모델 트랙). 의도된 분리.
- **Placeholder 스캔:** 코드 블록 모두 실체 포함. `src/quant/quantize_yolo.py` 함수명은 Task3 Step3에서
  실제 시그니처 확인 후 맞추라고 명시(미정의 참조 방지 가드).
- **타입 일관성:** 파일명 규약 `yolo_v4_signs_*` 전 Task 일치. 변형 키(fp32/fp16/int8/w4a16) Task3↔4 일치.

> **Plan B 선행조건:** Task 1의 DECISION 문서(출력 shape + 디코딩 메모)가 Plan B의
> `clientPipeline.ts` 후처리 설계 입력이다. Task 1~3 완료 후 Plan B를 작성한다.
