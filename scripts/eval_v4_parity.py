"""정밀도별 실프레임 검출 패리티 — Phase 11 교훈: 전체 텐서 CosSim은 검출 붕괴를 못 잡으므로
실프레임에서 (검출 박스 수, 평균 conf)로 비교한다.

YOLO26 출력은 NMS-free (1,300,6)=[x1,y1,x2,y2,conf,cls] (DECISION-yolo26-vs-yolo11.md).
→ conf 필터만으로 검출 수/평균 conf 산출 (NMS 불필요).

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
    "int8_fullhead": "yolo_v4_signs_int8_static.onnx",  # 헤드 포함 → 붕괴 시연
    "int8_headexcl": "yolo_v4_signs_int8_head_excluded.onnx",  # 헤드 제외 → 배포용
    "w4a16": "yolo_v4_signs_w4a16.onnx",
}
CONF = 0.25


def _pre(img: np.ndarray) -> np.ndarray:
    t = cv2.cvtColor(cv2.resize(img, (640, 640)), cv2.COLOR_BGR2RGB)
    return np.transpose(t.astype(np.float32) / 255.0, (2, 0, 1))[None]


def _decode(out: np.ndarray) -> tuple[int, float]:
    """YOLO26 (1,300,6) end-to-end. conf=col4. (폴백 검출기가 (1,4+nc,8400)이면 여기 교체.)"""
    o = out[0]  # (300, 6)
    if o.ndim == 2 and o.shape[1] == 6:
        conf = o[:, 4]
    else:  # v8형 (4+nc, 8400) 안전 폴백
        oo = o.T if o.shape[0] < o.shape[1] else o
        conf = oo[:, 4:].max(axis=1)
    keep = conf > CONF
    return int(keep.sum()), float(conf[keep].mean()) if keep.any() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    paths = sorted(VAL.glob("*.jpg"))[: args.n]
    sessions = {}
    for k, v in VARIANTS.items():
        if not (MS / v).exists():
            continue
        try:
            sessions[k] = ort.InferenceSession(
                str(MS / v), providers=["CPUExecutionProvider"]
            )
        except Exception as e:  # noqa: BLE001 — 로드 실패 variant는 건너뛰고 보고
            print(f"[skip] {k}: 로드 실패 — {str(e).splitlines()[-1][:90]}")
    if not sessions:
        raise SystemExit("variant ONNX가 없음 — 먼저 export_v4_variants.py 실행")
    print(f"{'variant':8} {'avg_det':>8} {'avg_conf':>9}  (n={len(paths)} frames, conf>{CONF})")
    for k, sess in sessions.items():
        inp = sess.get_inputs()[0]
        dt = np.float16 if "float16" in inp.type else np.float32
        dets, confs = [], []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            out = sess.run(None, {inp.name: _pre(img).astype(dt)})[0]
            d, c = _decode(out)
            dets.append(d)
            confs.append(c)
        print(f"{k:8} {np.mean(dets):8.1f} {np.mean(confs):9.3f}")


if __name__ == "__main__":
    main()
