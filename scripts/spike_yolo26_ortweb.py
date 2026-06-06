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
    ap.add_argument("--nms", action="store_true", help="export 시 NMS 내장(end-to-end) 강제")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    print(f"[spike] load {args.model} ...")
    model = YOLO(f"{args.model}.pt")  # 미존재 시 Ultralytics가 자동 다운로드 시도
    print(f"[spike] task={model.task} names={model.names}")

    kwargs = dict(format="onnx", imgsz=args.imgsz, opset=14, dynamic=False, simplify=True)
    if args.nms:
        kwargs["nms"] = True
    onnx_path = model.export(**kwargs)
    print(f"[spike] exported: {onnx_path}")

    import onnx

    g = onnx.load(str(onnx_path)).graph
    ins = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.input]
    outs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in g.output]
    print(f"[spike] inputs:  {ins}")
    print(f"[spike] outputs: {outs}")
    print("[spike] => 이 출력 shape를 DECISION 문서에 기록하라 (v8=(1,4+nc,8400) 형식과 비교).")


if __name__ == "__main__":
    main()
