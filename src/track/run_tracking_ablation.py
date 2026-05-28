"""
E1/E4/E5 양자화 모델 추적 평가 (ablation).

E0 FP32 기준선 대비 검출기 양자화가 MOT 메트릭에 미치는 영향 측정.

사용법:
  python src/track/run_tracking_ablation.py
  python src/track/run_tracking_ablation.py --models E1 E4   # 특정 실험만
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
MODEL_SPACE = ROOT / "model_space"

EXPERIMENTS = {
    "E0": ("yolov8s_signs_fp32.onnx",        "FP32 기준선"),
    "E1": ("yolov8s_signs_w8a8.onnx",        "W8A8 PTQ"),
    "E4": ("yolov8s_signs_w4a16.onnx",       "W4A16 PTQ"),
    "E5": ("yolov8s_signs_smoothquant.onnx", "SmoothQuant+W8A8"),
}

EVAL_SCRIPT = ROOT / "src" / "track" / "eval_tracking.py"


def run_eval(exp_id: str, onnx_name: str, label: str) -> dict | None:
    onnx_path = MODEL_SPACE / onnx_name
    if not onnx_path.exists():
        print(f"[{exp_id}] ⚠️  모델 없음: {onnx_path}")
        return None

    print(f"\n{'='*60}")
    print(f"[{exp_id}] {label}  →  {onnx_name}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-u", str(EVAL_SCRIPT), "--onnx", str(onnx_path)],
        capture_output=True   # bytes mode — Windows CP949 인코딩 문제 회피
    )
    elapsed = time.time() - t0

    # 바이트 → 문자열 (UTF-8 우선, 실패 시 CP949)
    def _decode(b: bytes) -> str:
        if not b:
            return ""
        for enc in ("utf-8", "cp949", "latin-1"):
            try:
                return b.decode(enc)
            except Exception:
                continue
        return b.decode("latin-1")

    stdout = _decode(result.stdout)
    stderr = _decode(result.stderr)

    if stdout:
        print(stdout)
    if stderr:
        for line in stderr.splitlines():
            if any(k in line for k in ["ERROR", "error", "Traceback", "Exception"]):
                print(f"[STDERR] {line}")

    # stdout에서 평균 결과 파싱
    metrics = _parse_metrics(stdout)
    if metrics:
        metrics["exp_id"] = exp_id
        metrics["label"] = label
        metrics["wall_time"] = round(elapsed, 1)
    return metrics


def _parse_metrics(stdout: str | None) -> dict | None:
    """eval_tracking.py '전체 평균 결과' 섹션에서 메트릭 파싱.

    출력 형식:
      MOTA:  0.219
      IDF1:  0.384
      HOTA:  0.487
      DetA:  ...
      FPS:   21.6
      GT:    340  FP: 7  FN: 265  IDSW: 0
    """
    if not stdout:
        return None

    import re
    result: dict = {}

    def _extract(label: str) -> float | None:
        m = re.search(rf"{label}:\s*([-\d.]+)", stdout)
        return float(m.group(1)) if m else None

    mota = _extract("MOTA")
    idf1 = _extract("IDF1")
    hota = _extract("HOTA")
    fps  = _extract("FPS")
    idsw = _extract("IDSW")

    if mota is None or idf1 is None or hota is None:
        return None

    return {
        "mota": mota,
        "idf1": idf1,
        "hota": hota,
        "idsw": int(idsw) if idsw is not None else 0,
        "fps":  fps if fps is not None else 0.0,
    }


def print_summary(all_results: list[dict]):
    print("\n" + "="*70)
    print("추적 ablation 결과 요약 (검출기 양자화 영향)")
    print("="*70)
    header = f"{'ID':<5} {'설명':<20} {'MOTA':>7} {'IDF1':>7} {'HOTA':>7} {'IDSW':>6} {'FPS':>7}"
    print(header)
    print("-"*70)

    e0 = next((r for r in all_results if r["exp_id"] == "E0"), None)

    for r in all_results:
        mota_delta = ""
        if e0 and r["exp_id"] != "E0":
            d = r["mota"] - e0["mota"]
            mota_delta = f" ({d:+.3f})"
        row = (f"{r['exp_id']:<5} {r['label']:<20} "
               f"{r['mota']:>7.3f}{mota_delta}")
        # idf1, hota, idsw, fps
        row2 = (f"  IDF1={r['idf1']:.3f}  HOTA={r['hota']:.3f}  "
                f"IDSW={r['idsw']}  FPS={r['fps']:.1f}")
        print(row)
        print("       " + row2)

    print("\n[민감도 분석]")
    if e0:
        for r in all_results:
            if r["exp_id"] == "E0":
                continue
            dm = r["mota"] - e0["mota"]
            di = r["idf1"] - e0["idf1"]
            dh = r["hota"] - e0["hota"]
            print(f"  {r['exp_id']}: MOTA {e0['mota']:.3f}→{r['mota']:.3f} ({dm:+.3f})"
                  f"  IDF1 {e0['idf1']:.3f}→{r['idf1']:.3f} ({di:+.3f})"
                  f"  HOTA {e0['hota']:.3f}→{r['hota']:.3f} ({dh:+.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*",
                        choices=list(EXPERIMENTS.keys()),
                        default=["E1", "E4", "E5"],
                        help="평가할 실험 ID (기본: E1 E4 E5)")
    parser.add_argument("--include_e0", action="store_true",
                        help="E0 FP32도 재평가 (이미 완료된 경우 불필요)")
    args = parser.parse_args()

    targets = args.models
    if args.include_e0:
        targets = ["E0"] + [t for t in targets if t != "E0"]

    all_results = []

    # E0 기준선 하드코딩 (이미 측정 완료)
    if "E0" not in targets:
        all_results.append({
            "exp_id": "E0", "label": "FP32 기준선",
            "mota": 0.219, "idf1": 0.384, "hota": 0.487,
            "idsw": 0, "fps": 21.6, "wall_time": 0,
        })

    for exp_id in targets:
        onnx_name, label = EXPERIMENTS[exp_id]
        metrics = run_eval(exp_id, onnx_name, label)
        if metrics:
            all_results.append(metrics)

    if all_results:
        print_summary(all_results)
        print("\n위 결과를 docs/EXPERIMENTS.md 추적 결과 표에 기입하세요.")
    else:
        print("결과 없음. ONNX 모델 경로를 확인하세요.")


if __name__ == "__main__":
    main()
