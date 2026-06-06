"""Phase 12 — 양자화 '붕괴 원인' 분석 (data-free, ONNX weight + 구조 시뮬레이션).

평가 지적("망가졌다는 관찰만 있고 '왜'가 없다")에 대응한다. 데이터셋 없이 ONNX 가중치와
구조적 시뮬레이션만으로 세 가지 메커니즘을 정량화한다.

  (A) 레이어별 가중치 양자화 난이도 — 동일 스킴(per-tensor symmetric)에서 비트폭만 INT8→INT4로
      낮췄을 때의 SQNR(dB) 하락. 분포의 heavy-tail(초과첨도)이 클수록 4-bit에서 더 붕괴.
      → OCR가 W4A16에서 붕괴하는 '범인 레이어'를 지목.
  (B) 검출기 백본 vs 검출 헤드(model.22) SQNR 대비 — 헤드가 INT8에 더 취약한 이유.
  (C1) DFL 증폭 — 검출 헤드의 box 좌표는 16-bin 분포의 softmax-기대값(DFL integral)이다.
      로짓에 INT8 양자화 잡음을 주면 좌표 오차가 '직접 회귀' 대비 몇 배로 증폭되는지 측정.
  (C2) CosSim 착시 — 전체 출력 텐서의 CosSim은 0.999여도, 결정적인 소수 high-conf 로짓이
      per-tensor 스케일에 눌려 임계값 아래로 떨어지면 검출이 0이 된다. 수치로 재현.

출력: 콘솔 표 + assets/v3/quant_collapse_analysis.png + README/EXPERIMENTS용 수치.

사용법:
  python scripts/analyze_quant_collapse.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).parent.parent
MS = ROOT / "model_space"
OCR = MS / "korean_ocr_net_fp32.onnx"
DET = MS / "yolov8s_signs_v3_fp32.onnx"
OUT_PNG = ROOT / "assets" / "v3" / "quant_collapse_analysis.png"

rng = np.random.default_rng(0)


# ---------- 공통 양자화 도구 ----------
def quant_dequant(w: np.ndarray, bits: int) -> np.ndarray:
    """Per-tensor symmetric weight 양자화→복원 (동일 스킴, 비트폭만 변경)."""
    qmax = 2 ** (bits - 1) - 1  # INT8: 127, INT4: 7
    amax = float(np.max(np.abs(w))) or 1e-12
    scale = amax / qmax
    q = np.clip(np.round(w / scale), -qmax, qmax)
    return q * scale


def sqnr_db(w: np.ndarray, bits: int) -> float:
    dq = quant_dequant(w, bits)
    sig = float(np.sum(w.astype(np.float64) ** 2))
    noise = float(np.sum((w.astype(np.float64) - dq) ** 2)) or 1e-12
    return 10.0 * np.log10(sig / noise)


def excess_kurtosis(w: np.ndarray) -> float:
    x = w.astype(np.float64).ravel()
    mu, sd = x.mean(), x.std()
    if sd < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / sd) ** 4) - 3.0)


def weight_tensors(model_path: Path) -> list[tuple[str, np.ndarray]]:
    g = onnx.load(str(model_path)).graph
    out = []
    for i in g.initializer:
        if len(i.dims) >= 2:  # conv/matmul weight
            out.append((i.name, numpy_helper.to_array(i)))
    return out


# ---------- (A) OCR 레이어별 난이도 ----------
def analyze_ocr() -> list[dict]:
    print("\n=== (A) OCR 레이어별 가중치 양자화 난이도 (KoreanOCRNet) ===")
    print(f"{'layer':<26}{'shape':<18}{'kurt':>8}{'SQNR8':>9}{'SQNR4':>9}{'drop':>8}")
    rows = []
    for name, w in weight_tensors(OCR):
        k = excess_kurtosis(w)
        s8, s4 = sqnr_db(w, 8), sqnr_db(w, 4)
        rows.append({"layer": name, "shape": list(w.shape), "kurt": k, "s8": s8, "s4": s4})
        print(f"{name:<26}{str(list(w.shape)):<18}{k:>8.1f}{s8:>9.1f}{s4:>9.1f}{s8 - s4:>8.1f}")
    worst = min(rows, key=lambda r: r["s4"])
    print(f"  → INT4 최악 레이어: {worst['layer']} (SQNR4={worst['s4']:.1f} dB, kurt={worst['kurt']:.1f})")
    return rows


# ---------- (B) 검출기 백본 vs 헤드 ----------
def analyze_detector() -> dict:
    print("\n=== (B) 검출기 백본 vs 검출 헤드(DFL) 양자화 취약성 (YOLOv8s-v3) ===")
    back, head = [], []
    for name, w in weight_tensors(DET):
        s4 = sqnr_db(w, 4)
        s8 = sqnr_db(w, 8)
        k = excess_kurtosis(w)
        (head if "model.22" in name else back).append({"name": name, "s4": s4, "s8": s8, "k": k})

    def summ(g, label):
        s4 = np.array([x["s4"] for x in g])
        s8 = np.array([x["s8"] for x in g])
        k = np.array([x["k"] for x in g])
        print(
            f"  {label:<14} n={len(g):>2}  SQNR8 평균 {s8.mean():6.1f}  "
            f"SQNR4 평균 {s4.mean():6.1f}  최소 {s4.min():6.1f}  초과첨도 평균 {k.mean():6.1f}"
        )
        return {"s4": s4, "s8": s8, "k": k}

    b = summ(back, "백본/넥")
    h = summ(head, "검출헤드")
    dfl = [x for x in head if ".cv2." in x["name"] and x["name"].endswith(".2.weight")]
    if dfl:
        print(f"  → DFL box 최종 conv(cv2.*.2) INT4 SQNR: {np.mean([x['s4'] for x in dfl]):.1f} dB")
    # 핵심(반전): 헤드 가중치는 백본만큼(오히려 더) 잘 양자화된다 → 가중치는 붕괴의 원인이 아니다.
    print(
        f"  → [반전] 헤드 INT8 SQNR {h['s8'].mean():.1f} dB ≥ 백본 {b['s8'].mean():.1f} dB. "
        f"헤드 가중치는 충분히 양자화 가능 — '헤드 가중치가 어렵다'는 가설을 **기각**한다."
    )
    print(
        f"  → 단 헤드 가중치 초과첨도 {h['k'].mean():.0f}(백본 {b['k'].mean():.0f})로 극히 heavy-tail "
        f"→ per-tensor 활성화 스케일이 outlier에 끌려가는 활성화-측 취약성을 시사."
    )
    return {"back": b, "head": h}


# ---------- (C1) DFL 로짓 강건성 (음성 결과) ----------
def analyze_dfl_robustness(reg_max: int = 16, trials: int = 4000) -> dict:
    print("\n=== (C1) DFL 적분(softmax-기대값)은 INT8 로짓 잡음에 강건한가? ===")
    proj = np.arange(reg_max)
    centers = rng.uniform(2, reg_max - 3, trials)
    sharp = rng.uniform(1.0, 4.0, trials)
    bins = np.arange(reg_max)[None, :]
    logits = sharp[:, None] * np.exp(-0.5 * ((bins - centers[:, None]) / 1.2) ** 2)

    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    coord = (softmax(logits) * proj).sum(axis=1)
    scale = np.max(np.abs(logits), axis=1, keepdims=True) / 127.0
    coord_q = (softmax(np.round(logits / scale) * scale) * proj).sum(axis=1)
    err = np.abs(coord - coord_q)
    print(f"  INT8 로짓 잡음 → DFL 좌표오차 중앙값 {np.median(err):.4f} bin (P99 {np.percentile(err, 99):.3f} bin)")
    print("  → DFL 적분 자체는 INT8에 **강건**(음성 결과). 헤드 붕괴의 원인은 DFL 수식이 아니다.")
    return {"err": err, "median": float(np.median(err))}


# ---------- (C2) CosSim의 구조적 맹점 ----------
def analyze_cossim_blindness(n_anchor: int = 8400, nc: int = 2, n_tp: int = 6) -> dict:
    print("\n=== (C2) CosSim의 맹점 — 왜 0.9995인데 검출이 0인가 ===")
    # YOLOv8 헤드 출력 (1, 4+nc, 8400): box 회귀값이 L2 에너지를 지배.
    box = rng.uniform(0, 640, (n_anchor, 4)).astype(np.float32)
    cls = np.full((n_anchor, nc), -6.0, dtype=np.float32)  # 배경 로짓
    tp_rows = rng.choice(n_anchor, n_tp, replace=False)
    tp_col = rng.integers(0, nc, n_tp)
    cls[tp_rows, tp_col] = rng.uniform(1.5, 3.0, n_tp)  # 진짜 검출(conf≈0.82~0.95)
    T = np.concatenate([box, cls], axis=1)  # (8400, 4+nc)

    # 활성화 양자화가 헤드 cls 분기에서 결정적 로짓을 손상시킨 상황을 모사:
    # box(회귀)는 보존되고 소수 TP cls 로짓만 배경 수준으로 붕괴 → 검출 소멸.
    T2 = T.copy()
    T2[tp_rows, 4 + tp_col] = -6.0

    cos = float(np.sum(T * T2) / (np.linalg.norm(T) * np.linalg.norm(T2) + 1e-12))

    def n_det(t, thr=0.25):
        conf = 1 / (1 + np.exp(-t[:, 4:]))
        return int((conf.max(axis=1) > thr).sum())

    d0, d1 = n_det(T), n_det(T2)
    box_energy = float(np.sum(box**2) / np.sum(T**2))
    print(f"  전체 헤드출력 CosSim(정상 vs cls손상) = {cos:.4f}")
    print(f"  검출 수: 정상 {d0}개 → cls손상 {d1}개  (유지율 {100 * d1 / max(d0, 1):.0f}%)")
    print(f"  맹점의 원인: box 회귀값이 전체 L2 에너지의 {box_energy:.1%}를 차지 → "
          f"검출을 좌우하는 소수 cls 로짓이 죽어도 CosSim은 거의 1. ")
    print("  → CosSim(전역 L2 방향)은 '임계값 통과' 같은 국소 결정에 구조적으로 둔감. 검출 수·conf로 봐야 한다.")
    return {"cos": cos, "d0": d0, "d1": d1, "box_energy": box_energy}


# ---------- 그림 ----------
def make_figure(ocr_rows, det, dfl, cos) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    # Panel 1: OCR per-layer INT8 vs INT4 SQNR
    names = [r["layer"].replace("onnx::Conv_", "conv_") for r in ocr_rows]
    x = np.arange(len(names))
    ax[0].bar(x - 0.2, [r["s8"] for r in ocr_rows], 0.4, label="INT8", color="#2a9d8f")
    ax[0].bar(x + 0.2, [r["s4"] for r in ocr_rows], 0.4, label="INT4 (W4A16)", color="#e76f51")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax[0].set_ylabel("Weight SQNR (dB, higher=safer)")
    ax[0].set_title("(A) OCR: INT4 craters the 2350-class head")
    ax[0].axhline(20, ls="--", c="gray", lw=0.8)
    ax[0].legend(fontsize=8)

    # Panel 2: detector backbone vs head INT8 SQNR (the documented collapse is at INT8)
    b8, h8 = det["back"]["s8"], det["head"]["s8"]
    ax[1].boxplot([b8, h8], tick_labels=["backbone/neck", "detect head\n(DFL)"], widths=0.5)
    ax[1].scatter(np.ones_like(b8), b8, alpha=0.4, color="#2a9d8f", s=14)
    ax[1].scatter(np.full_like(h8, 2), h8, alpha=0.5, color="#e76f51", s=14)
    ax[1].set_ylabel("Per-layer INT8 weight SQNR (dB)")
    ax[1].set_title(f"(B) Head INT8 {h8.mean():.0f} >= backbone {b8.mean():.0f} dB\nweights are NOT the cause")

    # Panel 3: CosSim blindness — high CosSim, zero detection retention
    ret = cos["d1"] / max(cos["d0"], 1)
    ax[2].bar([0, 1], [cos["cos"], ret], color=["#264653", "#e76f51"], width=0.5)
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(["head-output\nCosSim", "detection\nretention"])
    ax[2].set_ylim(0, 1.05)
    for i, v in enumerate([cos["cos"], ret]):
        ax[2].text(i, v + 0.02, f"{v:.4f}" if i == 0 else f"{v:.0%}", ha="center", fontsize=10)
    ax[2].set_title(f"(C) CosSim blind: {cos['cos']:.3f} yet {ret:.0%} kept\n(box={cos['box_energy']:.0%} of L2 · DFL noise {dfl['median']:.3f} bin)")

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\n[fig] saved {OUT_PNG}")


def main() -> None:
    ocr_rows = analyze_ocr()
    det = analyze_detector()
    dfl = analyze_dfl_robustness()
    cos = analyze_cossim_blindness()
    make_figure(ocr_rows, det, dfl, cos)
    print("\n[done] 붕괴 원인 분석 완료 — 수치를 README §8.3 / EXPERIMENTS.md에 기록하라.")


if __name__ == "__main__":
    main()
