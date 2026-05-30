"""
v2 분할 재학습/평가 완료 후 로그 파싱 및 결과 요약 마크다운 자동 생성.

입력:
  logs/val_v2split.log         — ultralytics val 출력
  logs/track_*.log             — eval_tracking.py 출력 4종
  logs/eval_e2e_v2.log         — eval_e2e.py 출력
  logs/quant_v2split.log       — run_experiments.py 출력
  logs/quant_int8_v2.log       — quantize_onnx_real.py 출력

출력:
  docs/RESULTS_V2.md           — 결과 요약 마크다운
  logs/results_v2.json         — 머신 판독용 결과 JSON

사용법:
  python scripts/_finalize_v2_results.py
"""
import io
import json
import re
import sys
from pathlib import Path
from datetime import datetime

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
LOGS = ROOT / "logs"
DOCS = ROOT / "docs"
OUT_MD   = DOCS / "RESULTS_V2.md"
OUT_JSON = LOGS / "results_v2.json"


def safe_read(p: Path) -> str:
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[ERROR reading {p}: {e}]"


def parse_val_log() -> dict:
    """ultralytics val 출력에서 mAP/P/R 추출."""
    log = safe_read(LOGS / "val_v2split.log")
    if not log:
        return {}
    # "all   N   N   P   R   mAP50   mAP50-95"
    pat = r"^\s*all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
    for line in log.splitlines():
        m = re.match(pat, line)
        if m:
            return {
                "precision": float(m.group(1)),
                "recall":    float(m.group(2)),
                "map50":     float(m.group(3)),
                "map":       float(m.group(4)),
            }
    return {}


def parse_tracking_log(name: str) -> dict:
    """eval_tracking.py '전체 평균 결과' 섹션에서 메트릭 추출."""
    log = safe_read(LOGS / f"track_{name}.log")
    if not log:
        return {}
    result = {}
    # "MOTA: 0.219" / "IDF1: 0.384" 등
    for key in ["MOTA", "IDF1", "HOTA", "DetA"]:
        m = re.search(rf"^\s*{key}:\s+([\d.]+)\s*$", log, re.MULTILINE)
        if m:
            result[key] = float(m.group(1))
    m = re.search(r"^\s*FPS:\s+([\d.]+)", log, re.MULTILINE)
    if m:
        result["FPS"] = float(m.group(1))
    m = re.search(r"IDSW:\s*(\d+)", log)
    if m:
        result["IDSW"] = int(m.group(1))
    return result


def parse_e2e_log() -> dict:
    """eval_e2e.py 결과 표에서 E0~E7 FPS / Final Score 추출."""
    log = safe_read(LOGS / "eval_e2e_v2.log")
    if not log:
        return {}
    result = {}
    # 라인 예: "  E0   FP32 All             21.2      24.3  1.0000  1.0000 1.0000  1.0000"
    pat = re.compile(
        r"^\s*(E[0-7])\s+\S.+?\s+"
        r"([\d.]+|—)\s+"          # FPS
        r"([\d.]+)\s+"            # Size MB
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"  # PerfN SpeedN MemN Score
    )
    for line in log.splitlines():
        m = pat.match(line)
        if not m:
            continue
        eid = m.group(1)
        fps = m.group(2)
        result[eid] = {
            "fps":         None if fps == "—" else float(fps),
            "size_mb":     float(m.group(3)),
            "perf_norm":   float(m.group(4)),
            "speed_norm":  float(m.group(5)),
            "mem_norm":    float(m.group(6)),
            "final_score": float(m.group(7)),
        }
    return result


def parse_quant_log() -> dict:
    """run_experiments.py 출력에서 E1/E4/E5 mAP@0.5 추출."""
    log = safe_read(LOGS / "quant_v2split.log")
    if not log:
        return {}
    result = {}
    # "E1: mAP@0.5 = 0.621" 형태 (실제 포맷 확인 필요)
    # run_experiments.py print_results 패턴: 가능성 모두 시도
    for exp in ["E1", "E4", "E5"]:
        # 시도 1: "E1 ... map50 0.621"
        m = re.search(rf"{exp}.*?map50['\":\s=]+\s*([\d.]+)", log, re.IGNORECASE)
        if m:
            result[exp] = {"map50": float(m.group(1))}
            continue
        # 시도 2: "E1 ... 0.621" (단순)
        m = re.search(rf"^{exp}\s+\S.*?\s+([\d.]+)", log, re.MULTILINE)
        if m:
            try:
                v = float(m.group(1))
                if 0 < v <= 1:
                    result[exp] = {"map50": v}
            except ValueError:
                pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 마크다운 생성
# ─────────────────────────────────────────────────────────────────────────────

def render_markdown(val, tracking, e2e, quant) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = []
    md.append("# Edge-Sign v2  Stratified Split Re-training Results")
    md.append("")
    md.append(f"자동 생성: `scripts/_finalize_v2_results.py`  |  {now}")
    md.append("")
    md.append("> 데이터 분할: train 5 시퀀스(주간 4 + 야간 1) / val 2(주간 1 + 야간 1) / test 2(주간 1 + 야간 1).")
    md.append("> v1(주간 학습 → 야간 단독 평가)과 달리 도메인 균형 평가가 가능하다.")
    md.append("> v1 모델 백업: `model_space/v1/`, `data/yolo_signs_v1/`.")
    md.append("")

    # ── 1. 검출 결과 ─────────────────────────────────────────────────────
    md.append("## 1. 검출 결과 (E0 FP32 v2)")
    md.append("")
    if val:
        md.append("| 지표 | v2 결과 | v1 결과 | Δ |")
        md.append("|------|---------|---------|---|")
        v1 = {"map50": 0.628, "map": 0.437, "precision": 0.722, "recall": 0.543}
        labels = [("mAP@0.5", "map50"), ("mAP@0.5:0.95", "map"),
                  ("Precision", "precision"), ("Recall", "recall")]
        for label, key in labels:
            v2v = val.get(key)
            v1v = v1[key]
            if v2v is None:
                md.append(f"| {label} | — | {v1v:.3f} | — |")
            else:
                delta = v2v - v1v
                md.append(f"| {label} | **{v2v:.3f}** | {v1v:.3f} | {delta:+.3f} |")
    else:
        md.append("_검출 결과 파싱 실패 — `logs/val_v2split.log` 직접 확인 필요_")
    md.append("")

    # ── 2. 양자화 mAP ────────────────────────────────────────────────────
    md.append("## 2. 검출기 양자화 결과 (E1 / E4 / E5)")
    md.append("")
    if quant:
        md.append("| ID | 양자화 | mAP@0.5 v2 | mAP@0.5 v1 |")
        md.append("|----|--------|-----------|-----------|")
        v1q = {"E1": 0.621, "E4": 0.581, "E5": 0.621}
        for eid, label in [("E1", "W8A8 PTQ"), ("E4", "W4A16 PTQ"), ("E5", "SmoothQuant+W8A8")]:
            q = quant.get(eid, {})
            v2v = q.get("map50")
            if v2v is None:
                md.append(f"| {eid} | {label} | — | {v1q[eid]:.3f} |")
            else:
                md.append(f"| {eid} | {label} | **{v2v:.3f}** | {v1q[eid]:.3f} |")
    else:
        md.append("_양자화 결과 파싱 실패 — `logs/quant_v2split.log` 직접 확인_")
    md.append("")

    # ── 3. 추적 결과 ─────────────────────────────────────────────────────
    md.append("## 3. 추적 평가 (v2 test 시퀀스, 주야간 균형)")
    md.append("")
    if tracking and any(tracking.values()):
        md.append("| ID | 양자화 | MOTA | IDF1 | HOTA | IDSW | FPS |")
        md.append("|----|--------|------|------|------|------|-----|")
        for name, eid, label in [
            ("E0_FP32",        "E0", "FP32"),
            ("E1_W8A8",        "E1", "W8A8"),
            ("E4_W4A16",       "E4", "W4A16"),
            ("E5_SmoothQuant", "E5", "SmoothQuant"),
        ]:
            t = tracking.get(name) or {}
            mota = t.get("MOTA")
            idf1 = t.get("IDF1")
            hota = t.get("HOTA")
            idsw = t.get("IDSW", "—")
            fps  = t.get("FPS")
            mota_s = f"{mota:.3f}" if mota is not None else "—"
            idf1_s = f"{idf1:.3f}" if idf1 is not None else "—"
            hota_s = f"{hota:.3f}" if hota is not None else "—"
            fps_s  = f"{fps:.1f}"  if fps  is not None else "—"
            md.append(f"| {eid} | {label} | {mota_s} | {idf1_s} | {hota_s} | {idsw} | {fps_s} |")
    else:
        md.append("_추적 결과 파싱 실패 — `logs/track_*.log` 직접 확인_")
    md.append("")

    # ── 4. E2E 종합 ──────────────────────────────────────────────────────
    md.append("## 4. E2E 종합 평가 (E0~E7 v2)")
    md.append("")
    if e2e:
        md.append("| ID | FPS | Size (MB) | PerfN | SpeedN | MemN | Final Score |")
        md.append("|----|-----|-----------|-------|--------|------|-------------|")
        for eid in sorted(e2e.keys()):
            d = e2e[eid]
            fps_s = f"{d['fps']:.1f}" if d['fps'] is not None else "—"
            md.append(f"| {eid} | {fps_s} | {d['size_mb']:.1f} | "
                      f"{d['perf_norm']:.4f} | {d['speed_norm']:.4f} | "
                      f"{d['mem_norm']:.4f} | **{d['final_score']:.4f}** |")
    else:
        md.append("_E2E 결과 파싱 실패 — `logs/eval_e2e_v2.log` 직접 확인_")
    md.append("")

    # ── 5. v1 vs v2 비교 ─────────────────────────────────────────────────
    md.append("## 5. v1 ↔ v2 비교 요약")
    md.append("")
    md.append("v1 결과 (참조):")
    md.append("- 검출 E0: mAP@0.5=0.628, mAP@.5:.95=0.437, P=0.722, R=0.543")
    md.append("- 추적 E0 ByteTrack: MOTA=0.219, IDF1=0.384, HOTA=0.487, IDSW=0 (야간 단독 평가)")
    md.append("- E2E E1 Best Final Score: 1.0335 (fake-quant)")
    md.append("- INT8 Static E3: 57.7 FPS @ CPU")
    md.append("")
    md.append("v2 변경의 영향:")
    md.append("- val/test에 주간 도메인이 포함되어 야간 단독 평가의 도메인 갭이 해소됨")
    md.append("- 학습 데이터는 13,387 → 13,387(주간 4+야간 1) 시퀀스 유지, 라벨 수는 다름")
    md.append("- 야간 학습 데이터 1개 시퀀스 포함으로 야간 검출 Recall 개선 기대")
    md.append("")

    # ── 6. 다음 단계 ─────────────────────────────────────────────────────
    md.append("## 6. 다음 단계 (사용자 검토 항목)")
    md.append("")
    md.append("- [ ] 위 v2 결과를 `docs/EXPERIMENTS.md` 본문에 통합 (v1 행은 보존)")
    md.append("- [ ] README 7.2~7.6 표/차트를 v2 결과로 재생성 — `scripts/plot_pareto.py`, `scripts/plot_sensitivity.py` 데이터 갱신 필요")
    md.append("- [ ] `scripts/quantize_onnx_real.py` 재실행으로 INT8 Static QDQ v2 확인")
    md.append("- [ ] v1 vs v2 mAP/MOTA 차이가 클 경우 분석 및 메모 추가")
    md.append("")

    md.append("---")
    md.append("")
    md.append("_파싱이 실패한 항목은 `logs/` 디렉토리의 원본 로그를 직접 확인할 것._")
    return "\n".join(md)


def main():
    print("=== v2 결과 파싱 ===")
    val      = parse_val_log()
    tracking = {name: parse_tracking_log(name) for name in
                ["E0_FP32", "E1_W8A8", "E4_W4A16", "E5_SmoothQuant"]}
    e2e      = parse_e2e_log()
    quant    = parse_quant_log()

    print(f"Val:      {val}")
    print(f"Quant:    {quant}")
    print(f"Tracking: { {k: bool(v) for k,v in tracking.items()} }")
    print(f"E2E:      {list(e2e.keys()) if e2e else 'none'}")

    # JSON 덤프
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "val": val, "tracking": tracking, "e2e": e2e, "quant": quant,
        "generated_at": datetime.now().isoformat(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] JSON saved: {OUT_JSON}")

    # 마크다운 생성
    md = render_markdown(val, tracking, e2e, quant)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"[OK] Markdown saved: {OUT_MD}")
    print(f"     {len(md.splitlines())} lines, {len(md)} chars")


if __name__ == "__main__":
    main()
