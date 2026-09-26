"""Aggregate runtime_matrix.py outputs into paper tables (JSON + Markdown).

Reads <matrix>/cpu_<key>/metrics.json, cpu_t<N>_<key>.json, the browser
{mode}_{tag}_ort{ver}_{key}.json files (tag: webgpu, wasm, wasmt<N>) and, for the
recognizer rows, recognizer_variants.py metrics (top-1 retention instead of mAP). Criteria (docs: EVALUATION_CRITERIA.md):
  accuracy  retention = mAP50-95(variant) / mAP50-95(FP32 of the same family); two quality
            tiers as MLPerf's yolo-95 / yolo-99: tier95 = retention >= 0.95, tier99 = >= 0.99
  latency   A: p90 <= 33.3 ms (30 FPS)   B: p90 <= 66.7 ms (15 FPS)
Operator placement is parsed from ORT's VerifyEachNodeIsAssignedToAnEp log lines
and written to placement.json, so the raw ops logs are not needed for the tables.

Usage: python scripts/paper/summarize_runtime_matrix.py --matrix paper_evidence/runtime/matrix \
           --artifact-root <checkout>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.runtime_matrix import MODELS

TIERS = {"tier95": 0.95, "tier99": 0.99}
P90_A, P90_B = 1000 / 30, 1000 / 15
# "Node(s) placed on [EP]" is followed by one line per node; when a single EP takes the
# whole graph ORT prints "All nodes placed on [EP]" with the count only.
PLACED = re.compile(r"(Node\(s\)|All nodes) placed on \[(\w+)\]\. Number of nodes: (\d+)")
NODE = re.compile(r"VerifyEachNodeIsAssignedToAnEp\]\s{3}(\w+) \((.*)\)")
RUN = re.compile(r"^(speed|ops|accuracy)_([a-z0-9]+)_ort(\d+)_(.+)$")
# ORT-Web 1.22 runs WebGPU through the JSEP provider; 1.30 has a native WebGPU provider
GPU_EPS = ("WebGpuExecutionProvider", "JsExecutionProvider")


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def placement(logs: list[str]) -> dict:
    eps: dict[str, dict] = {}
    current = None
    for line in logs:
        if m := PLACED.search(line):
            current = m.group(2)
            eps.setdefault(current, {"nodes": 0, "listed": Counter(), "all": m.group(1) == "All nodes"})
            eps[current]["nodes"] = int(m.group(3))
        elif current and (m := NODE.search(line)):
            eps[current]["listed"][m.group(1)] += 1
    return {ep: {"nodes": v["nodes"], "listed": v["nodes"] if v["all"] else sum(v["listed"].values()),
                 "op_types": dict(v["listed"].most_common()), "all_nodes": v["all"]}
            for ep, v in eps.items()}


def lat(result: dict | None) -> dict | None:
    if not result or result.get("status", "completed") != "completed":
        return None
    m = result["metrics"]["inference_ms"]
    return {"mean_ms": m["mean_ms"], "p90_ms": m["p90_ms"], "p99_ms": m["p99_ms"], "n": m["n"],
            "init_ms": result.get("session_init_ms"), "threads": result.get("effective_wasm_threads", result.get("threads")),
            "A_30fps": m["p90_ms"] <= P90_A, "B_15fps": m["p90_ms"] <= P90_B}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", type=Path, default=Path("paper_evidence/runtime/matrix"))
    parser.add_argument("--recognition", type=Path, default=Path("paper_evidence/recognition/variants"))
    parser.add_argument("--accuracy-from", type=Path, default=None,
                        help="folder with cpu_<key>/metrics.json (default: --matrix); lets a second device reuse them")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--coco", type=Path, default=Path("paper_evidence/coco"), help="coco_validation.py output")
    args = parser.parse_args()
    mx = args.matrix
    ax = args.accuracy_from or mx
    runs = {m.groups() for p in mx.glob("*.json") if (m := RUN.match(p.stem))}
    speed_cols = sorted({(tag, v) for mode, tag, v, _ in runs if mode == "speed"}, key=lambda t: (t[1], t[0]))
    ort_versions = sorted({v for _, _, v, _ in runs})

    places = {}
    for p in sorted(mx.glob("ops_*.json")):
        r = load(p)
        places[p.stem] = {"status": r.get("status"), "reason": r.get("reason"), "placement": placement(r.get("logs", []))}
    (mx / "placement.json").write_text(json.dumps(places, indent=2) + "\n", encoding="utf-8")

    rows = []
    for key, (family, rel) in MODELS.items():
        row = {"model": key, "family": family, "path": rel,
               "bytes": (args.artifact_root / rel).stat().st_size if (args.artifact_root / rel).exists() else None}
        acc = None
        if family == "rec":
            rec = load(args.recognition / f"{key[4:]}_metrics.json")
            if rec:
                row.update(top1=rec["top1_accuracy"], retention=rec["top1_retention_vs_fp32"],
                           agreement=rec["agreement_with_fp32"],
                           **{t: rec["top1_retention_vs_fp32"] >= v for t, v in TIERS.items()})
        else:
            if family == "coco":  # external validation: coco_validation.py (letterbox, pycocotools)
                acc = load(args.coco / f"cpu_{key[5:]}" / "metrics.json")
                ref = load(args.coco / "cpu_fp32" / "metrics.json")
            else:
                acc = load(ax / f"cpu_{key}" / "metrics.json")
                ref = load(ax / f"cpu_{family}_fp32" / "metrics.json")
            if acc:
                row.update(mAP50=acc["mAP50"], mAP50_95=acc["mAP50_95"], precision=acc.get("precision"),
                           recall=acc.get("recall"))
                if ref:
                    row["retention"] = acc["mAP50_95"] / ref["mAP50_95"]
                    row.update({t: row["retention"] >= v for t, v in TIERS.items()})
        row["cpu_t1"] = lat(load(mx / f"cpu_t1_{key}.json"))
        row["cpu_t4"] = lat(load(mx / f"cpu_t4_{key}.json"))
        for tag, v in speed_cols:
            row[f"{tag}_{v}"] = lat(load(mx / f"speed_{tag}_ort{v}_{key}.json"))
        for v in ort_versions:
            pl = places.get(f"ops_webgpu_ort{v}_{key}")
            if pl:
                row[f"place_{v}"] = {"status": pl["status"], "reason": (pl["reason"] or "")[:200],
                                     **{e: d["nodes"] for e, d in pl["placement"].items()}}
                # every placed node must also be listed, otherwise the captured log was truncated
                row[f"place_{v}_complete"] = all(d["listed"] == d["nodes"] for d in pl["placement"].values())
                cpu = pl["placement"].get("CPUExecutionProvider")
                row[f"place_{v}_cpu_ops"] = cpu["op_types"] if cpu else {}
            for tag in ("webgpu", "wasm", "wasmt4"):  # parity does not depend on the WASM thread count
                b = load(mx / f"accuracy_{tag}_ort{v}_{key}" / "metrics.json")
                if b and acc:
                    row[f"parity_{tag}_{v}"] = {"mAP50_95": b["mAP50_95"], "delta_vs_cpu": b["mAP50_95"] - acc["mAP50_95"],
                                                "mAP50": b["mAP50"], "delta50_vs_cpu": b["mAP50"] - acc["mAP50"]}
        rows.append(row)
    (mx / "summary.json").write_text(
        json.dumps({"ort_web_versions": ort_versions, "speed_columns": [f"{t}_{v}" for t, v in speed_cols], "rows": rows},
                   indent=2) + "\n", encoding="utf-8")

    def f(x, spec=".4f"):
        return "-" if x is None else format(x, spec)

    def l(x, det):
        if not x:
            return "-"
        mark = (" A" if x["A_30fps"] else " B" if x["B_15fps"] else "") if det else ""
        spec = ".1f" if x["mean_ms"] >= 10 else ".2f"
        return f"{x['mean_ms']:{spec}} / {x['p90_ms']:{spec}}{mark}"

    print("## Accuracy (ORT CPU, independent test)\n")
    print("| model | MB | mAP50 or top-1 | mAP50-95 | retention | >=95% | >=99% |")
    print("|---|---:|---:|---:|---:|:-:|:-:|")
    for r in rows:
        first = r.get("mAP50", r.get("top1"))
        print(f"| {r['model']} | {f(r['bytes'] and r['bytes'] / 1e6, '.3f')} | {f(first)} | {f(r.get('mAP50_95'))} "
              f"| {f(r.get('retention'), '.3f')} | "
              + " | ".join("Y" if r.get(t) else "N" if t in r else "-" for t in TIERS) + " |")
    cols = ["cpu_t1", "cpu_t4"] + [f"{t}_{v}" for t, v in speed_cols]
    print("\n## Latency mean / p90 ms, batch 1 (detector: A = p90<=33.3, B = p90<=66.7)\n")
    print("| model | " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * len(cols))
    for r in rows:
        print(f"| {r['model']} | " + " | ".join(l(r.get(c), r["family"] != "rec") for c in cols) + " |")
    print("\n## WebGPU node placement (WebGPU nodes / CPU nodes)\n")
    print("| model | " + " | ".join(f"ORT {v} | CPU op types ({v})" for v in ort_versions) + " |")
    print("|---|" + "---:|---|" * len(ort_versions))
    for r in rows:
        cells = []
        for v in ort_versions:
            p = r.get(f"place_{v}")
            if p is None:
                cells += ["-", "-"]
                continue
            if p["status"] != "completed":
                cells += [f"run failed ({p['status']})", p["reason"][:60] or "-"]
                continue
            mark = "" if r.get(f"place_{v}_complete", True) else " (log truncated)"
            gpu = sum(p.get(ep, 0) for ep in GPU_EPS)
            cells.append(f"{gpu}/{p.get('CPUExecutionProvider', 0)}{mark}")
            ops = r.get(f"place_{v}_cpu_ops") or {}
            cells.append(", ".join(f"{k}x{n}" for k, n in ops.items()) or "-")
        print(f"| {r['model']} | " + " | ".join(cells) + " |")
    print("\n## Browser numeric parity (vs ORT CPU, same frames and decoder)\n")
    print("| model | runtime | mAP50 | delta | mAP50-95 | delta |")
    print("|---|---|---:|---:|---:|---:|")
    for r in rows:
        for k, v in r.items():
            if k.startswith("parity_"):
                print(f"| {r['model']} | {k[7:]} | {v['mAP50']:.6f} | {v['delta50_vs_cpu']:+.6f} "
                      f"| {v['mAP50_95']:.6f} | {v['delta_vs_cpu']:+.6f} |")


if __name__ == "__main__":
    main()
