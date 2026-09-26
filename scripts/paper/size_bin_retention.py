"""Accuracy retention per object-size bin from saved predictions (no new inference).

Hypothesis (fixed before this script computed any retention): coordinate quantization costs
small objects more. An 8-bit tensor that spans 0-640 px has a step of about 2.5 px at the model
input, which is large for the road objects (median side 8.7 px at 640 input).
Decision rule: supported only if, in each of the three coordinate-quantized conditions
(v3 a8sim_decode_no_outconcat, v4 a8sim_decode_no_collapse, COCO a8sim_decode_no_outconcat),
retention in the smallest bin is below retention in the largest bin and the 95% bootstrap CI of
the paired difference (large - small) excludes 0. Otherwise the claim stays out of the paper.

Bins, fixed from the ground-truth size distribution only:
  road: side sqrt(w*h) at the 640x640 model input < 8 px, 8-16 px, >= 16 px (1,843 / 2,746 /
  2,183 boxes). Frames are resized (not letterboxed) from 1920x1080 or 1920x1200, so GT and
  detections are mapped into input coordinates first; IoU of axis-aligned boxes does not change
  under per-axis scaling, so matching is identical to the original-coordinate evaluation.
  (Correction before the full run: a first version used a uniform 1/3 scale; the thresholds are
  unchanged.)
  COCO: the standard small / medium / large areas (< 32^2, 32^2-96^2, >= 96^2 px).
AP is COCOeval AP@0.5:0.95 with its area-range rules (ground truth outside the bin is ignored,
unmatched detections outside the bin are ignored). Road uses maxDets 300 (the models emit up to
300), COCO uses 100 as in the paper. Frames/images are resampled with replacement (paired across
conditions, seed 0); retention = AP(variant) / AP(FP32) inside each resample.

Usage: python scripts/paper/size_bin_retention.py --coco-root C:/Users/leegy/Desktop/CNN_Quant
Output: paper_evidence/extra/size_bins/{size_bins.json, summary.md}
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MATRIX = Path("paper_evidence/runtime/matrix")
MANIFEST = Path("paper_evidence/splits/test_manifest.jsonl")
COCO_EVIDENCE = Path("paper_evidence/coco")
OUT = Path("paper_evidence/extra/size_bins")
INPUT = 640  # road frames are resized to 640x640; boxes are evaluated in input coordinates
ROAD_BINS = {"small": (0.0, 8.0), "medium": (8.0, 16.0), "large": (16.0, 1e5)}  # side px at 640 input
COCO_BINS = {"small": (0.0, 32.0), "medium": (32.0, 96.0), "large": (96.0, 1e5)}  # side px, COCO standard
PRIMARY = {"road_v3": "a8sim_decode_no_outconcat", "road_v4": "a8sim_decode_no_collapse",
           "coco": "a8sim_decode_no_outconcat"}
CONDITIONS = {
    "road_v3": ["fp32", "int8_head_excl", "int8_decode_excl", "a8sim_decode_no_outconcat", "w8sim_head"],
    "road_v4": ["fp32", "int8_head_excl", "int8_decode_excl", "a8sim_decode_no_collapse"],
    "coco": ["fp32", "int8_head_excl", "int8_decode_excl", "a8sim_decode_no_outconcat"],
}


class AreaWeightedAP:
    """COCOeval.evaluate() once, then accumulate() per area range with per-image weights.

    Generalizes WeightedCocoAP in coco_validation.py to every area range; with unit weights it
    must reproduce COCOeval.accumulate() for each range (checked in main)."""

    def __init__(self, gt, detections: list[dict], img_ids: list[int], area_rng: list[list[float]], max_det: int):
        from pycocotools.cocoeval import COCOeval

        ev = COCOeval(gt, gt.loadRes(detections), iouType="bbox")
        ev.params.imgIds = img_ids
        ev.params.areaRng = area_rng
        ev.params.areaRngLbl = [str(i) for i in range(len(area_rng))]
        ev.params.maxDets = [1, 10, max_det]
        with contextlib.redirect_stdout(io.StringIO()):
            ev.evaluate()
            ev.accumulate()
        p = ev.params
        self.rec_thrs, self.n_img, self.max_det = p.recThrs, len(p.imgIds), max_det
        n_area = len(p.areaRng)
        precision = ev.eval["precision"][:, :, :, :, -1]  # T x R x K x A at maxDets[-1]
        self.reference = [float(np.mean(v[v > -1])) if (v := precision[:, :, :, a]).size and (v > -1).any() else 0.0
                          for a in range(n_area)]
        self.per_area = []
        for a in range(n_area):
            cats = []
            for k in range(len(p.catIds)):
                base = k * n_area * self.n_img + a * self.n_img
                entries = [(i, e) for i in range(self.n_img) if (e := ev.evalImgs[base + i]) is not None]
                if not entries:
                    continue
                scores = np.concatenate([e["dtScores"][:max_det] for _, e in entries])
                owner = np.concatenate([np.full(len(e["dtScores"][:max_det]), i) for i, e in entries]).astype(np.int64)
                matches = np.concatenate([e["dtMatches"][:, :max_det] for _, e in entries], axis=1)
                ignore = np.concatenate([e["dtIgnore"][:, :max_det] for _, e in entries], axis=1)
                npig = np.zeros(self.n_img, dtype=np.int64)
                for i, e in entries:
                    npig[i] = np.count_nonzero(np.asarray(e["gtIgnore"]) == 0)
                order = np.argsort(-scores, kind="mergesort")  # the weighted path sorts once
                cats.append((scores, owner, matches.astype(bool), ignore.astype(bool), npig, order))
            self.per_area.append(cats)

    def ap(self, area: int, weights: np.ndarray, repeat: bool = False) -> float:
        """AP for one area range. Weighted cumulative sums over the once-sorted detections equal
        repeating each image's detections by its weight (repeat=True, as in coco_validation.py)."""
        values = []
        for scores, owner, matches, ignore, npig, order in self.per_area[area]:
            total = int((weights * npig).sum())
            if total == 0:
                continue
            if repeat:
                idx = np.repeat(np.arange(scores.size), weights[owner])
                sel = idx[np.argsort(-scores[idx], kind="mergesort")]
                tps = matches[:, sel] & ~ignore[:, sel]
                fps = ~matches[:, sel] & ~ignore[:, sel]
                tp_sum, fp_sum = np.cumsum(tps, axis=1, dtype=float), np.cumsum(fps, axis=1, dtype=float)
            else:
                w = weights[owner[order]].astype(float)
                keep = w > 0
                w = w[keep]
                m, ig = matches[:, order][:, keep], ignore[:, order][:, keep]
                tp_sum = np.cumsum((m & ~ig) * w, axis=1)
                fp_sum = np.cumsum((~m & ~ig) * w, axis=1)
            for tp, fp in zip(tp_sum, fp_sum, strict=True):
                recall = tp / total
                prec = tp / (fp + tp + np.spacing(1))
                prec = np.maximum.accumulate(prec[::-1])[::-1]
                inds = np.searchsorted(recall, self.rec_thrs, side="left")
                q = np.zeros(len(self.rec_thrs))
                valid = inds < len(prec)
                q[valid] = prec[inds[valid]]
                values.append(q)
        return float(np.mean(values)) if values else 0.0


def road_inputs(folder: Path) -> tuple[dict, list[dict], list[int], dict]:
    """COCO-format ground truth and detections for one road condition folder (predictions.jsonl.gz, metrics.json)."""
    rows = [json.loads(line) for line in gzip.open(folder / "predictions.jsonl.gz", "rt", encoding="utf-8")]
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    images, annotations, detections = [], [], []
    size = {m["image_rel"]: (m["width"], m["height"]) for m in map(json.loads, Path(MANIFEST).read_text(encoding="utf-8").splitlines())}
    for i, r in enumerate(rows):
        width, height = size[r["image_rel"]]
        sx, sy = INPUT / width, INPUT / height

        def box(b: list[float], sx: float = sx, sy: float = sy) -> list[float]:
            return [b[0] * sx, b[1] * sy, (b[2] - b[0]) * sx, (b[3] - b[1]) * sy]

        images.append({"id": i, "file_name": r["image_rel"]})
        for c, b in zip(r["ground_truth"]["classes"], r["ground_truth"]["boxes_xyxy"], strict=True):
            xywh = box(b)
            annotations.append({"id": len(annotations) + 1, "image_id": i, "category_id": int(c) + 1,
                                "bbox": xywh, "area": xywh[2] * xywh[3], "iscrowd": 0})
        for d in r["detections"]:
            detections.append({"image_id": i, "category_id": int(d["class_id"]) + 1,
                               "bbox": box(d["box_xyxy"]), "score": float(d["confidence"])})
    gt = {"images": images, "annotations": annotations,
          "categories": [{"id": 1, "name": "traffic_sign"}, {"id": 2, "name": "traffic_light"}]}
    meta = {k: metrics.get(k) for k in ("model", "model_path", "model_sha256", "mAP50_95", "mAP50", "frames_evaluated")}
    meta["image_order"] = [r["image_rel"] for r in rows]
    return gt, detections, list(range(len(rows))), meta


def coco_inputs(variant: str) -> tuple[list[dict], dict]:
    folder = COCO_EVIDENCE / f"cpu_{variant}"
    d = np.load(folder / "detections.npz")
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    dets = [{"image_id": int(i), "category_id": int(c), "bbox": [float(v) for v in b], "score": float(s)}
            for i, c, b, s in zip(d["image_id"], d["category_id"], d["bbox"], d["score"], strict=True)]
    meta = {k: metrics.get(k) for k in ("variant", "model", "model_sha256", "mAP50_95", "mAP50")}
    return dets, meta


def area_ranges(bins: dict[str, tuple[float, float]], scale: float) -> list[list[float]]:
    """[all, *bins] as pycocotools area ranges in original-image px^2 (side * scale squared)."""
    return [[0.0, 1e10]] + [[(lo * scale) ** 2, (hi * scale) ** 2] for lo, hi in bins.values()]


def extra(folder: Path, resamples: int, seed: int) -> None:
    """Same bins and bootstrap for new road conditions in <folder>/cpu_<v3|v4>_*/ (reported separately
    from the pre-declared test above); references are the paper's FP32 runs."""
    from pycocotools.coco import COCO

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    names = ["all", *ROAD_BINS]
    result: dict = {"commit": commit, "bins_side_px_at_640": ROAD_BINS, "resamples": resamples, "seed": seed, "families": {}}
    lines = [f"# Retention by object size: conditions in {folder.as_posix()}", "",
             f"Commit `{commit}`. Same bins, AP rules and paired bootstrap as `size_bins/` (reference: the paper's FP32 run).", ""]
    for family in ("v3", "v4"):
        conds = {"fp32": MATRIX / f"cpu_{family}_fp32"}
        conds.update({p.name[len(f"cpu_{family}_"):]: p for p in sorted(folder.glob(f"cpu_{family}_*"))
                      if (p / "predictions.jsonl.gz").exists()})
        evals, order = {}, None
        for name, path in conds.items():
            gt_dict, dets, img_ids, meta = road_inputs(path)
            order = order or meta["image_order"]
            if meta["image_order"] != order:
                raise SystemExit(f"{family} {name}: frame order differs from fp32")
            with contextlib.redirect_stdout(io.StringIO()):
                gt = COCO()
                gt.dataset = gt_dict
                gt.createIndex()
            evals[name] = AreaWeightedAP(gt, dets, img_ids, area_ranges(ROAD_BINS, 1.0), max_det=300)
        n_img = len(order)
        ones = np.ones(n_img, dtype=np.int64)
        rng = np.random.default_rng(seed)
        draws = [np.bincount(rng.integers(0, n_img, n_img), minlength=n_img) for _ in range(resamples)]
        rows = {}
        for name, e in evals.items():
            rows[name] = {"AP": {n: e.ap(a, ones) for a, n in enumerate(names)}}
            if name == "fp32":
                continue
            boot = {n: [] for n in names}
            for w in draws:
                for a, n in enumerate(names):
                    base = evals["fp32"].ap(a, w)
                    boot[n].append(e.ap(a, w) / base if base > 0 else np.nan)
            rows[name]["retention"] = {n: rows[name]["AP"][n] / rows["fp32"]["AP"][n] for n in names}
            rows[name]["retention_ci95"] = {n: [float(np.nanpercentile(boot[n], 2.5)), float(np.nanpercentile(boot[n], 97.5))] for n in names}
            diff = np.array(boot["large"]) - np.array(boot["small"])
            rows[name]["large_minus_small"] = {"point": rows[name]["retention"]["large"] - rows[name]["retention"]["small"],
                                               "ci95": [float(np.nanpercentile(diff, 2.5)), float(np.nanpercentile(diff, 97.5))]}
        result["families"][family] = rows
        lines += [f"## {family}", "", "| condition | " + " | ".join(f"retention {n} [95% CI]" for n in names) + " | large - small [95% CI] |",
                  "|---|" + "---:|" * (len(names) + 1)]
        for name, r in rows.items():
            if name == "fp32":
                continue
            cells = " | ".join(f"{r['retention'][n] * 100:.1f} [{r['retention_ci95'][n][0] * 100:.1f}-{r['retention_ci95'][n][1] * 100:.1f}]" for n in names)
            d = r["large_minus_small"]
            lines.append(f"| {name} | {cells} | {d['point'] * 100:+.1f} [{d['ci95'][0] * 100:+.1f}, {d['ci95'][1] * 100:+.1f}] |")
        lines.append("")
    (folder / "size_bins.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (folder / "size_bins.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco-root", type=Path, default=None, help="folder that holds data/coco/annotations")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extra", type=Path, default=None,
                        help="score the road conditions in this folder (cpu_<v3|v4>_*) instead of the pre-declared set")
    args = parser.parse_args()
    if args.extra:
        extra(args.extra, args.resamples, args.seed)
        return
    if args.coco_root is None:
        parser.error("--coco-root is required for the pre-declared analysis")
    from pycocotools.coco import COCO

    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    result: dict = {"commit": commit, "hypothesis": __doc__.split("Bins,")[0].strip(), "road_bins_side_px_at_640": ROAD_BINS,
                    "coco_bins_side_px": COCO_BINS, "resamples": args.resamples, "seed": args.seed, "workloads": {}}
    for workload, variants in CONDITIONS.items():
        bins = COCO_BINS if workload == "coco" else ROAD_BINS
        names = ["all", *bins]
        if workload == "coco":
            ann = args.coco_root / "data/coco/annotations/instances_val2017_safe.json"
            with contextlib.redirect_stdout(io.StringIO()):
                gt = COCO(str(ann))
            img_ids = sorted(gt.getImgIds())
            evals, metas = {}, {}
            for v in variants:
                dets, metas[v] = coco_inputs(v)
                evals[v] = AreaWeightedAP(gt, dets, img_ids, area_ranges(bins, 1.0), max_det=100)
            counts = {n: int(sum(1 for a in gt.dataset["annotations"] if not a.get("iscrowd", 0)
                                 and r[0] <= a["area"] <= r[1])) for n, r in zip(names, area_ranges(bins, 1.0), strict=True)}
        else:
            evals, metas, order = {}, {}, None
            for v in variants:
                gt_dict, dets, img_ids, metas[v] = road_inputs(MATRIX / f"cpu_{workload[5:]}_{v}")
                if order is None:
                    order = metas[v]["image_order"]
                if metas[v]["image_order"] != order:
                    raise SystemExit(f"{workload} {v}: frame order differs from fp32")
                with contextlib.redirect_stdout(io.StringIO()):
                    gt = COCO()
                    gt.dataset = gt_dict
                    gt.createIndex()
                evals[v] = AreaWeightedAP(gt, dets, img_ids, area_ranges(bins, 1.0), max_det=300)
            for m in metas.values():
                m.pop("image_order", None)
            counts = {n: int(sum(1 for a in gt_dict["annotations"] if r[0] <= a["area"] <= r[1]))
                      for n, r in zip(names, area_ranges(bins, 1.0), strict=True)}
        n_img = len(img_ids)
        ones = np.ones(n_img, dtype=np.int64)
        for v, e in evals.items():  # unit weights must reproduce COCOeval.accumulate()
            for a in range(len(names)):
                if abs(e.ap(a, ones) - e.reference[a]) > 1e-9:
                    raise SystemExit(f"{workload} {v} area {names[a]}: weighted AP {e.ap(a, ones)} != {e.reference[a]}")
        rng = np.random.default_rng(args.seed)
        draws = [np.bincount(rng.integers(0, n_img, n_img), minlength=n_img) for _ in range(args.resamples)]
        for w in draws[:3]:  # the fast weighted path must equal the repeat path used in coco_validation.py
            for v, e in evals.items():
                for a in range(len(names)):
                    if abs(e.ap(a, w) - e.ap(a, w, repeat=True)) > 1e-9:
                        raise SystemExit(f"{workload} {v} area {names[a]}: weighted path differs from repeat path")
        rows = {}
        for v, e in evals.items():
            rows[v] = {"meta": metas[v], "AP": {n: e.ap(a, ones) for a, n in enumerate(names)}}
            if v == "fp32":
                continue
            ref = evals["fp32"]
            boot = {n: [] for n in names}
            for w in draws:
                for a, n in enumerate(names):
                    base = ref.ap(a, w)
                    boot[n].append(e.ap(a, w) / base if base > 0 else np.nan)
            rows[v]["retention"] = {n: rows[v]["AP"][n] / rows["fp32"]["AP"][n] for n in names}
            rows[v]["retention_ci95"] = {n: [float(np.nanpercentile(boot[n], 2.5)), float(np.nanpercentile(boot[n], 97.5))]
                                         for n in names}
            diff = np.array(boot["large"]) - np.array(boot["small"])
            rows[v]["large_minus_small"] = {"point": rows[v]["retention"]["large"] - rows[v]["retention"]["small"],
                                            "ci95": [float(np.nanpercentile(diff, 2.5)), float(np.nanpercentile(diff, 97.5))]}
            print(f"{workload} {v}: " + ", ".join(f"{n} {rows[v]['retention'][n] * 100:.1f}%" for n in names), flush=True)
        result["workloads"][workload] = {"gt_counts": counts, "images": n_img, "conditions": rows}
    verdicts = {}
    for workload, v in PRIMARY.items():
        d = result["workloads"][workload]["conditions"][v]["large_minus_small"]
        verdicts[workload] = {"condition": v, "large_minus_small": d, "supports": d["point"] > 0 and d["ci95"][0] > 0}
    result["verdicts"] = verdicts
    result["hypothesis_supported"] = all(x["supports"] for x in verdicts.values())
    (OUT / "size_bins.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = ["# Retention by object size (saved predictions, COCOeval area rules)", "",
             f"Commit `{commit}`. Hypothesis and decision rule: see the docstring of `scripts/paper/size_bin_retention.py`.", ""]
    for workload, w in result["workloads"].items():
        names = list(w["gt_counts"])
        lines += [f"## {workload} ({w['images']} images; GT per bin: " + ", ".join(f"{n} {c}" for n, c in w["gt_counts"].items()) + ")", "",
                  "| condition | " + " | ".join(f"AP {n}" for n in names) + " | " + " | ".join(f"retention {n} [95% CI]" for n in names[1:]) + " | large - small [95% CI] |",
                  "|---|" + "---:|" * (2 * len(names) - 1 + 1)]
        for v, r in w["conditions"].items():
            aps = " | ".join(f"{r['AP'][n]:.4f}" for n in names)
            if "retention" in r:
                ret = " | ".join(f"{r['retention'][n] * 100:.1f} [{r['retention_ci95'][n][0] * 100:.1f}-{r['retention_ci95'][n][1] * 100:.1f}]" for n in names[1:])
                d = r["large_minus_small"]
                diff = f"{d['point'] * 100:+.1f} [{d['ci95'][0] * 100:+.1f}, {d['ci95'][1] * 100:+.1f}]"
            else:
                ret, diff = " | ".join("reference" for _ in names[1:]), "-"
            lines.append(f"| {v} | {aps} | {ret} | {diff} |")
        lines.append("")
    lines += ["## Verdict", "", *(f"- {k}: {v['condition']} large - small = {v['large_minus_small']['point'] * 100:+.1f} %p "
                                  f"[{v['large_minus_small']['ci95'][0] * 100:+.1f}, {v['large_minus_small']['ci95'][1] * 100:+.1f}] -> "
                                  f"{'supports' if v['supports'] else 'does not support'}" for k, v in verdicts.items()),
              f"- hypothesis supported (all three): **{result['hypothesis_supported']}**", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-6:]))


if __name__ == "__main__":
    main()
