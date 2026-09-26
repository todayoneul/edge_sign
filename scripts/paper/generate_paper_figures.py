"""Build publication candidates and source tables from newly recorded evidence."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.evaluate_qdq_detection import compute_metrics, iou

VARIANTS = ("yolo26_fp32", "yolo26_full_qdq", "yolo26_head_excluded_qdq")
LABELS = {
    "yolo26_fp32": "FP32",
    "yolo26_full_qdq": "Full QDQ",
    "yolo26_head_excluded_qdq": "Head-excluded QDQ",
}
COLORS = {0: "#33dd88", 1: "#ffb347"}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _crop_bounds(boxes: list[list[float]], width: int, height: int) -> tuple[int, int, int, int]:
    coords = np.asarray(boxes)
    x1, y1 = coords[:, :2].min(axis=0)
    x2, y2 = coords[:, 2:].max(axis=0)
    side = max(x2 - x1, y2 - y1, 80)
    margin = side * 0.8
    return (
        max(0, int(x1 - margin)),
        max(0, int(y1 - margin)),
        min(width, int(x2 + margin)),
        min(height, int(y2 + margin)),
    )


def _draw(
    ax, image: np.ndarray, record: dict, bounds: tuple[int, int, int, int], *, show_gt: bool = True
) -> None:
    x0, y0, x3, y3 = bounds
    ax.imshow(cv2.cvtColor(image[y0:y3, x0:x3], cv2.COLOR_BGR2RGB))
    if show_gt:
        for _cls, box in zip(
            record["ground_truth"]["classes"], record["ground_truth"]["boxes_xyxy"], strict=True
        ):
            x1, y1, x2, y2 = box
            ax.add_patch(
                plt.Rectangle(
                    (x1 - x0, y1 - y0),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="white",
                    linestyle="--",
                    linewidth=1.4,
                )
            )
    for detection in record["detections"]:
        if detection["confidence"] < 0.25:
            continue
        x1, y1, x2, y2 = detection["box_xyxy"]
        color = COLORS[detection["class_id"]]
        ax.add_patch(
            plt.Rectangle(
                (x1 - x0, y1 - y0), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2
            )
        )
        ax.text(
            x1 - x0,
            max(0, y1 - y0 - 4),
            f"{detection['class_name']} {detection['confidence']:.2f}",
            color="black",
            fontsize=7,
            bbox={"facecolor": color, "edgecolor": "none", "pad": 1.5},
        )
    ax.set_xlim(0, x3 - x0)
    ax.set_ylim(y3 - y0, 0)
    ax.axis("off")


def generate(artifact_root: Path, evidence: Path) -> None:
    root = artifact_root.resolve(strict=True)
    figures = evidence / "figures"
    tables = evidence / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    records = {
        name: _jsonl(evidence / "detection" / name / "predictions.jsonl") for name in VARIANTS
    }
    if len({len(value) for value in records.values()}) != 1:
        raise ValueError("detector prediction row counts differ")
    detection_rows = []
    for name in VARIANTS:
        metric = json.loads(
            (evidence / "detection" / name / "metrics.json").read_text(encoding="utf-8")
        )
        detection_rows.append(
            {
                "model": name,
                "frames": metric["frames_evaluated"],
                "mAP50": metric["mAP50"],
                "mAP50_95": metric["mAP50_95"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "detection_count": metric["detection_count"],
            }
        )
    _csv(tables / "detection_summary.csv", detection_rows)
    lighting_rows = []
    similarity_rows = []
    for name in VARIANTS:
        for lighting in ("daylight", "night"):
            subset = [row for row in records[name] if row["lighting"] == lighting]
            metric = compute_metrics(
                [row["ground_truth"] for row in subset], [row["detections"] for row in subset]
            )
            lighting_rows.append(
                {
                    "model": name,
                    "lighting": lighting,
                    "frames": len(subset),
                    "mAP50": metric["mAP50"],
                    "mAP50_95": metric["mAP50_95"],
                    "precision": metric["precision"],
                    "recall": metric["recall"],
                    "detection_count": metric["detection_count"],
                }
            )
        cosines = [
            row["raw_output_cosine_vs_fp32"]
            for row in records[name]
            if row["raw_output_cosine_vs_fp32"] is not None
        ]
        sqnr = [
            row["raw_output_sqnr_db_vs_fp32"]
            for row in records[name]
            if row["raw_output_sqnr_db_vs_fp32"] is not None
        ]
        matched = [
            det["matched_fp32_iou"]
            for row in records[name]
            for det in row["detections"]
            if det["confidence"] >= 0.25
        ]
        similarity_rows.append(
            {
                "model": name,
                "cosine_frames": len(cosines),
                "mean_raw_output_cosine": statistics.mean(cosines) if cosines else "",
                "mean_raw_output_sqnr_db": statistics.mean(sqnr) if sqnr else "",
                "mean_best_same_class_fp32_iou_at_conf_025": statistics.mean(matched)
                if matched
                else "",
                "task_detections": len(matched),
            }
        )
    _csv(tables / "detection_by_lighting.csv", lighting_rows)
    _csv(tables / "tensor_similarity_summary.csv", similarity_rows)
    recognition = json.loads(
        (evidence / "recognition/fp32_metrics.json").read_text(encoding="utf-8")
    )
    _csv(tables / "recognition_per_class.csv", recognition["per_class"])

    # Fig. 1: evidence scope, with historical observations explicitly separate.
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    boxes = [
        (0.3, 2.3, 2.8, 1.3, "Historical v2\nweight-only / summaries\nnot revalidated", "#eee9df"),
        (
            3.65,
            2.3,
            3.6,
            1.3,
            "Independent test\n2,417 frames / 2 classes\nsequence-disjoint",
            "#e5f1fa",
        ),
        (7.85, 2.3, 3.7, 1.3, "YOLO26 ONNX\nFP32 / Full QDQ / Head-excluded QDQ", "#e5f1fa"),
        (
            3.65,
            0.6,
            3.6,
            1.1,
            "Task evidence\nAP / frame predictions\nrecognition oracle ROIs",
            "#e8f5e9",
        ),
        (
            7.85,
            0.6,
            3.7,
            1.1,
            "Runtime & pipeline\nORT CPU / ORT Web\nByteTrack + KoreanSignNet",
            "#e8f5e9",
        ),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", fc=color, ec="#4b5563")
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
    ax.annotate("", (7.7, 2.95), (7.4, 2.95), arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.annotate("", (5.45, 1.78), (5.45, 2.15), arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.annotate("", (9.7, 1.78), (9.7, 2.15), arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.text(
        0.3,
        0.16,
        "Historical v2 is contextual only; all blue/green boxes use new raw evidence.",
        fontsize=9,
        color="#555",
    )
    fig.tight_layout()
    fig.savefig(figures / "fig1_evidence_scope.png", dpi=180)
    plt.close(fig)

    # Fig. 2: deterministic first frame meeting the three-way failure criterion.
    choice = next(
        i
        for i in range(len(records[VARIANTS[0]]))
        if records[VARIANTS[0]][i]["detection_count_task_threshold"] >= 2
        and records[VARIANTS[1]][i]["detection_count_task_threshold"] == 0
        and records[VARIANTS[2]][i]["detection_count_task_threshold"] >= 2
    )
    reference = records[VARIANTS[0]][choice]
    image = cv2.imread(str(root / reference["image_rel"]))
    gt_boxes = reference["ground_truth"]["boxes_xyxy"]
    bounds = _crop_bounds(gt_boxes, image.shape[1], image.shape[0])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    for ax, name in zip(axes, VARIANTS, strict=True):
        _draw(ax, image, records[name][choice], bounds)
        ax.set_title(
            f"{LABELS[name]}: {records[name][choice]['detection_count_task_threshold']} detections"
        )
    fig.suptitle("Same independent test frame; dashed white = ground truth; confidence >= 0.25")
    fig.tight_layout()
    fig.savefig(figures / "fig2_head_qdq_failure.png", dpi=200)
    plt.close(fig)

    # Fig. 3: scope-specific p50/p95 timing; unsupported combinations are labeled.
    runtime_rows = []
    sources = (
        ("ORT CPU", evidence / "runtime/ort", "cpu"),
        ("Chrome WASM", evidence / "runtime/browser_optimized", "wasm"),
        ("Chrome WebGPU", evidence / "runtime/browser_optimized", "webgpu"),
    )
    for surface, folder, prefix in sources:
        for name in ("fp32", "fp16", "full_qdq", "head_excluded_qdq"):
            data = json.loads((folder / f"{prefix}_{name}.json").read_text(encoding="utf-8"))
            metric = data.get("metrics", {}).get("detector_total_ms", {})
            config = data.get("config", data)
            runtime_rows.append(
                {
                    "surface": surface,
                    "model": name,
                    "status": data["status"],
                    "mean_ms": metric.get("mean_ms", ""),
                    "p50_ms": metric.get("p50_ms", ""),
                    "p95_ms": metric.get("p95_ms", ""),
                    "fps": metric.get("fps", ""),
                    "session_init_ms": config.get(
                        "session_init_ms", data.get("session_init_ms", "")
                    ),
                    "model_download_ms": data.get("model_download_ms", ""),
                    "first_inference_ms": config.get(
                        "first_inference_ms", data.get("first_inference_ms", "")
                    ),
                    "reason": data.get("reason", "")[:180],
                }
            )
    _csv(tables / "runtime_summary.csv", runtime_rows)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    for ax, (surface, _, _) in zip(axes, sources, strict=True):
        subset = [row for row in runtime_rows if row["surface"] == surface]
        for j, row in enumerate(subset):
            if row["status"] != "completed":
                ax.text(j, 5, "unsupported", rotation=90, ha="center", va="bottom", fontsize=8)
                continue
            ax.bar(j, row["p50_ms"], color="#497fb2")
            ax.vlines(j, row["p50_ms"], row["p95_ms"], color="#d95f02", linewidth=4)
        ax.set_title(surface)
        ax.set_xticks(range(4), ["FP32", "FP16", "Full\nQDQ", "Head-excl.\nQDQ"])
        ax.set_ylabel("Detector p50 / p95 latency (ms)")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("p50 bars and p95 ticks; each surface has its own preprocessing path")
    fig.tight_layout()
    fig.savefig(figures / "fig3_runtime_latency.png", dpi=180)
    plt.close(fig)

    # Fig. 4: two test-set miss cases, chosen by explicit rules.
    head = records[VARIANTS[2]]

    def missed_gt(index: int, gt_index: int) -> bool:
        coarse = head[index]["ground_truth"]["classes"][gt_index]
        target = head[index]["ground_truth"]["boxes_xyxy"][gt_index]
        return not any(
            d["class_id"] == coarse
            and d["confidence"] >= 0.25
            and iou(d["box_xyxy"], target) >= 0.5
            for d in head[index]["detections"]
        )

    night = next(
        (i, j)
        for i, row in enumerate(head)
        if row["lighting"] == "night"
        for j in range(len(row["ground_truth"]["classes"]))
        if missed_gt(i, j)
    )
    small = next(
        (i, j)
        for i, row in enumerate(head)
        if row["lighting"] == "daylight"
        for j, box in enumerate(row["ground_truth"]["boxes_xyxy"])
        if 0.00015 <= ((box[2] - box[0]) * (box[3] - box[1])) / (1920 * 1080) <= 0.001
        and missed_gt(i, j)
    )
    cases = [("Night miss", night), ("Small object miss", small)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (title, (i, j)) in zip(axes, cases, strict=True):
        row = head[i]
        image = cv2.imread(str(root / row["image_rel"]))
        bounds = _crop_bounds(
            [row["ground_truth"]["boxes_xyxy"][j]], image.shape[1], image.shape[0]
        )
        _draw(ax, image, row, bounds)
        ax.set_title(f"{title} (frame {i})")
    fig.suptitle("Head-excluded QDQ test misses; dashed white = ground truth")
    fig.tight_layout()
    fig.savefig(figures / "fig4_failure_cases.png", dpi=180)
    plt.close(fig)
    (tables / "failure_case_selection.json").write_text(
        json.dumps(
            {
                "fig2": {
                    "frame_index": choice,
                    "image_rel": reference["image_rel"],
                    "rule": "first frame with FP32>=2, full QDQ=0, head-excluded>=2 detections at confidence>=0.25",
                },
                "fig4_night": {
                    "frame_index": night[0],
                    "gt_index": night[1],
                    "rule": "first night GT missed by head-excluded QDQ at IoU>=0.5",
                },
                "fig4_small": {
                    "frame_index": small[0],
                    "gt_index": small[1],
                    "rule": "first daylight GT with area fraction 0.00015..0.001 missed at IoU>=0.5",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Fig. 5: class-normalized independent-test confusion matrix.
    matrix = np.asarray(recognition["confusion_matrix"], dtype=float)
    normalized = matrix / np.maximum(1, matrix.sum(axis=1))[:, None]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    display = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(14))
    ax.set_yticks(range(14))
    ax.set_xlabel("Predicted class index")
    ax.set_ylabel("Ground-truth class index")
    ax.set_title("KoreanSignNet FP32: independent test, oracle-box ROIs")
    fig.colorbar(display, ax=ax, label="Row-normalized frequency")
    fig.tight_layout()
    fig.savefig(figures / "fig5_recognition_confusion.png", dpi=180)
    plt.close(fig)
    print("generated five figures and source tables")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=Path("paper_evidence"))
    args = parser.parse_args()
    generate(args.artifact_root, args.evidence)


if __name__ == "__main__":
    main()
