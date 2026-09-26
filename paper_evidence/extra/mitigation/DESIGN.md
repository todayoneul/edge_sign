# Known-mitigation baseline: box-coordinate normalization (design, written before running)

Date: 2026-09-27. Scripts: `scripts/paper/normalization_baseline.py`. Results go to this folder only; existing evidence is not touched.

## 1. What the prior work does

**Moon et al., "YOLOv6+", Signal, Image and Video Processing 2025.**
Checked in the Research Square preprint rs-5738660/v1; the Version of Record is doi 10.1007/s11760-025-04234-0.
- **Fig. 1a.** YOLOv6's decoupled head computes box coordinates B = (A + R)·stride in [0, W). A is the anchor point and R the regression output. B is concatenated with class scores in [0, 1]. TFLite INT8 gives mAP 0.0%, attributed to the dynamic-range mismatch.
- **Fig. 1b, "vanilla rescaling".** They attribute this to YOLOv8 (Ultralytics). The pixel box is multiplied by 1/W, giving B in [0, 1), then concatenated with the scores. After the concat it is rescaled by [W, 1] back to input resolution. YOLOv6n INT8 gives 28.3%. They note that the tensor after "× stride" still has a large quantization error: its QE is 0.78.
- **Fig. 1c / 5b, Eq. (6), "Regression Normalization (RN)".** Anchor points and regression are normalized separately by stride'_l = stride_l / W before they are added: Q(B_RN) = Q(A·stride') + Q(R·stride'). The stride' factor is folded into the regression convolution, so the conv output is already in [0, 1). The figure keeps the [W, 1] rescaling after the concat. YOLOv6n INT8 gives 30.6%.
- **Table 5.** COCO val2017 at 640×640, TFLite 2.16.1 full-integer PTQ, calibrated on COCO val. YOLOv8n goes from 35.3% (FP16) to 28.4% INT8 and 31.9% INT8 + RN; YOLOv6n goes from 35.7% to 28.3% and 30.6%.

**Ultralytics 8.4.56**, as installed in convnext_env.
- `utils/export/tensorflow.py::_tf_decode_boxes` is used only for TensorFlow, TFLite and Edge TPU exports. It computes `norm = strides / (stride[0] * grid_size)`, i.e. stride_l / 640 for a 640 input, and `dbox = decode_bboxes(dfl(boxes) * norm, anchors * norm[:, :2])`. Distances and anchors are therefore normalized before they are combined, and the boxes leave the graph in [0, 1].
- The ONNX export (`Detect._get_decode_boxes`) returns `decode_bboxes(dfl(boxes), anchors) * strides`, i.e. pixel boxes. This is the graph used in the paper.

## 2. Mapping onto our ONNX graphs (FP32 sources, opset 14)

| | YOLOv8s (v3, `model.22`) | YOLO26-n (v4, `model.23`) |
|---|---|---|
| distances, grid units, [1,4,8400] | `/model.22/dfl/Reshape_1` (after DFL softmax + fixed 0..15 conv) | `/model.23/Concat` (direct regression, no DFL) |
| anchor constants, grid units, [1,2,8400] | `/model.22/Sub` in[0], `/model.22/Add_1` in[0] | `/model.23/Sub` in[0], `/model.23/Add_1` in[0] |
| × stride, [1,8400] | `/model.22/Mul_2` | `/model.23/Mul_2` |
| box ⊕ score concat | `/model.22/Concat_3` = graph output [1,6,8400] | `/model.23/Concat_3` → Transpose → Split → TopK/Gather → output [1,300,6] |

Variants:
- **N1 `post_norm` (vanilla rescaling), both models.** Insert `Mul(×1/640)` between `Mul_2` and `Concat_3`. The pixel box tensor after `Mul_2` still exists, and it is quantized.
- **N2 `pre_norm` (Ultralytics TF exporter), both models.** Distances are multiplied by stride_l/640 (new `Mul`), the anchor constants are replaced by anchor·stride_l/640, and `Mul_2` is removed. Every box tensor is then in normalized units.
- **N3 `fold_norm`, YOLO26-n only.** stride_l/640 is folded into the weights and bias of the last regression conv of each level (`one2one_cv2.{0,1,2}.2`). The anchors are normalized and `Mul_2` is removed. This is the structure of Moon et al.'s Eq. (6) and Fig. 5b. It is not possible for YOLOv8s, because the DFL softmax sits between the conv and the distances, and softmax logits cannot absorb a scale.
- **Rescaling location.** In every variant the ×640 rescaling happens in post-processing (the decoder), outside the quantized graph. Ultralytics' TFLite backend deploys it the same way. Moon et al.'s figure draws the [W, 1] multiply after the concat; inside an ONNX QDQ graph that multiply would re-create a mixed-range tensor.
  - Because of this difference, **no variant is called "Regression Normalization"**. N3 is reported as "stride folded into the regression conv (the structure of Moon et al.'s RN)".
- **FP32 check.** Each normalized FP32 graph must reproduce the original FP32 detections (boxes within 1e-3 px after rescaling, identical scores) and the original FP32 mAP before it is quantized.
- **Quantization.** Identical to `quantize_detector_variants.py`:
  - `quant_pre_process`, QDQ, MinMax;
  - per-channel symmetric INT8 weights, asymmetric UINT8 activations, INT32 bias;
  - the same 150 calibration frames (`paper_evidence/splits/calibration_manifest.jsonl`);
  - full graph, no node excluded.

Comparison rows: FP32, full INT8, head-excluded INT8 and decode-excluded INT8 (existing files and metrics), plus N1–N3 INT8.

Metrics on the test set (2,417 frames, same decoder as the paper):
- mAP@0.5:0.95, mAP@0.5, and retention vs FP32;
- file size and SHA-256;
- the quantization scale of the box and concat tensors.

Size-bin retention uses the same bins as `size_bin_retention.py`.

## 3. Predictions (recorded before running)

- **P-N1.** Vanilla rescaling removes the zero-score collapse, because the concat is in [0, 1]. Retention stays below both head-excluded and decode-excluded INT8, because the pixel-unit box tensor (s ≈ 2.5 px) is still quantized before normalization.
- **P-N2/N3.** Pre-normalization removes the large-range intermediate tensors. However, a normalized box tensor quantized over [0, 1] still has a step of about 1/255 of the input (≈ 2.5 px). Retention therefore stays below decode-excluded INT8, which keeps the whole decode stage in FP32. The loss is larger for small objects.
- **P-WebGPU.** Normalization does not change the number of `QuantizeLinear` nodes much. On ORT-Web 1.30 WebGPU the full-INT8 normalized models therefore remain far slower than FP32. Model-side correctness does not fix runtime-side fallback.

Latency rule (fixed now):
- If a normalized INT8 model reaches ≥ 90% retention for a detector, measure that model's latency on:
  - ORT CPU 1T/4T (1,024 runs);
  - WASM 4T and WebGPU 1.30 (1,024 runs; 128 for WebGPU INT8);
  - and log its WebGPU operator placement.
- Otherwise report accuracy only.
