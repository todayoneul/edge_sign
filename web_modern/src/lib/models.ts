/**
 * models.ts — 온디바이스 검출기 카탈로그 (모델 × 정밀도 × 실행 환경).
 *
 * 파일 크기와 정확도는 paper_evidence/reports/RUNTIME_MATRIX.md(독립 test 2,417장)의 값.
 * INT8은 검출 헤드를 FP32로 둔 정적 QDQ(헤드까지 INT8이면 검출 0 — 논문 4.2절).
 */

export type DetectorId = "yolo26n" | "yolov8s";
export type Precision = "fp32" | "fp16" | "int8";
export type ExecutionProvider = "webgpu" | "wasm";

export interface OnDeviceConfig {
  detector: DetectorId;
  precision: Precision;
  ep: ExecutionProvider;
}

interface ModelFile {
  file: string;
  mb: number;
  /** 독립 test mAP@0.5:0.95 */
  map: number;
  /** 입력 텐서 자료형 (YOLO26-n FP16만 FP16 입력, 출력은 모두 FP32) */
  inputDtype: "float32" | "float16";
}

export const DETECTORS: Record<
  DetectorId,
  { name: string; params: string; format: "yolo26" | "v8"; files: Record<Precision, ModelFile> }
> = {
  yolo26n: {
    name: "YOLO26-n",
    params: "2.4M",
    format: "yolo26",
    files: {
      fp32: { file: "yolo_v4_signs_fp32.onnx", mb: 9.8, map: 0.242, inputDtype: "float32" },
      fp16: { file: "yolo_v4_signs_fp16.onnx", mb: 5.0, map: 0.242, inputDtype: "float16" },
      int8: { file: "yolo_v4_signs_int8_head_excluded.onnx", mb: 3.4, map: 0.235, inputDtype: "float32" },
    },
  },
  yolov8s: {
    name: "YOLOv8s",
    params: "11.2M",
    format: "v8",
    files: {
      fp32: { file: "yolov8s_signs_v3_fp32.onnx", mb: 44.7, map: 0.281, inputDtype: "float32" },
      fp16: { file: "yolov8s_signs_v3_fp16.onnx", mb: 22.4, map: 0.281, inputDtype: "float32" },
      int8: { file: "yolov8s_signs_v3_int8_static.onnx", mb: 18.0, map: 0.277, inputDtype: "float32" },
    },
  },
};

export const DEFAULT_ONDEVICE: OnDeviceConfig = { detector: "yolo26n", precision: "fp32", ep: "webgpu" };

export const PRECISION_LABEL: Record<Precision, string> = { fp32: "FP32", fp16: "FP16", int8: "INT8" };
export const EP_LABEL: Record<ExecutionProvider, string> = { webgpu: "WebGPU", wasm: "WASM" };

export function modelFile(cfg: OnDeviceConfig): ModelFile {
  return DETECTORS[cfg.detector].files[cfg.precision];
}

export function modelUrl(cfg: OnDeviceConfig): string {
  return `/models/${modelFile(cfg).file}`;
}

/** 실행 환경 시도 순서: WebGPU를 고르면 미지원 브라우저에서 WASM으로 폴백. */
export function executionProviders(cfg: OnDeviceConfig): ExecutionProvider[] {
  return cfg.ep === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];
}

/** "YOLO26-n FP32" */
export function modelName(cfg: Pick<OnDeviceConfig, "detector" | "precision">): string {
  return `${DETECTORS[cfg.detector].name} ${PRECISION_LABEL[cfg.precision]}`;
}

/** INT8 QDQ는 WebGPU에서 QuantizeLinear가 CPU로 폴백해 수십 배 느리다 (논문 4.3절). */
export function slowCombination(cfg: OnDeviceConfig): boolean {
  return cfg.precision === "int8" && cfg.ep === "webgpu";
}
