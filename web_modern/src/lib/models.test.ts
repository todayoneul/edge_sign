import { expect, test } from "vitest";
import { DEFAULT_ONDEVICE, DETECTORS, executionProviders, modelName, modelUrl, slowCombination } from "./models";

test("기본 구성은 논문 파이프라인 기준 YOLO26-n FP32 · WebGPU", () => {
  expect(DEFAULT_ONDEVICE).toEqual({ detector: "yolo26n", precision: "fp32", ep: "webgpu" });
  expect(modelUrl(DEFAULT_ONDEVICE)).toBe("/models/yolo_v4_signs_fp32.onnx");
  expect(modelName(DEFAULT_ONDEVICE)).toBe("YOLO26-n FP32");
});

test("WebGPU를 고르면 WASM으로 폴백, WASM은 단독", () => {
  expect(executionProviders({ ...DEFAULT_ONDEVICE, ep: "webgpu" })).toEqual(["webgpu", "wasm"]);
  expect(executionProviders({ ...DEFAULT_ONDEVICE, ep: "wasm" })).toEqual(["wasm"]);
});

test("INT8 + WebGPU만 느린 조합으로 표시", () => {
  expect(slowCombination({ detector: "yolov8s", precision: "int8", ep: "webgpu" })).toBe(true);
  expect(slowCombination({ detector: "yolov8s", precision: "int8", ep: "wasm" })).toBe(false);
  expect(slowCombination({ detector: "yolov8s", precision: "fp16", ep: "webgpu" })).toBe(false);
});

test("모든 검출기 × 정밀도에 파일이 있고 FP16 입력은 YOLO26-n FP16뿐", () => {
  const fp16Inputs = Object.entries(DETECTORS).flatMap(([id, d]) =>
    Object.entries(d.files).filter(([, f]) => f.inputDtype === "float16").map(([p]) => `${id}/${p}`),
  );
  expect(fp16Inputs).toEqual(["yolo26n/fp16"]);
});
