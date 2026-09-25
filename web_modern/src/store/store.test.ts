import { afterEach, expect, test, vi } from "vitest";
import { FPS_STALE_MS, useStore } from "./index";
import type { FrameResult } from "../lib/types";

const frame = (variant?: string): FrameResult => ({
  frame_id: 1,
  inference_ms: 12,
  tracks: [{ id: 1, class: 0, class_name: "traffic_sign", conf: 0.9, bbox: [0, 0, 1, 1] }],
  variant,
});

afterEach(() => {
  vi.restoreAllMocks();
  useStore.getState().reset();
});

test("setFrame 시 트랙·텔레메트리·누적 갱신", () => {
  useStore.getState().setFrame(frame());
  const st = useStore.getState();
  expect(st.tracks.length).toBe(1);
  expect(st.telemetry.inferenceMs).toBe(12);
  expect(st.totalDetections).toBe(1);
});

test("처리 FPS: 결과 간격으로 모든 모드에서 계산", () => {
  const now = vi.spyOn(performance, "now");
  for (let i = 0; i < 20; i++) {
    now.mockReturnValue(1000 + i * 100); // 결과 100 ms 간격 = 10 FPS
    useStore.getState().setFrame(frame("yolov8s_v3_int8"));
  }
  const st = useStore.getState();
  expect(st.telemetry.fps).toBe(10);
  expect(st.fpsByVariant["yolov8s_v3_int8"]).toBe(10);
});

test("처리 FPS: 결과가 끊기면 0으로 떨어지고 재개 시 새로 계산", () => {
  const now = vi.spyOn(performance, "now");
  now.mockReturnValue(1000);
  useStore.getState().setFrame(frame());
  now.mockReturnValue(1050);
  useStore.getState().setFrame(frame());
  expect(useStore.getState().telemetry.fps).toBe(20);

  now.mockReturnValue(1050 + FPS_STALE_MS + 1);
  useStore.getState().decayFps();
  expect(useStore.getState().telemetry.fps).toBe(0);

  // 재개 첫 결과는 이전 간격을 쓰지 않는다
  now.mockReturnValue(5000);
  useStore.getState().setFrame(frame());
  expect(useStore.getState().telemetry.fps).toBe(0);
  now.mockReturnValue(5200);
  useStore.getState().setFrame(frame());
  expect(useStore.getState().telemetry.fps).toBe(5);
});
