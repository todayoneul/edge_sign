import { useStore } from "./index";

test("setFrame 시 트랙·텔레메트리·누적 갱신", () => {
  useStore.getState().setFrame({ frame_id: 1, inference_ms: 12, tracks: [{ id: 1, class: 0, class_name: "traffic_sign", conf: 0.9, bbox: [0,0,1,1] }] });
  const st = useStore.getState();
  expect(st.tracks.length).toBe(1);
  expect(st.telemetry.inferenceMs).toBe(12);
  expect(st.totalDetections).toBe(1);
});
