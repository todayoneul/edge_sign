/**
 * clientPipeline.test.ts — 온디바이스 파이프라인 순수 함수 + 통합(스텁 ORT) 검증.
 */

import { describe, it, expect } from "vitest";
import {
  imageDataToNCHW,
  postprocessDetections,
  scaleDetections,
  preprocessClsRoi,
  softmax,
  decodeClsTop1,
  ClientPipeline,
  type OrtNamespace,
  type OrtSession,
  type RecognizerLabels,
} from "./clientPipeline";

// 합성 YOLO 출력 [1, 6, A] (chFirst: 채널<앵커) — 앵커0에만 검출 심기.
function makeOutput(A = 10): { data: Float32Array; dims: number[] } {
  const C = 6;
  const data = new Float32Array(C * A);
  const set = (c: number, a: number, v: number) => (data[c * A + a] = v);
  // 앵커0: cx,cy,w,h=320,320,100,100  cls0=0.9 cls1=0.1
  set(0, 0, 320);
  set(1, 0, 320);
  set(2, 0, 100);
  set(3, 0, 100);
  set(4, 0, 0.9);
  set(5, 0, 0.1);
  return { data, dims: [1, C, A] };
}

describe("clientPipeline 순수 함수", () => {
  it("imageDataToNCHW: 평면(NCHW) + /255", () => {
    // 2×2 RGBA: 픽셀0 (255,0,0), 픽셀1 (0,255,0), 픽셀2 (0,0,255), 픽셀3 (255,255,255)
    const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255]);
    const f = imageDataToNCHW(rgba, 2);
    // R 평면(0..3), G 평면(4..7), B 평면(8..11)
    expect(Array.from(f.slice(0, 4))).toEqual([1, 0, 0, 1]); // R
    expect(Array.from(f.slice(4, 8))).toEqual([0, 1, 0, 1]); // G
    expect(Array.from(f.slice(8, 12))).toEqual([0, 0, 1, 1]); // B
  });

  it("postprocessDetections: 임계값 위 앵커 → 1 검출, bbox/cls 정확", () => {
    const { data, dims } = makeOutput();
    const dets = postprocessDetections(data, dims);
    expect(dets).toHaveLength(1);
    expect(dets[0].cls).toBe(0);
    expect(dets[0].score).toBeCloseTo(0.9);
    expect(dets[0].bbox).toEqual([270, 270, 370, 370]); // cxcywh→xyxy
  });

  it("postprocessDetections: 같은 클래스 겹침 → NMS로 1개만", () => {
    const A = 10;
    const { data, dims } = makeOutput(A);
    // 앵커1: 거의 같은 박스, 약간 낮은 점수 → NMS 제거 대상
    data[0 * A + 1] = 325;
    data[1 * A + 1] = 325;
    data[2 * A + 1] = 100;
    data[3 * A + 1] = 100;
    data[4 * A + 1] = 0.8;
    const dets = postprocessDetections(data, dims);
    expect(dets).toHaveLength(1);
    expect(dets[0].score).toBeCloseTo(0.9); // 높은 점수만 유지
  });

  it("scaleDetections: 640 좌표 → 원본 스케일", () => {
    const scaled = scaleDetections([{ bbox: [320, 320, 640, 640], score: 0.9, cls: 0 }], 1280, 1280);
    expect(scaled[0].bbox).toEqual([640, 640, 1280, 1280]);
  });
});

describe("clientPipeline 인식 순수 함수", () => {
  it("preprocessClsRoi: (rgb/255-0.5)/0.5 정규화 + NCHW", () => {
    // 1×1 RGBA: (255, 0, 128)
    const f = preprocessClsRoi(new Uint8ClampedArray([255, 0, 128, 255]), 1);
    expect(f[0]).toBeCloseTo(1.0); // R: (1-0.5)/0.5
    expect(f[1]).toBeCloseTo(-1.0); // G: (0-0.5)/0.5
    expect(f[2]).toBeCloseTo((128 / 255 - 0.5) / 0.5); // B
  });

  it("softmax: 합 1·단조", () => {
    const p = softmax([1, 2, 3]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(p[2]).toBeGreaterThan(p[0]);
  });

  it("decodeClsTop1: det_cls 서브셋으로 후보 제한", () => {
    const labels: RecognizerLabels = {
      names: ["s0", "s1", "L0", "L1"],
      sign_ids: [0, 1],
      light_ids: [2, 3],
    };
    // 로짓 최대는 idx2(L0)지만 det_cls=0(표지판)이면 sign_ids[0,1]만 후보
    const logits = [0.1, 5.0, 9.0, 0.2];
    expect(decodeClsTop1(logits, labels, 0)?.label).toBe("s1"); // 표지판 서브셋 최대
    expect(decodeClsTop1(logits, labels, 1)?.label).toBe("L0"); // 신호등 서브셋 최대
  });
});

// ── 스텁 ORT — 항상 합성 검출 출력 반환 ──────────────────────────────────────
function stubOrt(): OrtNamespace {
  const session: OrtSession = {
    inputNames: ["images"],
    outputNames: ["output0"],
    run: async () => ({ output0: makeOutput() }),
  };
  return {
    Tensor: class {
      constructor(_type: string, _data: Float32Array, _dims: number[]) {
        void _type;
        void _data;
        void _dims;
      }
    },
    InferenceSession: { create: async () => session },
    env: { wasm: {} },
  };
}

describe("ClientPipeline 통합(스텁 ORT)", () => {
  it("load → processFrame: 검출+추적 → FrameResult 트랙 생성", async () => {
    const pipe = new ClientPipeline();
    await pipe.load(stubOrt(), new Uint8Array([0]), ["wasm"]);
    expect(pipe.loaded).toBe(true);
    expect(pipe.activeEP).toBe("wasm");

    const rgba = new Uint8ClampedArray(640 * 640 * 4);
    const r = await pipe.processFrame(rgba, 640, 640);
    expect(r.tracks).toHaveLength(1);
    expect(r.tracks[0].class_name).toBe("표지판");
    expect(r.tracks[0].id).toBe(1);
    expect(r.variant).toBe("ondevice-wasm");
    // 분류기 미로드 → 라벨=클래스명, recognize 단계는 루프 오버헤드만(≥0)
    expect(r.tracks[0].label).toBe("표지판");
    expect(r.stage_ms?.recognize).toBeGreaterThanOrEqual(0);
    expect(pipe.recognizerLoaded).toBe(false);
  });
});
