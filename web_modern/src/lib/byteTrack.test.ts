/**
 * byteTrack.test.ts — TS 포팅이 서버 Python ByteTracker와 동일 거동인지 검증.
 *
 * golden(byteTrack.golden.json)은 src/track/bytetrack.py 를 결정적 시퀀스에 돌린
 * 출력(scripts/export_bytetrack_golden.py 생성). 같은 검출을 TS 추적기에 먹여
 * 트랙 ID(정확히 일치) + 박스(부동소수 float32↔64 차 허용 tolerance)를 대조한다.
 */

import { describe, it, expect, beforeEach } from "vitest";
import { ByteTracker, resetTrackId, type Detection } from "./byteTrack";
import golden from "./byteTrack.golden.json";

interface GoldenFrame {
  dets: number[][]; // [x1,y1,x2,y2,conf,cls]
  tracks: { id: number; cls: number; tlbr: number[] }[];
}

const BOX_TOL = 1.0; // px — Python float32 vs JS float64 Kalman 수치차 허용

describe("ByteTracker TS 포팅 ↔ Python 골든", () => {
  beforeEach(() => resetTrackId());

  it("프레임별 트랙 ID·박스가 골든과 일치", () => {
    const frames = golden as GoldenFrame[];
    const tracker = new ByteTracker({
      trackThresh: 0.5,
      matchThresh: 0.8,
      trackBuffer: 30,
      frameRate: 30,
      lowThresh: 0.1,
    });

    frames.forEach((frame, fi) => {
      const dets: Detection[] = frame.dets.map((d) => ({
        bbox: [d[0], d[1], d[2], d[3]],
        score: d[4],
        cls: d[5],
      }));
      const out = tracker.update(dets).sort((a, b) => a.trackId - b.trackId);

      // ID 집합 정확히 일치
      expect(out.map((t) => t.trackId), `frame ${fi + 1} track ids`).toEqual(
        frame.tracks.map((t) => t.id),
      );

      // 각 트랙 박스 tolerance 내 일치 + 클래스 동일
      out.forEach((t, i) => {
        const g = frame.tracks[i];
        expect(t.cls, `frame ${fi + 1} id ${t.trackId} cls`).toBe(g.cls);
        const tlbr = t.tlbr;
        for (let k = 0; k < 4; k++) {
          expect(
            Math.abs(tlbr[k] - g.tlbr[k]),
            `frame ${fi + 1} id ${t.trackId} tlbr[${k}] (${tlbr[k]} vs ${g.tlbr[k]})`,
          ).toBeLessThanOrEqual(BOX_TOL);
        }
      });
    });
  });

  it("lost→reactivate 시 트랙 ID 보존 (객체 A: 프레임6-7 소실, 8 복귀)", () => {
    const frames = golden as GoldenFrame[];
    // 골든 시나리오 검증: 프레임1의 id1이 프레임8에서도 동일 id로 복귀
    expect(frames[0].tracks.some((t) => t.id === 1)).toBe(true);
    expect(frames[5].tracks.some((t) => t.id === 1)).toBe(false); // 프레임6 소실
    expect(frames[7].tracks.some((t) => t.id === 1)).toBe(true); // 프레임8 복귀, 같은 id
  });
});
