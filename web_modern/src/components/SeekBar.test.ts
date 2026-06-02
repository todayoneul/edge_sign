import { fmtTime } from "./SeekBar";

test("fmtTime: 0초 → 0:00", () => {
  expect(fmtTime(0)).toBe("0:00");
});

test("fmtTime: 9초 → 0:09 (0 패딩)", () => {
  expect(fmtTime(9)).toBe("0:09");
});

test("fmtTime: 75초 → 1:15", () => {
  expect(fmtTime(75)).toBe("1:15");
});

test("fmtTime: 음수 → --:--", () => {
  expect(fmtTime(-1)).toBe("--:--");
});

test("fmtTime: Infinity → --:--", () => {
  expect(fmtTime(Infinity)).toBe("--:--");
});

test("fmtTime: NaN → --:--", () => {
  expect(fmtTime(NaN)).toBe("--:--");
});

test("seekPercent: pos/total → 0~1000 정수", () => {
  const seekPercent = (pos: number, total: number) =>
    total > 0 ? Math.round(Math.min(1, pos / total) * 1000) : 0;
  expect(seekPercent(0, 300)).toBe(0);
  expect(seekPercent(150, 300)).toBe(500);
  expect(seekPercent(300, 300)).toBe(1000);
  expect(seekPercent(310, 300)).toBe(1000); // 클램프
  expect(seekPercent(0, 0)).toBe(0);        // total=0 → 0
});
