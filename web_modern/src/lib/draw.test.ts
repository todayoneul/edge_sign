import { mapBox, classColor } from "./draw";

test("원본 bbox를 표시 크기로 스케일", () => {
  expect(mapBox([100, 50, 200, 150], 640, 480, 1280, 960)).toEqual([200, 100, 400, 300]);
});

test("클래스별 시그널색", () => {
  expect(classColor(0)).toBe("#22c55e");
  expect(classColor(1)).toBe("#ef4444");
  expect(classColor(2)).toBe("#f59e0b");
});
