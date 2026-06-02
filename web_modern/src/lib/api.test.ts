import { parseSSELine } from "./api";
test("SSE data 라인을 QAEvent로 파싱", () => {
  expect(parseSSELine('data: {"type":"token","text":"안"}')).toEqual({ type: "token", text: "안" });
  expect(parseSSELine("event: ping")).toBeNull();
  expect(parseSSELine("")).toBeNull();
});
