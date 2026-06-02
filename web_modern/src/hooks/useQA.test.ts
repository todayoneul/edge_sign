import { renderHook, act, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import * as api from "../lib/api";
import { useQA } from "./useQA";

test("ask 시 토큰을 순서대로 누적", async () => {
  vi.spyOn(api, "askQA").mockImplementation(async (_t, _q, _k, onEvent) => {
    onEvent({ type: "token", text: "안" });
    onEvent({ type: "token", text: "녕" });
    onEvent({ type: "done" });
  });
  const { result } = renderHook(() => useQA());
  await act(async () => { await result.current.ask([], "?"); });
  await waitFor(() => expect(result.current.answer).toBe("안녕"));
});
