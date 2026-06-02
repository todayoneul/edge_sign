import { renderHook, act } from "@testing-library/react";
import { useTheme } from "./useTheme";

beforeEach(() => localStorage.clear());

test("기본 테마는 light (저장값 없을 때)", () => {
  const { result } = renderHook(() => useTheme());
  expect(result.current.theme).toBe("light");
  expect(document.documentElement.getAttribute("data-theme")).toBe("light");
});
test("toggle 시 dark로 전환되고 localStorage에 저장", () => {
  const { result } = renderHook(() => useTheme());
  act(() => result.current.toggle());
  expect(result.current.theme).toBe("dark");
  expect(localStorage.getItem("edge-sign-theme")).toBe("dark");
  expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
});
test("저장된 dark를 복원", () => {
  localStorage.setItem("edge-sign-theme", "dark");
  const { result } = renderHook(() => useTheme());
  expect(result.current.theme).toBe("dark");
});
