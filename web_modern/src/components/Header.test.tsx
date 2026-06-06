/**
 * Header.test.tsx — Header 렌더 테스트 (T6)
 *
 * 검증: useStore.setState로 tracks 주입 시 활성 트랙 KPI 표시 확인.
 */

import { render, screen } from "@testing-library/react";
import { act } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import Header from "./Header";
import { useStore } from "../store";

// ResizeObserver mock (jsdom에 없음)
beforeEach(() => {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };

  // canvas getContext mock
  HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({
    clearRect: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    arc: vi.fn(),
    stroke: vi.fn(),
    fill: vi.fn(),
    setTransform: vi.fn(),
    measureText: vi.fn(() => ({ width: 50 })),
    fillText: vi.fn(),
    fillRect: vi.fn(),
    strokeRect: vi.fn(),
    roundRect: vi.fn(),
    closePath: vi.fn(),
    arcTo: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    translate: vi.fn(),
  } as unknown as CanvasRenderingContext2D);
});

afterEach(() => {
  // store reset after each test
  useStore.setState({
    tracks: [],
    totalDetections: 0,
    telemetry: { fps: 0, inferenceMs: 0 },
    connected: false,
    playing: false,
  });
});

test("Header 렌더: idle(미재생) 시 KPI 격자 숨김 + '분석 준비됨' 뱃지", () => {
  render(<Header onToggleTheme={() => {}} />);
  // Direction A: idle엔 KPI 격자 대신 ready 뱃지 노출
  expect(document.getElementById("kpi-tracks")).toBeNull();
  expect(screen.getByText("분석 준비됨")).toBeTruthy();
});

test("Header 렌더: 재생 중 store에 tracks 주입 시 KPI 갱신", () => {
  render(<Header onToggleTheme={() => {}} />);

  act(() => {
    useStore.setState({
      playing: true,
      tracks: [
        { id: 1, class: 0, class_name: "traffic_sign", conf: 0.9, bbox: [0, 0, 10, 10] },
        { id: 2, class: 1, class_name: "traffic_light", conf: 0.8, bbox: [5, 5, 15, 15] },
        { id: 3, class: 2, class_name: "signboard", conf: 0.7, bbox: [20, 20, 40, 40] },
      ],
    });
  });

  // 재생 중에는 활성 트랙 KPI (#kpi-tracks) 값이 3이어야 함
  const kpiEl = document.getElementById("kpi-tracks");
  expect(kpiEl).not.toBeNull();
  expect(kpiEl!.textContent).toBe("3");
});

test("Header 렌더: 연결 상태 필 표시", () => {
  render(<Header onToggleTheme={() => {}} />);

  // 초기 상태: 대기
  expect(screen.getByText("대기")).toBeTruthy();

  act(() => {
    useStore.setState({ connected: true });
  });

  expect(screen.getByText("연결됨")).toBeTruthy();
});

test("Header 렌더: 테마 토글 버튼 존재", () => {
  const onToggle = vi.fn();
  render(<Header onToggleTheme={onToggle} />);
  const btn = document.getElementById("theme-toggle");
  expect(btn).not.toBeNull();
});
