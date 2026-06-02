/**
 * useHotkeys — 전역 키보드 단축키 (app.js document.addEventListener("keydown") 포팅)
 *
 * 단축키 목록 (app.js 806-817):
 *  Escape      — shortcuts 모달 닫기 / 텍스트 필드 blur
 *  Space       — play/pause 토글
 *  /           — Q&A 탭 포커스 (chat input)
 *  ? / Shift+/ — shortcuts 모달 열기/닫기
 *  ←           — 5초(≈75프레임) 뒤로
 *  →           — 5초(≈75프레임) 앞으로
 *  T / t       — 테마 토글
 *
 * DOM 직접 조작 없음 — 콜백 주입으로 처리.
 */

import { useEffect } from "react";

export interface HotkeyHandlers {
  onTogglePlay?: () => void;
  onFocusChat?: () => void;
  onToggleShortcuts?: (open?: boolean) => void;
  onStepBack?: () => void;   // 5초(≈75프레임) 뒤로
  onStepForward?: () => void;
  onToggleTheme?: () => void;
}

export function useHotkeys(handlers: HotkeyHandlers) {
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const ae = document.activeElement;
      // 텍스트 입력(텍스트 input·textarea)만 단축키 차단. range 슬라이더(시크·속도)에
      // 포커스가 있어도 Space·←·→ 가 동작하도록 통과시킨다.
      const typing =
        (ae instanceof HTMLInputElement && ae.type !== "range") ||
        ae instanceof HTMLTextAreaElement;

      // Escape — 항상 처리
      if (e.key === "Escape") {
        handlers.onToggleShortcuts?.(false);
        if (typing) (document.activeElement as HTMLElement).blur();
        return;
      }

      // 텍스트 입력 중에는 나머지 단축키 비활성 (app.js 패턴)
      if (typing) return;

      if (e.key === "?" || (e.shiftKey && e.key === "/")) {
        e.preventDefault();
        handlers.onToggleShortcuts?.();
      } else if (e.key === "/") {
        e.preventDefault();
        handlers.onFocusChat?.();
      } else if (e.key === " ") {
        e.preventDefault();
        handlers.onTogglePlay?.();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();   // range 포커스 시 네이티브 슬라이더 이동 방지
        handlers.onStepBack?.();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        handlers.onStepForward?.();
      } else if (e.key.toLowerCase() === "t") {
        handlers.onToggleTheme?.();
      }
    }

    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [handlers]);
}
