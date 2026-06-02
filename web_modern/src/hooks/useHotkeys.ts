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
      const typing =
        document.activeElement instanceof HTMLInputElement ||
        document.activeElement instanceof HTMLTextAreaElement;

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
        handlers.onStepBack?.();
      } else if (e.key === "ArrowRight") {
        handlers.onStepForward?.();
      } else if (e.key.toLowerCase() === "t") {
        handlers.onToggleTheme?.();
      }
    }

    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [handlers]);
}
