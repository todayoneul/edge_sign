/**
 * Toast.tsx — 자동 소멸 알림 큐 (app.js toast() 포팅)
 *
 * store.toasts → DOM 포탈로 #toast-wrap에 렌더.
 * 4초 후 자동 dismiss; out 애니메이션(CSS .toast.out) 후 제거.
 */

import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { useStore, type ToastItem } from "../store";

const AUTO_MS = 4000;
const ANIM_MS = 300; // .toast.out 애니메이션 duration (CSS 280ms + 버퍼)

function ToastEl({ item }: { item: ToastItem }) {
  const dismiss = useStore((s) => s.dismissToast);
  const elRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    // 4초 후 out 애니메이션 → dismiss
    timerRef.current = setTimeout(() => {
      const el = elRef.current;
      if (el) {
        el.classList.add("out");
        el.addEventListener(
          "animationend",
          () => dismiss(item.id),
          { once: true },
        );
        // 폴백: 애니메이션 없는 환경
        setTimeout(() => dismiss(item.id), ANIM_MS);
      } else {
        dismiss(item.id);
      }
    }, AUTO_MS);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [dismiss, item.id]);

  return (
    <div ref={elRef} className={`toast${item.kind ? ` ${item.kind}` : ""}`}>
      <span className="tdot" />
      <span>{item.msg}</span>
    </div>
  );
}

export default function Toast() {
  const toasts = useStore((s) => s.toasts);
  const wrap = document.getElementById("toast-wrap");
  if (!wrap) return null;
  return createPortal(
    <>
      {toasts.map((t) => (
        <ToastEl key={t.id} item={t} />
      ))}
    </>,
    wrap,
  );
}
