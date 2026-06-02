/**
 * ShortcutsModal.tsx — 키보드 단축키 도움말 모달 (app.js toggleShortcuts 포팅)
 *
 * Radix Dialog 사용. open/onOpenChange는 App.tsx가 제어.
 * 항목은 index.html #shortcuts .kb-row 와 동일.
 */

import * as Dialog from "@radix-ui/react-dialog";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const SHORTCUTS = [
  { label: "재생 / 일시정지", keys: ["Space"] },
  { label: "질문 입력으로 이동", keys: ["/"] },
  { label: "5초 뒤로 / 앞으로", keys: ["←", "→"] },
  { label: "테마 전환", keys: ["T"] },
  { label: "이 도움말 열기 / 닫기", keys: ["?", "Esc"] },
] as const;

export default function ShortcutsModal({ open, onOpenChange }: Props) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        {/* 배경 스크림 */}
        <Dialog.Overlay
          style={{
            position: "fixed",
            inset: 0,
            zIndex: "var(--z-modal)",
            background: "var(--scrim)",
            backdropFilter: "blur(3px)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            animation: "fadeIn 0.2s var(--ease)",
          }}
        >
          {/* 모달 카드 */}
          <Dialog.Content
            className="modal-card"
            aria-labelledby="sc-title"
            onOpenAutoFocus={(e) => e.preventDefault()}
          >
            <div className="modal-head">
              <Dialog.Title id="sc-title" style={{ fontSize: "0.92rem", fontWeight: 600 }}>
                키보드 단축키
              </Dialog.Title>
              <Dialog.Close asChild>
                <button
                  className="icon-btn"
                  aria-label="닫기"
                  id="sc-close"
                >
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={2}
                    strokeLinecap="round"
                  >
                    <path d="M6 6l12 12M18 6 6 18" />
                  </svg>
                </button>
              </Dialog.Close>
            </div>
            <div className="modal-body">
              {SHORTCUTS.map(({ label, keys }) => (
                <div className="kb-row" key={label}>
                  <span>{label}</span>
                  <span style={{ display: "flex", gap: "var(--sp-1)", alignItems: "center" }}>
                    {keys.map((k) => (
                      <span className="kbd" key={k}>{k}</span>
                    ))}
                  </span>
                </div>
              ))}
            </div>
          </Dialog.Content>
        </Dialog.Overlay>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
