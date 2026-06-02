/**
 * QAPanel.tsx — Q&A 주행 어시스턴트 (app.js sendQuestion / BYOK 포팅)
 *
 * - useQA.ask(store.tracks, question) → SSE 스트리밍 답변
 * - quick chips: 지금 보이는 표지판은 뭐야? / 지금 멈춰야 해? / 현재 장면을 요약해줘.
 * - Enter 전송 (Shift+Enter 줄바꿈)
 * - BYOK password input + 표시 toggle + state label (ok/off)
 * - chatInputRef: 부모에서 / 단축키로 포커스하기 위해 forwardRef
 */

import { forwardRef, useCallback, useRef, useState } from "react";
import { useQA } from "../hooks/useQA";
import { useStore } from "../store";

interface ChatMsg {
  role: "user" | "assistant";
  text: string;
  streaming?: boolean;
}

const QUICK_CHIPS = [
  "지금 보이는 표지판은 뭐야?",
  "지금 멈춰야 해?",
  "현재 장면을 요약해줘.",
] as const;

const QAPanel = forwardRef<HTMLTextAreaElement>((_, chatRef) => {
  const tracks = useStore((s) => s.tracks);
  const byokKey = useStore((s) => s.byokKey);
  const setByok = useStore((s) => s.setByok);
  const setTab = useStore((s) => s.setTab);
  const pushToast = useStore((s) => s.pushToast);

  const { ask, streaming } = useQA();
  const [msgs, setMsgs] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [byokVisible, setByokVisible] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);
  const internalRef = useRef<HTMLTextAreaElement>(null);
  // Merge forwarded ref with internal ref
  const textareaRef = (chatRef as React.RefObject<HTMLTextAreaElement>) ?? internalRef;

  const scrollToBottom = useCallback(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, []);

  const sendQuestion = useCallback(
    async (qOverride?: string) => {
      const question = (qOverride ?? input).trim();
      if (!question || streaming) return;
      setInput("");

      if (!tracks.length) {
        setMsgs((m) => [
          ...m,
          { role: "user", text: question },
          { role: "assistant", text: "아직 인식된 객체가 없습니다. 먼저 영상을 재생해 주세요." },
        ]);
        scrollToBottom();
        return;
      }

      // Add user bubble
      setMsgs((m) => [...m, { role: "user", text: question }]);
      scrollToBottom();

      // Add streaming assistant bubble
      const idx = { current: -1 };
      setMsgs((m) => {
        idx.current = m.length + 1; // will be this index after user push
        return [...m, { role: "assistant", text: "", streaming: true }];
      });

      // We use a local accumulator and update state on each token
      let full = "";
      try {
        await ask(tracks, question, (chunk) => {
          full += chunk;
          setMsgs((m) => {
            const copy = [...m];
            const last = copy[copy.length - 1];
            if (last?.role === "assistant") {
              copy[copy.length - 1] = { ...last, text: full };
            }
            return copy;
          });
          scrollToBottom();
        });
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        setMsgs((m) => {
          const copy = [...m];
          const last = copy[copy.length - 1];
          if (last?.role === "assistant") {
            copy[copy.length - 1] = { ...last, text: `오류: ${msg}`, streaming: false };
          }
          return copy;
        });
        pushToast("답변 생성 실패: " + msg, "err");
      } finally {
        // Mark streaming done
        setMsgs((m) => {
          const copy = [...m];
          const last = copy[copy.length - 1];
          if (last?.role === "assistant") {
            copy[copy.length - 1] = { ...last, streaming: false };
          }
          return copy;
        });
      }
    },
    [input, tracks, streaming, ask, scrollToBottom, pushToast],
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void sendQuestion();
    }
  };

  const handleChip = (q: string) => {
    setTab("qa");
    void sendQuestion(q);
  };

  return (
    <div id="qa-panel" className="panel active" style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0 }}>
      {/* 퀵 칩 */}
      <div className="quick-chips" id="quick-chips">
        {QUICK_CHIPS.map((q) => (
          <button
            key={q}
            className="chip"
            type="button"
            onClick={() => handleChip(q)}
          >
            {q}
          </button>
        ))}
      </div>

      {/* 채팅 로그 */}
      <div id="chat-log" ref={logRef} style={{ flex: 1, overflowY: "auto", padding: "var(--sp-4)", display: "flex", flexDirection: "column", gap: "var(--sp-3)" }}>
        {msgs.length === 0 && (
          <div className="empty" style={{ margin: "auto", textAlign: "center", color: "var(--ink-3)", fontFamily: "var(--f-mono)", fontSize: "0.78rem", lineHeight: 1.8 }}>
            질문을 입력하거나 위의 빠른 질문을 클릭하세요
          </div>
        )}
        {msgs.map((m, i) => (
          <div
            key={i}
            className={`chat-msg ${m.role}`}
          >
            {m.text}
            {m.role === "assistant" && m.streaming && (
              <span className="cursor" style={{ display: "inline" }}>
                <span
                  style={{
                    display: "inline-block",
                    color: "var(--ink-3)",
                    animation: "blink 0.9s step-end infinite",
                    marginLeft: 2,
                  }}
                >
                  ▍
                </span>
              </span>
            )}
          </div>
        ))}
      </div>

      {/* 입력 영역 */}
      <div id="chat-input-area" style={{ display: "flex", gap: "var(--sp-2)", alignItems: "flex-end", padding: "var(--sp-3) var(--sp-4)", borderTop: "1px solid var(--line)", background: "var(--surface)" }}>
        <textarea
          ref={textareaRef}
          id="chat-input"
          placeholder="표지판에 대해 질문해보세요…"
          rows={1}
          value={input}
          onChange={(e) => {
            setInput(e.target.value);
            // Auto-resize (app.js chatInput.addEventListener('input'))
            e.target.style.height = "auto";
            e.target.style.height = Math.min(120, e.target.scrollHeight) + "px";
          }}
          onKeyDown={handleKeyDown}
          style={{ resize: "none", minHeight: 44, maxHeight: 120, lineHeight: 1.5 }}
        />
        <button
          id="send-btn"
          className="btn btn-primary"
          type="button"
          disabled={streaming || !input.trim()}
          onClick={() => void sendQuestion()}
          aria-label="전송"
          style={{ minHeight: 44, padding: "0 var(--sp-3)", flexShrink: 0 }}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" style={{ width: 16, height: 16 }}>
            <path d="M22 2 11 13M22 2 15 22l-4-9-9-4 20-7z" />
          </svg>
        </button>
      </div>

      {/* BYOK */}
      <div className="byok">
        <div className="byok-label">
          <span>Groq API 키 (BYOK)</span>
          <span className={`byok-state ${byokKey ? "ok" : "off"}`} id="byok-state">
            {byokKey ? "저장됨" : "미설정"}
          </span>
        </div>
        <div className="byok-field">
          <input
            id="byok-input"
            type={byokVisible ? "text" : "password"}
            placeholder="gsk_..."
            value={byokKey}
            onChange={(e) => setByok(e.target.value.trim())}
            autoComplete="off"
            spellCheck={false}
          />
          <button
            id="byok-show"
            className="btn btn-ghost"
            type="button"
            onClick={() => setByokVisible((v) => !v)}
            style={{ minHeight: 42, padding: "0 var(--sp-3)", fontSize: "0.72rem", fontFamily: "var(--f-mono)", flexShrink: 0 }}
          >
            {byokVisible ? "숨김" : "표시"}
          </button>
        </div>
        <p className="byok-help">
          키 없이도 서버 기본 키를 사용합니다.{" "}
          <a href="https://console.groq.com/keys" target="_blank" rel="noopener noreferrer">
            Groq 콘솔
          </a>에서 발급.
        </p>
      </div>
    </div>
  );
});

QAPanel.displayName = "QAPanel";
export default QAPanel;
