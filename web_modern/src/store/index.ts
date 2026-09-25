import { create } from "zustand";
import type { FrameResult, Track, VariantInfo } from "../lib/types";

// ── Toast ────────────────────────────────────────────────────────────────────
export interface ToastItem {
  id: number;
  msg: string;
  kind: "ok" | "warn" | "err" | "";
}

interface State {
  connected: boolean;
  sourceKind: "none" | "stream" | "session";
  /** 추론 위치: server=서버 WS 추론, ondevice=브라우저 ORT-Web(WebGPU) 추론 */
  pipelineMode: "server" | "ondevice";
  /** 온디바이스 검출기 정밀도: fp32=빠름(43MB), fp16=작음(22MB) */
  ondeviceModel: "fp32" | "fp16";
  playing: boolean;
  tracks: Track[];
  totalDetections: number;
  telemetry: {
    fps: number;
    inferenceMs: number;
    stageMs?: FrameResult["stage_ms"];
    variant?: string;
    modelMb?: number;
  };
  /** Selected variant name (set by PerfStrip toggle, read by useStream/useSession) */
  selectedVariant: string | null;
  /** Available variants from /api/status */
  variants: VariantInfo[];
  /** Per-variant FPS history for Δ display */
  fpsByVariant: Record<string, number>;
  activeTab: "tracks" | "qa";
  byokKey: string;
  /** Hovered track id for canvas ↔ rail highlight */
  hoverId: number | null;
  /** Toast queue */
  toasts: ToastItem[];
  _toastSeq: number;
  /** performance.now() of the last result and the unrounded FPS average (setFrame) */
  _lastFrameAt: number;
  _fpsEma: number;

  setFrame: (r: FrameResult) => void;
  /** Drop FPS to 0 once no result has arrived for FPS_STALE_MS (called by Header while playing). */
  decayFps: () => void;
  setPipelineMode: (m: "server" | "ondevice") => void;
  setOndeviceModel: (m: "fp32" | "fp16") => void;
  setConnected: (b: boolean) => void;
  setTab: (t: "tracks" | "qa") => void;
  setByok: (k: string) => void;
  reset: () => void;
  setHoverId: (id: number | null) => void;
  setSelectedVariant: (name: string) => void;
  setVariants: (v: VariantInfo[], active: string | null) => void;
  pushToast: (msg: string, kind?: ToastItem["kind"]) => void;
  dismissToast: (id: number) => void;
}

// Processing FPS = rate of results. Server stream, server session and on-device inference
// all deliver results through setFrame, so it is measured once here for every mode.
// A gap longer than this (pause, seek, stalled server) restarts the average.
export const FPS_STALE_MS = 2000;

export const useStore = create<State>((set) => ({
  connected: false,
  sourceKind: "none",
  // 기본 온디바이스(WebGPU) — 공개 데모가 무료 CPU 서버에 의존하지 않고 방문자 GPU에서 추론(빠름).
  pipelineMode: "ondevice",
  ondeviceModel: "fp32",
  playing: false,
  tracks: [],
  totalDetections: 0,
  telemetry: { fps: 0, inferenceMs: 0 },
  selectedVariant: null,
  variants: [],
  fpsByVariant: {},
  activeTab: "tracks",
  byokKey: localStorage.getItem("edge-sign-byok") ?? "",
  hoverId: null,
  toasts: [],
  _toastSeq: 0,
  _lastFrameAt: 0,
  _fpsEma: 0,

  setFrame: (r) =>
    set((s) => {
      const now = performance.now();
      const gap = now - s._lastFrameAt;
      const fresh = s._lastFrameAt > 0 && gap < FPS_STALE_MS;
      const inst = fresh ? 1000 / Math.max(1, gap) : 0;
      const ema = !fresh ? 0 : s._fpsEma > 0 ? s._fpsEma * 0.8 + inst * 0.2 : inst;
      const fps = Math.round(ema * 10) / 10;
      // keyed by the variant that produced the result (server A/B or ondevice-<ep>)
      const key = r.variant ?? s.selectedVariant;
      return {
        _lastFrameAt: now,
        _fpsEma: ema,
        tracks: r.tracks,
        totalDetections: s.totalDetections + r.tracks.length,
        telemetry: {
          ...s.telemetry,
          fps,
          inferenceMs: r.inference_ms,
          stageMs: r.stage_ms,
          variant: r.variant,
          modelMb: r.model_mb,
        },
        fpsByVariant: key && fps > 0 ? { ...s.fpsByVariant, [key]: fps } : s.fpsByVariant,
      };
    }),

  decayFps: () =>
    set((s) =>
      s.telemetry.fps > 0 && performance.now() - s._lastFrameAt > FPS_STALE_MS
        ? { _fpsEma: 0, telemetry: { ...s.telemetry, fps: 0 } }
        : s,
    ),

  setPipelineMode: (m) => set({ pipelineMode: m }),
  setOndeviceModel: (m) => set({ ondeviceModel: m }),
  setConnected: (b) => set({ connected: b }),
  setTab: (t) => set({ activeTab: t }),
  setByok: (k) => {
    localStorage.setItem("edge-sign-byok", k);
    set({ byokKey: k });
  },
  reset: () =>
    set((s) => ({
      tracks: [],
      totalDetections: 0,
      sourceKind: "none",
      playing: false,
      hoverId: null,
      fpsByVariant: {},
      _lastFrameAt: 0,
      _fpsEma: 0,
      telemetry: { ...s.telemetry, fps: 0 },
    })),

  setHoverId: (id) => set({ hoverId: id }),

  setSelectedVariant: (name) => set({ selectedVariant: name }),

  setVariants: (v, active) =>
    set((s) => ({
      variants: v,
      selectedVariant: active ?? s.selectedVariant ?? v[0]?.name ?? null,
    })),

  pushToast: (msg, kind = "") => {
    set((s) => {
      const id = s._toastSeq + 1;
      return {
        _toastSeq: id,
        toasts: [...s.toasts, { id, msg, kind }],
      };
    });
  },

  dismissToast: (id) =>
    set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}));
