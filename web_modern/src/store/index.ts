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

  setFrame: (r: FrameResult) => void;
  setPipelineMode: (m: "server" | "ondevice") => void;
  setConnected: (b: boolean) => void;
  setTab: (t: "tracks" | "qa") => void;
  setByok: (k: string) => void;
  reset: () => void;
  setHoverId: (id: number | null) => void;
  setSelectedVariant: (name: string) => void;
  setVariants: (v: VariantInfo[], active: string | null) => void;
  recordFps: (fps: number) => void;
  pushToast: (msg: string, kind?: ToastItem["kind"]) => void;
  dismissToast: (id: number) => void;
}

export const useStore = create<State>((set, get) => ({
  connected: false,
  sourceKind: "none",
  pipelineMode: "server",
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

  setFrame: (r) =>
    set((s) => {
      const fps = s.telemetry.fps; // keep existing fps; Header updates it separately
      return {
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
      };
    }),

  setPipelineMode: (m) => set({ pipelineMode: m }),
  setConnected: (b) => set({ connected: b }),
  setTab: (t) => set({ activeTab: t }),
  setByok: (k) => {
    localStorage.setItem("edge-sign-byok", k);
    set({ byokKey: k });
  },
  reset: () =>
    set({
      tracks: [],
      totalDetections: 0,
      sourceKind: "none",
      playing: false,
      hoverId: null,
      fpsByVariant: {},
    }),

  setHoverId: (id) => set({ hoverId: id }),

  setSelectedVariant: (name) => set({ selectedVariant: name }),

  setVariants: (v, active) =>
    set((s) => ({
      variants: v,
      selectedVariant: active ?? s.selectedVariant ?? v[0]?.name ?? null,
    })),

  recordFps: (fps) => {
    const { selectedVariant } = get();
    set((s) => ({
      telemetry: { ...s.telemetry, fps },
      fpsByVariant: selectedVariant
        ? { ...s.fpsByVariant, [selectedVariant]: fps }
        : s.fpsByVariant,
    }));
  },

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
