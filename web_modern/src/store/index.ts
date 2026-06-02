import { create } from "zustand";
import type { FrameResult, Track } from "../lib/types";

interface State {
  connected: boolean;
  sourceKind: "none" | "stream" | "session";
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
  activeTab: "tracks" | "qa";
  byokKey: string;
  setFrame: (r: FrameResult) => void;
  setConnected: (b: boolean) => void;
  setTab: (t: "tracks" | "qa") => void;
  setByok: (k: string) => void;
  reset: () => void;
}

export const useStore = create<State>((set) => ({
  connected: false,
  sourceKind: "none",
  playing: false,
  tracks: [],
  totalDetections: 0,
  telemetry: { fps: 0, inferenceMs: 0 },
  activeTab: "tracks",
  byokKey: localStorage.getItem("edge-sign-byok") ?? "",

  setFrame: (r) =>
    set((s) => ({
      tracks: r.tracks,
      totalDetections: s.totalDetections + r.tracks.length,
      telemetry: {
        ...s.telemetry,
        inferenceMs: r.inference_ms,
        stageMs: r.stage_ms,
        variant: r.variant,
        modelMb: r.model_mb,
      },
    })),

  setConnected: (b) => set({ connected: b }),
  setTab: (t) => set({ activeTab: t }),
  setByok: (k) => {
    localStorage.setItem("edge-sign-byok", k);
    set({ byokKey: k });
  },
  reset: () =>
    set({ tracks: [], totalDetections: 0, sourceKind: "none", playing: false }),
}));
