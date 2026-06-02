export interface Track { id: number; class: number; class_name: string; conf: number; label?: string; bbox: [number, number, number, number]; }
export interface StageMs { detect: number; track: number; recognize: number; }
export interface FrameResult { frame_id: number; inference_ms: number; tracks: Track[]; variant?: string; model_mb?: number; stage_ms?: StageMs; }
export interface VariantInfo { name: string; mb: number; }
export interface Status { yolo: boolean; ocr: boolean; tsign: boolean; taxonomy: string; variants: VariantInfo[]; active_variant: string | null; }
export type QAEvent = { type: "context" | "token"; text: string } | { type: "done" };
