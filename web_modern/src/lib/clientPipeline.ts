/**
 * clientPipeline.ts — 브라우저 온디바이스 추론 파이프라인 (검출 + 추적).
 *
 * 서버 e2e_pipeline.process_frame 의 클라이언트판: ORT-Web 검출(WebGPU) →
 * byteTrack.ts 추적 → 서버와 동일한 FrameResult 형태로 변환(기존 store/오버레이 재사용).
 * 인식(OCR/분류기)은 후속 — 현재 label=클래스명.
 *
 * 전처리/후처리는 서버 _preprocess_yolo / postprocess_yolo 를 그대로 옮겼다
 * (640 단순 resize, [1,4+nc,8400] 디코딩, 클래스별 greedy NMS). 순수 함수라 단위 테스트됨.
 */

import { ByteTracker, type Detection } from "./byteTrack";
import type { FrameResult, Track } from "./types";

export const DET_INPUT = 640;
// v3 택소노미 (e2e_pipeline.py class_names). 한국어 표기 — 온디바이스 라벨.
export const DET_CLASS_NAMES = ["표지판", "신호등", "간판"];
export const DET_CONF_THRES = 0.25;
export const DET_IOU_THRES = 0.45;

// ── 전처리: RGBA ImageData → Float32 NCHW [1,3,H,W] ───────────────────────────
/** 서버 _preprocess_yolo 동일: RGB/255, 채널 평면(NCHW). 입력은 640×640 RGBA. */
export function imageDataToNCHW(rgba: Uint8ClampedArray, size = DET_INPUT): Float32Array {
  const n = size * size;
  const f = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    f[i] = rgba[i * 4] / 255; // R
    f[n + i] = rgba[i * 4 + 1] / 255; // G
    f[2 * n + i] = rgba[i * 4 + 2] / 255; // B
  }
  return f;
}

// ── 후처리: ONNX 출력 → Detection[] (입력 640 좌표계) ─────────────────────────
/**
 * 서버 postprocess_yolo 동일. out dims [1,C,A] 또는 [1,A,C], C=4+nc.
 * @returns Detection[]  bbox=[x1,y1,x2,y2] (640 좌표), score, cls
 */
export function postprocessDetections(
  data: ArrayLike<number>,
  dims: readonly number[],
  confThres = DET_CONF_THRES,
  iouThres = DET_IOU_THRES,
): Detection[] {
  if (dims.length !== 3) return [];
  let C: number, A: number, chFirst: boolean;
  if (dims[1] < dims[2]) {
    C = dims[1];
    A = dims[2];
    chFirst = true;
  } else {
    C = dims[2];
    A = dims[1];
    chFirst = false;
  }
  const nc = C - 4;
  if (nc <= 0) return [];
  const at = (a: number, c: number) => (chFirst ? data[c * A + a] : data[a * C + c]);

  const cand: Detection[] = [];
  for (let a = 0; a < A; a++) {
    let best = 0;
    let bestC = 0;
    for (let c = 0; c < nc; c++) {
      const s = at(a, 4 + c);
      if (s > best) {
        best = s;
        bestC = c;
      }
    }
    if (best <= confThres) continue;
    const cx = at(a, 0);
    const cy = at(a, 1);
    const w = at(a, 2);
    const h = at(a, 3);
    cand.push({
      bbox: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
      score: best,
      cls: bestC,
    });
  }
  return nmsPerClass(cand, iouThres);
}

function iou(a: Detection["bbox"], b: Detection["bbox"]): number {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-7;
  return inter / ua;
}

function nmsPerClass(boxes: Detection[], iouThres: number): Detection[] {
  const out: Detection[] = [];
  const classes = new Set(boxes.map((b) => b.cls));
  for (const c of classes) {
    const group = boxes.filter((b) => b.cls === c).sort((p, q) => q.score - p.score);
    const keep: Detection[] = [];
    for (const b of group) {
      if (keep.every((k) => iou(k.bbox, b.bbox) <= iouThres)) keep.push(b);
    }
    out.push(...keep);
  }
  return out;
}

// ── 검출 좌표(640) → 원본 프레임 좌표 스케일 ─────────────────────────────────
export function scaleDetections(dets: Detection[], srcW: number, srcH: number): Detection[] {
  const sx = srcW / DET_INPUT;
  const sy = srcH / DET_INPUT;
  return dets.map((d) => ({
    ...d,
    bbox: [d.bbox[0] * sx, d.bbox[1] * sy, d.bbox[2] * sx, d.bbox[3] * sy],
  }));
}

// ── 분류기(한국 표지판/신호등 14클래스) 인식 — 서버 _run_tsign 포팅 ────────────
export interface RecognizerLabels {
  names: string[];
  sign_ids: number[];
  light_ids: number[];
}

/** ROI RGBA(32×32) → [1,3,32,32] ((rgb/255-0.5)/0.5). 서버 _run_tsign 전처리 동일. */
export function preprocessClsRoi(rgba32: Uint8ClampedArray, size = 32): Float32Array {
  const n = size * size;
  const f = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    f[i] = (rgba32[i * 4] / 255 - 0.5) / 0.5;
    f[n + i] = (rgba32[i * 4 + 1] / 255 - 0.5) / 0.5;
    f[2 * n + i] = (rgba32[i * 4 + 2] / 255 - 0.5) / 0.5;
  }
  return f;
}

export function softmax(logits: ArrayLike<number>): number[] {
  let mx = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
  let sum = 0;
  const e = new Array<number>(logits.length);
  for (let i = 0; i < logits.length; i++) {
    e[i] = Math.exp(logits[i] - mx);
    sum += e[i];
  }
  return e.map((v) => v / sum);
}

/** 분류기 로짓 → top-1 라벨. det_cls로 후보 서브셋 제한(1=신호등→light_ids, 그 외 sign_ids). */
export function decodeClsTop1(
  logits: ArrayLike<number>,
  labels: RecognizerLabels,
  detCls: number,
): { label: string; conf: number } | null {
  const probs = softmax(logits);
  let cand = (detCls === 1 ? labels.light_ids : labels.sign_ids).filter((i) => i < probs.length);
  if (cand.length === 0) cand = probs.map((_, i) => i);
  let best = cand[0];
  for (const i of cand) if (probs[i] > probs[best]) best = i;
  const name = labels.names[best];
  if (name == null) return null;
  return { label: name, conf: probs[best] };
}

/** ROI 샘플러 — bbox(원본 좌표) 영역을 size×size RGBA로 잘라 반환(없으면 null). */
export type RoiSampler = (
  bbox: [number, number, number, number],
  size: number,
) => Uint8ClampedArray | null;

// ── ORT 세션 최소 인터페이스 (onnxruntime-web 하드 의존 회피, 동적 import) ────
export interface OrtTensorLike {
  data: ArrayLike<number>;
  dims: readonly number[];
}
export interface OrtSession {
  inputNames: string[];
  outputNames: string[];
  run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensorLike>>;
}
export interface OrtNamespace {
  Tensor: new (type: string, data: Float32Array, dims: number[]) => unknown;
  InferenceSession: {
    create(model: Uint8Array, opts: Record<string, unknown>): Promise<OrtSession>;
  };
  env?: { wasm?: { numThreads?: number } };
}

// ── ClientPipeline — 검출 세션 + ByteTrack 통합 ───────────────────────────────
export class ClientPipeline {
  private session: OrtSession | null = null;
  private ort: OrtNamespace | null = null;
  private inputName = "images";
  private outputName = "output0";
  private tracker = new ByteTracker({ frameRate: 30 });
  private frameId = 0;
  activeEP = "—";
  // 분류기(인식) — 선택적. 미로드면 라벨=클래스명.
  private clsSession: OrtSession | null = null;
  private clsInput = "input";
  private labels: RecognizerLabels | null = null;
  // temporal 안정화: trackId → (label → 누적 conf) (서버 _stable_recognition)
  private trackBuf = new Map<number, Map<string, number>>();

  /** ORT 네임스페이스 + 모델 바이트로 세션 생성. EP 순서대로 워밍업 검증 후 확정. */
  async load(ort: OrtNamespace, modelBytes: Uint8Array, eps: string[]): Promise<void> {
    this.ort = ort;
    if (ort.env?.wasm) ort.env.wasm.numThreads = 1;
    let lastErr: unknown = null;
    for (const ep of eps) {
      try {
        const s = await ort.InferenceSession.create(modelBytes, {
          executionProviders: [ep],
          graphOptimizationLevel: "all",
        });
        // 워밍업 — WebGPU INT8 등 런타임 미지원을 로드 단계에서 감지(스파이크 교훈)
        const warm = new ort.Tensor(
          "float32",
          new Float32Array(3 * DET_INPUT * DET_INPUT),
          [1, 3, DET_INPUT, DET_INPUT],
        );
        await s.run({ [s.inputNames[0]]: warm });
        this.session = s;
        this.inputName = s.inputNames[0];
        this.outputName = s.outputNames[0];
        this.activeEP = ep;
        return;
      } catch (e) {
        lastErr = e;
      }
    }
    throw lastErr ?? new Error("세션 생성 실패");
  }

  /** 분류기 세션 + 라벨 메타 로드(선택적 인식 단계 활성화). */
  async loadClassifier(
    ort: OrtNamespace,
    modelBytes: Uint8Array,
    eps: string[],
    labels: RecognizerLabels,
  ): Promise<void> {
    let lastErr: unknown = null;
    for (const ep of eps) {
      try {
        const s = await ort.InferenceSession.create(modelBytes, {
          executionProviders: [ep],
          graphOptimizationLevel: "all",
        });
        const warm = new ort.Tensor("float32", new Float32Array(3 * 32 * 32), [1, 3, 32, 32]);
        await s.run({ [s.inputNames[0]]: warm });
        this.clsSession = s;
        this.clsInput = s.inputNames[0];
        this.labels = labels;
        return;
      } catch (e) {
        lastErr = e;
      }
    }
    throw lastErr ?? new Error("분류기 세션 생성 실패");
  }

  get loaded(): boolean {
    return this.session !== null;
  }

  get recognizerLoaded(): boolean {
    return this.clsSession !== null && this.labels !== null;
  }

  /** temporal 안정화: 누적 conf 최대 라벨. */
  private stableLabel(trackId: number, label: string, conf: number): string {
    let m = this.trackBuf.get(trackId);
    if (!m) {
      m = new Map();
      this.trackBuf.set(trackId, m);
    }
    m.set(label, (m.get(label) ?? 0) + conf);
    let best = label;
    let bestV = -Infinity;
    for (const [k, v] of m) if (v > bestV) [best, bestV] = [k, v];
    return best;
  }

  /**
   * 640×640 RGBA 프레임 + 원본 크기 → FrameResult(검출+추적[+인식]).
   * @param roiSampler 제공 + 분류기 로드 시 ROI별 한글 라벨 인식(temporal 안정화).
   */
  async processFrame(
    rgba640: Uint8ClampedArray,
    srcW: number,
    srcH: number,
    roiSampler?: RoiSampler,
  ): Promise<FrameResult> {
    if (!this.session || !this.ort) throw new Error("미로드");
    this.frameId += 1;
    const t0 = performance.now();

    const input = imageDataToNCHW(rgba640);
    const tensor = new this.ort.Tensor("float32", input, [1, 3, DET_INPUT, DET_INPUT]);
    const out = await this.session.run({ [this.inputName]: tensor });
    const tDetect = performance.now();

    const o = out[this.outputName];
    let dets = postprocessDetections(o.data, o.dims);
    dets = scaleDetections(dets, srcW, srcH);

    const stracks = this.tracker.update(dets);
    const tTrack = performance.now();

    // ── 인식(선택): 분류기 로드 + roiSampler 제공 시 ROI별 한글 라벨 ──────────
    const tracks: Track[] = [];
    for (const t of stracks) {
      const [x1, y1, x2, y2] = t.tlbr;
      const className = DET_CLASS_NAMES[t.cls] ?? String(t.cls);
      let label = className;
      if (this.recognizerLoaded && roiSampler && this.clsSession && this.labels) {
        const rgba = roiSampler([x1, y1, x2, y2], 32);
        if (rgba) {
          try {
            const ci = preprocessClsRoi(rgba);
            const ct = new this.ort.Tensor("float32", ci, [1, 3, 32, 32]);
            const cout = await this.clsSession.run({ [this.clsInput]: ct });
            const logits = cout[this.clsSession.outputNames[0]];
            const dec = decodeClsTop1(logits.data, this.labels, t.cls);
            if (dec) label = this.stableLabel(t.trackId, dec.label, dec.conf);
          } catch {
            /* 인식 실패 → 클래스명 유지 */
          }
        }
      }
      tracks.push({
        id: t.trackId,
        class: t.cls,
        class_name: className,
        conf: Math.round(t.score * 1000) / 1000,
        label,
        bbox: [Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2)],
      });
    }
    const tRecognize = performance.now();

    return {
      frame_id: this.frameId,
      inference_ms: Math.round((tRecognize - t0) * 10) / 10,
      tracks,
      variant: `ondevice-${this.activeEP}`,
      stage_ms: {
        detect: Math.round((tDetect - t0) * 10) / 10,
        track: Math.round((tTrack - tDetect) * 10) / 10,
        recognize: Math.round((tRecognize - tTrack) * 10) / 10,
      },
    };
  }

  reset(): void {
    this.tracker.reset();
    this.trackBuf.clear();
    this.frameId = 0;
  }
}
