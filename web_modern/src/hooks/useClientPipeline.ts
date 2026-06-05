/**
 * useClientPipeline — 브라우저 온디바이스 추론(ORT-Web WebGPU) 훅.
 *
 * ORT-Web을 ESM 동적 import(webgpu 자기완결 번들 — 스파이크에서 검증된 방식),
 * /models/ 에서 검출기 ONNX를 fetch해 ClientPipeline 로드. Viewport가 640 캡처 프레임을
 * processFrame 으로 넘기면 검출+추적 결과를 store(setFrame)로 흘려보낸다(서버 모드와 동일 렌더).
 *
 * 기본 모델 fp32(WebGPU 62FPS 실측). INT8은 WebGPU 불가(스파이크), fp16은 더 느림 →
 * 속도 우선 fp32 기본. EP는 webgpu→wasm 폴백(워밍업으로 런타임 검증).
 */

import { useCallback, useRef, useState } from "react";
import {
  ClientPipeline,
  type OrtNamespace,
  type RecognizerLabels,
  type RoiSampler,
} from "../lib/clientPipeline";
import type { FrameResult } from "../lib/types";

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.bundle.min.mjs";
const DEFAULT_MODEL = "/models/yolov8s_signs_v3_fp32.onnx";
const CLS_MODEL = "/models/korean_sign_net_fp32.onnx"; // 한국 표지판/신호등 14클래스
const LABELS_URL = "/api/labels";

export interface ClientStatus {
  loading: boolean;
  loaded: boolean;
  ep: string;
  error: string | null;
}

export function useClientPipeline() {
  const pipeRef = useRef<ClientPipeline | null>(null);
  const ortRef = useRef<OrtNamespace | null>(null);
  const loadingRef = useRef<Promise<void> | null>(null);
  const [status, setStatus] = useState<ClientStatus>({
    loading: false,
    loaded: false,
    ep: "—",
    error: null,
  });

  /** ORT + 모델을 1회 로드(중복 호출 시 진행 중 Promise 공유). */
  const ensureLoaded = useCallback(
    async (modelUrl: string = DEFAULT_MODEL, eps: string[] = ["webgpu", "wasm"]): Promise<void> => {
      if (pipeRef.current?.loaded) return;
      if (loadingRef.current) return loadingRef.current;

      const task = (async () => {
        setStatus((s) => ({ ...s, loading: true, error: null }));
        try {
          if (!ortRef.current) {
            const mod = (await import(/* @vite-ignore */ ORT_URL)) as unknown as OrtNamespace;
            ortRef.current = mod;
          }
          const resp = await fetch(modelUrl);
          if (!resp.ok) throw new Error(`모델 HTTP ${resp.status} (/models 마운트·서버 확인)`);
          const bytes = new Uint8Array(await resp.arrayBuffer());

          const pipe = new ClientPipeline();
          await pipe.load(ortRef.current, bytes, eps);
          pipeRef.current = pipe;
          setStatus({ loading: false, loaded: true, ep: pipe.activeEP, error: null });

          // 인식기(분류기 + 라벨) — 비치명적. 실패 시 라벨=클래스명으로 동작.
          try {
            const [labelsResp, clsResp] = await Promise.all([
              fetch(LABELS_URL),
              fetch(CLS_MODEL),
            ]);
            if (labelsResp.ok && clsResp.ok) {
              const labels = (await labelsResp.json()) as RecognizerLabels;
              const clsBytes = new Uint8Array(await clsResp.arrayBuffer());
              if (Array.isArray(labels.names) && labels.names.length > 0) {
                await pipe.loadClassifier(ortRef.current, clsBytes, eps, labels);
              }
            }
          } catch {
            /* 인식기 로드 실패 — 검출+추적만으로 계속 */
          }
        } catch (e) {
          setStatus({ loading: false, loaded: false, ep: "—", error: String(e) });
          throw e;
        } finally {
          loadingRef.current = null;
        }
      })();
      loadingRef.current = task;
      return task;
    },
    [],
  );

  /** 640×640 RGBA 프레임 → FrameResult. roiSampler 제공 시 인식(라벨)까지. 미로드면 null. */
  const processFrame = useCallback(
    async (
      rgba640: Uint8ClampedArray,
      srcW: number,
      srcH: number,
      roiSampler?: RoiSampler,
    ): Promise<FrameResult | null> => {
      const pipe = pipeRef.current;
      if (!pipe?.loaded) return null;
      return pipe.processFrame(rgba640, srcW, srcH, roiSampler);
    },
    [],
  );

  const reset = useCallback(() => pipeRef.current?.reset(), []);

  return { ensureLoaded, processFrame, reset, status };
}
