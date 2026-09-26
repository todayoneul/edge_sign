/**
 * useClientPipeline — 브라우저 온디바이스 추론(ORT-Web) 훅.
 *
 * ORT-Web을 ESM 동적 import(webgpu 자기완결 번들 — 스파이크에서 검증된 방식),
 * /models/ 에서 검출기 ONNX를 fetch해 ClientPipeline 로드. Viewport가 640 캡처 프레임을
 * processFrame 으로 넘기면 검출+추적 결과를 store(setFrame)로 흘려보낸다(서버 모드와 동일 렌더).
 *
 * 검출기는 화면에서 고른 모델 × 정밀도 × 실행 환경(lib/models.ts)으로 로드한다.
 * 인식기(KoreanSignNet, 2.9만 파라미터)는 항상 WASM: WebGPU 디스패치 비용이 연산보다 커서
 * WASM 0.25 ms vs WebGPU 3.5 ms (논문 RUNTIME_MATRIX 2.2절).
 * ORT-Web 1.30: 1.22는 FP16을 CPU로 폴백해 3–4배 느리고 INT32 bias INT8을 WebGPU에서 실행하지 못함.
 */

import { useCallback, useRef, useState } from "react";
import {
  ClientPipeline,
  type OrtNamespace,
  type RecognizerLabels,
  type RoiSampler,
} from "../lib/clientPipeline";
import { DETECTORS, executionProviders, modelFile, modelUrl, type OnDeviceConfig } from "../lib/models";
import type { FrameResult } from "../lib/types";

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.bundle.min.mjs";
const CLS_MODEL = "/models/korean_sign_net_fp32.onnx"; // 한국 표지판/신호등 14클래스
const CLS_EPS = ["wasm"];
const LABELS_URL = "/api/labels";

export interface ClientStatus {
  loading: boolean;
  loaded: boolean;
  ep: string;
  error: string | null;
}

const configKey = (cfg: OnDeviceConfig) => `${modelUrl(cfg)}@${executionProviders(cfg).join(">")}`;

export function useClientPipeline() {
  const pipeRef = useRef<ClientPipeline | null>(null);
  const ortRef = useRef<OrtNamespace | null>(null);
  const loadingRef = useRef<Promise<void> | null>(null);
  const loadedKeyRef = useRef<string | null>(null);
  // ORT-Web's WebGPU build (JSEP/asyncify) is not re-entrant: a create/run/release that starts
  // while another call is suspended on the GPU can corrupt the shared WASM heap
  // ("memory access out of bounds"). Every ORT call goes through this queue, one at a time.
  const queueRef = useRef<Promise<unknown>>(Promise.resolve());
  const serial = useCallback(<T,>(fn: () => Promise<T>): Promise<T> => {
    const run = queueRef.current.then(fn, fn);
    queueRef.current = run.catch(() => {});
    return run;
  }, []);
  const [status, setStatus] = useState<ClientStatus>({
    loading: false,
    loaded: false,
    ep: "—",
    error: null,
  });

  /**
   * 선택한 구성의 검출기(+인식기)를 로드하고 실제 실행 환경(WebGPU 미지원이면 wasm)을 돌려준다.
   * 같은 구성이면 캐시, 다른 구성이 로드 중이면 끝난 뒤 다시 로드.
   */
  const ensureLoaded = useCallback(async (cfg: OnDeviceConfig): Promise<string> => {
    const key = configKey(cfg);
    while (loadingRef.current) await loadingRef.current.catch(() => {});
    if (pipeRef.current?.loaded && loadedKeyRef.current === key) return pipeRef.current.activeEP;

    const task = (async () => {
      setStatus((s) => ({ ...s, loading: true, error: null }));
      try {
        if (!ortRef.current) {
          const mod = (await import(/* @vite-ignore */ ORT_URL)) as { default?: OrtNamespace } & OrtNamespace;
          ortRef.current = mod.default ?? mod;
        }
        const ort = ortRef.current;
        const resp = await fetch(modelUrl(cfg));
        if (!resp.ok) throw new Error(`모델 HTTP ${resp.status} (/models 마운트·서버 확인)`);
        const bytes = new Uint8Array(await resp.arrayBuffer());
        // 인식기(분류기 + 라벨) — 비치명적. 실패 시 라벨=클래스명으로 동작.
        let recognizer: { labels: RecognizerLabels; bytes: Uint8Array } | null = null;
        try {
          const [labelsResp, clsResp] = await Promise.all([fetch(LABELS_URL), fetch(CLS_MODEL)]);
          if (labelsResp.ok && clsResp.ok) {
            const labels = (await labelsResp.json()) as RecognizerLabels;
            if (Array.isArray(labels.names) && labels.names.length > 0)
              recognizer = { labels, bytes: new Uint8Array(await clsResp.arrayBuffer()) };
          }
        } catch {
          /* 검출+추적만으로 계속 */
        }

        const pipe = new ClientPipeline();
        await serial(async () => {
          await pipe.load(ort, bytes, executionProviders(cfg), DETECTORS[cfg.detector].format, modelFile(cfg).inputDtype);
          if (recognizer) await pipe.loadClassifier(ort, recognizer.bytes, CLS_EPS, recognizer.labels).catch(() => {});
          const previous = pipeRef.current;
          pipeRef.current = pipe;
          loadedKeyRef.current = key;
          await previous?.release(); // 이전 모델 세션 해제 (누적되면 내장 GPU에서 멈춤)
        });
        setStatus({ loading: false, loaded: true, ep: pipe.activeEP, error: null });
      } catch (e) {
        setStatus({ loading: false, loaded: false, ep: "—", error: String(e) });
        throw e;
      } finally {
        loadingRef.current = null;
      }
    })();
    loadingRef.current = task;
    await task;
    return pipeRef.current?.activeEP ?? "—";
  }, [serial]);

  /** 640×640 RGBA 프레임 → FrameResult. roiSampler 제공 시 인식(라벨)까지. 미로드면 null. */
  const processFrame = useCallback(
    async (
      rgba640: Uint8ClampedArray,
      srcW: number,
      srcH: number,
      roiSampler?: RoiSampler,
    ): Promise<FrameResult | null> => {
      return serial(async () => {
        const pipe = pipeRef.current;
        if (!pipe?.loaded) return null;
        return pipe.processFrame(rgba640, srcW, srcH, roiSampler);
      });
    },
    [serial],
  );

  const reset = useCallback(() => pipeRef.current?.reset(), []);

  return { ensureLoaded, processFrame, reset, status };
}
