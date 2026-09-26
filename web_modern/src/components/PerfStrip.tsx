/**
 * PerfStrip.tsx — 양자화 A/B 토글 + 파이프라인 단계 레이턴시 플로우
 * (app.js buildVariantToggle / updateVariantDelta / renderStages 포팅)
 *
 * - variants from store (set by App.tsx via getStatus())
 * - variant A/B toggle → store.setSelectedVariant; useStream/useSession read selectedVariant
 * - pipeline-flow: stageMs.detect/track/recognize 비율 바 + bottleneck 강조
 */

import { useStore } from "../store";
import {
  DETECTORS,
  EP_LABEL,
  PRECISION_LABEL,
  modelFile,
  modelName,
  slowCombination,
  type DetectorId,
  type ExecutionProvider,
  type OnDeviceConfig,
  type Precision,
} from "../lib/models";

function varName(name: string): string {
  if (name.includes("int8") || name.includes("INT8")) return "INT8";
  if (name.includes("fp32") || name.includes("FP32")) return "FP32";
  return name.toUpperCase();
}

/** 온디바이스 검출기 선택: 모델 × 정밀도 × 실행 환경. 재생 중이면 Viewport가 즉시 다시 로드. */
function OnDeviceSelector() {
  const ondevice = useStore((s) => s.ondevice);
  const setOndevice = useStore((s) => s.setOndevice);
  const pushToast = useStore((s) => s.pushToast);
  const file = modelFile(ondevice);

  const choose = (change: Partial<OnDeviceConfig>) => {
    const next = { ...ondevice, ...change };
    if (JSON.stringify(next) === JSON.stringify(ondevice)) return;
    setOndevice(change);
    pushToast(`${modelName(next)} · ${EP_LABEL[next.ep]} (${modelFile(next).mb} MB)`, "ok");
  };

  const group = <T extends string>(
    label: string,
    options: readonly T[],
    current: T,
    text: (o: T) => string,
    sub: (o: T) => string,
    pick: (o: T) => void,
  ) => (
    <div className="seg-toggle" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o} type="button" aria-pressed={o === current} onClick={() => pick(o)}>
          {text(o)}
          {sub(o) && <span className="vmb">{sub(o)}</span>}
        </button>
      ))}
    </div>
  );

  return (
    <div className="perf-block" id="ondevice-block">
      <span className="perf-tag">온디바이스 검출기</span>
      <div className="var-row">
        {group<DetectorId>(
          "검출기",
          ["yolo26n", "yolov8s"],
          ondevice.detector,
          (d) => DETECTORS[d].name,
          (d) => DETECTORS[d].params,
          (d) => choose({ detector: d }),
        )}
        {group<Precision>(
          "정밀도",
          ["fp32", "fp16", "int8"],
          ondevice.precision,
          (p) => PRECISION_LABEL[p],
          (p) => `${DETECTORS[ondevice.detector].files[p].mb}MB`,
          (p) => choose({ precision: p }),
        )}
        {group<ExecutionProvider>(
          "실행 환경",
          ["webgpu", "wasm"],
          ondevice.ep,
          (e) => EP_LABEL[e],
          () => "",
          (e) => choose({ ep: e }),
        )}
      </div>
      <span className="var-delta" id="ondevice-info">
        <b>{modelName(ondevice)}</b>
        {` · ${file.mb}MB · test mAP50-95 ${file.map.toFixed(3)} · 인식기 WASM`}
        {slowCombination(ondevice) && (
          <span className="var-warn" title="QuantizeLinear가 WebGPU에서 CPU로 폴백 (논문 4.3절)">
            INT8+WebGPU는 매우 느림
          </span>
        )}
      </span>
    </div>
  );
}

export default function PerfStrip() {
  const telemetry = useStore((s) => s.telemetry);
  const variants = useStore((s) => s.variants);
  const selectedVariant = useStore((s) => s.selectedVariant);
  const fpsByVariant = useStore((s) => s.fpsByVariant);
  const setSelectedVariant = useStore((s) => s.setSelectedVariant);
  const pushToast = useStore((s) => s.pushToast);
  // 서버가 입력을 처리할 때(브라우저가 못 여는 영상·URL·이미지 인제스트)만 서버 A/B 토글
  const serverProcessing = useStore((s) => s.sourceKind === "session");

  const stageMs = telemetry.stageMs;
  const d = stageMs?.detect ?? 0;
  const t = stageMs?.track ?? 0;
  const r = stageMs?.recognize ?? 0;
  const tot = Math.max(d + t + r, 0.001);
  const maxStage = Math.max(d, t, r);
  const bottleneck = d === maxStage ? "detect" : t === maxStage ? "track" : "recog";

  // Variant delta display (app.js updateVariantDelta)
  function renderDelta() {
    if (variants.length < 2 || !selectedVariant) return null;
    const cur = variants.find((v) => v.name === selectedVariant);
    const other = variants.find((v) => v.name !== selectedVariant);
    if (!cur || !other) return null;

    const dMb = cur.mb - other.mb;
    const fa = fpsByVariant[selectedVariant];
    const fb = fpsByVariant[other.name];

    return (
      <span className="var-delta" id="variant-delta">
        <b>{cur.mb}MB</b>
        {" · "}
        {dMb <= 0 ? (
          <span className="good">{dMb.toFixed(1)}MB</span>
        ) : (
          `+${dMb.toFixed(1)}MB`
        )}
        {" vs "}
        {varName(other.name)}
        {fa != null && fb != null && (
          <>
            {" · "}
            {fa - fb >= 0 ? (
              <span className="good">+{(fa - fb).toFixed(1)} FPS</span>
            ) : (
              `${(fa - fb).toFixed(1)} FPS`
            )}
          </>
        )}
      </span>
    );
  }

  const handleVariantClick = (name: string) => {
    if (name === selectedVariant) return;
    setSelectedVariant(name);
    pushToast(`검출기 → ${varName(name)}`, "ok");
  };

  const stages: Array<{ key: "detect" | "track" | "recog"; label: string; ms: number; frac: number }> = [
    { key: "detect", label: "검출", ms: d, frac: d / tot },
    { key: "track",  label: "추적", ms: t, frac: t / tot },
    { key: "recog",  label: "인식", ms: r, frac: r / tot },
  ];

  return (
    <div className="perf-strip" id="perf-strip">
      {!serverProcessing && <OnDeviceSelector />}

      {/* 서버 처리 중 양자화 A/B 세그먼트 토글 — variants가 2개 이상일 때만 표시 */}
      {serverProcessing && variants.length >= 2 && (
        <div className="perf-block" id="variant-block">
          <span className="perf-tag">서버 처리 · 검출기 양자화</span>
          <div className="var-row">
            <div className="seg-toggle" id="variant-toggle" role="group" aria-label="양자화 variant 선택">
              {variants.map((v) => (
                <button
                  key={v.name}
                  type="button"
                  data-variant={v.name}
                  aria-pressed={v.name === selectedVariant}
                  onClick={() => handleVariantClick(v.name)}
                >
                  {varName(v.name)}
                  <span className="vmb">{v.mb}MB</span>
                </button>
              ))}
            </div>
            {renderDelta()}
          </div>
        </div>
      )}

      {/* 파이프라인 단계 레이턴시 플로우 */}
      <div className="perf-block" style={{ flex: 1 }}>
        <span className="perf-tag">파이프라인 단계 · 프레임당 소요</span>
        <div className="pipeline-flow" id="pipeline-flow">
          {stages.map(({ key, label, ms, frac }) => (
            <div
              className={`pstage${maxStage > 0 && key === bottleneck ? " bottleneck" : ""}`}
              data-stage={key}
              key={key}
            >
              <div className="pstage-head">
                <span className="pstage-name">
                  {label}
                  <span className="bn">병목</span>
                </span>
                <span className="pstage-ms" id={`ms-${key}`}>
                  {ms > 0 ? (
                    <>
                      {ms.toFixed(1)}
                      <small>ms</small>
                    </>
                  ) : (
                    "—"
                  )}
                </span>
              </div>
              <div className="pbar">
                <i
                  id={`bar-${key}`}
                  style={{ width: `${Math.round(Math.min(1, frac) * 100)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
