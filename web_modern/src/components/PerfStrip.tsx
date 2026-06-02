/**
 * PerfStrip.tsx — 양자화 A/B 토글 + 파이프라인 단계 레이턴시 플로우
 * (app.js buildVariantToggle / updateVariantDelta / renderStages 포팅)
 *
 * - variants from store (set by App.tsx via getStatus())
 * - variant A/B toggle → store.setSelectedVariant; useStream/useSession read selectedVariant
 * - pipeline-flow: stageMs.detect/track/recognize 비율 바 + bottleneck 강조
 */

import { useStore } from "../store";

function varName(name: string): string {
  if (name.includes("int8") || name.includes("INT8")) return "INT8";
  if (name.includes("fp32") || name.includes("FP32")) return "FP32";
  return name.toUpperCase();
}

export default function PerfStrip() {
  const telemetry = useStore((s) => s.telemetry);
  const variants = useStore((s) => s.variants);
  const selectedVariant = useStore((s) => s.selectedVariant);
  const fpsByVariant = useStore((s) => s.fpsByVariant);
  const setSelectedVariant = useStore((s) => s.setSelectedVariant);
  const pushToast = useStore((s) => s.pushToast);

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
      {/* 양자화 A/B 세그먼트 토글 — variants가 2개 이상일 때만 표시 */}
      {variants.length >= 2 && (
        <div className="perf-block" id="variant-block">
          <span className="perf-tag">검출기 양자화</span>
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
