/*
 * spike.js — 브라우저 온디바이스 YOLO 추론 타당성 스파이크
 *
 * 서버 e2e_pipeline.py 의 _preprocess_yolo / postprocess_yolo 를 JS로 정확히 복제:
 *   - 전처리: 프레임 → 640×640 단순 resize(레터박스 아님), RGB, /255, NCHW float32
 *   - 출력: [1, 4+nc, 8400] (nc=3) → 채널: cx,cy,w,h, cls0..cls2
 *   - conf=max(cls), cls=argmax, conf_thres 필터 → cxcywh→xyxy → 클래스별 NMS(iou 0.45)
 *
 * 목적: FPS / 활성 EP(WebGPU·WASM) / INT8 구동 여부를 실측해 전면 전환 타당성을 판단.
 */

// ORT-Web은 ES 모듈로 동적 import (webgpu 자기완결 ESM 번들 — wasm 인라인).
let ort = null;

const INPUT = 640;
const CONF_THRES = 0.25;
const IOU_THRES = 0.45;
const CLASS_NAMES = ["표지판", "신호등", "간판"];
const CLASS_COLORS = ["#22c55e", "#ef4444", "#f59e0b"];

const $ = (id) => document.getElementById(id);
const videoEl = $("video");
const stillEl = $("still");
const overlay = $("overlay");
const octx = overlay.getContext("2d");

// 추론 소스: 'video'(웹캠·동영상파일) | 'image'(정지 이미지) | 'synth'(합성 프레임)
let sourceKind = "video";
// 합성 프레임용 캔버스 (웹캠/파일 없이 추론 throughput 측정)
const synth = document.createElement("canvas");
synth.width = INPUT;
synth.height = INPUT;

// 전처리용 오프스크린 640×640
const pre = document.createElement("canvas");
pre.width = INPUT;
pre.height = INPUT;
const pctx = pre.getContext("2d", { willReadFrequently: true });

let session = null;
let inputName = null;
let outputName = null;
let activeEP = "—";
let running = false;
let rafId = 0;

// 롤링 평균
const inferHist = [];
const fpsHist = [];
let lastFrameTs = 0;

function log(msg) {
  const el = $("log");
  const t = new Date().toLocaleTimeString();
  el.textContent += `[${t}] ${msg}\n`;
  el.scrollTop = el.scrollHeight;
}

function avg(arr) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

// 치명적 오류를 화면 상단 빨간 배너 + 로그 + 콘솔에 크게 노출
function showFatal(msg) {
  const el = $("fatal");
  el.style.display = "block";
  el.textContent = `⚠ ${msg}`;
  log(`✗ ${msg}`);
  console.error("[spike]", msg);
}

window.addEventListener("error", (e) => showFatal(`스크립트 오류: ${e.message}`));
window.addEventListener("unhandledrejection", (e) =>
  showFatal(`처리 안 된 Promise 거부: ${e.reason}`),
);

// ── ORT-Web 동적 로드 (webgpu ESM 번들) ──────────────────────────────────────
async function initOrt() {
  const url = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.bundle.min.mjs";
  try {
    log("ORT-Web(webgpu ESM) import 중…");
    const mod = await import(/* @vite-ignore */ url);
    ort = mod.default ?? mod; // 네임스페이스: InferenceSession, env, Tensor
    if (ort?.env?.wasm) ort.env.wasm.numThreads = 1; // COOP/COEP 없음 → 단일 스레드
    log(`✓ ORT-Web 로드 완료. WebGPU 지원: ${"gpu" in navigator ? "예" : "아니오 (WASM만)"}`);
    log("① 모델 로드 → 소스(테스트 프레임/파일/웹캠) 순으로 진행하세요.");
    $("load-btn").disabled = false;
  } catch (e) {
    showFatal(`ORT-Web 로드 실패 (CDN 차단·네트워크 확인): ${e}`);
  }
}

// ── 모델 로드 ────────────────────────────────────────────────────────────────
async function loadModel() {
  if (!ort) {
    log("✗ ORT-Web(window.ort) 로드 실패 — CDN 스크립트 차단? 네트워크/광고차단 확인.");
    $("m-ep").innerHTML = `<span class="badge err">ORT 미로드</span>`;
    return;
  }
  const url = $("model-sel").value;
  const epChoice = $("ep-sel").value;
  $("load-btn").disabled = true;
  $("cam-btn").disabled = true;

  // EP 후보 구성
  let providers;
  if (epChoice === "webgpu") providers = ["webgpu"];
  else if (epChoice === "wasm") providers = ["wasm"];
  else providers = ["webgpu", "wasm"]; // auto

  const sizeMB = url.includes("fp32") ? 43 : url.includes("fp16") ? 22 : 18;
  const t0 = performance.now();

  // ── 1단계: 모델 다운로드 (서버 /models 마운트 확인 — 실패 시 명확히 구분) ──
  let modelBytes;
  try {
    log(`모델 다운로드 시작: ${url} (~${sizeMB}MB)`);
    $("m-load").textContent = "다운로드 중…";
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(
        `HTTP ${resp.status} ${resp.statusText} — /models 마운트 없음? **서버를 재시작**했는지 확인하세요.`,
      );
    }
    modelBytes = new Uint8Array(await resp.arrayBuffer());
    log(`✓ 다운로드 완료: ${(modelBytes.byteLength / 1e6).toFixed(1)}MB`);
  } catch (e) {
    $("m-load").textContent = "다운로드 실패";
    $("m-ep").innerHTML = `<span class="badge err">다운로드 실패</span>`;
    log(`✗ 모델 다운로드 실패: ${e}`);
    $("load-btn").disabled = false;
    return;
  }

  // ── 2단계: 세션 생성 (EP를 순서대로 시도해 실제 성공 EP 확정) ──
  try {
    let lastErr = null;
    for (const ep of providers) {
      try {
        log(`InferenceSession 생성 시도 — EP=${ep}`);
        const s = await ort.InferenceSession.create(modelBytes, {
          executionProviders: [ep],
          graphOptimizationLevel: "all",
        });
        // 워밍업 1회 — 세션 생성은 돼도 런타임 커널이 해당 EP에서 미지원일 수 있음.
        // (예: INT8 모델의 int32 DequantizeLinear는 WebGPU 미지원 → run에서 throw)
        // 실제 run으로 검증해야 auto 모드가 webgpu→wasm 로 정확히 폴백한다.
        const inName = s.inputNames[0];
        const warm = new ort.Tensor(
          "float32",
          new Float32Array(3 * INPUT * INPUT),
          [1, 3, INPUT, INPUT],
        );
        await s.run({ [inName]: warm });
        session = s;
        activeEP = ep;
        log(`  ✓ ${ep} 워밍업 성공`);
        break;
      } catch (e) {
        lastErr = e;
        log(`  ✗ ${ep} 실패(생성·워밍업): ${String(e).slice(0, 160)}`);
        session = null;
      }
    }
    if (!session) throw lastErr ?? new Error("세션 생성/워밍업 실패");

    inputName = session.inputNames[0];
    outputName = session.outputNames[0];
    const loadMs = Math.round(performance.now() - t0);
    $("m-load").textContent = `${(loadMs / 1000).toFixed(1)}s`;
    $("m-ep").innerHTML = `<span class="badge ok">${activeEP.toUpperCase()}</span>`;
    log(`✓ 로드 완료 (${loadMs}ms). input=${inputName}, output=${outputName}, EP=${activeEP}`);
    if (url.includes("int8") && activeEP === "webgpu") {
      log("주의: WebGPU에서 INT8 양자화 연산은 지원이 불완전 — 정상 추론/속도 여부를 결과로 확인하세요.");
    }
    $("cam-btn").disabled = false;
    $("file-btn").disabled = false;
    $("synth-btn").disabled = false;
  } catch (e) {
    $("m-load").textContent = "실패";
    $("m-ep").innerHTML = `<span class="badge err">로드 실패</span>`;
    log(`✗ 모델 로드 실패: ${e}`);
    $("load-btn").disabled = false;
    return;
  }
  $("load-btn").disabled = false;
}

// 활성 소스 엘리먼트 + 원본 해상도(레터박스 계산용)
function activeSource() {
  if (sourceKind === "image") return { el: stillEl, w: stillEl.naturalWidth, h: stillEl.naturalHeight };
  if (sourceKind === "synth") return { el: synth, w: INPUT, h: INPUT };
  return { el: videoEl, w: videoEl.videoWidth, h: videoEl.videoHeight };
}

// ── 전처리: 활성 소스 프레임 → ort.Tensor [1,3,640,640] ──────────────────────
function preprocess() {
  pctx.drawImage(activeSource().el, 0, 0, INPUT, INPUT); // 단순 resize (서버와 동일)
  const { data } = pctx.getImageData(0, 0, INPUT, INPUT); // RGBA
  const n = INPUT * INPUT;
  const f = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    f[i] = data[i * 4] / 255; // R 평면
    f[n + i] = data[i * 4 + 1] / 255; // G 평면
    f[2 * n + i] = data[i * 4 + 2] / 255; // B 평면
  }
  return new ort.Tensor("float32", f, [1, 3, INPUT, INPUT]);
}

// ── 후처리: 출력 → [{x1,y1,x2,y2,conf,cls}] (640 좌표계) ──────────────────────
function postprocess(out) {
  const dims = out.dims; // [1, C, A] 또는 [1, A, C]
  const d = out.data;
  let C, A, chFirst;
  if (dims.length === 3) {
    if (dims[1] < dims[2]) {
      C = dims[1];
      A = dims[2];
      chFirst = true;
    } else {
      C = dims[2];
      A = dims[1];
      chFirst = false;
    }
  } else {
    return [];
  }
  const nc = C - 4;
  const at = (a, c) => (chFirst ? d[c * A + a] : d[a * C + c]);

  const cand = [];
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
    if (best <= CONF_THRES) continue;
    const cx = at(a, 0);
    const cy = at(a, 1);
    const w = at(a, 2);
    const h = at(a, 3);
    cand.push({
      x1: cx - w / 2,
      y1: cy - h / 2,
      x2: cx + w / 2,
      y2: cy + h / 2,
      conf: best,
      cls: bestC,
    });
  }
  return nmsPerClass(cand);
}

function iou(a, b) {
  const ix1 = Math.max(a.x1, b.x1),
    iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2),
    iy2 = Math.min(a.y2, b.y2);
  const iw = Math.max(0, ix2 - ix1),
    ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0;
}

function nmsPerClass(boxes) {
  const out = [];
  for (let c = 0; c < CLASS_NAMES.length; c++) {
    const group = boxes.filter((b) => b.cls === c).sort((p, q) => q.conf - p.conf);
    const keep = [];
    for (const b of group) {
      if (keep.every((k) => iou(k, b) <= IOU_THRES)) keep.push(b);
    }
    out.push(...keep);
  }
  return out;
}

// ── 박스 렌더 (640 좌표 → 표시 캔버스) ───────────────────────────────────────
function draw(dets) {
  const cw = overlay.clientWidth;
  const ch = overlay.clientHeight;
  if (overlay.width !== cw || overlay.height !== ch) {
    overlay.width = cw;
    overlay.height = ch;
  }
  octx.clearRect(0, 0, cw, ch);

  // 소스가 object-fit:contain 이므로 표시 영역(레터박스) 계산
  const src = activeSource();
  const vAR = src.w / src.h || 4 / 3;
  const cAR = cw / ch;
  let dW, dH, oX, oY;
  if (vAR > cAR) {
    dW = cw;
    dH = cw / vAR;
    oX = 0;
    oY = (ch - dH) / 2;
  } else {
    dH = ch;
    dW = ch * vAR;
    oY = 0;
    oX = (cw - dW) / 2;
  }
  const sx = dW / INPUT;
  const sy = dH / INPUT;

  octx.lineWidth = 2;
  octx.font = "600 14px 'Fira Code', monospace";
  for (const b of dets) {
    const color = CLASS_COLORS[b.cls] ?? "#a1a1aa";
    const x = oX + b.x1 * sx,
      y = oY + b.y1 * sy;
    const w = (b.x2 - b.x1) * sx,
      h = (b.y2 - b.y1) * sy;
    octx.strokeStyle = color;
    octx.strokeRect(x, y, w, h);
    const label = `${CLASS_NAMES[b.cls]} ${(b.conf * 100).toFixed(0)}%`;
    const tw = octx.measureText(label).width;
    octx.fillStyle = color;
    octx.fillRect(x, y - 20, tw + 12, 20);
    octx.fillStyle = "#0a0a0a";
    octx.textBaseline = "middle";
    octx.fillText(label, x + 6, y - 10);
  }
}

// ── 추론 루프 ────────────────────────────────────────────────────────────────
async function loop() {
  if (!running || !session) return;
  const frameStart = performance.now();
  if (lastFrameTs) {
    const dt = frameStart - lastFrameTs;
    fpsHist.push(1000 / dt);
    if (fpsHist.length > 20) fpsHist.shift();
  }
  lastFrameTs = frameStart;

  try {
    const tPre0 = performance.now();
    const tensor = preprocess();
    const tInfer0 = performance.now();
    const out = await session.run({ [inputName]: tensor });
    const tInfer1 = performance.now();
    const dets = postprocess(out[outputName]);
    const tPost1 = performance.now();

    inferHist.push(tInfer1 - tInfer0);
    if (inferHist.length > 20) inferHist.shift();

    draw(dets);
    $("m-fps").textContent = avg(fpsHist).toFixed(1);
    $("m-infer").textContent = `${avg(inferHist).toFixed(1)} ms`;
    $("m-det").textContent = String(dets.length);
    $("m-pp").textContent = `${(tPre0 === tInfer0 ? 0 : tInfer0 - tPre0 + (tPost1 - tInfer1)).toFixed(1)} ms`;
  } catch (e) {
    log(`✗ 추론 오류: ${e}`);
    running = false;
    return;
  }
  rafId = requestAnimationFrame(loop);
}

// ── 소스 시작 공통 ───────────────────────────────────────────────────────────
function beginLoop() {
  running = true;
  lastFrameTs = 0;
  inferHist.length = 0;
  fpsHist.length = 0;
  $("cam-btn").disabled = true;
  $("file-btn").disabled = true;
  $("synth-btn").disabled = true;
  $("stop-btn").disabled = false;
  loop();
}

function showEl(kind) {
  videoEl.style.display = kind === "image" || kind === "synth" ? "none" : "";
  stillEl.style.display = kind === "image" ? "" : "none";
}

// ── 웹캠 ─────────────────────────────────────────────────────────────────────
async function startCam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
      audio: false,
    });
    sourceKind = "video";
    showEl("video");
    videoEl.srcObject = stream;
    await videoEl.play();
    log(`웹캠 시작: ${videoEl.videoWidth}×${videoEl.videoHeight}`);
    beginLoop();
  } catch (e) {
    log(`✗ 웹캠 접근 실패(웹캠 없음?): ${e}`);
  }
}

// ── 이미지/동영상 파일 ───────────────────────────────────────────────────────
function startFile(file) {
  const url = URL.createObjectURL(file);
  if (file.type.startsWith("image/")) {
    sourceKind = "image";
    showEl("image");
    stillEl.onload = () => {
      log(`이미지 로드: ${stillEl.naturalWidth}×${stillEl.naturalHeight} — 같은 프레임 반복 추론으로 throughput 측정`);
      beginLoop();
    };
    stillEl.src = url;
  } else {
    sourceKind = "video";
    showEl("video");
    videoEl.srcObject = null;
    videoEl.src = url;
    videoEl
      .play()
      .then(() => {
        log(`동영상 로드: ${videoEl.videoWidth}×${videoEl.videoHeight} (루프 재생)`);
        beginLoop();
      })
      .catch((e) => log(`✗ 동영상 재생 실패(코덱 미지원?): ${e.message ?? e}`));
  }
}

// ── 합성 테스트 프레임 (웹캠·파일 없이 FPS 측정) ─────────────────────────────
function startSynth() {
  sourceKind = "synth";
  showEl("synth");
  const c = synth.getContext("2d");
  const g = c.createLinearGradient(0, 0, INPUT, INPUT);
  g.addColorStop(0, "#1e293b");
  g.addColorStop(1, "#475569");
  c.fillStyle = g;
  c.fillRect(0, 0, INPUT, INPUT);
  for (let i = 0; i < 40; i++) {
    c.fillStyle = `hsl(${(i * 37) % 360},60%,55%)`;
    c.fillRect(Math.random() * INPUT, Math.random() * INPUT, 40, 40);
  }
  log("합성 프레임으로 추론 throughput 측정 (검출은 무의미, FPS·ms만 유효)");
  beginLoop();
}

function stop() {
  running = false;
  cancelAnimationFrame(rafId);
  const s = videoEl.srcObject;
  if (s) s.getTracks().forEach((t) => t.stop());
  videoEl.srcObject = null;
  videoEl.removeAttribute("src");
  octx.clearRect(0, 0, overlay.width, overlay.height);
  $("cam-btn").disabled = false;
  $("file-btn").disabled = false;
  $("synth-btn").disabled = false;
  $("stop-btn").disabled = true;
  log("정지됨");
}

$("load-btn").addEventListener("click", loadModel);
$("cam-btn").addEventListener("click", startCam);
$("file-btn").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (e) => {
  if (e.target.files[0]) startFile(e.target.files[0]);
  e.target.value = "";
});
$("synth-btn").addEventListener("click", startSynth);
$("stop-btn").addEventListener("click", stop);

log("BUILD v5 · fp16 추가 (이 줄이 보이면 최신 코드)");
log("페이지 로드됨. ORT-Web 가져오는 중…");
initOrt();
