/**
 * Edge-Sign Console — 주행 인지 관제 클라이언트
 *
 * 백엔드 계약(불변):
 *  1. WebSocket /ws/stream 으로 프레임(base64 JPEG) 전송 → 결과 JSON 수신
 *  2. POST /api/qa (SSE) 로 질문 전송 → 토큰 스트리밍 답변
 *
 * 프론트 추가: KPI count-up + FPS 스파크라인, 트랙↔박스 호버 연동,
 *             키보드 단축키, 퀵 질문 칩, 토스트, 세그먼트 탭, 스플래시.
 */

// ── 설정 ──────────────────────────────────────────────────────────────────────
const WS_URL    = `ws://${location.host}/ws/stream`;
const QA_URL    = `/api/qa`;
const FRAME_FPS = 10;            // 서버 전송 프레임레이트
const REDUCED   = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const COLORS = {
  traffic_sign:  '#22c55e',
  traffic_light: '#ef4444',
  signboard:     '#f59e0b',
};
const KIND_KR = { 0: '교통표지판', 1: '신호등', 2: '간판' };

// ── 상태 ──────────────────────────────────────────────────────────────────────
const state = {
  ws: null, stream: null, videoSrc: null,
  isPlaying: false, sendTimer: null,
  lastResult: null,
  sentW: 0, sentH: 0,
  playbackRate: 1.0,
  mode: 'client',           // 'client'(웹캠/호환영상) | 'server'(비호환코덱/URL/이미지)
  sessionWS: null,
  streamBlobUrl: null,
  hoverId: null,            // 호버된 트랙 id (트랙↔박스 연동)
  seenIds: new Set(),       // 누적 검출 (리셋 시 초기화)
  fpsHistory: [],           // 스파크라인용
  displayed: {},            // KPI 현재 표시값 (count-up)
};

// ── DOM ───────────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const videoEl = $('video-el'), overlay = $('overlay-canvas'), ctx = overlay.getContext('2d');
const viewport = $('viewport');
const splash = $('splash'), splashStatus = $('splash-status');
const statusDot = $('status-dot'), statusText = $('status-text'), stageStatus = $('stage-status');
const kpiFps = $('kpi-fps'), kpiLat = $('kpi-lat'), kpiTracks = $('kpi-tracks'), kpiTotal = $('kpi-total');
const spark = $('fps-spark'), sctx = spark.getContext('2d');
const trackList = $('track-list'), trackTally = $('track-tally');
const chatLog = $('chat-log'), chatInput = $('chat-input'), sendBtn = $('send-btn');
const frameInfo = $('frame-info'), timeInfo = $('time-info'), trackCount = $('track-count'), wsInfo = $('ws-info');
const speedRange = $('speed-range'), speedVal = $('speed-val');
const fileInput = $('file-input'), dropzone = $('dropzone');
const toastWrap = $('toast-wrap');
const streamImg = $('stream-img'), urlInput = $('url-input'), urlBtn = $('url-btn');

// ════════════════════════════════════════════════════════════════════════════
//  토스트
// ════════════════════════════════════════════════════════════════════════════
function toast(msg, kind = '') {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.innerHTML = `<span class="tdot"></span><span>${msg}</span>`;
  toastWrap.appendChild(el);
  setTimeout(() => {
    el.classList.add('out');
    el.addEventListener('animationend', () => el.remove(), { once: true });
  }, 4000);
}

// ════════════════════════════════════════════════════════════════════════════
//  KPI count-up + 스파크라인
// ════════════════════════════════════════════════════════════════════════════
function animateVal(el, key, to, { decimals = 0, suffix = '' } = {}) {
  const from = state.displayed[key] ?? to;
  state.displayed[key] = to;
  if (REDUCED || Math.abs(to - from) < (decimals ? 0.05 : 1)) {
    el.textContent = to.toFixed(decimals) + suffix; return;
  }
  const t0 = performance.now(), dur = 360;
  function step(now) {
    const p = Math.min(1, (now - t0) / dur);
    const e = 1 - Math.pow(1 - p, 3);               // ease-out cubic
    el.textContent = (from + (to - from) * e).toFixed(decimals) + suffix;
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function drawSpark() {
  const dpr = window.devicePixelRatio || 1;
  const w = spark.clientWidth || 40, h = spark.clientHeight || 16;
  if (spark.width !== w * dpr) { spark.width = w * dpr; spark.height = h * dpr; }
  sctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  sctx.clearRect(0, 0, w, h);
  const data = state.fpsHistory;
  if (data.length < 2) return;
  const max = Math.max(15, ...data), n = data.length;
  const css = getComputedStyle(document.documentElement);
  const stroke = css.getPropertyValue('--spark').trim() || '#71717a';
  sctx.beginPath();
  data.forEach((v, i) => {
    const x = (i / (n - 1)) * w;
    const y = h - (v / max) * (h - 2) - 1;
    i ? sctx.lineTo(x, y) : sctx.moveTo(x, y);
  });
  sctx.strokeStyle = stroke; sctx.lineWidth = 1.25; sctx.lineJoin = 'round';
  sctx.stroke();
  // 끝점 강조
  const lx = w, ly = h - (data[n - 1] / max) * (h - 2) - 1;
  sctx.beginPath(); sctx.arc(lx - 1, ly, 1.6, 0, Math.PI * 2);
  sctx.fillStyle = css.getPropertyValue('--ink').trim() || '#fafafa'; sctx.fill();
}

// ════════════════════════════════════════════════════════════════════════════
//  WebSocket
// ════════════════════════════════════════════════════════════════════════════
function connectWS() {
  state.ws = new WebSocket(WS_URL);

  state.ws.onopen = () => {
    statusDot.classList.add('connected');
    statusText.textContent = '연결됨';
    wsInfo.textContent = 'WS 연결됨';
    stageStatus.textContent = '연결됨 — 영상을 시작하세요';
    hideSplash();
    toast('파이프라인 서버에 연결되었습니다', 'ok');
    startSendLoop();
  };
  state.ws.onclose = () => {
    statusDot.classList.remove('connected');
    statusText.textContent = '재연결 중';
    wsInfo.textContent = 'WS 끊김';
    stopSendLoop();
    setTimeout(connectWS, 3000);
  };
  state.ws.onerror = () => { hideSplash(); };
  state.ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'result') handleResult(msg.data);
    else if (msg.type === 'error') console.warn('[WS]', msg.message);
  };
}

function startSendLoop() {
  stopSendLoop();
  state.sendTimer = setInterval(sendFrame, Math.round(1000 / FRAME_FPS));
}
function stopSendLoop() {
  if (state.sendTimer) { clearInterval(state.sendTimer); state.sendTimer = null; }
}
function wsSend(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(obj));
}
function resetPipeline() {
  wsSend({ type: 'reset' });
  state.seenIds.clear();
  state.fpsHistory = [];
  animateVal(kpiTotal, 'total', 0);
}

const _cap = document.createElement('canvas');
const _cctx = _cap.getContext('2d');
function sendFrame() {
  if (!state.isPlaying || videoEl.readyState < 2) return;
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  const vw = videoEl.videoWidth || 640, vh = videoEl.videoHeight || 480;
  const tw = Math.min(vw, 1280);
  _cap.width = tw; _cap.height = Math.round(tw * vh / vw);
  _cctx.drawImage(videoEl, 0, 0, _cap.width, _cap.height);
  state.sentW = _cap.width; state.sentH = _cap.height;
  wsSend({ type: 'frame', data: _cap.toDataURL('image/jpeg', 0.8) });
}

// ════════════════════════════════════════════════════════════════════════════
//  결과 처리
// ════════════════════════════════════════════════════════════════════════════
let _fpsTs = performance.now(), _fpsCount = 0;

function handleResult(result) {
  state.lastResult = result;
  const tracks = result.tracks || [];

  // FPS
  _fpsCount++;
  const now = performance.now();
  if (now - _fpsTs > 1000) {
    const fps = _fpsCount * 1000 / (now - _fpsTs);
    animateVal(kpiFps, 'fps', fps, { decimals: 1 });
    stageStatus.textContent = `처리 중 · ${fps.toFixed(1)} FPS`;
    stageStatus.classList.add('live');
    state.fpsHistory.push(fps);
    if (state.fpsHistory.length > 48) state.fpsHistory.shift();
    drawSpark();
    _fpsCount = 0; _fpsTs = now;
  }

  // KPI
  animateVal(kpiLat, 'lat', Number(result.inference_ms) || 0, { decimals: 0 });
  animateVal(kpiTracks, 'tracks', tracks.length);
  tracks.forEach(t => state.seenIds.add(t.id));
  animateVal(kpiTotal, 'total', state.seenIds.size);

  // 푸터
  frameInfo.textContent = `frame ${result.frame_id ?? '—'}`;
  timeInfo.textContent  = `추론 ${result.inference_ms ?? '—'} ms`;
  trackCount.textContent = `tracks ${tracks.length}`;
  trackTally.textContent = tracks.length;

  drawOverlay(tracks);
  updateTrackList(tracks);
}

function roundRect(c, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  if (c.roundRect) { c.beginPath(); c.roundRect(x, y, w, h, r); return; }
  c.beginPath();
  c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r); c.closePath();
}

let _geo = null;   // 박스 화면 좌표 캐시 (호버 히트테스트용)

function drawOverlay(tracks) {
  const refW = state.sentW || videoEl.videoWidth || 1280;
  const refH = state.sentH || videoEl.videoHeight || 720;
  const cw = overlay.clientWidth, ch = overlay.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  if (overlay.width !== cw * dpr || overlay.height !== ch * dpr) {
    overlay.width = cw * dpr; overlay.height = ch * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);

  // object-fit: contain 레터박스 계산
  const vAR = refW / refH, cAR = cw / ch;
  let dW, dH, oX, oY;
  if (vAR > cAR) { dW = cw; dH = cw / vAR; oX = 0; oY = (ch - dH) / 2; }
  else           { dH = ch; dW = ch * vAR; oY = 0; oX = (cw - dW) / 2; }
  const sx = dW / refW, sy = dH / refH;

  _geo = [];
  for (const t of tracks) {
    const [x1, y1, x2, y2] = t.bbox;
    const bx = oX + x1 * sx, by = oY + y1 * sy;
    const bw = (x2 - x1) * sx, bh = (y2 - y1) * sy;
    _geo.push({ id: t.id, x: bx, y: by, w: bw, h: bh });

    const color = COLORS[t.class_name] || '#a1a1aa';
    const hot = state.hoverId === t.id;
    ctx.globalAlpha = (state.hoverId == null || hot) ? 1 : 0.4;

    // 박스
    ctx.strokeStyle = color;
    ctx.lineWidth = hot ? 3 : 2;
    if (hot) { ctx.shadowColor = color; ctx.shadowBlur = 12; } else { ctx.shadowBlur = 0; }
    roundRect(ctx, bx, by, bw, bh, 7); ctx.stroke();
    ctx.shadowBlur = 0;

    // 라벨 pill
    const label = t.label || t.class_name;
    const txt = `#${t.id}  ${label}  ${(t.conf * 100).toFixed(0)}%`;
    ctx.font = "600 12px 'Fira Code','Pretendard',monospace";
    const tw = ctx.measureText(txt).width;
    const padX = 7, ph = 19, ly = Math.max(by - ph - 3, 2);
    roundRect(ctx, bx, ly, tw + padX * 2 + 10, ph, 6);
    ctx.fillStyle = color; ctx.fill();
    // 점
    ctx.fillStyle = 'rgba(0,0,0,0.85)';
    ctx.beginPath(); ctx.arc(bx + padX + 2, ly + ph / 2, 2.5, 0, Math.PI * 2); ctx.fill();
    // 텍스트
    ctx.fillStyle = '#0a0a0a';
    ctx.textBaseline = 'middle';
    ctx.fillText(txt, bx + padX + 10, ly + ph / 2 + 0.5);

    // 신뢰도 바 (박스 하단 내부)
    ctx.globalAlpha = (state.hoverId == null || hot) ? 0.9 : 0.4;
    ctx.fillStyle = color;
    ctx.fillRect(bx, by + bh - 2.5, bw * Math.min(1, t.conf), 2.5);
  }
  ctx.globalAlpha = 1;
}

function redrawOverlay() { if (state.lastResult) drawOverlay(state.lastResult.tracks || []); }

function updateTrackList(tracks) {
  if (!tracks.length) {
    trackList.innerHTML = `<div class="empty" id="no-tracks">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M3 12h4l2 6 4-12 2 6h6"/></svg>
      인식된 객체가 여기 표시됩니다<br/>영상을 시작하세요</div>`;
    return;
  }
  trackList.innerHTML = tracks.map(t => {
    const label = t.label || t.class_name;
    const clsKr = KIND_KR[t.class] || '객체';
    const pct = (t.conf * 100).toFixed(0);
    return `<div class="track-item ${t.class_name}" data-id="${t.id}">
        <span class="track-dot"></span>
        <div class="track-info">
          <span class="track-id">#${t.id} · ${clsKr}</span>
          <span class="track-label">${label}</span>
        </div>
        <div class="track-conf-wrap">
          <span class="track-conf">${pct}%</span>
          <span class="conf-bar"><i style="width:${pct}%"></i></span>
        </div>
      </div>`;
  }).join('');

  // 트랙 행 → 박스 하이라이트
  trackList.querySelectorAll('.track-item').forEach(row => {
    const id = Number(row.dataset.id);
    row.addEventListener('mouseenter', () => { state.hoverId = id; redrawOverlay(); });
    row.addEventListener('mouseleave', () => { state.hoverId = null; redrawOverlay(); });
  });
}

// 박스 → 트랙 행 하이라이트 (뷰포트 위 마우스 이동, 네이티브 컨트롤 막지 않음)
viewport.addEventListener('mousemove', (e) => {
  if (!_geo || !_geo.length) return;
  const r = overlay.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  let hit = null;
  for (const g of _geo) if (mx >= g.x && mx <= g.x + g.w && my >= g.y && my <= g.y + g.h) { hit = g.id; break; }
  if (hit !== state.hoverId) {
    state.hoverId = hit;
    trackList.querySelectorAll('.track-item').forEach(row =>
      row.classList.toggle('hl', Number(row.dataset.id) === hit));
    redrawOverlay();
  }
});
viewport.addEventListener('mouseleave', () => {
  if (state.hoverId != null) {
    state.hoverId = null;
    trackList.querySelectorAll('.track-item').forEach(row => row.classList.remove('hl'));
    redrawOverlay();
  }
});

const ro = new ResizeObserver(() => { redrawOverlay(); drawSpark(); });
ro.observe(viewport);

// ════════════════════════════════════════════════════════════════════════════
//  영상 입력
// ════════════════════════════════════════════════════════════════════════════
async function startWebcam() {
  stopMedia();
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    videoEl.srcObject = state.stream;
    videoEl.removeAttribute('controls');
    viewport.classList.add('playing');
    state.isPlaying = true; $('stop-btn').disabled = false;
    resetPipeline();
    toast('웹캠을 시작했습니다', 'ok');
  } catch (e) { toast('웹캠 접근 실패: ' + e.message, 'err'); }
}

function loadVideoFile(file) {
  if (!file) return;
  // 이미지는 브라우저 디코딩 대신 서버 인제스트(단일 프레임 인식)
  if (file.type && file.type.startsWith('image/')) {
    ingest('image', file, file.name);
    fileInput.value = '';
    return;
  }
  stopMedia({ keepInput: true });
  if (state.videoSrc) URL.revokeObjectURL(state.videoSrc);
  state.videoSrc = URL.createObjectURL(file);
  videoEl.srcObject = null;
  videoEl.src = state.videoSrc;
  videoEl.setAttribute('controls', '');
  videoEl.playbackRate = state.playbackRate;
  viewport.classList.add('playing');

  // 모드 자동 판별: 브라우저가 디코딩 못 하면(비호환 코덱) 서버 인제스트로 폴백
  let fellBack = false;
  const fallback = () => {
    if (fellBack || state.mode === 'server') return;
    fellBack = true;
    videoEl.onerror = null;
    toast('브라우저 비호환 코덱 — 서버 디코딩으로 전환', 'warn');
    ingest('video', file, file.name);
  };
  videoEl.onerror = fallback;                 // MEDIA_ERR_SRC_NOT_SUPPORTED 등
  videoEl.play().then(() => {
    state.mode = 'client';                    // 디코딩 성공 → 클라 캡처
  }).catch(() => fallback());

  state.isPlaying = true; $('stop-btn').disabled = false;
  stageStatus.textContent = `재생 중 · ${file.name}`;
  fileInput.value = '';
  resetPipeline();
}

// ── 서버 인제스트 + 서버 스트림 표시 (모드②) ──────────────────────────────────
async function ingest(kind, fileOrUrl, label) {
  stopServerStream();
  stopMedia({ keepInput: true, keepServer: true });
  const fd = new FormData();
  fd.append('kind', kind);
  if (kind === 'url') fd.append('url', fileOrUrl);
  else fd.append('file', fileOrUrl);
  stageStatus.textContent = '서버 디코딩 준비 중…';
  try {
    const resp = await fetch('/api/ingest', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok) { toast('입력 열기 실패: ' + (data.error || resp.status), 'err'); return; }
    startServerStream(label || kind);
  } catch (e) {
    toast('인제스트 오류: ' + e.message, 'err');
  }
}

function startServerStream(label) {
  state.mode = 'server';
  state.isPlaying = true;
  videoEl.style.display = 'none';
  streamImg.style.display = 'block';
  viewport.classList.add('playing');
  ctx.clearRect(0, 0, overlay.width, overlay.height);  // 서버가 박스를 그려 보내므로 클라 오버레이 비움
  _geo = null; state.hoverId = null;
  $('stop-btn').disabled = false;
  resetPipeline();
  stageStatus.textContent = `서버 스트림 · ${label}`;
  stageStatus.classList.add('live');
  toast(`서버 디코딩 시작: ${label}`, 'ok');

  state.sessionWS = new WebSocket(`ws://${location.host}/ws/session`);
  state.sessionWS.binaryType = 'arraybuffer';
  state.sessionWS.onmessage = (e) => {
    if (typeof e.data === 'string') {
      const msg = JSON.parse(e.data);
      if (msg.type === 'frame') handleServerFrame(msg);
      else if (msg.type === 'ended') { stageStatus.textContent = '재생 완료 — 다른 입력을 시도하세요'; stageStatus.classList.remove('live'); }
      else if (msg.type === 'error') toast('세션 오류: ' + msg.message, 'err');
    } else {
      const blob = new Blob([e.data], { type: 'image/jpeg' });
      if (state.streamBlobUrl) URL.revokeObjectURL(state.streamBlobUrl);
      state.streamBlobUrl = URL.createObjectURL(blob);
      streamImg.src = state.streamBlobUrl;
    }
  };
  state.sessionWS.onclose = () => { if (state.mode === 'server') stageStatus.classList.remove('live'); };
}

function sessionControl(action, value) {
  if (state.sessionWS && state.sessionWS.readyState === WebSocket.OPEN)
    state.sessionWS.send(JSON.stringify({ type: 'control', action, value }));
}

function stopServerStream() {
  if (state.sessionWS) {
    sessionControl('stop');
    try { state.sessionWS.close(); } catch (e) {}
    state.sessionWS = null;
  }
  if (state.streamBlobUrl) { URL.revokeObjectURL(state.streamBlobUrl); state.streamBlobUrl = null; }
  streamImg.style.display = 'none';
  streamImg.removeAttribute('src');
  videoEl.style.display = '';
}

// 서버 프레임 메타(JSON) → KPI/트랙목록/푸터 갱신 (박스는 서버가 JPEG에 이미 그림)
let _sFpsTs = performance.now(), _sFpsCount = 0;
function handleServerFrame(msg) {
  const tracks = msg.tracks || [];
  state.lastResult = { tracks };
  state.serverFrameId = msg.frame_id || 0;
  _sFpsCount++;
  const now = performance.now();
  if (now - _sFpsTs > 1000) {
    const fps = _sFpsCount * 1000 / (now - _sFpsTs);
    animateVal(kpiFps, 'fps', fps, { decimals: 1 });
    stageStatus.textContent = `서버 스트림 · ${fps.toFixed(1)} FPS`;
    state.fpsHistory.push(fps); if (state.fpsHistory.length > 48) state.fpsHistory.shift();
    drawSpark();
    _sFpsCount = 0; _sFpsTs = now;
  }
  animateVal(kpiLat, 'lat', Number(msg.inference_ms) || 0, { decimals: 0 });
  animateVal(kpiTracks, 'tracks', tracks.length);
  tracks.forEach(t => state.seenIds.add(t.id));
  animateVal(kpiTotal, 'total', state.seenIds.size);
  frameInfo.textContent = `frame ${msg.frame_id ?? '—'}`;
  timeInfo.textContent  = `추론 ${msg.inference_ms ?? '—'} ms`;
  trackCount.textContent = `tracks ${tracks.length}`;
  trackTally.textContent = tracks.length;
  updateTrackList(tracks);
}

function stopMedia(opts = {}) {
  state.isPlaying = false;
  if (!opts.keepServer) { stopServerStream(); state.mode = 'client'; }
  if (state.stream) { state.stream.getTracks().forEach(t => t.stop()); state.stream = null; }
  videoEl.pause(); videoEl.srcObject = null; videoEl.src = '';
  viewport.classList.remove('playing');
  $('stop-btn').disabled = true;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  _geo = null; state.hoverId = null; state.lastResult = null;
  updateTrackList([]); trackTally.textContent = '0';
  stageStatus.textContent = '정지됨 — 영상을 시작하세요';
  stageStatus.classList.remove('live');
  if (!opts.keepInput) fileInput.value = '';
}

$('webcam-btn').addEventListener('click', startWebcam);
$('hero-webcam').addEventListener('click', startWebcam);
$('file-btn').addEventListener('click', () => fileInput.click());
$('hero-file').addEventListener('click', () => fileInput.click());
$('stop-btn').addEventListener('click', () => stopMedia());
fileInput.addEventListener('change', () => loadVideoFile(fileInput.files[0]));
dropzone.addEventListener('click', () => fileInput.click());

['dragover'].forEach(ev => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); }));
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); loadVideoFile(e.dataTransfer.files[0]); });
// 뷰포트 전체에도 드롭 허용
viewport.addEventListener('dragover', (e) => e.preventDefault());
viewport.addEventListener('drop', (e) => { e.preventDefault(); loadVideoFile(e.dataTransfer.files[0]); });

videoEl.addEventListener('ended', () => { state.isPlaying = false; stageStatus.textContent = '재생 완료 — 다른 영상을 올려보세요'; stageStatus.classList.remove('live'); });
videoEl.addEventListener('pause', () => { state.isPlaying = false; });
videoEl.addEventListener('play', () => { if (videoEl.src || videoEl.srcObject) state.isPlaying = true; });
videoEl.addEventListener('seeked', resetPipeline);
videoEl.addEventListener('loadedmetadata', () => { videoEl.playbackRate = state.playbackRate; });
videoEl.addEventListener('ratechange', () => {
  const v = videoEl.playbackRate;
  if (Math.abs(v - state.playbackRate) > 0.001) {
    state.playbackRate = v; speedRange.value = v; speedVal.textContent = `${v.toFixed(2)}×`;
  }
});

speedRange.addEventListener('input', () => {
  const v = parseFloat(speedRange.value) || 1;
  state.playbackRate = v; videoEl.playbackRate = v; speedVal.textContent = `${v.toFixed(2)}×`;
  if (state.mode === 'server') sessionControl('speed', v);   // 서버 스트림 속도
});
// 5초 점프 — 클라 모드는 video.currentTime, 서버 모드는 frame seek(≈5초×fps)
$('step-back-btn').addEventListener('click', () => {
  if (state.mode === 'server') sessionControl('seek', Math.max(0, (state.serverFrameId || 0) - 75));
  else if (videoEl.src) videoEl.currentTime = Math.max(0, videoEl.currentTime - 5);
});
$('step-fwd-btn').addEventListener('click', () => {
  if (state.mode === 'server') sessionControl('seek', (state.serverFrameId || 0) + 75);
  else if (videoEl.src && isFinite(videoEl.duration)) videoEl.currentTime = Math.min(videoEl.duration, videoEl.currentTime + 5);
});

// URL 입력 → 서버 인제스트
function submitUrl() { const u = urlInput.value.trim(); if (u) ingest('url', u, u); }
urlBtn.addEventListener('click', submitUrl);
urlInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') submitUrl(); });

// ════════════════════════════════════════════════════════════════════════════
//  세그먼트 탭
// ════════════════════════════════════════════════════════════════════════════
const tabs = [$('tab-tracks'), $('tab-qa')];
const panels = { 'tab-tracks': $('tracks-panel'), 'tab-qa': $('qa-panel') };
tabs.forEach(tab => tab.addEventListener('click', () => {
  tabs.forEach(t => t.setAttribute('aria-selected', String(t === tab)));
  Object.entries(panels).forEach(([id, p]) => p.classList.toggle('active', id === tab.id));
}));
function showTab(id) { $(id).click(); }

// ════════════════════════════════════════════════════════════════════════════
//  Q&A
// ════════════════════════════════════════════════════════════════════════════
function addMsg(role, text) {
  const empty = chatLog.querySelector('.empty');
  if (empty) empty.remove();
  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}

async function sendQuestion(qOverride) {
  const question = (qOverride ?? chatInput.value).trim();
  if (!question) return;
  if (!qOverride) chatInput.value = '';
  chatInput.style.height = 'auto';
  sendBtn.disabled = true;
  addMsg('user', question);

  const tracks = state.lastResult?.tracks ?? [];
  if (!tracks.length) {
    addMsg('assistant', '아직 인식된 객체가 없습니다. 먼저 영상을 재생해 주세요.');
    sendBtn.disabled = false; return;
  }

  const bubble = addMsg('assistant', '');
  bubble.classList.add('cursor');
  let full = '';
  try {
    const resp = await fetch(QA_URL, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tracks, question }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const reader = resp.body.getReader(), dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n'); buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const s = line.slice(5).trim(); if (!s) continue;
        const d = JSON.parse(s);
        if (d.type === 'token') { full += d.text; bubble.textContent = full; chatLog.scrollTop = chatLog.scrollHeight; }
        else if (d.type === 'done') break;
      }
    }
  } catch (e) {
    bubble.textContent = `⚠ 오류: ${e.message}`;
    toast('답변 생성 실패: ' + e.message, 'err');
  } finally {
    bubble.classList.remove('cursor');
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener('click', () => sendQuestion());
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuestion(); }
});
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(120, chatInput.scrollHeight) + 'px';
});
$('quick-chips').querySelectorAll('.chip').forEach(chip =>
  chip.addEventListener('click', () => { showTab('tab-qa'); sendQuestion(chip.dataset.q); }));

// ════════════════════════════════════════════════════════════════════════════
//  테마 · 스플래시 · 단축키 모달
// ════════════════════════════════════════════════════════════════════════════
const THEME_KEY = 'edge-sign-theme';
function toggleTheme() {
  const root = document.documentElement;
  const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  root.setAttribute('data-theme', next);
  localStorage.setItem(THEME_KEY, next);
  redrawOverlay(); drawSpark();
}
$('theme-toggle').addEventListener('click', toggleTheme);

let splashHidden = false;
function hideSplash() {
  if (splashHidden) return;
  splashHidden = true;
  splashStatus.textContent = '준비 완료';
  splash.classList.add('hide');
  setTimeout(() => splash.remove(), 500);
}
// 서버가 늦어도 UI는 노출 (최대 2.5초)
setTimeout(hideSplash, 2500);

const scModal = $('shortcuts');
function toggleShortcuts(open) { scModal.classList.toggle('open', open ?? !scModal.classList.contains('open')); }
$('help-btn').addEventListener('click', () => toggleShortcuts(true));
$('sc-close').addEventListener('click', () => toggleShortcuts(false));
scModal.addEventListener('click', (e) => { if (e.target === scModal) toggleShortcuts(false); });

// ════════════════════════════════════════════════════════════════════════════
//  키보드 단축키
// ════════════════════════════════════════════════════════════════════════════
document.addEventListener('keydown', (e) => {
  const typing = ['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName);
  if (e.key === 'Escape') { toggleShortcuts(false); if (typing) document.activeElement.blur(); return; }
  if (typing) return;

  if (e.key === '?' || (e.shiftKey && e.key === '/')) { e.preventDefault(); toggleShortcuts(); }
  else if (e.key === '/') { e.preventDefault(); showTab('tab-qa'); chatInput.focus(); }
  else if (e.key === ' ') {
    if (videoEl.src || videoEl.srcObject) { e.preventDefault(); videoEl.paused ? videoEl.play() : videoEl.pause(); }
  }
  else if (e.key === 'ArrowLeft' && videoEl.src) { videoEl.currentTime = Math.max(0, videoEl.currentTime - 5); }
  else if (e.key === 'ArrowRight' && videoEl.src && isFinite(videoEl.duration)) { videoEl.currentTime = Math.min(videoEl.duration, videoEl.currentTime + 5); }
  else if (e.key.toLowerCase() === 't') { toggleTheme(); }
});

// ── 초기화 ──────────────────────────────────────────────────────────────────
connectWS();
