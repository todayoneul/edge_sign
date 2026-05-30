/**
 * Edge-Sign v2 — 주행 Q&A 데모 클라이언트
 *
 * 동작:
 *  1. WebSocket /ws/stream 으로 프레임(base64 JPEG) 전송
 *  2. 서버에서 파이프라인 처리 결과(JSON) 수신 → Canvas 오버레이
 *  3. POST /api/qa (SSE) 로 질문 전송 → 스트리밍 답변 표시
 */

// ── 설정 ──────────────────────────────────────────────────────────────────────
const WS_URL    = `ws://${location.host}/ws/stream`;
const QA_URL    = `/api/qa`;
const FRAME_FPS = 10;   // 서버로 전송할 프레임레이트 (처리 속도 기준)

const COLORS = {
  traffic_sign:  '#48bb78',  // 초록
  traffic_light: '#f56565',  // 빨강 (신호등)
  signboard:     '#ed8936',  // 주황
};

// ── 상태 ──────────────────────────────────────────────────────────────────────
const state = {
  ws:           null,
  stream:       null,       // MediaStream (웹캠)
  videoSrc:     null,       // 동영상 파일 URL
  sending:      false,
  lastResult:   null,       // 최신 파이프라인 결과
  sendTimer:    null,
  isPlaying:    false,
  sentW:        0,          // 마지막으로 서버에 보낸 프레임 너비 (bbox 좌표 기준)
  sentH:        0,
  playbackRate: 1.0,        // 사용자가 설정한 재생 속도 (metadata 로드 시 리셋 방지용)
};

// ── DOM 참조 ──────────────────────────────────────────────────────────────────
const videoEl       = document.getElementById('video-el');
const overlayCanvas = document.getElementById('overlay-canvas');
const ctx           = overlayCanvas.getContext('2d');
const noVideoMsg    = document.getElementById('no-video-msg');
const webcamBtn     = document.getElementById('webcam-btn');
const fileBtn       = document.getElementById('file-btn');
const fileInput     = document.getElementById('file-input');
const stopBtn       = document.getElementById('stop-btn');
const dropzone      = document.getElementById('dropzone');
const fpsInfo       = document.getElementById('fps-info');
const frameInfo     = document.getElementById('frame-info');
const timeInfo      = document.getElementById('time-info');
const trackCount    = document.getElementById('track-count');
const wsInfo        = document.getElementById('ws-info');
const statusDot     = document.getElementById('status-dot');
const trackList     = document.getElementById('track-list');
const chatLog       = document.getElementById('chat-log');
const chatInput     = document.getElementById('chat-input');
const sendBtn       = document.getElementById('send-btn');
const speedRange    = document.getElementById('speed-range');
const speedVal      = document.getElementById('speed-val');
const stepBackBtn   = document.getElementById('step-back-btn');
const stepFwdBtn    = document.getElementById('step-fwd-btn');
const videoWrapper  = document.getElementById('video-wrapper');

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connectWS() {
  state.ws = new WebSocket(WS_URL);

  state.ws.onopen = () => {
    statusDot.classList.add('connected');
    wsInfo.textContent = 'WS: 연결됨';
    fpsInfo.textContent = '연결됨 — 영상을 시작하세요';
    startSendLoop();
  };

  state.ws.onclose = () => {
    statusDot.classList.remove('connected');
    wsInfo.textContent = 'WS: 연결 끊김';
    stopSendLoop();
    // 3초 후 재연결
    setTimeout(connectWS, 3000);
  };

  state.ws.onerror = (e) => {
    console.warn('[WS] 오류:', e);
  };

  state.ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'result') {
      handleResult(msg.data);
    } else if (msg.type === 'error') {
      console.warn('[WS] 서버 오류:', msg.message);
    }
  };
}

// ── 프레임 전송 루프 ──────────────────────────────────────────────────────────
function startSendLoop() {
  stopSendLoop();
  const interval = Math.round(1000 / FRAME_FPS);
  state.sendTimer = setInterval(sendFrame, interval);
}

function stopSendLoop() {
  if (state.sendTimer) {
    clearInterval(state.sendTimer);
    state.sendTimer = null;
  }
}

const _capCanvas = document.createElement('canvas');
const _capCtx    = _capCanvas.getContext('2d');

function sendFrame() {
  if (!state.isPlaying) return;
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  if (videoEl.readyState < 2) return;

  const vw = videoEl.videoWidth  || 640;
  const vh = videoEl.videoHeight || 480;
  const targetW = Math.min(vw, 1280);
  _capCanvas.width  = targetW;
  _capCanvas.height = Math.round(targetW * vh / vw);

  _capCtx.drawImage(videoEl, 0, 0, _capCanvas.width, _capCanvas.height);
  const b64 = _capCanvas.toDataURL('image/jpeg', 0.8);

  state.sentW = _capCanvas.width;
  state.sentH = _capCanvas.height;
  state.ws.send(JSON.stringify({ type: 'frame', data: b64 }));
}

// ── 결과 처리 → Canvas 오버레이 + 트랙 목록 ──────────────────────────────────
let _lastFpsTs = performance.now();
let _frameCount = 0;

function handleResult(result) {
  state.lastResult = result;

  // FPS 계산
  _frameCount++;
  const now = performance.now();
  if (now - _lastFpsTs > 1000) {
    const fps = (_frameCount * 1000 / (now - _lastFpsTs)).toFixed(1);
    fpsInfo.textContent = `처리: ${fps} FPS`;
    _frameCount = 0;
    _lastFpsTs = now;
  }

  // 하단 정보 바
  frameInfo.textContent   = `frame: ${result.frame_id}`;
  timeInfo.textContent    = `추론: ${result.inference_ms} ms`;
  trackCount.textContent  = `tracks: ${result.tracks.length}`;

  // Canvas 오버레이 그리기
  drawOverlay(result.tracks);

  // 트랙 목록 업데이트
  updateTrackList(result.tracks);
}

function drawOverlay(tracks) {
  // bbox 좌표 기준 = 서버에 전송한 프레임 크기 (state.sentW × sentH)
  const refW = state.sentW || videoEl.videoWidth  || 1280;
  const refH = state.sentH || videoEl.videoHeight || 720;

  // 캔버스 내부 해상도 = 캔버스가 화면에 차지하는 실제 픽셀(clientWidth/Height).
  // 내부 해상도와 표시 크기를 일치시켜야 박스가 찌그러지지 않는다.
  const cw = overlayCanvas.clientWidth;
  const ch = overlayCanvas.clientHeight;
  if (overlayCanvas.width !== cw || overlayCanvas.height !== ch) {
    overlayCanvas.width  = cw;
    overlayCanvas.height = ch;
  }

  // video는 object-fit:contain → 캔버스 영역 내에서 레터박스 계산
  const videoAR = refW / refH;
  const canvasAR = cw / ch;
  let dispW, dispH, offsetX, offsetY;
  if (videoAR > canvasAR) {
    dispW = cw;  dispH = cw / videoAR;  offsetX = 0;  offsetY = (ch - dispH) / 2;
  } else {
    dispH = ch;  dispW = ch * videoAR;  offsetY = 0;  offsetX = (cw - dispW) / 2;
  }

  const scaleX = dispW / refW;
  const scaleY = dispH / refH;

  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  for (const t of tracks) {
    const [x1, y1, x2, y2] = t.bbox;
    const sx1 = offsetX + x1 * scaleX, sy1 = offsetY + y1 * scaleY;
    const sw  = (x2 - x1) * scaleX, sh = (y2 - y1) * scaleY;

    const color = COLORS[t.class_name] || '#63b3ed';

    // 박스
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.strokeRect(sx1, sy1, sw, sh);

    // 레이블 배경 (한글 지원 폰트)
    const label = t.label || t.class_name;
    const text  = `#${t.id} ${label} ${(t.conf * 100).toFixed(0)}%`;
    ctx.font = "13px 'Malgun Gothic','Apple SD Gothic Neo',sans-serif";
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = color;
    ctx.fillRect(sx1 - 1, sy1 - 18, tw + 6, 18);

    // 레이블 텍스트
    ctx.fillStyle = '#000';
    ctx.fillText(text, sx1 + 2, sy1 - 4);
  }
}

function updateTrackList(tracks) {
  if (tracks.length === 0) {
    trackList.innerHTML = '<div id="no-tracks">인식된 객체 없음</div>';
    return;
  }

  const KIND = { 0: '교통표지판', 1: '신호등', 2: '간판' };
  trackList.innerHTML = tracks.map(t => {
    const label = t.label || t.class_name;
    const clsKr = KIND[t.class] || '객체';
    const confPct = (t.conf * 100).toFixed(0);
    return `
      <div class="track-item ${t.class_name}">
        <div class="track-info">
          <span class="track-id">#${t.id} · ${clsKr}</span>
          <span class="track-label">${label}</span>
        </div>
        <span class="track-conf">${confPct}%</span>
      </div>
    `;
  }).join('');
}

// ── 영상 입력 ─────────────────────────────────────────────────────────────────
webcamBtn.addEventListener('click', async () => {
  stopMedia();
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 }, audio: false,
    });
    videoEl.srcObject = state.stream;
    videoEl.style.display = 'block';
    noVideoMsg.style.display = 'none';
    state.isPlaying = true;
    stopBtn.disabled = false;
    // 시퀀스 리셋
    if (state.ws?.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: 'reset' }));
    }
  } catch (e) {
    alert('웹캠 접근 실패: ' + e.message);
  }
});

fileBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) loadVideoFile(file);
});

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) loadVideoFile(file);
});

function loadVideoFile(file) {
  stopMedia({ keepFileInput: true });
  if (state.videoSrc) URL.revokeObjectURL(state.videoSrc);
  state.videoSrc = URL.createObjectURL(file);
  videoEl.srcObject = null;
  videoEl.src       = state.videoSrc;
  videoEl.style.display = 'block';
  videoEl.playbackRate = state.playbackRate;
  noVideoMsg.style.display = 'none';
  videoWrapper.classList.add('has-controls');
  videoEl.play().catch((e) => console.warn('[video] play() 거부:', e));
  state.isPlaying = true;
  stopBtn.disabled = false;
  fpsInfo.textContent = `재생 중: ${file.name}`;
  // 파일 input 재선택 가능하도록 value 초기화 (같은 파일 다시 올리기 허용)
  fileInput.value = '';
  if (state.ws?.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: 'reset' }));
  }
}

stopBtn.addEventListener('click', () => stopMedia());

function stopMedia(opts = {}) {
  state.isPlaying = false;
  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }
  videoEl.pause();
  videoEl.srcObject = null;
  videoEl.src = '';
  videoEl.style.display = 'none';
  videoWrapper.classList.remove('has-controls');
  noVideoMsg.style.display = 'flex';
  stopBtn.disabled = true;
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  updateTrackList([]);
  if (!opts.keepFileInput) fileInput.value = '';
}

// 비디오 종료 시: 프레임 전송 중단, 새 영상 안내
videoEl.addEventListener('ended', () => {
  state.isPlaying = false;
  fpsInfo.textContent = '재생 완료 — 다른 동영상을 올려 계속 테스트하세요';
});

// 일시정지/재생 토글에 따라 isPlaying 동기화 (네이티브 컨트롤 대응)
videoEl.addEventListener('pause', () => { state.isPlaying = false; });
videoEl.addEventListener('play',  () => {
  if (videoEl.src) state.isPlaying = true;
});
// seek 시 추적기 리셋 (시간 점프하면 ID 연속성 깨짐)
videoEl.addEventListener('seeked', () => {
  if (state.ws?.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: 'reset' }));
  }
});

// 재생 속도 슬라이더 — 원하는 속도를 state에 저장하고 즉시 적용
speedRange.addEventListener('input', () => {
  const v = parseFloat(speedRange.value) || 1.0;
  state.playbackRate = v;
  videoEl.playbackRate = v;
  speedVal.textContent = `${v.toFixed(2)}×`;
});

// 브라우저는 새 소스/메타데이터 로드 시 playbackRate를 1로 리셋 → 저장값 재적용
videoEl.addEventListener('loadedmetadata', () => {
  videoEl.playbackRate = state.playbackRate;
});
videoEl.addEventListener('ratechange', () => {
  // 네이티브 컨트롤 등으로 바뀐 실제 속도를 슬라이더/표시에 동기화
  const v = videoEl.playbackRate;
  if (Math.abs(v - state.playbackRate) > 0.001) {
    state.playbackRate = v;
    speedRange.value = v;
    speedVal.textContent = `${v.toFixed(2)}×`;
  }
});

// 5초 앞/뒤 이동
stepBackBtn.addEventListener('click', () => {
  if (videoEl.src) videoEl.currentTime = Math.max(0, videoEl.currentTime - 5);
});
stepFwdBtn.addEventListener('click', () => {
  if (videoEl.src && isFinite(videoEl.duration)) {
    videoEl.currentTime = Math.min(videoEl.duration, videoEl.currentTime + 5);
  }
});

// ── Q&A ───────────────────────────────────────────────────────────────────────
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});
sendBtn.addEventListener('click', sendQuestion);

function addChatMsg(role, text) {
  // 처음 빈 상태 메시지 제거
  const empty = chatLog.querySelector('.chat-empty');
  if (empty) empty.remove();

  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}

async function sendQuestion() {
  const question = chatInput.value.trim();
  if (!question) return;
  chatInput.value = '';
  sendBtn.disabled = true;

  addChatMsg('user', question);

  // 현재 트랙 상태
  const tracks = state.lastResult?.tracks ?? [];
  if (tracks.length === 0) {
    addChatMsg('assistant', '아직 인식된 객체가 없습니다. 먼저 영상을 재생해주세요.');
    sendBtn.disabled = false;
    return;
  }

  const assistantDiv = addChatMsg('assistant', '');
  assistantDiv.classList.add('typing');
  let fullText = '';

  try {
    const resp = await fetch(QA_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tracks, question }),
    });

    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // 마지막 불완전한 줄 보관

      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const json_str = line.slice(5).trim();
        if (!json_str) continue;
        const data = JSON.parse(json_str);
        if (data.type === 'token') {
          fullText += data.text;
          assistantDiv.textContent = fullText;
          chatLog.scrollTop = chatLog.scrollHeight;
        } else if (data.type === 'done') {
          break;
        }
      }
    }
  } catch (e) {
    fullText = `⚠️ 오류: ${e.message}`;
    assistantDiv.textContent = fullText;
  } finally {
    assistantDiv.classList.remove('typing');
    sendBtn.disabled = false;
  }
}

// ── 초기화 ────────────────────────────────────────────────────────────────────
connectWS();
