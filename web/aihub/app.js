const video = document.getElementById("video");
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const capture = document.getElementById("capture");
const captureCtx = capture.getContext("2d");

const wsBadge = document.getElementById("wsBadge");
const statusBadge = document.getElementById("statusBadge");
const messageEl = document.getElementById("message");
const detectedEl = document.getElementById("detectedLabel");
const confidenceEl = document.getElementById("confidenceLabel");
const stableEl = document.getElementById("stableLabel");
const fpsEl = document.getElementById("fpsLabel");

const wsUrlInput = document.getElementById("wsUrl");
const sendFpsInput = document.getElementById("sendFps");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

let ws = null;
let running = false;
let sendTimer = null;
let stream = null;

function setMessage(text, isError = false) {
  messageEl.textContent = text;
  messageEl.style.color = isError ? "#ff9b9b" : "";
}

function updateBadges() {
  wsBadge.textContent = `WS: ${ws ? wsUrlInput.value : "-"}`;
  statusBadge.textContent = `Status: ${running ? "running" : "idle"}`;
}

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
}

function drawLoop() {
  if (!running) {
    return;
  }
  if (video.readyState >= 2) {
    const ratio = video.videoWidth / video.videoHeight;
    const width = canvas.width;
    const height = Math.round(width / ratio);
    canvas.height = height;
    ctx.drawImage(video, 0, 0, width, height);
  }
  requestAnimationFrame(drawLoop);
}

function sendFrame() {
  if (!running || !ws || ws.readyState !== WebSocket.OPEN) {
    return;
  }
  if (video.readyState < 2) {
    return;
  }
  const targetWidth = 640;
  const ratio = video.videoWidth / video.videoHeight;
  const targetHeight = Math.round(targetWidth / ratio);
  capture.width = targetWidth;
  capture.height = targetHeight;
  captureCtx.drawImage(video, 0, 0, targetWidth, targetHeight);
  capture.toBlob(
    (blob) => {
      if (blob && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(blob);
      }
    },
    "image/jpeg",
    0.7
  );
}

function startStreaming() {
  const wsUrl = wsUrlInput.value.trim();
  if (!wsUrl) {
    setMessage("WebSocket URL is required.", true);
    return;
  }

  ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    running = true;
    updateBadges();
    setMessage("WebSocket connected.");
    const fps = Math.max(1, Math.min(15, Number(sendFpsInput.value) || 5));
    const interval = Math.round(1000 / fps);
    sendTimer = setInterval(sendFrame, interval);
    requestAnimationFrame(drawLoop);
  };

  ws.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      detectedEl.textContent = payload.label || "-";
      confidenceEl.textContent = payload.confidence
        ? payload.confidence.toFixed(3)
        : "-";
      stableEl.textContent = payload.stable || "-";
      fpsEl.textContent = payload.fps ? payload.fps.toFixed(1) : "-";
    } catch (err) {
      setMessage("Failed to parse server response.", true);
    }
  };

  ws.onerror = () => {
    setMessage("WebSocket error.", true);
  };

  ws.onclose = () => {
    stopStreaming();
  };
}

function stopStreaming() {
  running = false;
  updateBadges();
  if (sendTimer) {
    clearInterval(sendTimer);
    sendTimer = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
  stopCamera();
  setMessage("Stopped.");
}

startBtn.addEventListener("click", async () => {
  try {
    await startCamera();
    startStreaming();
  } catch (err) {
    setMessage("Camera permission denied.", true);
  }
});

stopBtn.addEventListener("click", () => {
  stopStreaming();
});

updateBadges();
