const CONFIG = {
  inputSize: 224,
  minConfidence: 0.4,
  bufferSize: 15,
  minVotes: 10,
  minGapMs: 1500,
  sentenceResetMs: 5000,
  fpsWindow: 30,
};

const ROI_CONFIG = {
  updateMs: 120,
  padding: 0.3,
  minSize: 120,
};

const DEFAULT_CONFIG = {
  hfRepoId: "",
  hfRevision: "main",
  modelFile: "convnextv2_ksl_int8.onnx",
  labelsFile: "labels.json",
  localModelPath: "./model/convnextv2_ksl_int8.onnx",
  localLabelsPath: "./labels.json",
};

const runtimeConfig = window.EDGE_SIGN_CONFIG || {};
const hubConfig = { ...DEFAULT_CONFIG, ...runtimeConfig };

function buildHubUrl(fileName) {
  if (!hubConfig.hfRepoId) {
    return "";
  }
  return `https://huggingface.co/${hubConfig.hfRepoId}/resolve/${hubConfig.hfRevision}/${fileName}`;
}

const MODEL_DEFAULT_PATH =
  buildHubUrl(hubConfig.modelFile) || hubConfig.localModelPath;
const LABELS_PATH =
  buildHubUrl(hubConfig.labelsFile) || hubConfig.localLabelsPath;
const WAITING_LABEL = "동작 대기 중";

const state = {
  session: null,
  inputName: null,
  outputName: null,
  labels: [],
  isRunning: false,
  inFlight: false,
  predictionBuffer: [],
  sentenceBuffer: [],
  lastAddedWord: null,
  lastAddedTime: performance.now(),
  fpsHistory: [],
  lastFrameTime: performance.now(),
  inputBuffer: null,
  roiEnabled: false,
  useHands: true,
  useFace: true,
  roi: null,
  roiStatus: "Off",
  hands: null,
  faceMesh: null,
  handLandmarks: [],
  faceLandmarks: [],
  mediaPipeReady: false,
  lastRoiUpdate: 0,
  handsBusy: false,
  faceBusy: false,
};

const video = document.getElementById("video");
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const inputCanvas = document.getElementById("inputCanvas");
const inputCtx = inputCanvas.getContext("2d", { willReadFrequently: true });

const providerBadge = document.getElementById("providerBadge");
const modelBadge = document.getElementById("modelBadge");
const messageEl = document.getElementById("message");
const detectedLabelEl = document.getElementById("detectedLabel");
const confidenceLabelEl = document.getElementById("confidenceLabel");
const fpsLabelEl = document.getElementById("fpsLabel");
const roiStatusLabelEl = document.getElementById("roiStatusLabel");
const sentenceLabelEl = document.getElementById("sentenceLabel");

const modelPathInput = document.getElementById("modelPath");
const loadModelBtn = document.getElementById("loadModelBtn");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const roiToggle = document.getElementById("roiToggle");
const handsToggle = document.getElementById("handsToggle");
const faceToggle = document.getElementById("faceToggle");

function setMessage(text, isError = false) {
  messageEl.textContent = text;
  messageEl.style.color = isError ? "#ff9b9b" : "";
}

function updateRoiStatus(text) {
  state.roiStatus = text;
  roiStatusLabelEl.textContent = text;
}

function getBoundsFromLandmarks(landmarksList) {
  if (!landmarksList || landmarksList.length === 0) {
    return null;
  }
  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;

  for (const landmarks of landmarksList) {
    for (const point of landmarks) {
      if (point.x < minX) minX = point.x;
      if (point.y < minY) minY = point.y;
      if (point.x > maxX) maxX = point.x;
      if (point.y > maxY) maxY = point.y;
    }
  }

  if (maxX <= minX || maxY <= minY) {
    return null;
  }

  return { minX, minY, maxX, maxY };
}

function mergeBounds(a, b) {
  if (!a) return b;
  if (!b) return a;
  return {
    minX: Math.min(a.minX, b.minX),
    minY: Math.min(a.minY, b.minY),
    maxX: Math.max(a.maxX, b.maxX),
    maxY: Math.max(a.maxY, b.maxY),
  };
}

function buildRoiFromBounds(bounds, width, height) {
  if (!bounds) return null;

  const centerX = ((bounds.minX + bounds.maxX) / 2) * width;
  const centerY = ((bounds.minY + bounds.maxY) / 2) * height;
  const boxWidth = (bounds.maxX - bounds.minX) * width;
  const boxHeight = (bounds.maxY - bounds.minY) * height;
  const size = Math.max(boxWidth, boxHeight) * (1 + ROI_CONFIG.padding);
  const finalSize = Math.max(size, ROI_CONFIG.minSize);

  let x = centerX - finalSize / 2;
  let y = centerY - finalSize / 2;
  x = Math.max(0, Math.min(x, width - finalSize));
  y = Math.max(0, Math.min(y, height - finalSize));

  return { x, y, size: finalSize };
}

function updateRoiFromLandmarks() {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) {
    return;
  }

  let bounds = null;
  if (state.useHands) {
    bounds = mergeBounds(bounds, getBoundsFromLandmarks(state.handLandmarks));
  }
  if (state.useFace) {
    bounds = mergeBounds(bounds, getBoundsFromLandmarks(state.faceLandmarks));
  }

  const roi = buildRoiFromBounds(bounds, width, height);
  state.roi = roi;
  if (state.roiEnabled) {
    updateRoiStatus(roi ? "MediaPipe" : "Center");
  }
}

async function initMediaPipe() {
  if (state.mediaPipeReady) {
    return;
  }
  if (!window.Hands || !window.FaceMesh) {
    setMessage("MediaPipe scripts not loaded.", true);
    return;
  }

  state.hands = new Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });
  state.hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  state.hands.onResults((results) => {
    state.handsBusy = false;
    state.handLandmarks = results.multiHandLandmarks || [];
    updateRoiFromLandmarks();
  });

  state.faceMesh = new FaceMesh({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
  });
  state.faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  state.faceMesh.onResults((results) => {
    state.faceBusy = false;
    state.faceLandmarks = results.multiFaceLandmarks || [];
    updateRoiFromLandmarks();
  });

  state.mediaPipeReady = true;
}

function maybeUpdateRoi() {
  if (!state.roiEnabled || !state.mediaPipeReady) {
    return;
  }
  if (video.readyState < 2) {
    return;
  }
  const now = performance.now();
  if (now - state.lastRoiUpdate < ROI_CONFIG.updateMs) {
    return;
  }
  state.lastRoiUpdate = now;

  if (state.useHands && state.hands && !state.handsBusy) {
    state.handsBusy = true;
    state.hands.send({ image: video }).catch(() => {
      state.handsBusy = false;
    });
  }

  if (state.useFace && state.faceMesh && !state.faceBusy) {
    state.faceBusy = true;
    state.faceMesh.send({ image: video }).catch(() => {
      state.faceBusy = false;
    });
  }
}

function getCropRegion(width, height) {
  if (state.roiEnabled && state.roi) {
    return {
      sx: state.roi.x,
      sy: state.roi.y,
      size: state.roi.size,
      source: "roi",
    };
  }
  const cropSize = Math.min(width, height);
  return {
    sx: (width - cropSize) / 2,
    sy: (height - cropSize) / 2,
    size: cropSize,
    source: "center",
  };
}

async function loadLabels() {
  try {
    const response = await fetch(LABELS_PATH);
    if (!response.ok) {
      throw new Error("Labels file not found");
    }
    const labels = await response.json();
    if (!Array.isArray(labels) || labels.length === 0) {
      throw new Error("Labels file is empty");
    }
    state.labels = labels;
    setMessage(`Loaded ${labels.length} labels.`);
  } catch (error) {
    state.labels = [];
    setMessage("Labels missing. Using index-based labels.", true);
  }
}

async function loadModel() {
  const modelPath = modelPathInput.value.trim() || MODEL_DEFAULT_PATH;
  try {
    const providers = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
    ort.env.wasm.numThreads = Math.min(
      navigator.hardwareConcurrency || 4,
      4
    );

    setMessage("Loading model...");
    state.session = await ort.InferenceSession.create(modelPath, {
      executionProviders: providers,
    });
    state.inputName = state.session.inputNames[0];
    state.outputName = state.session.outputNames[0];
    providerBadge.textContent = `Provider: ${providers[0]}`;
    modelBadge.textContent = `Model: ${modelPath}`;
    setMessage("Model loaded. Ready.");
  } catch (error) {
    setMessage(`Model load failed: ${error.message}`, true);
    state.session = null;
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setMessage("Camera API not available in this browser.", true);
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    resizeCanvas();
    state.isRunning = true;
    setMessage("Camera started.");
  } catch (error) {
    setMessage(`Camera start failed: ${error.message}`, true);
  }
}

function stopCamera() {
  state.isRunning = false;
  const stream = video.srcObject;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
  video.srcObject = null;
  setMessage("Camera stopped.");
}

function resizeCanvas() {
  const width = video.videoWidth || 960;
  const height = video.videoHeight || 540;
  canvas.width = width;
  canvas.height = height;
}

function drawFrame() {
  if (!state.isRunning || video.readyState < 2) {
    return;
  }
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const region = getCropRegion(canvas.width, canvas.height);
  ctx.strokeStyle =
    region.source === "roi"
      ? "rgba(32, 214, 165, 0.9)"
      : "rgba(255, 255, 255, 0.7)";
  ctx.lineWidth = 2;
  ctx.strokeRect(region.sx, region.sy, region.size, region.size);
}

function updateFps() {
  const now = performance.now();
  const delta = now - state.lastFrameTime;
  state.lastFrameTime = now;
  const fps = 1000 / Math.max(delta, 1);
  state.fpsHistory.push(fps);
  if (state.fpsHistory.length > CONFIG.fpsWindow) {
    state.fpsHistory.shift();
  }
  const avgFps =
    state.fpsHistory.reduce((sum, value) => sum + value, 0) /
    state.fpsHistory.length;
  fpsLabelEl.textContent = avgFps.toFixed(1);
}

function imageDataToTensor(imageData) {
  const { data } = imageData;
  const size = CONFIG.inputSize * CONFIG.inputSize;
  if (!state.inputBuffer || state.inputBuffer.length !== size * 3) {
    state.inputBuffer = new Float32Array(size * 3);
  }
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < size; i += 1) {
    const offset = i * 4;
    const r = data[offset] / 255;
    const g = data[offset + 1] / 255;
    const b = data[offset + 2] / 255;

    state.inputBuffer[i] = (r - mean[0]) / std[0];
    state.inputBuffer[size + i] = (g - mean[1]) / std[1];
    state.inputBuffer[size * 2 + i] = (b - mean[2]) / std[2];
  }

  return new ort.Tensor("float32", state.inputBuffer, [1, 3, 224, 224]);
}

function getTopPrediction(logits) {
  let maxLogit = -Infinity;
  let bestIdx = 0;
  for (let i = 0; i < logits.length; i += 1) {
    if (logits[i] > maxLogit) {
      maxLogit = logits[i];
      bestIdx = i;
    }
  }

  let sum = 0;
  for (let i = 0; i < logits.length; i += 1) {
    sum += Math.exp(logits[i] - maxLogit);
  }

  const prob = Math.exp(logits[bestIdx] - maxLogit) / Math.max(sum, 1e-9);
  const label = state.labels[bestIdx] || `Class ${bestIdx}`;
  return { label, prob };
}

function updateBuffers(label, prob) {
  const now = performance.now();
  const word = prob > CONFIG.minConfidence ? label : WAITING_LABEL;
  state.predictionBuffer.push(word);
  if (state.predictionBuffer.length > CONFIG.bufferSize) {
    state.predictionBuffer.shift();
  }

  const counts = new Map();
  let mostCommon = word;
  let maxCount = 1;
  for (const entry of state.predictionBuffer) {
    const nextCount = (counts.get(entry) || 0) + 1;
    counts.set(entry, nextCount);
    if (nextCount > maxCount) {
      maxCount = nextCount;
      mostCommon = entry;
    }
  }

  if (
    maxCount >= CONFIG.minVotes &&
    mostCommon !== WAITING_LABEL &&
    mostCommon !== state.lastAddedWord &&
    now - state.lastAddedTime > CONFIG.minGapMs
  ) {
    state.sentenceBuffer.push(mostCommon);
    state.lastAddedWord = mostCommon;
    state.lastAddedTime = now;
  }

  if (now - state.lastAddedTime > CONFIG.sentenceResetMs) {
    state.sentenceBuffer = [];
    state.lastAddedWord = null;
  }

  return { mostCommon, maxCount };
}

async function runInference() {
  if (!state.session || !state.isRunning || video.readyState < 2) {
    return;
  }

  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) {
    return;
  }

  const region = getCropRegion(width, height);
  inputCtx.drawImage(
    video,
    region.sx,
    region.sy,
    region.size,
    region.size,
    0,
    0,
    CONFIG.inputSize,
    CONFIG.inputSize
  );

  const imageData = inputCtx.getImageData(
    0,
    0,
    CONFIG.inputSize,
    CONFIG.inputSize
  );
  const inputTensor = imageDataToTensor(imageData);
  const feeds = { [state.inputName]: inputTensor };
  const results = await state.session.run(feeds);
  const output = results[state.outputName];
  const { label, prob } = getTopPrediction(output.data);
  const { mostCommon } = updateBuffers(label, prob);

  detectedLabelEl.textContent = mostCommon;
  confidenceLabelEl.textContent = `${(prob * 100).toFixed(1)}%`;
  sentenceLabelEl.textContent = state.sentenceBuffer.join(" ") || "-";
}

function renderLoop() {
  drawFrame();
  updateFps();
  maybeUpdateRoi();

  if (!state.roiEnabled) {
    updateRoiStatus("Off");
  }

  if (state.isRunning && state.session && !state.inFlight) {
    state.inFlight = true;
    runInference()
      .catch((error) => {
        setMessage(`Inference error: ${error.message}`, true);
      })
      .finally(() => {
        state.inFlight = false;
      });
  }

  requestAnimationFrame(renderLoop);
}

loadModelBtn.addEventListener("click", () => {
  loadModel();
});
startBtn.addEventListener("click", () => {
  startCamera();
});
stopBtn.addEventListener("click", () => {
  stopCamera();
});
modelPathInput.addEventListener("change", () => {
  state.session = null;
});
roiToggle.addEventListener("change", async () => {
  state.roiEnabled = roiToggle.checked;
  if (state.roiEnabled) {
    updateRoiStatus("Loading");
    await initMediaPipe();
    updateRoiFromLandmarks();
  } else {
    state.roi = null;
    updateRoiStatus("Off");
  }
});
handsToggle.addEventListener("change", () => {
  state.useHands = handsToggle.checked;
  updateRoiFromLandmarks();
});
faceToggle.addEventListener("change", () => {
  state.useFace = faceToggle.checked;
  updateRoiFromLandmarks();
});

async function bootstrap() {
  modelPathInput.value = MODEL_DEFAULT_PATH;
  roiToggle.checked = false;
  handsToggle.checked = true;
  faceToggle.checked = true;
  updateRoiStatus("Off");
  await loadLabels();
  requestAnimationFrame(renderLoop);
}

bootstrap();
