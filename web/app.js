const CONFIG = {
  inputSize: 224,
  minConfidence: 0.4,
  bufferSize: 15,
  minVotes: 10,
  minGapMs: 1500,
  sentenceResetMs: 5000,
  fpsWindow: 30,
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
const sentenceLabelEl = document.getElementById("sentenceLabel");

const modelPathInput = document.getElementById("modelPath");
const loadModelBtn = document.getElementById("loadModelBtn");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

function setMessage(text, isError = false) {
  messageEl.textContent = text;
  messageEl.style.color = isError ? "#ff9b9b" : "";
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

  const cropSize = Math.min(canvas.width, canvas.height);
  const x1 = (canvas.width - cropSize) / 2;
  const y1 = (canvas.height - cropSize) / 2;
  ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, cropSize, cropSize);
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

  const cropSize = Math.min(width, height);
  const sx = (width - cropSize) / 2;
  const sy = (height - cropSize) / 2;
  inputCtx.drawImage(
    video,
    sx,
    sy,
    cropSize,
    cropSize,
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

async function bootstrap() {
  modelPathInput.value = MODEL_DEFAULT_PATH;
  await loadLabels();
  requestAnimationFrame(renderLoop);
}

bootstrap();
