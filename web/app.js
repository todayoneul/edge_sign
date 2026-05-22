// Global App State
const state = {
    selectedModel: "mediapipe", // "mediapipe" or "landmark"
    ws: null,
    camera: null,
    holistic: null,
    isStreaming: false,
    sentenceBuffer: [],
    sentTimestamps: [],
    lastFrameTime: performance.now(),
    fpsHistory: [],
    loadingMP: false
};

// UI Elements
const videoEl = document.getElementById("inputVideo");
const canvasEl = document.getElementById("outputCanvas");
const canvasCtx = canvasEl.getContext("2d");
const loadingOverlay = document.getElementById("loadingOverlay");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const tabMediapipe = document.getElementById("tabMediapipe");
const tabLandmark = document.getElementById("tabLandmark");

const wsStatusBadge = document.getElementById("wsStatusBadge");
const wsStatusText = document.getElementById("wsStatusText");

const detectedLabel = document.getElementById("detectedLabel");
const confidenceLabel = document.getElementById("confidenceLabel");
const confidenceBar = document.getElementById("confidenceBar");
const stableResult = document.getElementById("stableResult");
const sentenceBufferEl = document.getElementById("sentenceBuffer");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");

const fpsLabel = document.getElementById("fpsLabel");
const latencyLabel = document.getElementById("latencyLabel");
const quantizationLabel = document.getElementById("quantizationLabel");

// Settings Elements
const settingsToggle = document.getElementById("settingsToggle");
const settingsBody = document.getElementById("settingsBody");
const settingsCard = document.querySelector(".settings-card");
const wsUrlInput = document.getElementById("wsUrlInput");
const inferIntervalInput = document.getElementById("inferIntervalInput");
const windowSizeInput = document.getElementById("windowSizeInput");
const voteSizeInput = document.getElementById("voteSizeInput");
const minVotesInput = document.getElementById("minVotesInput");
const minConfInput = document.getElementById("minConfInput");
const minGapInput = document.getElementById("minGapInput");

// MediaPipe 33 keypoints mapping to OpenPose 25 keypoints
const POSE_MAPPING = [
    0, // 0: Nose
    [11, 12], // 1: Neck (midpoint of left/right shoulders)
    12, // 2: R-Sho
    14, // 3: R-Elb
    16, // 4: R-Wr
    11, // 5: L-Sho
    13, // 6: L-Elb
    15, // 7: L-Wr
    [23, 24], // 8: MidHip (midpoint of left/right hips)
    24, // 9: R-Hip
    26, // 10: R-Knee
    28, // 11: R-Ank
    23, // 12: L-Hip
    25, // 13: L-Knee
    27, // 14: L-Ank
    5, // 15: R-Eye
    2, // 16: L-Eye
    8, // 17: R-Ear
    7, // 18: L-Ear
    31, // 19: L-BigToe
    29, // 20: L-SmallToe
    31, // 21: L-Heel
    32, // 22: R-BigToe
    30, // 23: R-SmallToe
    32, // 24: R-Heel
];

// OpenPose Layout Dims:
// pose_2d(75) + face_2d(210) + left_hand_2d(63) + right_hand_2d(63) = 411
// pose_3d(100) + face_3d(280) + left_hand_3d(84) + right_hand_3d(84) = 548
// Total = 959 dimensions

function extractFeatures(results) {
    const features = new Float32Array(959);
    let offset = 0;

    // Helper to add 2D Point (x, y, visibility)
    function add2DPoint(point) {
        if (point) {
            features[offset++] = point.x;
            features[offset++] = point.y;
            features[offset++] = point.visibility !== undefined ? point.visibility : 1.0;
        } else {
            offset += 3;
        }
    }

    // Helper to add 3D Point (x, y, z, visibility)
    function add3DPoint(point) {
        if (point) {
            features[offset++] = point.x;
            features[offset++] = point.y;
            features[offset++] = point.z || 0.0;
            features[offset++] = point.visibility !== undefined ? point.visibility : 1.0;
        } else {
            offset += 4;
        }
    }

    // --- 2D Coordinates (411 dimensions) ---
    // 1. Pose 2D (25 * 3 = 75)
    if (results.poseLandmarks) {
        for (let i = 0; i < 25; i++) {
            const mapIdx = POSE_MAPPING[i];
            if (Array.isArray(mapIdx)) {
                const p1 = results.poseLandmarks[mapIdx[0]];
                const p2 = results.poseLandmarks[mapIdx[1]];
                if (p1 && p2) {
                    add2DPoint({ 
                        x: (p1.x + p2.x) / 2, 
                        y: (p1.y + p2.y) / 2, 
                        visibility: Math.min(p1.visibility, p2.visibility) 
                    });
                } else {
                    add2DPoint(null);
                }
            } else {
                add2DPoint(results.poseLandmarks[mapIdx]);
            }
        }
    } else { 
        offset += 75; 
    }

    // 2. Face 2D (70 * 3 = 210)
    // MediaPipe provides 468/478 points, but the model maps the first 70 points
    if (results.faceLandmarks) {
        for (let i = 0; i < 70; i++) { 
            add2DPoint(results.faceLandmarks[i]); 
        }
    } else { 
        offset += 210; 
    }

    // 3. Left Hand 2D (21 * 3 = 63)
    if (results.leftHandLandmarks) {
        for (let i = 0; i < 21; i++) { 
            add2DPoint(results.leftHandLandmarks[i]); 
        }
    } else { 
        offset += 63; 
    }

    // 4. Right Hand 2D (21 * 3 = 63)
    if (results.rightHandLandmarks) {
        for (let i = 0; i < 21; i++) { 
            add2DPoint(results.rightHandLandmarks[i]); 
        }
    } else { 
        offset += 63; 
    }

    // --- 3D Coordinates (548 dimensions) ---
    // 5. Pose 3D (25 * 4 = 100)
    if (results.poseWorldLandmarks) {
        for (let i = 0; i < 25; i++) {
            const mapIdx = POSE_MAPPING[i];
            if (Array.isArray(mapIdx)) {
                const p1 = results.poseWorldLandmarks[mapIdx[0]];
                const p2 = results.poseWorldLandmarks[mapIdx[1]];
                if (p1 && p2) {
                    add3DPoint({ 
                        x: (p1.x + p2.x) / 2, 
                        y: (p1.y + p2.y) / 2, 
                        z: (p1.z + p2.z) / 2, 
                        visibility: Math.min(p1.visibility, p2.visibility) 
                    });
                } else {
                    add3DPoint(null);
                }
            } else {
                add3DPoint(results.poseWorldLandmarks[mapIdx]);
            }
        }
    } else { 
        offset += 100; 
    }

    // 6. Face 3D (70 * 4 = 280)
    if (results.faceLandmarks) {
        for (let i = 0; i < 70; i++) { 
            add3DPoint(results.faceLandmarks[i]); 
        }
    } else { 
        offset += 280; 
    }

    // 7. Left Hand 3D (21 * 4 = 84)
    if (results.leftHandLandmarks) {
        for (let i = 0; i < 21; i++) { 
            add3DPoint(results.leftHandLandmarks[i]); 
        }
    } else { 
        offset += 84; 
    }

    // 8. Right Hand 3D (21 * 4 = 84)
    if (results.rightHandLandmarks) {
        for (let i = 0; i < 21; i++) { 
            add3DPoint(results.rightHandLandmarks[i]); 
        }
    } else { 
        offset += 84; 
    }

    return features;
}

// Draw skeleton and connections overlay on canvas
function onResults(results) {
    if (!state.isStreaming) return;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    
    // Draw raw video frame (mirrored automatically by canvas layout scaleX)
    canvasCtx.drawImage(results.image, 0, 0, canvasEl.width, canvasEl.height);
    
    // Draw MediaPipe landmark overlays
    if (results.poseLandmarks) {
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { color: '#a855f7', lineWidth: 2 });
        drawLandmarks(canvasCtx, results.poseLandmarks, { color: '#3b82f6', lineWidth: 1, radius: 2 });
    }
    if (results.leftHandLandmarks) {
        drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: '#06b6d4', lineWidth: 2 });
        drawLandmarks(canvasCtx, results.leftHandLandmarks, { color: '#10b981', lineWidth: 1, radius: 2 });
    }
    if (results.rightHandLandmarks) {
        drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: '#06b6d4', lineWidth: 2 });
        drawLandmarks(canvasCtx, results.rightHandLandmarks, { color: '#a855f7', lineWidth: 1, radius: 2 });
    }
    canvasCtx.restore();

    // Stream 959-dimensional Float32 landmark features via WebSocket
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        const featureArray = extractFeatures(results);
        state.sentTimestamps.push(performance.now());
        if (state.sentTimestamps.length > 100) {
            state.sentTimestamps.shift();
        }
        state.ws.send(featureArray.buffer); // binary send
    }
}

// Initialize and Start camera & connection
async function startStreaming() {
    startBtn.disabled = true;
    tabMediapipe.disabled = true;
    tabLandmark.disabled = true;
    loadingOverlay.classList.remove("hidden");

    const wsUrl = wsUrlInput.value.trim();
    if (!wsUrl) {
        alert("WebSocket URL을 정확히 입력해주세요.");
        resetUI();
        return;
    }

    // Build URL with query params for configuration
    const queryParams = new URLSearchParams({
        window_size: windowSizeInput.value,
        vote_size: voteSizeInput.value,
        min_votes: minVotesInput.value,
        min_conf: minConfInput.value,
        min_gap: minGapInput.value,
        infer_interval: inferIntervalInput.value
    });

    const fullWsUrl = `${wsUrl}?${queryParams.toString()}`;
    console.log(`Connecting to: ${fullWsUrl}`);

    // Establish WebSocket Connection
    try {
        state.ws = new WebSocket(fullWsUrl);
        state.ws.binaryType = "arraybuffer";
    } catch (e) {
        alert(`WebSocket 연결에 실패했습니다: ${e.message}`);
        resetUI();
        return;
    }

    state.ws.onopen = () => {
        wsStatusBadge.className = "status-badge connected";
        wsStatusText.textContent = "연결됨";
        state.isStreaming = true;
        stopBtn.disabled = false;
        stableResult.textContent = "카메라 준비 중...";
    };

    state.ws.onmessage = (event) => {
        const receiveTime = performance.now();
        const sendTime = state.sentTimestamps.shift();
        if (sendTime) {
            const latency = receiveTime - sendTime;
            latencyLabel.textContent = Math.round(latency);
        }

        try {
            const payload = JSON.parse(event.data);
            detectedLabel.textContent = payload.label || "-";
            
            const conf = payload.confidence || 0.0;
            confidenceLabel.textContent = `${Math.round(conf * 100)}%`;
            confidenceBar.style.width = `${Math.round(conf * 100)}%`;
            
            fpsLabel.textContent = payload.fps ? payload.fps.toFixed(1) : "0.0";

            if (payload.quantized !== undefined) {
                const qType = payload.quantized ? "W8A8" : "FP32";
                const suffix = state.selectedModel === "mediapipe" ? " (MeanStd)" : " (Raw)";
                quantizationLabel.textContent = qType + suffix;
            }

            if (payload.stable && payload.stable !== "-" && payload.stable !== "") {
                stableResult.textContent = payload.stable;
                stableResult.style.color = "var(--success-color)";
                stableResult.style.transform = "scale(1.15)";
                setTimeout(() => { stableResult.style.transform = "scale(1)"; }, 150);

                // Add to sentence buffer
                state.sentenceBuffer.push(payload.stable);
                updateSentenceBufferUI();
            }
        } catch (e) {
            console.error("JSON 파싱 오류:", e);
        }
    };

    state.ws.onerror = (e) => {
        console.error("WebSocket 오류 발생:", e);
    };

    state.ws.onclose = () => {
        console.log("WebSocket 연결 닫힘.");
        stopStreaming();
    };

    // Load MediaPipe Holistic model
    if (!state.holistic) {
        state.loadingMP = true;
        state.holistic = new Holistic({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
        });

        state.holistic.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            refineFaceLandmarks: false,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        state.holistic.onResults(onResults);
        state.loadingMP = false;
    }

    // Initialize Camera Utilities
    if (!state.camera) {
        state.camera = new Camera(videoEl, {
            onFrame: async () => {
                if (state.isStreaming && state.holistic) {
                    await state.holistic.send({ image: videoEl });
                }
            },
            width: 640,
            height: 480
        });
        
        try {
            await state.camera.start();
            loadingOverlay.classList.add("hidden");
            stableResult.textContent = "동작을 시작하세요!";
        } catch (err) {
            alert(`카메라 스트림 시작 실패: ${err.message}`);
            stopStreaming();
        }
    } else {
        loadingOverlay.classList.add("hidden");
        stableResult.textContent = "동작을 시작하세요!";
    }
}

function stopStreaming() {
    state.isStreaming = false;
    
    if (state.ws) {
        state.ws.close();
        state.ws = null;
    }
    if (state.camera) {
        state.camera.stop();
        state.camera = null;
    }

    resetUI();
}

function resetUI() {
    startBtn.disabled = false;
    stopBtn.disabled = true;
    tabMediapipe.disabled = false;
    tabLandmark.disabled = false;
    loadingOverlay.classList.add("hidden");

    wsStatusBadge.className = "status-badge disconnected";
    wsStatusText.textContent = "연결 끊김";
    
    detectedLabel.textContent = "-";
    confidenceLabel.textContent = "0%";
    confidenceBar.style.width = "0%";
    stableResult.textContent = "대기 중...";
    fpsLabel.textContent = "0.0";
    latencyLabel.textContent = "0";

    canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);
}

function updateSentenceBufferUI() {
    if (state.sentenceBuffer.length === 0) {
        sentenceBufferEl.textContent = "동작을 시작하면 여기에 수어 단어가 누적되어 문장으로 표현됩니다.";
        sentenceBufferEl.style.color = "var(--text-muted)";
    } else {
        sentenceBufferEl.textContent = state.sentenceBuffer.join(" ");
        sentenceBufferEl.style.color = "#fff";
        // Scroll to bottom
        sentenceBufferEl.scrollTop = sentenceBufferEl.scrollHeight;
    }
}

// Tab Switching Handler
function selectModel(modelName) {
    if (state.isStreaming) {
        alert("먼저 인식을 정지한 후 모델을 변경해주세요.");
        return;
    }

    state.selectedModel = modelName;

    if (modelName === "mediapipe") {
        tabMediapipe.classList.add("active");
        tabLandmark.classList.remove("active");
        wsUrlInput.value = "ws://localhost:8000/ws/mediapipe";
        
        // Update recommended defaults for MediaPipe Model
        inferIntervalInput.value = "0.1";
        windowSizeInput.value = "30";
        voteSizeInput.value = "10";
        minVotesInput.value = "6";
        minConfInput.value = "0.3";
        minGapInput.value = "1.0";
        quantizationLabel.textContent = "FP32 (MeanStd)";
    } else {
        tabMediapipe.classList.remove("active");
        tabLandmark.classList.add("active");
        wsUrlInput.value = "ws://localhost:8000/ws/landmark";

        // Update recommended defaults for AIHub Model
        inferIntervalInput.value = "0.15";
        windowSizeInput.value = "40";
        voteSizeInput.value = "15";
        minVotesInput.value = "9";
        minConfInput.value = "0.45";
        minGapInput.value = "1.5";
        quantizationLabel.textContent = "FP32 (Raw)";
    }
}

// Collapsible Settings Event
settingsToggle.addEventListener("click", () => {
    settingsCard.classList.toggle("collapsed");
});

// Tab Buttons Click
tabMediapipe.addEventListener("click", () => selectModel("mediapipe"));
tabLandmark.addEventListener("click", () => selectModel("landmark"));

// Control Buttons Click
startBtn.addEventListener("click", startStreaming);
stopBtn.addEventListener("click", stopStreaming);
clearHistoryBtn.addEventListener("click", () => {
    state.sentenceBuffer = [];
    updateSentenceBufferUI();
});

// Setup Initial UI state
selectModel("mediapipe");
settingsCard.classList.add("collapsed"); // start collapsed for clean UI
updateSentenceBufferUI();
