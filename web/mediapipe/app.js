const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const wsUrlInput = document.getElementById('wsUrl');
const wsStatus = document.getElementById('wsStatus');

const detectedLabel = document.getElementById('detectedLabel');
const confidenceLabel = document.getElementById('confidenceLabel');
const stableResult = document.getElementById('stableResult');
const fpsLabel = document.getElementById('fpsLabel');

let ws = null;
let camera = null;
let holistic = null;
let isStreaming = false;

// MediaPipe Holistic을 OpenPose 959 차원으로 매핑
// OpenPose 959 차원: 
// pose_2d(75) + face_2d(210) + left_hand_2d(63) + right_hand_2d(63) = 411
// pose_3d(100) + face_3d(280) + left_hand_3d(84) + right_hand_3d(84) = 548

// MediaPipe 33개 포인트를 OpenPose 25개 포인트로 매핑
const POSE_MAPPING = [
    0, // 0: Nose
    [11, 12], // 1: Neck (양어깨의 중간)
    12, // 2: R-Sho
    14, // 3: R-Elb
    16, // 4: R-Wr
    11, // 5: L-Sho
    13, // 6: L-Elb
    15, // 7: L-Wr
    [23, 24], // 8: MidHip (양골반의 중간)
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
    31, // 21: L-Heel (가까운 값으로 대체)
    32, // 22: R-BigToe
    30, // 23: R-SmallToe
    32, // 24: R-Heel (가까운 값으로 대체)
];

function extractFeatures(results) {
    // 959 길이의 Float32Array (4바이트 실수형 배열)
    const features = new Float32Array(959);
    let offset = 0;

    // 2D 포인트 추가 헬퍼 함수
    function add2DPoint(point) {
        if (point) {
            features[offset++] = point.x;
            features[offset++] = point.y;
            features[offset++] = point.visibility !== undefined ? point.visibility : 1.0;
        } else {
            offset += 3;
        }
    }

    // 3D 포인트 추가 헬퍼 함수
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

    // --- 2D 특징 (411 차원) ---
    // 1. Pose 2D (25 * 3 = 75)
    if (results.poseLandmarks) {
        for (let i = 0; i < 25; i++) {
            const mapIdx = POSE_MAPPING[i];
            if (Array.isArray(mapIdx)) {
                const p1 = results.poseLandmarks[mapIdx[0]];
                const p2 = results.poseLandmarks[mapIdx[1]];
                if (p1 && p2) {
                    add2DPoint({ x: (p1.x + p2.x)/2, y: (p1.y + p2.y)/2, visibility: Math.min(p1.visibility, p2.visibility) });
                } else {
                    add2DPoint(null);
                }
            } else {
                add2DPoint(results.poseLandmarks[mapIdx]);
            }
        }
    } else { offset += 75; }

    // 2. Face 2D (70 * 3 = 210)
    // MediaPipe는 468 포인트를 주지만, OpenPose 형식에 맞춰 처음 70개만 사용
    if (results.faceLandmarks) {
        for (let i = 0; i < 70; i++) { add2DPoint(results.faceLandmarks[i]); }
    } else { offset += 210; }

    // 3. Left Hand 2D (21 * 3 = 63)
    if (results.leftHandLandmarks) {
        for (let i = 0; i < 21; i++) { add2DPoint(results.leftHandLandmarks[i]); }
    } else { offset += 63; }

    // 4. Right Hand 2D (21 * 3 = 63)
    if (results.rightHandLandmarks) {
        for (let i = 0; i < 21; i++) { add2DPoint(results.rightHandLandmarks[i]); }
    } else { offset += 63; }


    // --- 3D 특징 (548 차원) ---
    // 5. Pose 3D (25 * 4 = 100)
    if (results.poseWorldLandmarks) {
        for (let i = 0; i < 25; i++) {
            const mapIdx = POSE_MAPPING[i];
            if (Array.isArray(mapIdx)) {
                const p1 = results.poseWorldLandmarks[mapIdx[0]];
                const p2 = results.poseWorldLandmarks[mapIdx[1]];
                if (p1 && p2) {
                    add3DPoint({ x: (p1.x + p2.x)/2, y: (p1.y + p2.y)/2, z: (p1.z + p2.z)/2, visibility: Math.min(p1.visibility, p2.visibility) });
                } else {
                    add3DPoint(null);
                }
            } else {
                add3DPoint(results.poseWorldLandmarks[mapIdx]);
            }
        }
    } else { offset += 100; }

    // 6. Face 3D (70 * 4 = 280)
    if (results.faceLandmarks) {
        for (let i = 0; i < 70; i++) { add3DPoint(results.faceLandmarks[i]); }
    } else { offset += 280; }

    // 7. Left Hand 3D (21 * 4 = 84)
    if (results.leftHandLandmarks) {
        for (let i = 0; i < 21; i++) { add3DPoint(results.leftHandLandmarks[i]); }
    } else { offset += 84; }

    // 8. Right Hand 3D (21 * 4 = 84)
    if (results.rightHandLandmarks) {
        for (let i = 0; i < 21; i++) { add3DPoint(results.rightHandLandmarks[i]); }
    } else { offset += 84; }

    return features;
}

function onResults(results) {
    // 1. 디버깅 및 시각화를 위한 랜드마크 렌더링
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // 비디오 프레임 그리기 (CSS에서 이미 좌우 반전 처리됨)
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    
    // 포즈 및 손 랜드마크 오버레이
    if (results.poseLandmarks) {
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {color: '#00FF00', lineWidth: 2});
        drawLandmarks(canvasCtx, results.poseLandmarks, {color: '#FF0000', lineWidth: 1, radius: 2});
    }
    if (results.leftHandLandmarks) {
        drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, {color: '#CC0000', lineWidth: 2});
        drawLandmarks(canvasCtx, results.leftHandLandmarks, {color: '#00FF00', lineWidth: 1, radius: 2});
    }
    if (results.rightHandLandmarks) {
        drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, {color: '#00CC00', lineWidth: 2});
        drawLandmarks(canvasCtx, results.rightHandLandmarks, {color: '#FF0000', lineWidth: 1, radius: 2});
    }
    canvasCtx.restore();

    // 2. 서버로 바이너리 데이터 전송
    if (isStreaming && ws && ws.readyState === WebSocket.OPEN) {
        const featureArray = extractFeatures(results);
        ws.send(featureArray.buffer); // Float32Array 전송
    }
}

async function startApp() {
    startBtn.disabled = true;
    stableResult.textContent = "모델 로딩 및 연결 중...";

    // WebSocket 초기화
    const wsUrl = wsUrlInput.value.trim();
    if (!wsUrl) {
        alert("WebSocket URL을 입력하세요.");
        startBtn.disabled = false;
        return;
    }
    
    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
        wsStatus.textContent = "연결됨";
        wsStatus.className = "badge connected";
        isStreaming = true;
        stopBtn.disabled = false;
        stableResult.textContent = "카메라를 보고 수어를 시작하세요!";
    };

    ws.onmessage = (event) => {
        try {
            const payload = JSON.parse(event.data);
            detectedLabel.textContent = payload.label || "-";
            confidenceLabel.textContent = payload.confidence ? payload.confidence.toFixed(2) : "0.00";
            fpsLabel.textContent = payload.fps ? payload.fps.toFixed(1) : "0.0";
            
            if (payload.stable && payload.stable !== "-") {
                stableResult.textContent = payload.stable;
                stableResult.style.color = "#28a745";
                // Pop 효과
                stableResult.style.transform = "scale(1.1)";
                setTimeout(() => { stableResult.style.transform = "scale(1)"; }, 200);
            }
        } catch (err) {
            console.error("서버 메시지 파싱 오류:", err);
        }
    };

    ws.onerror = (err) => {
        console.error("WebSocket 오류:", err);
        wsStatus.textContent = "연결 오류";
        wsStatus.className = "badge disconnected";
    };

    ws.onclose = () => {
        stopApp();
    };

    // MediaPipe Holistic 초기화
    if (!holistic) {
        holistic = new Holistic({locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
        }});
        
        holistic.setOptions({
            modelComplexity: 1, // Edge 환경을 위해 1 유지
            smoothLandmarks: true,
            enableSegmentation: false,
            smoothSegmentation: false,
            refineFaceLandmarks: false,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        
        holistic.onResults(onResults);
    }

    // 카메라 초기화
    if (!camera) {
        camera = new Camera(videoElement, {
            onFrame: async () => {
                if (isStreaming) {
                    await holistic.send({image: videoElement});
                }
            },
            width: 640,
            height: 480
        });
        
        camera.start();
    }
}

function stopApp() {
    isStreaming = false;
    if (ws) {
        ws.close();
        ws = null;
    }
    if (camera) {
        camera.stop();
        camera = null;
    }
    wsStatus.textContent = "연결 끊김";
    wsStatus.className = "badge disconnected";
    startBtn.disabled = false;
    stopBtn.disabled = true;
    stableResult.textContent = "종료되었습니다.";
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
}

startBtn.addEventListener('click', startApp);
stopBtn.addEventListener('click', stopApp);
