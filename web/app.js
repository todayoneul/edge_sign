// App State
const state = {
    session: null,
    idxToChar: null,
    currentMode: 'canvas', // 'canvas' or 'camera'
    canvasRecognitionMode: 'single', // 'single' (entire canvas) or 'multi' (horizontal segmentation)
    canvasMergeGap: 30, // Horizontal segmentation merge threshold
    isDrawing: false,
    hasDrawn: false,
    lastX: 0,
    lastY: 0,
    webcamStream: null,
    cameraAnimId: null,
    threshold: 128,
    isModelLoading: true,
    currentTool: 'pen', // 'pen' or 'eraser'
    penSize: 14,        // Adjustable pen size
    eraserSize: 36      // Adjustable eraser size
};

// Elements
const el = {
    modelStatusBadge: document.getElementById('modelStatusBadge'),
    modelStatusText: document.getElementById('modelStatusText'),
    tabCanvas: document.getElementById('tabCanvas'),
    tabCamera: document.getElementById('tabCamera'),
    canvasArea: document.getElementById('canvasArea'),
    cameraArea: document.getElementById('cameraArea'),
    drawingCanvas: document.getElementById('drawingCanvas'),
    canvasOverlay: document.getElementById('canvasOverlay'),
    multiGapControl: document.getElementById('multiGapControl'),
    gapSlider: document.getElementById('gapSlider'),
    gapVal: document.getElementById('gapVal'),
    canvasPreviewPanel: document.getElementById('canvasPreviewPanel'),
    drawingPreviewCanvas: document.getElementById('drawingPreviewCanvas'),
    clearCanvasBtn: document.getElementById('clearCanvasBtn'),
    addCharBtn: document.getElementById('addCharBtn'),
    webcamVideo: document.getElementById('webcamVideo'),
    cameraOverlayCanvas: document.getElementById('cameraOverlayCanvas'),
    thresholdSlider: document.getElementById('thresholdSlider'),
    thresholdVal: document.getElementById('thresholdVal'),
    thresholdPreviewCanvas: document.getElementById('thresholdPreviewCanvas'),
    startCamBtn: document.getElementById('startCamBtn'),
    stopCamBtn: document.getElementById('stopCamBtn'),
    detectedLabel: document.getElementById('detectedLabel'),
    candidatesList: document.getElementById('candidatesList'),
    textBuffer: document.getElementById('textBuffer'),
    copyTextBtn: document.getElementById('copyTextBtn'),
    clearTextBtn: document.getElementById('clearTextBtn'),
    btnSpace: document.getElementById('btnSpace'),
    btnBackspace: document.getElementById('btnBackspace'),
    inferenceTimeLabel: document.getElementById('inferenceTimeLabel'),
    pwaInstallBanner: document.getElementById('pwaInstallBanner'),
    pwaInstallBtn: document.getElementById('pwaInstallBtn'),
    btnModeSingle: document.getElementById('btnModeSingle'),
    btnModeMulti: document.getElementById('btnModeMulti'),
    btnToolPen: document.getElementById('btnToolPen'),
    btnToolEraser: document.getElementById('btnToolEraser'),
    headerInstallBtn: document.getElementById('headerInstallBtn'),
    installInstructionsModal: document.getElementById('installInstructionsModal'),
    closeModalBtn: document.getElementById('closeModalBtn'),
    closeModalBtnFooter: document.getElementById('closeModalBtnFooter'),
    btnSelectAndroid: document.getElementById('btnSelectAndroid'),
    btnSelectIOS: document.getElementById('btnSelectIOS'),
    instructionsAndroid: document.getElementById('instructionsAndroid'),
    instructionsIOS: document.getElementById('instructionsIOS'),
    toolSizeSlider: document.getElementById('toolSizeSlider'),
    toolSizeVal: document.getElementById('toolSizeVal')
};

// Canvas drawing context configuration
const ctx = el.drawingCanvas.getContext('2d', { willReadFrequently: true });
ctx.lineWidth = 14;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = '#000000'; // Draw black strokes

const overlayCtx = el.canvasOverlay ? el.canvasOverlay.getContext('2d') : null;

// Initialize drawing canvas background to white
function clearCanvas() {
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, el.drawingCanvas.width, el.drawingCanvas.height);
    clearOverlay();
    state.hasDrawn = false;
    clearPredictions();
    if (el.drawingPreviewCanvas) {
        const pCtx = el.drawingPreviewCanvas.getContext('2d');
        pCtx.fillStyle = '#FFFFFF';
        pCtx.fillRect(0, 0, 64, 64);
    }
}

function clearOverlay() {
    if (overlayCtx) {
        overlayCtx.clearRect(0, 0, el.canvasOverlay.width, el.canvasOverlay.height);
    }
}

function clearPredictions() {
    el.detectedLabel.textContent = '-';
    el.candidatesList.innerHTML = '<div class="candidate-item-empty">대기 중...</div>';
}

// Setup Event Listeners
function initEvents() {
    // Tab switching
    el.tabCanvas.addEventListener('click', () => switchTab('canvas'));
    el.tabCamera.addEventListener('click', () => switchTab('camera'));

    // Canvas drawing (Mouse)
    el.drawingCanvas.addEventListener('mousedown', startDrawing);
    el.drawingCanvas.addEventListener('mousemove', draw);
    el.drawingCanvas.addEventListener('mouseup', stopDrawing);
    el.drawingCanvas.addEventListener('mouseleave', stopDrawing);

    // Canvas drawing (Touch)
    el.drawingCanvas.addEventListener('touchstart', (e) => {
        startDrawing(e);
        e.preventDefault();
    }, { passive: false });
    el.drawingCanvas.addEventListener('touchmove', (e) => {
        draw(e);
        e.preventDefault();
    }, { passive: false });
    el.drawingCanvas.addEventListener('touchend', (e) => {
        stopDrawing();
        e.preventDefault();
    });

    // Clear and insert actions
    el.clearCanvasBtn.addEventListener('click', clearCanvas);
    el.addCharBtn.addEventListener('click', () => {
        const topChar = el.detectedLabel.textContent;
        if (topChar && topChar !== '-') {
            insertText(topChar);
            clearCanvas();
        }
    });

    // Keyboard helper actions
    el.btnSpace.addEventListener('click', () => insertText(' '));
    el.btnBackspace.addEventListener('click', () => {
        const text = el.textBuffer.textContent;
        if (text.length > 0) {
            el.textBuffer.textContent = text.slice(0, -1);
        }
    });
    el.copyTextBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(el.textBuffer.textContent);
        const originalText = el.copyTextBtn.textContent;
        el.copyTextBtn.textContent = '복사됨!';
        setTimeout(() => el.copyTextBtn.textContent = originalText, 1500);
    });
    el.clearTextBtn.addEventListener('click', () => {
        el.textBuffer.textContent = '';
    });

    // Drawing Tool toggles
    el.btnToolPen.addEventListener('click', () => {
        state.currentTool = 'pen';
        el.btnToolPen.classList.add('active');
        el.btnToolEraser.classList.remove('active');
        
        // Update slider context for Pen
        if (el.toolSizeSlider) {
            el.toolSizeSlider.min = 5;
            el.toolSizeSlider.max = 30;
            el.toolSizeSlider.value = state.penSize;
        }
        if (el.toolSizeVal) el.toolSizeVal.textContent = state.penSize;
        ctx.lineWidth = state.penSize;
    });
    el.btnToolEraser.addEventListener('click', () => {
        state.currentTool = 'eraser';
        el.btnToolEraser.classList.add('active');
        el.btnToolPen.classList.remove('active');
        
        // Update slider context for Eraser
        if (el.toolSizeSlider) {
            el.toolSizeSlider.min = 10;
            el.toolSizeSlider.max = 80;
            el.toolSizeSlider.value = state.eraserSize;
        }
        if (el.toolSizeVal) el.toolSizeVal.textContent = state.eraserSize;
        ctx.lineWidth = state.eraserSize;
    });

    // Tool Size Slider
    if (el.toolSizeSlider) {
        el.toolSizeSlider.addEventListener('input', (e) => {
            const size = parseInt(e.target.value);
            if (el.toolSizeVal) el.toolSizeVal.textContent = size;
            
            if (state.currentTool === 'eraser') {
                state.eraserSize = size;
            } else {
                state.penSize = size;
            }
            ctx.lineWidth = size;
        });
    }

    // Recognition Mode switching
    el.btnModeSingle.addEventListener('click', () => {
        state.canvasRecognitionMode = 'single';
        el.btnModeSingle.classList.add('active');
        el.btnModeMulti.classList.remove('active');
        if (el.multiGapControl) el.multiGapControl.classList.add('hidden');
        if (el.canvasPreviewPanel) el.canvasPreviewPanel.classList.add('hidden');
        clearOverlay();
        if (state.hasDrawn) runInferenceOnCanvas();
    });
    el.btnModeMulti.addEventListener('click', () => {
        state.canvasRecognitionMode = 'multi';
        el.btnModeMulti.classList.add('active');
        el.btnModeSingle.classList.remove('active');
        if (el.multiGapControl) el.multiGapControl.classList.remove('hidden');
        if (el.canvasPreviewPanel) el.canvasPreviewPanel.classList.remove('hidden');
        clearOverlay();
        if (state.hasDrawn) runInferenceOnCanvas();
    });

    // Merge Gap Slider
    if (el.gapSlider) {
        el.gapSlider.addEventListener('input', (e) => {
            state.canvasMergeGap = parseInt(e.target.value);
            if (el.gapVal) el.gapVal.textContent = state.canvasMergeGap;
        });
        el.gapSlider.addEventListener('change', () => {
            if (state.hasDrawn) runInferenceOnCanvas();
        });
    }

    // Camera actions
    el.startCamBtn.addEventListener('click', startWebcam);
    el.stopCamBtn.addEventListener('click', stopWebcam);
    el.thresholdSlider.addEventListener('input', (e) => {
        state.threshold = parseInt(e.target.value);
        el.thresholdVal.textContent = state.threshold;
    });

    // PWA Install Button handlers
    if (el.headerInstallBtn) {
        el.headerInstallBtn.addEventListener('click', () => {
            handleInstallClick();
        });
    }
    if (el.pwaInstallBtn) {
        el.pwaInstallBtn.addEventListener('click', () => {
            handleInstallClick();
        });
    }

    // Modal control actions
    if (el.closeModalBtn) {
        el.closeModalBtn.addEventListener('click', () => {
            el.installInstructionsModal.classList.add('hidden');
        });
    }
    if (el.closeModalBtnFooter) {
        el.closeModalBtnFooter.addEventListener('click', () => {
            el.installInstructionsModal.classList.add('hidden');
        });
    }
    if (el.installInstructionsModal) {
        el.installInstructionsModal.addEventListener('click', (e) => {
            if (e.target === el.installInstructionsModal) {
                el.installInstructionsModal.classList.add('hidden');
            }
        });
    }
    if (el.btnSelectAndroid) {
        el.btnSelectAndroid.addEventListener('click', () => switchInstructionTab('android'));
    }
    if (el.btnSelectIOS) {
        el.btnSelectIOS.addEventListener('click', () => switchInstructionTab('ios'));
    }
}

function switchTab(mode) {
    if (mode === state.currentMode) return;
    
    state.currentMode = mode;
    clearPredictions();

    if (mode === 'canvas') {
        el.tabCanvas.classList.add('active');
        el.tabCamera.classList.remove('active');
        el.canvasArea.classList.remove('hidden');
        el.canvasArea.classList.add('active');
        el.cameraArea.classList.remove('active');
        el.cameraArea.classList.add('hidden');
        stopWebcam();
        clearCanvas();
    } else {
        el.tabCamera.classList.add('active');
        el.tabCanvas.classList.remove('active');
        el.cameraArea.classList.remove('hidden');
        el.cameraArea.classList.add('active');
        el.canvasArea.classList.remove('active');
        el.canvasArea.classList.add('hidden');
    }
}

// Coordinate translation helper handling responsive canvas bounds and scaling
function getCanvasCoords(e, canvas) {
    const rect = canvas.getBoundingClientRect();
    let clientX, clientY;
    if (e.touches && e.touches.length > 0) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
    } else if (e.changedTouches && e.changedTouches.length > 0) {
        clientX = e.changedTouches[0].clientX;
        clientY = e.changedTouches[0].clientY;
    } else {
        clientX = e.clientX;
        clientY = e.clientY;
    }
    
    return {
        x: (clientX - rect.left) * (canvas.width / rect.width),
        y: (clientY - rect.top) * (canvas.height / rect.height)
    };
}

// Drawing Logic
function startDrawing(e) {
    state.isDrawing = true;
    state.hasDrawn = true;
    
    // Set properties according to tool and size
    if (state.currentTool === 'eraser') {
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = state.eraserSize;
    } else {
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = state.penSize;
    }
    
    const coords = getCanvasCoords(e, el.drawingCanvas);
    state.lastX = coords.x;
    state.lastY = coords.y;
    clearOverlay();
}

function draw(e) {
    if (!state.isDrawing) return;
    const coords = getCanvasCoords(e, el.drawingCanvas);

    ctx.beginPath();
    ctx.moveTo(state.lastX, state.lastY);
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();

    state.lastX = coords.x;
    state.lastY = coords.y;
}

function stopDrawing() {
    if (state.isDrawing) {
        state.isDrawing = false;
        runInferenceOnCanvas();
    }
}

function insertText(text) {
    if (el.textBuffer.textContent === '손글씨를 쓰거나 카메라로 스캔하여 완성해 보세요.' ||
        el.textBuffer.textContent === '손글씨를 입력하면 여기에 텍스트가 완성됩니다.') {
        el.textBuffer.textContent = '';
    }
    el.textBuffer.textContent += text;
}

// Vertical projection-based segmentation to support multiple characters side-by-side
function segmentCanvas(canvas, mergeGapThreshold) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Get full canvas image data
    const imgData = ctx.getImageData(0, 0, width, height);
    
    // 1. Calculate vertical projection (number of ink pixels in each column)
    const projection = new Int32Array(width);
    for (let x = 0; x < width; x++) {
        let count = 0;
        for (let y = 0; y < height; y++) {
            const idx = (y * width + x) * 4;
            const r = imgData.data[idx];
            const g = imgData.data[idx + 1];
            const b = imgData.data[idx + 2];
            // Dark pixel check (ink)
            if (r < 240 || g < 240 || b < 240) {
                count++;
            }
        }
        projection[x] = count;
    }
    
    // 2. Find ink intervals
    const minPixelThreshold = 1; // Ignore single-pixel noise
    let intervals = [];
    let inInterval = false;
    let start = 0;
    
    for (let x = 0; x < width; x++) {
        const isInk = projection[x] >= minPixelThreshold;
        if (isInk && !inInterval) {
            start = x;
            inInterval = true;
        } else if (!isInk && inInterval) {
            intervals.push({ start, end: x - 1 });
            inInterval = false;
        }
    }
    if (inInterval) {
        intervals.push({ start, end: width - 1 });
    }
    
    if (intervals.length === 0) return [];
    
    // 3. Merge intervals separated by gaps smaller than mergeGapThreshold
    // This groups strokes of the same character (e.g. 'ㅇ' and 'ㅣ' in '아')
    let merged = [intervals[0]];
    
    for (let i = 1; i < intervals.length; i++) {
        const prev = merged[merged.length - 1];
        const curr = intervals[i];
        const gap = curr.start - prev.end;
        
        if (gap <= mergeGapThreshold) {
            prev.end = curr.end;
        } else {
            merged.push(curr);
        }
    }
    
    // 4. For each merged horizontal interval, find vertical bounds to crop tightly
    let charBoxes = [];
    merged.forEach(interval => {
        let minY = height;
        let maxY = 0;
        let found = false;
        
        for (let x = interval.start; x <= interval.end; x++) {
            for (let y = 0; y < height; y++) {
                const idx = (y * width + x) * 4;
                const r = imgData.data[idx];
                const g = imgData.data[idx + 1];
                const b = imgData.data[idx + 2];
                if (r < 240 || g < 240 || b < 240) {
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                    found = true;
                }
            }
        }
        
        if (found) {
            // Add padding around bounds
            const margin = 8;
            minY = Math.max(0, minY - margin);
            maxY = Math.min(height, maxY + margin);
            const minX = Math.max(0, interval.start - margin);
            const maxX = Math.min(width, interval.end + margin);
            
            charBoxes.push({ minX, maxX, minY, maxY });
        }
    });
    
    return charBoxes;
}

// Image Preprocessing & Inference
async function runInferenceOnCanvas() {
    if (!state.session || !state.idxToChar || !state.hasDrawn) return;

    // Segment the drawing canvas into multiple character boxes
    const rect = el.drawingCanvas.getBoundingClientRect();
    const scaleFactor = rect.width ? (el.drawingCanvas.width / rect.width) : 1.0;
    const mergeGapThreshold = state.canvasRecognitionMode === 'single' ? 9999 : (state.canvasMergeGap * scaleFactor);
    
    const charBoxes = segmentCanvas(el.drawingCanvas, mergeGapThreshold);
    if (charBoxes.length === 0) {
        clearPredictions();
        clearOverlay();
        return;
    }
    
    let recognizedChars = [];
    let multiCandidatesHTML = '';
    const startOverallTime = performance.now();
    
    el.candidatesList.innerHTML = ''; // Clear candidates panel

    // Show preview panel and clear it
    if (el.canvasPreviewPanel) {
        el.canvasPreviewPanel.classList.remove('hidden');
    }

    for (let c = 0; c < charBoxes.length; c++) {
        const bounds = charBoxes[c];
        
        // Create temporary canvas for cropping and resizing
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 64;
        tempCanvas.height = 64;
        const tCtx = tempCanvas.getContext('2d');
        
        // Fill white background
        tCtx.fillStyle = '#FFFFFF';
        tCtx.fillRect(0, 0, 64, 64);
        
        const srcWidth = bounds.maxX - bounds.minX;
        const srcHeight = bounds.maxY - bounds.minY;
        const maxLen = Math.max(srcWidth, srcHeight);
        
        // Dynamic Scaling to fix the stroke thickness and character size mismatch!
        // We want stroke width in 64x64 to be ~2.8px (ideal training width).
        // Since canvas stroke is 14px, the ideal scale is 2.8 / 14 = 0.20.
        // We cap the maximum size to 44px (to ensure it fits within 64x64 with a margin),
        // and cap the minimum size to 28px (so small drawings are legible).
        let scale = 2.8 / 14; 
        const targetSize = 44; 
        if (maxLen * scale > targetSize) {
            scale = targetSize / maxLen;
        } else if (maxLen * scale < 28) {
            scale = 28 / maxLen;
        }
        
        const targetW = srcWidth * scale;
        const targetH = srcHeight * scale;
        const targetX = (64 - targetW) / 2;
        const targetY = (64 - targetH) / 2;
        
        tCtx.drawImage(
            el.drawingCanvas,
            bounds.minX, bounds.minY, srcWidth, srcHeight, // Source
            targetX, targetY, targetW, targetH // Destination
        );

        // Update the 64x64 preview canvas in the UI with the first character
        if (c === 0 && el.drawingPreviewCanvas) {
            const pCtx = el.drawingPreviewCanvas.getContext('2d');
            pCtx.fillStyle = '#FFFFFF';
            pCtx.fillRect(0, 0, 64, 64);
            pCtx.drawImage(tempCanvas, 0, 0);
        }

        // Get 64x64 pixel data and convert to normalized float array [-1.0, 1.0]
        const preprocData = tCtx.getImageData(0, 0, 64, 64);
        const floatBuffer = new Float32Array(64 * 64);
        for (let i = 0; i < 64 * 64; i++) {
            const r = preprocData.data[i * 4];
            const g = preprocData.data[i * 4 + 1];
            const b = preprocData.data[i * 4 + 2];
            const gray = 0.299 * r + 0.587 * g + 0.114 * b;
            floatBuffer[i] = (gray / 255.0 - 0.5) / 0.5;
        }

        // Run ONNX prediction
        const tensor = new ort.Tensor('float32', floatBuffer, [1, 1, 64, 64]);
        const results = await state.session.run({ input: tensor });
        const output = results.output.data;
        const top5 = getTopK(output, 5);
        
        const top1Char = state.idxToChar[top5[0].index];
        recognizedChars.push(top1Char);
        
        // Build candidates list HTML
        if (charBoxes.length > 1) {
            multiCandidatesHTML += `
                <div class="multi-char-candidates-group">
                    <div class="multi-char-header">글자 ${c + 1} (${top1Char}) 후보군:</div>
            `;
        }
        
        top5.forEach((cand, rank) => {
            const char = state.idxToChar[cand.index];
            const probPercent = (cand.prob * 100).toFixed(1);
            
            if (charBoxes.length > 1) {
                multiCandidatesHTML += `
                    <div class="candidate-item" data-char="${char}">
                        <span class="candidate-rank">${rank + 1}</span>
                        <span class="candidate-char">${char}</span>
                        <div class="progress-bar-container">
                            <div class="progress-bar" style="width: ${probPercent}%"></div>
                        </div>
                        <span class="candidate-prob">${probPercent}%</span>
                    </div>
                `;
            } else {
                const item = document.createElement('div');
                item.className = 'candidate-item';
                item.innerHTML = `
                    <span class="candidate-rank">${rank + 1}</span>
                    <span class="candidate-char">${char}</span>
                    <div class="progress-bar-container">
                        <div class="progress-bar" style="width: ${probPercent}%"></div>
                    </div>
                    <span class="candidate-prob">${probPercent}%</span>
                `;
                item.addEventListener('click', () => {
                    insertText(char);
                    clearCanvas();
                });
                el.candidatesList.appendChild(item);
            }
        });
        
        if (charBoxes.length > 1) {
            multiCandidatesHTML += `</div>`;
        }
    }
    
    // Performance display
    const overallTime = performance.now() - startOverallTime;
    el.inferenceTimeLabel.textContent = `${overallTime.toFixed(1)} ms`;
    
    // Render overall string
    const resultString = recognizedChars.join('');
    el.detectedLabel.textContent = resultString;
    
    if (charBoxes.length > 1) {
        el.candidatesList.innerHTML = multiCandidatesHTML;
        
        // Attach click triggers to multi-candidates
        el.candidatesList.querySelectorAll('.candidate-item').forEach(item => {
            item.addEventListener('click', () => {
                const char = item.getAttribute('data-char');
                insertText(char);
                clearCanvas();
            });
        });
    }

    // Draw the bounding boxes and their labels on the overlay canvas
    drawBoundingBoxes(charBoxes, recognizedChars);
}

function drawBoundingBoxes(charBoxes, recognizedChars) {
    clearOverlay();
    if (!overlayCtx) return;
    if (state.canvasRecognitionMode === 'single') return; // In single mode, don't draw bounding box to avoid clutter
    
    charBoxes.forEach((box, idx) => {
        // Draw bounding box rectangle
        overlayCtx.strokeStyle = 'rgba(59, 130, 246, 0.8)'; // sleek blue with opacity
        overlayCtx.lineWidth = 2;
        overlayCtx.strokeRect(box.minX, box.minY, box.maxX - box.minX, box.maxY - box.minY);
        
        // Draw a subtle background for the box
        overlayCtx.fillStyle = 'rgba(59, 130, 246, 0.05)';
        overlayCtx.fillRect(box.minX, box.minY, box.maxX - box.minX, box.maxY - box.minY);
        
        // Draw text label (Character index + prediction if available)
        const predChar = recognizedChars && recognizedChars[idx] ? recognizedChars[idx] : `글자 ${idx + 1}`;
        overlayCtx.fillStyle = '#3b82f6';
        overlayCtx.font = 'bold 12px sans-serif';
        overlayCtx.textBaseline = 'bottom';
        
        // Background for the label text
        const textWidth = overlayCtx.measureText(predChar).width;
        overlayCtx.fillStyle = 'rgba(10, 12, 22, 0.8)';
        overlayCtx.fillRect(box.minX, Math.max(0, box.minY - 18), textWidth + 10, 18);
        
        // Label border
        overlayCtx.strokeStyle = 'rgba(59, 130, 246, 0.4)';
        overlayCtx.strokeRect(box.minX, Math.max(0, box.minY - 18), textWidth + 10, 18);
        
        // Draw the text
        overlayCtx.fillStyle = '#60a5fa';
        overlayCtx.fillText(predChar, box.minX + 5, Math.max(0, box.minY - 3));
    });
}

// Prediction executor
async function predict(inputData) {
    const startTime = performance.now();
    try {
        // Create ONNX tensor: shape [1, 1, 64, 64]
        const tensor = new ort.Tensor('float32', inputData, [1, 1, 64, 64]);
        
        // Run inference
        const results = await state.session.run({ input: tensor });
        const output = results.output.data; // Float32 logits of size 2350
        
        // Compute softmax or simple argmax/topk
        const top5 = getTopK(output, 5);
        
        const inferTime = performance.now() - startTime;
        el.inferenceTimeLabel.textContent = `${inferTime.toFixed(1)} ms`;

        // Render top-1 result
        const top1Char = state.idxToChar[top5[0].index];
        el.detectedLabel.textContent = top1Char;
        
        // Render top-5 lists
        el.candidatesList.innerHTML = '';
        top5.forEach((cand, rank) => {
            const char = state.idxToChar[cand.index];
            const probPercent = (cand.prob * 100).toFixed(1);
            
            const item = document.createElement('div');
            item.className = 'candidate-item';
            item.innerHTML = `
                <span class="candidate-rank">${rank + 1}</span>
                <span class="candidate-char">${char}</span>
                <div class="progress-bar-container">
                    <div class="progress-bar" style="width: ${probPercent}%"></div>
                </div>
                <span class="candidate-prob">${probPercent}%</span>
            `;
            
            // Add click support to manually override or insert the candidate
            item.addEventListener('click', () => {
                insertText(char);
                if (state.currentMode === 'canvas') {
                    clearCanvas();
                }
            });
            el.candidatesList.appendChild(item);
        });

    } catch (e) {
        console.error("Prediction error:", e);
    }
}

// Helper to compute Softmax and Top-K
function getTopK(logits, k) {
    // Convert Float32Array to regular JavaScript array to allow mapping to objects
    const logitsArray = Array.from(logits);
    
    // Calculate max logit manually to avoid stack size limits with Math.max(...logits)
    let maxLogit = -Infinity;
    for (let i = 0; i < logitsArray.length; i++) {
        if (logitsArray[i] > maxLogit) {
            maxLogit = logitsArray[i];
        }
    }
    
    // Softmax
    const exps = logitsArray.map(x => Math.exp(x - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(x => x / (sumExps || 1)); // prevent division by zero
    
    // Map to { prob, index } objects and sort
    const probIndices = probs.map((prob, index) => ({ prob, index }));
    probIndices.sort((a, b) => b.prob - a.prob);
    
    return probIndices.slice(0, k);
}

// Webcam Logic
async function startWebcam() {
    try {
        el.startCamBtn.disabled = true;
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'environment' // Rear camera on mobile
            },
            audio: false
        };
        
        state.webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
        el.webcamVideo.srcObject = state.webcamStream;
        el.stopCamBtn.disabled = false;
        
        // Start process loop
        processCameraFrame();
        
    } catch (e) {
        console.error("Error opening camera:", e);
        alert("카메라를 열 수 없습니다. 권한을 확인해주세요.");
        el.startCamBtn.disabled = false;
    }
}

function stopWebcam() {
    if (state.cameraAnimId) {
        cancelAnimationFrame(state.cameraAnimId);
        state.cameraAnimId = null;
    }
    if (state.webcamStream) {
        state.webcamStream.getTracks().forEach(track => track.stop());
        state.webcamStream = null;
    }
    el.webcamVideo.srcObject = null;
    el.startCamBtn.disabled = false;
    el.stopCamBtn.disabled = true;
}

function processCameraFrame() {
    if (!state.webcamStream || el.webcamVideo.paused || el.webcamVideo.ended) {
        state.cameraAnimId = requestAnimationFrame(processCameraFrame);
        return;
    }

    // Sync overlay canvas size
    const vW = el.webcamVideo.videoWidth;
    const vH = el.webcamVideo.videoHeight;
    
    if (vW > 0 && vH > 0) {
        if (el.cameraOverlayCanvas.width !== vW || el.cameraOverlayCanvas.height !== vH) {
            el.cameraOverlayCanvas.width = vW;
            el.cameraOverlayCanvas.height = vH;
        }

        const oCtx = el.cameraOverlayCanvas.getContext('2d');
        oCtx.clearRect(0, 0, vW, vH);
        
        // Calculate crop box in video coordinate space
        // Typically a square centered in the frame
        const boxSize = Math.min(vW, vH) * 0.5; // 50% of the smallest dimension
        const boxX = (vW - boxSize) / 2;
        const boxY = (vH - boxSize) / 2;
        
        // 1. Draw target box overlay on screen
        oCtx.strokeStyle = '#3b82f6';
        oCtx.lineWidth = 4;
        oCtx.strokeRect(boxX, boxY, boxSize, boxSize);
        
        // Add corner ticks
        oCtx.fillStyle = '#3b82f6';
        const tick = 20;
        oCtx.fillRect(boxX - 2, boxY - 2, tick, 6);
        oCtx.fillRect(boxX - 2, boxY - 2, 6, tick);
        oCtx.fillRect(boxX + boxSize - tick + 2, boxY - 2, tick, 6);
        oCtx.fillRect(boxX + boxSize - 4, boxY - 2, 6, tick);
        oCtx.fillRect(boxX - 2, boxY + boxSize - 4, tick, 6);
        oCtx.fillRect(boxX - 2, boxY + boxSize - tick + 2, 6, tick);
        oCtx.fillRect(boxX + boxSize - tick + 2, boxY + boxSize - 4, tick, 6);
        oCtx.fillRect(boxX + boxSize - 4, boxY + boxSize - tick + 2, 6, tick);

        // 2. Extract cropped area and apply thresholding for inference
        const cropCanvas = document.createElement('canvas');
        cropCanvas.width = 64;
        cropCanvas.height = 64;
        const cCtx = cropCanvas.getContext('2d');
        
        // Draw cropped area into 64x64
        cCtx.drawImage(el.webcamVideo, boxX, boxY, boxSize, boxSize, 0, 0, 64, 64);
        
        const cropImgData = cCtx.getImageData(0, 0, 64, 64);
        const floatBuffer = new Float32Array(64 * 64);
        
        // Thresholding Preview Canvas context
        const pCtx = el.thresholdPreviewCanvas.getContext('2d');
        const pImgData = pCtx.createImageData(64, 64);

        // Apply thresholding filter (grayscale -> binary white/black)
        for (let i = 0; i < 64 * 64; i++) {
            const idx = i * 4;
            const r = cropImgData.data[idx];
            const g = cropImgData.data[idx + 1];
            const b = cropImgData.data[idx + 2];
            const gray = 0.299 * r + 0.587 * g + 0.114 * b;
            
            // If the luminance is below threshold, it's black (ink), otherwise white (paper background)
            const binaryVal = gray < state.threshold ? 0 : 255;
            
            // Fill preview image
            pImgData.data[idx] = binaryVal;
            pImgData.data[idx + 1] = binaryVal;
            pImgData.data[idx + 2] = binaryVal;
            pImgData.data[idx + 3] = 255; // Alpha
            
            // Normalize: (binaryVal / 255.0 - 0.5) / 0.5
            floatBuffer[i] = (binaryVal / 255.0 - 0.5) / 0.5;
        }
        
        // Draw binary preview on screen
        pCtx.putImageData(pImgData, 0, 0);

        // Run prediction
        if (state.session && state.idxToChar) {
            predict(floatBuffer);
        }
    }
    
    // Loop
    state.cameraAnimId = requestAnimationFrame(processCameraFrame);
}

// Load Model and Data
async function loadModelAndData() {
    try {
        console.log("Loading mapping and model...");
        el.modelStatusText.textContent = '인덱스 맵 로딩 중...';

        // 1. Fetch idx_to_char map
        const mapRes = await fetch('idx_to_char.json');
        state.idxToChar = await mapRes.json();
        
        // 2. Fetch quantized ONNX model
        el.modelStatusText.textContent = 'ONNX 모델 로딩 중...';
        state.session = await ort.InferenceSession.create('korean_ocr_quant.onnx', {
            executionProviders: ['wasm'] // Use WebAssembly for on-device inference
        });

        console.log("Model loaded successfully!");
        state.isModelLoading = false;
        
        // Update badge UI
        el.modelStatusBadge.classList.remove('disconnected');
        el.modelStatusBadge.classList.add('connected');
        el.modelStatusText.textContent = '준비 완료 (On-Device)';
        
        // Initial drawing canvas setup
        clearCanvas();
        
    } catch (e) {
        console.error("Failed to load model/data:", e);
        el.modelStatusText.textContent = '로딩 에러 (오류 발생)';
        alert("모델을 불러오는데 실패했습니다. idx_to_char.json 및 korean_ocr_quant.onnx가 올바른 위치에 있는지 확인해주세요.");
    }
}

let deferredPrompt = null;

// PWA Install helper
async function handleInstallClick() {
    if (deferredPrompt) {
        deferredPrompt.prompt();
        const { outcome } = await deferredPrompt.userChoice;
        console.log(`User response to install prompt: ${outcome}`);
        if (outcome === 'accepted') {
            deferredPrompt = null;
            if (el.pwaInstallBanner) el.pwaInstallBanner.classList.add('hidden');
            if (el.headerInstallBtn) el.headerInstallBtn.classList.add('hidden');
        }
    } else {
        showInstallModal();
    }
}

function showInstallModal() {
    if (!el.installInstructionsModal) return;
    
    el.installInstructionsModal.classList.remove('hidden');
    
    // Detect iOS
    const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
    if (isIOS) {
        switchInstructionTab('ios');
    } else {
        switchInstructionTab('android');
    }
}

function switchInstructionTab(os) {
    if (os === 'ios') {
        el.btnSelectIOS.classList.add('active');
        el.btnSelectAndroid.classList.remove('active');
        el.instructionsIOS.classList.remove('hidden');
        el.instructionsAndroid.classList.add('hidden');
    } else {
        el.btnSelectAndroid.classList.add('active');
        el.btnSelectIOS.classList.remove('active');
        el.instructionsAndroid.classList.remove('hidden');
        el.instructionsIOS.classList.add('hidden');
    }
}

function checkStandalone() {
    const isStandalone = window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone;
    if (isStandalone) {
        if (el.headerInstallBtn) el.headerInstallBtn.classList.add('hidden');
        if (el.pwaInstallBanner) el.pwaInstallBanner.classList.add('hidden');
    }
}

// PWA Installation events
window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    if (el.pwaInstallBanner) el.pwaInstallBanner.classList.remove('hidden');
    if (el.headerInstallBtn) el.headerInstallBtn.classList.remove('hidden');
});

window.addEventListener('appinstalled', () => {
    console.log('PWA was installed');
    deferredPrompt = null;
    if (el.pwaInstallBanner) el.pwaInstallBanner.classList.add('hidden');
    if (el.headerInstallBtn) el.headerInstallBtn.classList.add('hidden');
});

// App Entry Point
window.addEventListener('DOMContentLoaded', () => {
    initEvents();
    loadModelAndData();
    checkStandalone();
});
