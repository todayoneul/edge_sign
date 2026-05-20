# Edge-Sign Web Live

이 웹 데모는 `onnxruntime-web`을 사용하여 브라우저에서 직접 INT8 ONNX 한국수어(KSL) 모델을 실행합니다.

## 옵션 A) Hugging Face Hub 사용 (권장)

1) 정답 라벨을 추출하고 에셋을 Hugging Face 리포지토리에 업로드합니다.

```powershell
python scripts/export_labels.py --data-dir ./dataset/train --output ./web/labels.json
```

다음 파일들을 Hugging Face 리포지토리에 업로드합니다 (웹 UI 또는 git-lfs 이용):
- `convnextv2_ksl_int8.onnx`
- `labels.json`

2) 웹 설정 파일에 리포지토리 ID를 구성합니다.

[web/config.js](config.js) 파일을 열어 수정합니다:

```js
window.EDGE_SIGN_CONFIG = {
	hfRepoId: "gyann/edge-sign-ksl",
	hfRevision: "main",
	modelFile: "convnextv2_ksl_int8.onnx",
	labelsFile: "labels.json",
	localModelPath: "./model/convnextv2_ksl_int8.onnx",
	localLabelsPath: "./labels.json",
};
```

3) 로컬 웹 서버를 실행합니다.

```powershell
python -m http.server 8000 --directory web
```

크롬(Chrome) 브라우저에서 `http://localhost:8000`에 접속한 뒤 카메라 접근을 허용합니다. "Load model"을 클릭하여 모델을 로드한 후 "Start camera"를 눌러 실시간 감지를 시작합니다.

*(선택 사항)* 오프라인 환경에서 사용하려면 아래 스크립트를 통해 에셋을 로컬로 다운로드할 수 있습니다.

```powershell
python scripts/fetch_hf_assets.py --repo your-username/edge-sign-ksl
```

## 옵션 B) 로컬 에셋 전용 실행

Hugging Face 연결 없이 로컬 파일만으로 실행하려면 아래 명령어들을 순차적으로 실행합니다.

```powershell
python scripts/export_labels.py --data-dir ./dataset/train --output ./web/labels.json
Copy-Item -Force .\model_space\convnextv2_ksl_int8.onnx .\web\model\convnextv2_ksl_int8.onnx
python -m http.server 8000 --directory web
```

## 참고 사항

- WebGPU 환경이 지원되는 브라우저에서는 최상의 프레임(FPS) 속도로 동작하며, 지원되지 않을 경우 WASM 백엔드로 자동 대체(Fallback)됩니다.
- 모델 경로나 설정이 변경된 경우, 웹 페이지에서 모델을 다시 로드해야 정상적으로 반영됩니다.
- MediaPipe ROI를 켜면 손/얼굴 랜드마크를 기준으로 크롭하며, 카메라 거리/배경 변화에 더 강해집니다. 성능이 느리면 끄는 것을 권장합니다.

## AIhub Keypoint Live (OpenPose)

이 데모는 OpenPose로 AIhub 스타일 키포인트를 추출하고, 학습된 랜드마크 분류기로 실시간 인식을 수행합니다.

### 1) 서버 실행 (FastAPI)

필요 패키지:

```powershell
pip install fastapi uvicorn opencv-python
```

OpenPose 경로를 환경 변수로 지정합니다:

```powershell
$env:OPENPOSE_BIN="C:\openpose\bin\OpenPoseDemo.exe"
$env:OPENPOSE_MODEL_DIR="C:\openpose\models"
```

서버 실행:

```powershell
python scripts/aihub_web_server.py --weights ./checkpoints/landmark_best.pth --labels ./dataset/landmarks_top50/labels.json
```

### 2) 웹 UI 실행

```powershell
python -m http.server 8000 --directory web
```

브라우저에서 `http://localhost:8000/aihub/`로 접속합니다.

### 참고

- W8A8 PTQ는 기본 활성화입니다. 끄려면 `--no-w8a8` 옵션을 사용하세요.
- OpenPose는 프레임당 비용이 크므로 `Send FPS`를 낮게 설정하는 것을 권장합니다.
