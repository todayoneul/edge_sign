window.EDGE_SIGN_CONFIG = {
  // 기본 로딩 소스 설정: 'local' (로컬 파일) 또는 'hf' (Hugging Face Hub)
  defaultSource: "hf",
  
  // 기본 Hugging Face 사용자 이름 (업로드 후 해당 아이디로 변경 가능)
  hfUsername: "gyann",
  
  // 1. MediaPipe Model 설정 (2,771 단어)
  mediapipe: {
    // 로컬 경로
    localModelUrl: "./model/mediapipe_best.onnx",
    localLabelsUrl: "./model/mediapipe_labels.json",
    localStatsUrl: "./model/mediapipe_stats.json",
    
    // Hugging Face 저장소 정보
    hfRepo: "edge-sign-ksl-mediapipe",
    hfRevision: "main",
    modelFile: "mediapipe_best.onnx",
    labelsFile: "mediapipe_labels.json",
    statsFile: "mediapipe_stats.json"
  },
  
  // 2. AIHub Landmark Model 설정 (50 단어)
  landmark: {
    // 로컬 경로
    localModelUrl: "./model/landmark_best.onnx",
    localLabelsUrl: "./model/landmark_labels.json",
    
    // Hugging Face 저장소 정보
    hfRepo: "edge-sign-ksl-landmark",
    hfRevision: "main",
    modelFile: "landmark_best.onnx",
    labelsFile: "landmark_labels.json"
  }
};
