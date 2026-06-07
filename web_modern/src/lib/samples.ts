/**
 * samples.ts — 내장 데모 샘플 클립 목록.
 *
 * 모두 H.264(브라우저 디코딩 가능) · 720p · 짧은 동(同)도메인 주행 클립이라
 * 온디바이스(WebGPU) 경로로 매끄럽게 돈다. MPEG-4/HEVC 등 브라우저 비호환
 * 코덱은 서버 인제스트로 폴백되며, 공개 데모(HF) 서버는 CPU 전용이라 느릴 수 있다.
 * → 샘플은 "바로 매끄럽게 도는" 기준 입력 역할.
 */
export interface SampleClip {
  /** public/sample/ 아래 파일명 */
  file: string;
  /** 버튼/칩에 표시할 짧은 라벨 */
  label: string;
}

export const SAMPLES: SampleClip[] = [
  { file: "seoul_daylight.mp4", label: "주간 도심" },
  { file: "seoul_drive.mp4", label: "도로주행" },
];
