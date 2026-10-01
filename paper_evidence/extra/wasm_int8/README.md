# Mac WASM에서 INT8 가속이 사라지는 원인 (2026-10-01, 0–1단계)

**질문.** 같은 Mac에서 네이티브 CPU 4T는 INT8이 빠르다(FP32/INT8: YOLO26-n 1.68, YOLOv8s 1.97). 그런데 브라우저 WASM 4T에서는 느리다(0.84, 0.95; [RUNTIME_MATRIX §2.6](../../reports/RUNTIME_MATRIX.md)). Windows는 WASM에서도 빠르다(1.34, 1.62). 이 차이가 어느 실행 경로(ORT-Web 커널, WASM SIMD, V8 lowering)에서 생기는가?

계획, 가설, 예측은 실행 전에 [ROADMAP](../../../docs/ROADMAP.md)에 기록했다(`70268ba`). 이 폴더는 0단계와 1단계(Mac)의 결과이다.

## 0단계: 측정에 쓴 WASM 바이너리 ([step0_wasm_scan.json](step0_wasm_scan.json))

- **바이너리:** 논문의 WASM·WebGPU 측정은 모두 `ort.webgpu.bundle.min.mjs`를 썼다. 이 번들은 `ort-wasm-simd-threaded.asyncify.wasm`(ORT-Web 1.30.0, 26.8 MB, SHA-256 `39f9f089…`)을 읽는다. 프로파일 페이지의 resource timing으로도 확인했다.
- **relaxed SIMD 명령은 0개이다.** `wasm_simd_scan.py`로 함수 15,571개를 모두 디코딩했다(본문 끝 불일치 0). SIMD 명령은 97,989개이다.
- **INT8 GEMM은 fixed SIMD의 `i32x4.dot_i16x8_s`로 내적한다.** 이 명령 19개는 모두 함수 하나에 인라인되어 있다(MLAS U8X8 QGEMM).
  - ORT v1.30.0 소스(`f2c39fe`)의 `qgemm_kernel_wasmsimd.cpp`는 u8 값을 i16으로 넓힌 뒤 `acc = i32x4.add(acc, i32x4.dot_i16x8(B, A))`를 반복한다. 명령 하나가 8 MAC이다.
  - 같은 소스에 relaxed SIMD 커널(`qgemm_kernel_wasmrelaxedsimd.cpp`, `i32x4.relaxed_dot_i8x16_i7x16_add`, 명령 하나가 16 MAC)도 있다. 하지만 빌드 옵션 `onnxruntime_ENABLE_WEBASSEMBLY_RELAXED_SIMD`가 기본 OFF여서 배포판에는 들어 있지 않다.
- 함수 이름 섹션이 제거된 빌드여서, 1단계는 함수를 번호와 명령 구성으로 구분한다.

## 1단계: Mac 커널별 시간 분해 ([breakdown_mac.json](breakdown_mac.json), 원시 [mac/](mac/))

**방법.**
- **조건:** Chrome 154.0.8037.92 headless, ORT-Web 1.30.0, WASM EP, 1스레드, 교차 출처 격리, test 첫 프레임, 웜업 20회 뒤 60회 추론.
- **프로파일러:** V8 CPU 프로파일러를 100 µs 간격으로, 측정 구간에만 켰다(`wasm_kernel_profile.mjs`, CDP).
- **반복:** 모델당 2라운드이고, 라운드마다 모델 순서를 바꿨다.
- **1스레드로 잰 이유:** 프로파일러는 메인 스레드만 샘플링한다. Mac은 1스레드에서도 같은 현상을 보인다(첫 측정 0.91).
- **분류:** 샘플의 `wasm-function[N]`을 0단계 바이너리의 함수 번호에 맞추고, 각 함수를 담긴 명령으로 분류했다(`wasm_kernel_breakdown.py`).
  - `int8_dot_gemm`: `dot_i16x8_s`를 포함
  - `quant_convert`: 정수↔실수 변환·narrow를 포함(Q/DQ, 재양자화)
  - `fp32_simd_mac`: `f32x4.mul`·`add`를 포함(SGEMM, 합성곱)
  - 나머지: 기타 SIMD, 스칼라, WASM 밖
- **검증:** 프로파일로 잡힌 시간은 추론당 실측 지연보다 0.2–0.9% 크다(측정 구간 앞뒤의 호출 포함).

**결과 (추론 1회당 ms, 2라운드 중앙값).**

| 모델 | FP32 전체 | INT8 전체 | FP32/INT8 | 양자화된 합성곱: FP32 SGEMM → INT8 QGEMM | Q/DQ·재양자화 | 기타 SIMD | 스칼라 |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO26-n (v4) | 166.4 | 184.5 | 0.90 | 134.6 → 122.4 (**1.10배**) | +13.9 | +7.8 | +8.6 |
| YOLOv8s (v3) | 749.8 | 765.2 | 0.98 | 512.4 → 491.9 (**1.04배**) | +19.6 | +7.4 | +9.0 |

- "양자화된 합성곱" 열의 계산: FP32 모델의 `fp32_simd_mac` 시간에서, INT8 모델에 FP32로 남은 헤드의 `fp32_simd_mac`(v4 14.0, v3 198.7 ms)를 뺀 값이 INT8로 바뀐 부분이다. 그 부분이 `int8_dot_gemm` 시간으로 바뀌었다.
- 라운드 간 차이: 전체 지연 기준 v4 0.1–1.3%, v3 0.1–0.5%.

**해석.**
- **Mac에서 INT8 QGEMM은 자기가 대체한 FP32 SGEMM보다 4–10%만 빠르다.**
  - 명령당 MAC 수로는 `dot_i16x8_s`(8 MAC)가 `f32x4.mul`+`f32x4.add`(4 MAC, 명령 2개)보다 4배 많다.
  - 그런데 이 이론상의 이득이 거의 나타나지 않는다.
- **순손실(+15–18 ms)은 INT8 그래프가 추가로 하는 일에서 나온다.**
  - Q/DQ와 재양자화(+14–20 ms), 기타 SIMD·스칼라 함수(+16–17 ms)이다.
  - GEMM에서 얻은 12–20 ms로 이를 메우지 못한다.
- **사전 예측과 비교:**
  - 예측은 "Mac INT8의 추가 시간은 Q/DQ 변환이 아니라 정수 GEMM 커널에 몰린다"였다. **이 예측은 문자 그대로는 맞지 않았다.** 정수 GEMM은 FP32보다 느리지 않고 조금 빠르며, 추가 시간은 변환과 기타 함수에 있다.
  - 가설의 핵심, 즉 "ARM의 WASM에서는 INT8 내적이 FP32 대비 이득을 거의 못 낸다"는 이 결과와 맞는다.
  - **이것이 ARM에 특유한 현상인지는 아직 모른다.** Windows에서 같은 분해를 해서, x86에서는 GEMM 부분의 배율이 크고 변환 비용은 비슷한지 확인해야 한다.
  - Windows 1T 첫 측정은 FP32 157.1 ms 대 INT8 90.8 ms였으므로 GEMM 배율이 클 것으로 예상한다. 이는 측정 전 예측이다.

**한계.**
- 이름 없는 바이너리라 함수 분류는 명령 구성에 따른 것이다.
  - `int8_dot_gemm` 함수에는 A/B 패킹과 열 합 계산이 함께 인라인되어 있다.
  - `fp32_simd_mac`에는 GEMM이 아닌 실수 SIMD 함수도 섞일 수 있다.
- 조건 범위: 1스레드, 라운드 2회, 한 프레임 입력이다. Chrome 154로, 논문 측정 때(153)와 버전이 다르다. 조용한 환경 절차는 쓰지 않았다(측정 직전 1분 load average 2.3–4.6, 10코어).

## Windows에서 1단계 실행 (남은 일)

```
# 0) 같은 바이너리 (SHA-256 39f9f0894d478800487ed9f7dbe92618498db320cf55c8e3d89adff8dce658da)
curl -o ort-asyncify.wasm https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort-wasm-simd-threaded.asyncify.wasm
# 1) 서버 (원본 체크아웃의 model_space를 쓰는 기존 측정과 같은 서버)
python scripts/paper/runtime_matrix.py serve --artifact-root <checkout> --output <임시 폴더> --port 8791 --isolate
# 2) 프로파일 (Node 22 이상, scripts/paper/wait_quiet.ps1로 배경 부하 확인 후)
node scripts/paper/wasm_kernel_profile.mjs --out paper_evidence/extra/wasm_int8/windows --models v4_fp32 v4_int8_head_excl v3_fp32 v3_int8_head_excl --rounds 2 --iterations 60
# 3) 두 기기 분해 (폴더 이름이 기기 이름이 된다)
python scripts/paper/wasm_kernel_breakdown.py --wasm ort-asyncify.wasm --profiles paper_evidence/extra/wasm_int8/mac paper_evidence/extra/wasm_int8/windows --output paper_evidence/extra/wasm_int8/breakdown.json
```
