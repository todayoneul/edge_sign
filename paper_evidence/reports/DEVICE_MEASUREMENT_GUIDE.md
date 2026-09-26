# 두 번째 기기(Mac) 측정 안내

**목적.** 논문의 실행 환경별 결과는 Windows 데스크톱 한 대(Ryzen 5 9600X + RTX 5070)에서 측정했다. 같은 모델 파일, 같은 입력, 같은 절차로 Apple Silicon CPU와 Metal 기반 WebGPU에서 한 번 더 측정한다. 이렇게 하면 결과가 특정 기기나 GPU 백엔드에만 해당하는지 확인할 수 있다.

**측정 항목.** Windows와 같은 절차이며, 스크립트 `scripts/paper/run_device_matrix.sh`가 모두 순서대로 실행한다.

| 단계 | 내용 | 논문과의 관계 |
|---|---|---|
| 2 | WebGPU 연산자 배치 (ORT-Web 1.30) | RQ2 원인 |
| 3 | 네이티브 ORT CPU 지연 (1·4 스레드) | RQ2 |
| 4 | WebGPU 지연 (1.30) | RQ2 |
| 4b | WebGPU 1.22의 YOLO26-n FP32·FP16 지연과 배치 | FP16 효과의 런타임 버전 의존성 |
| 5 | WASM 지연 (1·4 스레드) | RQ2 |
| 6 | 브라우저 파이프라인, 10개 배치 1회 | RQ3 |
| 6b | 주요 4개 배치를 브라우저 재기동으로 4번 더 측정(총 5회) | Windows와 같은 반복 프로토콜 |
| 6c | YOLO11l(COCO) 대표 조건: FP32·FP16 WebGPU, INT8 헤드 제외 WebGPU·WASM, CPU 4T, INT8 배치 로그 | 외부 검증 워크로드의 실행 환경 효과 |
| 7 | (선택) WebGPU 수치 일치성 | 정확도가 실행 환경과 무관한지 |

- 정확도는 같은 모델 파일(SHA-256 동일)이라 Windows 결과를 그대로 쓴다. COCO 정확도 측정은 Mac에서 반복하지 않는다.
- 전체 그래프 INT8 검출기는 모든 실행 환경에서 mAP 0이라 배포 후보가 아니므로 제외한다.
- 측정 동안 10초마다 CPU 사용 상위 프로세스를 `cpu_load.log`에 기록한다. Windows에서는 배경 프로세스가 코어 하나를 쓰면 WASM 지연이 20% 넘게 늘었다.

## 1. 준비물

| 항목 | 내용 |
|---|---|
| Mac | Apple Silicon 권장, 전원 어댑터 연결 |
| 브라우저 | Google Chrome 최신판(`/Applications/Google Chrome.app`). Windows 측정은 Chrome 153이다. 가능하면 같은 주 버전을 쓰고, 버전은 결과에 자동 기록된다 |
| Python | 3.10 이상 (`python3 --version`) |
| 저장소 | `main` 최신판에서 만든 측정 전용 브랜치(아래 2절) |
| 측정 번들 | Windows의 `C:\Users\leegy\Desktop\edge_sign_device_bundle` (약 1.8 GB, 2026-09-26 재생성) |

**번들.**
- 구성: `model_space/` 24개 모델(YOLO11l COCO 변형 포함), `data/aihub_traffic/test/images/` test 2,417프레임, `test_manifest.jsonl`, `SHA256SUMS`
- 스크립트가 시작할 때 SHA-256으로 모든 파일을 검증한다. 이전 번들(`edge_sign_device_bundle_old`)에는 COCO 모델이 없으니 쓰지 않는다.
- 번들에는 AI Hub 이미지가 들어 있다(약관상 재배포 금지). USB 메모리·외장 SSD, 또는 **본인만 접근 가능한** 저장소로 옮기고 공개 저장소나 공개 링크에는 올리지 않는다.
- 아래 예시는 Mac의 `~/edge_sign_device_bundle`에 두었다고 가정한다.

## 2. 저장소와 환경 (한 번만)

Mac 결과는 `main`에 바로 커밋하지 않는다. `main` 최신판에서 측정 전용 브랜치를 만든다.

```bash
git clone https://github.com/todayoneul/edge_sign.git   # 이미 있으면 생략
cd edge_sign
git fetch origin
git switch main && git pull --ff-only
git switch -c paper/mac-device-validation              # 측정 전용 브랜치
git rev-parse HEAD                                      # 측정 기준 커밋을 기록해 둔다
python3 -m venv .venv-bench
.venv-bench/bin/pip install numpy "opencv-python-headless==4.13.*" onnx "onnxruntime==1.23.2" psutil
```

- onnxruntime은 Windows와 같은 1.23.2를 쓴다.
- opencv 4.13을 설치할 수 없으면 최신판을 쓰고, `environment.txt`에 기록된 버전을 보고서에 적는다.

## 3. 측정 전 점검

- 전원 어댑터를 연결하고, 시스템 설정 → 배터리에서 **저전력 모드를 끈다**.
- 다른 앱을 모두 종료한다(브라우저, 화상회의, 백업).
- iCloud Drive·Dropbox·OneDrive 동기화를 일시정지하고, Spotlight 색인이 끝났는지 확인한다.
- 활동 모니터에서 CPU가 거의 쉬고 있는지 확인한 뒤 시작한다.
- 측정 중에는 Mac을 쓰지 않는다. 잠자기는 아래 명령의 `caffeinate`가 막는다.

## 4. 빠른 점검 (약 10–15분)

반복 횟수와 프레임 수를 줄여 모든 단계가 도는지만 확인한다. 이 결과는 논문에 쓰지 않는다.

```bash
QUICK=1 PYTHON=.venv-bench/bin/python bash scripts/paper/run_device_matrix.sh ~/edge_sign_device_bundle /tmp/edge_quick
```

- 마지막 줄이 `=== ... device done`이면 성공이다.
- `/tmp/edge_quick/summary.md`에 표가 있는지 확인한다.
- `/tmp/edge_quick/environment.txt`의 `HEADED=`를 확인한다.
  - `no`이면 headless Chrome이 Apple GPU의 WebGPU를 잡은 것이다.
  - `--headed`이면 스크립트가 "headless Chrome has no hardware WebGPU adapter"를 출력하고, 측정마다 작은 Chrome 창을 띄운다. **그 창을 가리거나 최소화하지 않는다.**

## 5. 본측정 (수 시간)

```bash
caffeinate -dimsu env PYTHON=.venv-bench/bin/python \
  bash scripts/paper/run_device_matrix.sh ~/edge_sign_device_bundle paper_evidence/runtime/matrix_mac --with-accuracy \
  2>&1 | tee matrix_mac.log
```

- 중간에 멈추면 **같은 명령을 다시 실행**한다. 이미 끝난 측정은 건너뛴다.
- 실패한 한 건만 다시 재려면 `paper_evidence/runtime/matrix_mac/`에서 해당 `*.json`을 지우고 같은 명령을 다시 실행한다.
- `--with-accuracy`(test 전체 WebGPU 수치 일치성)는 시간이 오래 걸리면 빼도 된다.
- 끝나면 `paper_evidence/runtime/matrix_mac/summary.md`를 먼저 확인한다. 지연표, 배치표, 5회 반복 요약이 들어 있다.
- `cpu_load.log`에서 측정과 무관한 프로세스가 CPU를 오래 쓴 구간이 있으면 해당 측정의 `*.json`을 지우고 다시 실행한다. 어떤 측정을 지웠는지는 기록해 둔다.

## 6. 결과 올리기

측정 전용 브랜치에 커밋하고 푸시한다. **`main`에는 병합하지 않는다.** Windows 쪽에서 결과를 검토한 뒤 논문에 반영하면서 병합한다.

```bash
cp matrix_mac.log paper_evidence/runtime/matrix_mac/
git add -f paper_evidence/runtime/matrix_mac
git commit -m "feat(paper): record the runtime matrix on a Mac"
git push -u origin paper/mac-device-validation
```

`matrix_mac/`에 들어가는 것:
- 지연 trace, 연산자 배치 로그, CPU 부하 기록, 기기 정보(`environment.txt`)
- 프레임별 예측(GT 박스 좌표와 경로만 있고 이미지는 없음)

번들 폴더는 커밋하지 않는다.

## 7. 문제 해결

| 증상 | 조치 |
|---|---|
| `bundle checksum mismatch` | 번들 복사 중 손상되었거나 이전 번들이다. 새 번들을 다시 복사한다 |
| WebGPU 결과의 `status`가 `unsupported`, 사유가 `navigator.gpu unavailable` | Chrome을 최신판으로 올린다. 그래도 같으면 `chrome://gpu`에서 WebGPU 상태를 캡처해 둔다 |
| 모든 브라우저 측정이 `timeout` | 방화벽이 127.0.0.1:8791을 막는지 확인한다. 포트를 바꾸려면 `PORT=8795`를 앞에 붙인다 |
| `crossOriginIsolated`가 false이고 WASM 4스레드 결과의 스레드 수가 1 | 서버가 `--isolate`로 떠 있는지 `server.log`에서 확인한다 |
| YOLO11l WASM 단계가 매우 오래 걸림 | 정상이다(Windows 기준 추론 1회 약 0.7초). 중단했다가 같은 명령으로 이어서 실행해도 된다 |

## 8. 추가 재측정: WASM FP32 대 INT8 (약 1시간, 2026-09-27 추가)

**목적.** 1차 Mac 측정에서 WASM INT8은 FP32보다 빠르지 않았다(0.89–1.01배). Windows와 반대 방향이라 논문 초록에 들어간 결과이지만, 근거가 약하다.
- 모델마다 한 번씩만 측정했다.
- 그 구간에 다른 앱(ChatGPT/Codex, Antigravity IDE, 평소 쓰는 Chrome 창)이 CPU를 썼다.
- 파이프라인의 FP32@WASM도 1회만 측정했다.

이 비교만 조용한 환경에서 반복해 결과를 확정한다. 스크립트는 `scripts/paper/run_mac_wasm_recheck.sh`이다.

| 단계 | 내용 | 시간(대략) |
|---|---|---|
| 1 | YOLO26-n FP32·INT8 헤드 제외, WASM 4스레드, 5회 | 10분 |
| 2 | YOLOv8s FP32·INT8 헤드 제외, WASM 4스레드, 3회 | 20분 |
| 3 | 파이프라인 FP32@WASM 대 INT8@WASM, 5회 | 10분 |
| 4 | WASM 수치 일치성(YOLO26-n FP32, INT8 헤드 제외 2종; test 전체) | 10분 |
| 선택 | `WITH_1T=1`을 붙이면 YOLO26-n 1스레드 3회 추가 | +20분 |

- 측정마다 Chrome을 새로 띄운다. 회차마다 FP32와 INT8의 순서를 바꿔, 발열이나 배경 작업이 한쪽에만 유리하지 않게 한다.
- 측정 전마다 다른 프로세스의 CPU 사용 합이 30% 미만인 상태가 15초 이어질 때까지 기다린다(최대 5분). 기다린 결과는 `quiet.log`에 남는다.
  - 제외하는 것은 스크립트 자신, 벤치마크 서버, 부하 기록 프로세스뿐이다(PID 기준).
  - 평소 쓰는 Chrome 창이나 다른 python도 배경 부하로 센다. 그러니 측정 전에 반드시 닫는다.

**준비.**
1. 전원 어댑터를 연결하고 저전력 모드를 끈다.
2. **모든 앱을 종료한다.** 특히 ChatGPT(Codex), Antigravity IDE, 평소 쓰는 Chrome 창(측정용 Chrome은 스크립트가 따로 띄운다), 다른 터미널의 python 작업. iCloud 동기화도 일시정지한다.
3. 저장소를 최신으로 받는다(새 스크립트 포함).

```bash
cd edge_sign && git switch main && git pull --ff-only
```

4. 번들은 1차와 같은 `~/edge_sign_device_bundle`을 쓴다.

**빠른 점검 (약 5분, 논문에 쓰지 않음).**

```bash
QUICK=1 PYTHON=.venv-bench/bin/python bash scripts/paper/run_mac_wasm_recheck.sh ~/edge_sign_device_bundle /tmp/recheck_quick
```

마지막 줄이 `=== ... recheck done`이고 요약 표가 나오면 된다. 빠른 점검은 32프레임만 쓰므로 일치성 표의 차이(delta)가 크게 나오는데, 정상이다.

**본측정.** 측정 중에는 Mac을 쓰지 않는다. 중간에 멈추면 같은 명령을 다시 실행한다. 끝난 측정은 건너뛴다.

```bash
caffeinate -dimsu env PYTHON=.venv-bench/bin/python bash scripts/paper/run_mac_wasm_recheck.sh ~/edge_sign_device_bundle paper_evidence/runtime/matrix_mac_recheck 2>&1 | tee matrix_mac_recheck.log
```

**결과 확인.** `paper_evidence/runtime/matrix_mac_recheck/summary.md`를 본다.
- 첫 표의 "FP32/INT8 per round"가 모든 회차에서 1보다 작으면, 1차 결과(INT8이 빠르지 않음)가 조용한 환경에서도 재현된 것이다. 1보다 크면 1차 결과는 배경 부하의 영향이었을 수 있다.
- `Quiet gate` 줄의 `not quiet` 개수가 0인지 확인한다.

**결과 올리기.** 예측 파일은 스크립트가 이미 압축해 둔다.

```bash
cp matrix_mac_recheck.log paper_evidence/runtime/matrix_mac_recheck/
git add -f paper_evidence/runtime/matrix_mac_recheck
git commit -m "feat(paper): quiet re-check of WASM FP32 vs INT8 on the Mac"
git push origin main
```
