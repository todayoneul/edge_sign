# 두 번째 기기(Mac) 측정 안내

**목적.** 논문의 "실행 환경별" 주장은 Windows 데스크톱 한 대(Ryzen 5 9600X + RTX 5070)의 측정에 기대고 있다. 같은 모델 파일, 같은 입력, 같은 절차로 다른 CPU(Apple Silicon)와 다른 GPU 백엔드(Metal 기반 WebGPU)에서 한 번 더 측정한다. 이렇게 하면 결과가 특정 기기에만 해당하는지 확인할 수 있다.

측정 항목은 Windows와 같다: WebGPU 연산자 배치, 네이티브 ORT CPU(1·4 스레드), WebGPU·WASM(1·4 스레드) 지연, 브라우저 파이프라인. `--with-accuracy`를 주면 WebGPU 수치 일치성도 잰다. 전체 그래프 INT8 검출기는 모든 실행 환경에서 mAP 0이라 배포 후보가 아니므로 Mac에서는 제외한다.

## 1. 준비물

| 항목 | 내용 |
|---|---|
| Mac | Apple Silicon 권장. 전원 어댑터 연결 |
| 브라우저 | Google Chrome 최신판(`/Applications/Google Chrome.app`) |
| Python | 3.10 이상 (`python3 --version`) |
| 저장소 | `paper/tiis-evidence-revalidation` 브랜치 |
| 측정 번들 | Windows에서 만든 `C:\Users\leegy\Desktop\edge_sign_device_bundle` (약 1.5 GB) |

**번들 옮기기.** 모델 ONNX와 AI Hub 프레임은 git에 없다(용량, AI Hub 약관상 재배포 금지). USB 메모리·외장 SSD, 또는 **본인만 접근 가능한** 클라우드 폴더로 옮긴다. 공개 저장소·공개 링크에 올리지 않는다. 아래 예시는 Mac의 `~/edge_sign_device_bundle`에 두었다고 가정한다.

번들 구성: `model_space/`(18개 모델), `data/aihub_traffic/test/images/`(test 2,417프레임), `test_manifest.jsonl`, `SHA256SUMS`. 번들 안의 파일은 스크립트가 시작할 때 SHA-256으로 모두 검증한다.

## 2. 환경 준비 (한 번만)

```bash
cd ~/edge_sign                       # 저장소 위치에 맞게
git fetch origin
git checkout paper/tiis-evidence-revalidation
git pull
python3 -m venv .venv-bench
.venv-bench/bin/pip install numpy "opencv-python-headless==4.13.*" onnx "onnxruntime==1.23.2" psutil
```

onnxruntime은 Windows 측정과 같은 1.23.2를 쓴다. opencv 4.13을 설치할 수 없으면 최신판을 쓰고, `environment.txt`에 기록된 버전을 보고서에 적는다.

## 3. 측정 전 점검

- 다른 앱을 모두 종료한다(특히 브라우저, 화상회의, 백업, Spotlight 색인).
- 시스템 설정 → 배터리에서 저전력 모드를 끈다.
- 측정 중에는 Mac을 쓰지 않는다. 잠자기는 아래 명령의 `caffeinate`가 막는다.

## 4. 빠른 점검 (약 5–10분)

반복 횟수와 프레임 수를 줄여 모든 단계가 도는지만 확인한다. 결과는 논문에 쓰지 않는다.

```bash
QUICK=1 PYTHON=.venv-bench/bin/python bash scripts/paper/run_device_matrix.sh ~/edge_sign_device_bundle /tmp/edge_quick
```

- 마지막 줄이 `=== ... device done`이면 성공이다.
- `/tmp/edge_quick/summary.md`에 표가 있는지 확인한다.
- `/tmp/edge_quick/environment.txt`의 `HEADED=`를 확인한다. `no`이면 headless Chrome이 GPU를 잡은 것이다. `--headed`이면 측정마다 작은 Chrome 창이 뜨는데, **그 창을 가리거나 최소화하지 않는다.**

## 5. 본측정 (수 시간)

```bash
caffeinate -dimsu env PYTHON=.venv-bench/bin/python \
  bash scripts/paper/run_device_matrix.sh ~/edge_sign_device_bundle paper_evidence/runtime/matrix_mac --with-accuracy \
  2>&1 | tee matrix_mac.log
```

- 중간에 멈추면 **같은 명령을 다시 실행**한다. 이미 끝난 측정은 건너뛴다.
- 실패한 한 건만 다시 재려면 `paper_evidence/runtime/matrix_mac/`에서 해당 `*.json`을 지우고 같은 명령을 다시 실행한다.
- `--with-accuracy`(test 전체 WebGPU 수치 일치성)는 시간이 오래 걸리면 빼도 된다. 나머지 결과만으로도 기기 간 비교는 가능하다.

## 6. 결과 올리기

```bash
git add -f paper_evidence/runtime/matrix_mac
git commit -m "feat(paper): record the runtime matrix on a Mac"
git push
```

`matrix_mac/`에는 지연 trace, 연산자 배치 로그, 프레임별 예측(GT 박스 좌표와 경로만 있고 이미지는 없음), 기기 정보가 들어간다. 번들 폴더는 커밋하지 않는다.

Windows 쪽에서는 이 결과로 기기별 지연 비교표와 그림을 만들어 논문 4장에 넣는다.

## 7. 문제 해결

| 증상 | 조치 |
|---|---|
| `bundle checksum mismatch` | 번들 복사 중 손상. 다시 복사한다 |
| WebGPU 결과의 `status`가 `unsupported`, 사유가 `navigator.gpu unavailable` | Chrome을 최신판으로 올린다. 그래도 같으면 `chrome://gpu`에서 WebGPU 상태를 캡처해 둔다 |
| 모든 브라우저 측정이 `timeout` | 방화벽이 127.0.0.1:8791을 막는지 확인한다. 포트를 바꾸려면 `PORT=8795`를 앞에 붙인다 |
| `crossOriginIsolated`가 false이고 WASM 4스레드 결과의 스레드 수가 1 | 서버가 `--isolate`로 떠 있는지 `server.log`에서 확인한다 |
