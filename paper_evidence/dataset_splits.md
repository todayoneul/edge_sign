# Dataset split 검증

조사일 2026-09-23. 아래 수량은 **현재 디스크 snapshot**이다. 파일명과 라벨 메타데이터만 읽었으며 원본 영상/이미지를 복사·디코딩·추론하지 않았다. 역사적 실험 당시 manifest가 없어 현재 수량을 과거 실험 수량으로 자동 대입하지 않는다.

## YAML / 생성 코드

- data/yolo_signs/dataset.yaml:1-8: traffic_sign/signboard 2클래스. data/yolo_signs_v2/dataset.yaml:1-8: traffic_sign/traffic_light 2클래스. 두 이름의 v2는 실험 세대 v2와 같지 않다. yolo_signs_v2는 v3/v4용 데이터다.
- **CONFLICT** data/yolo_signs_v1/dataset.yaml:1이 data/yolo_signs를 가리킨다. 보관 폴더를 이 YAML로 재평가하면 현재 v2 분할을 읽을 수 있다.
- **CONFLICT** src/detect/prepare_dataset.py:51-65,78-85는 현재 3클래스(sign/light/signboard)인데 보존 yolo_signs YAML은2클래스다. script 상단 docstring:10-12의 2클래스 설명도 현재 상수와 다르다. 현재 script를 실행해 과거 v2를 그대로 재현할 수 없다.
- scripts/extract_frames.py:150-219는 주/야간별 TAR 크기 내림차순 시퀀스 분할. 난수 seed는 사용하지 않는다. :94-105 default sample_rate6, train_ratio.67, val_ratio.11; :245-270 정렬 JPG를 매6번째 선택하고 기존 파일은 건너뜀. 이는 코드 default이지 과거 실제 실행 인자 증거가 아니다. 시퀀스 비율은 주야 이미지/객체 비율의 균형을 보장하지 않는다.
- src/detect/prepare_dataset.py:154-157 GTSDB image-level shuffle seed42; :223-225/:336-338 --max_images 사용시 seed42. 간판은 :313-316 원래 Training/Validation 분할 유지. traffic은 :210-280 기존 train/val 시퀀스 승계.
- scripts/prepare_korean_traffic.py:130-201 train/val만 처리; sign/light 이외 객체 제외, 양의 유효 박스가 없는 frame 제외. 검출 2클래스(:35-36), ROI14클래스(:39-54), ROI32x32(:33). --max는 정렬된 앞부분만 선택(:145-147), 무작위 seed 없음. 출력 폴더를 비우지 않는 mkdir/copy 구현이므로 재실행 stale 파일 가능성은 배제할 수 없다.

## 현재 YOLO 이미지 수

| 폴더 | train | val | test |
|---|---:|---:|---|
| yolo_signs | 39937 | 7167 | test directory absent; YAML test key absent |
| yolo_signs_v1 | 44696 | 4667 | test directory absent; YAML test key absent |
| yolo_signs_v2 | 12375 | 1903 | test directory absent; YAML test key absent |

## 현재 AI Hub traffic 시퀀스

| Split | 시퀀스 | 주/야 | JPG | JSON |
|---|---|---|---:|---:|
| train | c_validation_1280_720_daylight_1 | 주간 | 5000 | 5000 |
| train | c_validation_1280_720_daylight_2 | 주간 | 5000 | 5000 |
| train | c_validation_1280_720_daylight_3 | 주간 | 1093 | 1093 |
| train | c_validation_1280_720_night_1 | 야간 | 142 | 142 |
| train | c_validation_1920_1200_daylight_1 | 주간 | 2152 | 2152 |
| val | d_validation_1920_1080_daylight_1 | 주간 | 2500 | 2500 |
| val | d_validation_1920_1080_night_1 | 야간 | 184 | 184 |
| test | c_validation_1920_1200_night_1 | 야간 | 16 | 16 |
| test | d_validation_1920_1080_daylight_2 | 주간 | 2401 | 2401 |

현재 시퀀스명 집합 train∩val=train∩test=val∩test=∅. train5(주간4/야간1), val2(각1), test2(각1)로 README.md:327-339와 일치한다. 이는 sequence-name leakage가 없다는 증거이며 동일 픽셀·인접 촬영·중복 scene까지 배제한 증거는 아니다. 모든 원본 이미지 content hashing은 수행하지 않았다.

생성일/실제 sampling rate: 실행 명령 기록이나 immutable split manifest가 없어 **MISSING**. 문서 docs/EXPERIMENTS.md:231은 2026-05-27 최초 준비, README.md:342는 2026-05-30 v2 재측정을 기록(VERIFIED_DOC_ONLY). v1 시퀀스 구성 설명은 docs/ROADMAP.md:34이며 현재 v2 원천 폴더와 섞어 사용하지 않는다.

## 현재 라벨 객체 수 (원시 비어 있지 않은 YOLO TXT 행)

| Dataset | Split | class0 | class1 | 합계 | TXT 파일수 |
|---|---|---:|---:|---:|---:|
| yolo_signs | train | 45416 | 93329 | 138745 | 39937 |
| yolo_signs | val | 9115 | 15171 | 24286 | 7167 |
| yolo_signs_v1 | train | 60152 | 93329 | 153481 | 44696 |
| yolo_signs_v1 | val | 943 | 15171 | 16114 | 4667 |
| yolo_signs_v2 | train | 21637 | 22040 | 43677 | 12375 |
| yolo_signs_v2 | val | 4318 | 4305 | 8623 | 1903 |

각 행 source: `data/<Dataset>/labels/<Split>/*.txt`, 모든 nonempty 행의 첫 class token 집계. 해당 images 폴더 JPG 수와 TXT 수 일치. 이는 디스크 라벨 행 수이며 evaluator가 사용하는 유효/중복제거 instance 수와 다를 수 있다. 실제 `logs/train_v4.log:41`은 `d_validation_1920_1080_daylight_1__14452258.jpg` 중복 라벨1개 제거를 명시; :67077-67079 평가 8622개(sign4317/light4305)와 디스크8623개 차이가 설명된다.

## AI Hub 원본 JSON annotation 수

아래 각 source는 `data/aihub_traffic/<split>/labels/<sequence>/*.json`, `$.annotation[*].class` 전체 항목 집계다. traffic_information을 포함한 합계는 tracking GT와 다르다.

| split | sequence | traffic_sign | traffic_light | traffic_information | total |
|---|---|---:|---:|---:|---:|
| train | c_validation_1280_720_daylight_1 | 8119 | 8994 | 305 | 17418 |
| train | c_validation_1280_720_daylight_2 | 7245 | 6766 | 296 | 14307 |
| train | c_validation_1280_720_daylight_3 | 1871 | 1985 | 39 | 3895 |
| train | c_validation_1280_720_night_1 | 207 | 100 | 8 | 315 |
| train | c_validation_1920_1200_daylight_1 | 4195 | 4204 | 137 | 8536 |
| val | d_validation_1920_1080_daylight_1 | 3984 | 3960 | 228 | 8172 |
| val | d_validation_1920_1080_night_1 | 334 | 345 | 5 | 684 |
| test | c_validation_1920_1200_night_1 | 16 | 17 | 0 | 33 |
| test | d_validation_1920_1080_daylight_2 | 3350 | 3389 | 140 | 6879 |

현재 test는 2417장 중 주간2401/야간16으로 **시퀀스 개수 균형과 프레임 균형은 다르다**. sign+light GT는 6739+33=6772, 시퀀스당 단순 평균3386. 문서3386을 전체 test GT로 적으면 안 된다.

## 세대 사이 overlap 및 leakage 해석

`data/yolo_signs_v1/images/train/`에는 `d_validation_1920_1080_daylight_2__*.jpg` 2401장이 존재한다. 이 시퀀스는 **현재 v2 test**다. 따라서 v1 모델에 v2 test를 사용하여 학습 미사용이라고 주장할 수 없다. 이것은 v2 모델 자체에 leakage가 있다는 뜻이 아니다. v1 train 주간6시퀀스, val 야간1시퀀스인 보존 이미지 구성은 docs/ROADMAP.md:34와 일치한다. v1 당시 test 원본 폴더를 별도 보존한 증거는 확인하지 못했다.

현재 yolo_signs_v2 필터 후 시퀀스별 images: train c1280_daylight1=4845, daylight2=4444, daylight3=991, night1=125, c1920_daylight1=1970; val d1920_daylight1=1754, d1920_night1=149. 원천 시퀀스와 동일 split 유지; 유효 sign/light가 없는 프레임이 제외되어 AI Hub 원천 counts보다 감소했다. source: `data/yolo_signs_v2/images/{train,val}/*.jpg` filename prefix 집계; filtering 코드 scripts/prepare_korean_traffic.py:159-201.

- `yolo_signs` train/val 동일 stem 교집합: 0 (현재 파일명 기준; 내용 hash 미검증).

- `yolo_signs_v1` train/val 동일 stem 교집합: 0 (현재 파일명 기준; 내용 hash 미검증).

- `yolo_signs_v2` train/val 동일 stem 교집합: 0 (현재 파일명 기준; 내용 hash 미검증).
