# Experiment: A2 가지치기·증류 P0 (데이터와 실현 가능성 게이트)

- 날짜: 2026-10-15 · branch `a2-prune-distill` · 계획 `plans/2026-10-15_a2-prune-distill-multitask.md`
- 코드: `experiments/distill/`
- 산출물: `outputs/distill_sets/`, `outputs/distill_g0a/`, `outputs/distill_g0b/`, `outputs/distill_g0c/`, `outputs/distill_refboxes/`, `outputs/distill_p0_detprobe/`
- 캐시: `/mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed/distill_a2/<set>/`

## Purpose

P1(teacher 라벨링)과 P2(가지치기)에 들어가기 전에 다섯 가지를 확인한다.

- **데이터**: 계획대로 학습·평가 세트를 만들 수 있는가.
- **G0a**: 320×576 캐시가 teacher 출력을 바꾸지 않는가.
- **G0b**: 가지치기 slicing 코드가 정확한가.
- **G0c**: 3D obstacle 라벨을 front-wide 2D 참조 박스로 투영할 수 있는가.
- **G0d**: A2가 다객체 2D detection pseudo label을 만들 수 있는가. 이 결과를 보고 detection 라벨 소스(D5)를 사용자가 결정한다.

## Setup

- **Teacher**: `nvidia/Alpamayo2-Super@00554695`, bf16, Blackwell 2장 1쌍(카드당 30/34층), bf16 autocast.
- **driving 시퀀스**: release `meta_action` 과제. 샘플링은 temperature 0.6, top_p 0.98이다.
- **detection 시퀀스**: vqa 메시지 레이아웃, 6cam 4frame grounding profile. 샘플링은 temperature 0.1, top_p 1.0이다.
- **세트 선정 규칙**:
  - train split, LiDAR 있음, A2 6카메라·egomotion·calibration 있음.
  - calib_100과 OOD manifest clip은 제외한다.
  - chunk당 최대 16 clip을 후보로 뽑고, t0 grid(2–13 s, 1 s 간격)에서 bucket을 계산한다.
  - 가장 부족한 bucket부터 채워 clip당 t0 1개를 고르고, chunk당 최대 10 clip으로 제한한다. seed는 20261015.

## Results (Facts)

### 데이터 세트 (`outputs/distill_sets/summary.txt`)

| 세트 | clip | LiDAR | 비고 |
|---|---:|---:|---|
| train_10k | 10,000 | 10,000 | 1,573 chunk(chunk당 최대 10), 5 bucket 각 2,000, 25개국(US 4,926, DE 1,515 등), hyperion_8.1 7,459 / 8 2,541 |
| dev_238 | 238 | 232 | `val_official_238`, t0 5.1 s |
| val_500 | 500 | 482 | t0 5.1 s |
| ood_262 | 262 | 240 | OOD-val |
| calib_100 | 100 | 96 | 가지치기 calib와 G0d |

- 세트 간 clip 중복은 모두 0이다.
- 후보 25,216 clip × 12 t0에서 bucket을 계산했다(302,590 샘플, 오류 2건은 t0가 egomotion 범위를 벗어난 경우).
- 같은 pool로 두 번 실행해 같은 manifest가 나오는 것을 확인했다.

### G0a: 캐시 충실도 (calib 20 clip, `outputs/distill_g0a/`)

| 입력 경로 (기준: 원본 full-res 프레임) | KL(raw‖x) 평균 | top-1 일치 | 같은 seed ADE (m) | minADE@8 |
|---|---:|---:|---:|---:|
| full-res JPEG q95 (기존 oodval·importance 캐시) | 4.16e-4 | 0.9987 | 0.033 | 1.168 |
| 320×576 uint8 무손실 | 1.73e-4 | 0.9986 | 0.020 | 1.170 |
| 320×576 JPEG q95 (첫 캐시) | 1.42e-3 | 0.9941 | 0.064 | 1.161 |

- 원본 입력에서 seed만 바꿨을 때의 ADE는 1.090 m였다(raw minADE@8 1.166).
- 기준은 두 가지다. (i) 같은 seed ADE < 0.5 × seed 노이즈, (ii) KL ≤ 3 × full-res JPEG의 KL.
- JPEG q95 캐시는 (i) 통과, (ii) 실패했다(3.4배).
- 사전에 정한 대체안대로 캐시를 **무손실(WebP lossless, uint8과 픽셀 단위로 같음)**로 다시 만들었다. 무손실은 (i)·(ii) 모두 통과한다.
- 무손실 캐시는 샘플당 3.9 MB이고, 10k 기준 약 39 GB다.

### G0b: slicing 정확성 (`outputs/distill_g0b/`)

- **정적 검사**: 모두 남기는 spec으로 자르면 config가 같고, 층 0/31/63과 층 밖 텐서 433개가 비트 단위로 같다. **PASS**.
- **동치 검사**: 같은 부분을 teacher에서 hook으로 제거한 결과와 비교했다(3 clip, 같은 seed 4회). **PASS**.
  - 자른 spec: L=9(0–2층 유지), KV 4/8, Q 4/group, MLP 1/2, expert Q 1/group, expert MLP 1/2, V′=20,963.
  - 실제 파라미터 수와 공식이 정확히 일치했다(예산 부분 2,376,996,354).
  - top-1 일치 0.990, KL 2.9e-4, 같은 seed 궤적 ADE 0.023 m (seed 노이즈 6.68 m).
- 잔차 폭 d의 slicing은 RMSNorm 때문에 정확한 동치가 없다. 그래서 정적 검사와 shape 검사로만 확인했다.

### G0c: obstacle 투영 (calib_100, `outputs/distill_g0c/`)

- 96 clip이 성공했고 4 clip은 라벨이 없다.
- 투영 박스 1,614개 중 필터를 통과한 박스는 461개(clip당 4.8개)다. 클래스는 vehicle 413, pedestrian 34, cyclist 11, other 3이다.
- 필터: 가시율 ≥ 0.5, 높이 ≥ 12 px@320, 거리 ≤ 80 m.
- 라벨과 frame의 시각 차이는 중앙값 26.6 ms, p99 49.7 ms다(기준 50 ms 미만).
- overlay 20장에서 박스가 객체 위에 놓인다. 다만 가까운 차량에서는 cuboid를 감싼 사각형이라 실제 보이는 영역보다 크다. **PASS**.

### G0d: A2 다객체 detection (calib 96 clip, front-wide t0 frame, `outputs/distill_p0_detprobe/`)

| prompt | 박스/clip | 토큰 | s/clip | P@0.5 | R@0.5 | R@0.3 | R 0–30 m | R 30–60 m | 차량 R | 보행자 R (n=34) | cyclist R (n=11) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1 Qwen JSON | 2.5 | 82 | 11.9 | 0.886 | 0.336 | 0.341 | 0.47 | 0.20 | 0.36 | 0.06 | 0.36 |
| p2 release 스타일 | 2.8 | 94 | 13.5 | 0.904 | 0.369 | 0.373 | 0.49 | 0.25 | 0.40 | 0.06 | 0.36 |
| p3 클래스 목록 | 2.8 | 92 | 12.9 | 0.892 | 0.358 | 0.364 | 0.49 | 0.23 | 0.39 | 0.00 | 0.18 |

- **좌표계**: A2 출력은 0–1000 정규화다. 모델 입력 픽셀(576×320)로 해석하면 모든 prompt에서 매칭이 0이다.
- **파싱**: 288회 중 288회 모두 JSON으로 파싱됐다.
- **예측 박스 구성(p2, 272개)**: 참조 박스와 매칭 170, 필터로 제외된 참조 박스와 겹침 84(작거나 멀거나 잘린 박스), 그 외 18.
- **가림 분석** (`occlusion_analysis.json`, p2): 참조 박스를 깊이 순으로 쌓아 앞쪽 박스에 덮이지 않은 비율(가시율)을 계산했다.

| 참조 박스 조건 | recall@0.5 | n |
|---|---:|---:|
| 전체 | 0.369 | 461 |
| 가시율 ≥ 0.3 | 0.491 | 318 |
| 가시율 ≥ 0.5 | 0.511 | 280 |
| 가시율 ≥ 0.5, 높이 12–30 px | 0.396 | 182 |
| 가시율 ≥ 0.5, 높이 30–60 px | 0.638 | 47 |
| 가시율 ≥ 0.5, 높이 ≥ 60 px | 0.804 | 51 |

- G0d는 무손실 재빌드 전의 JPEG q95 calib 캐시로 실행했다(G0a 기준 KL 1.4e-3 수준 차이).

### 처리량 (Blackwell 1쌍)

| 작업 | 시간 |
|---|---:|
| driving 라벨 (평가 세트, denoise 8회 포함) | 20.7 s/샘플 |
| driving 라벨 (train, 생성만) | 약 6.4 s/샘플 (평균 32 토큰) |
| detection 라벨 (p2) | 13.5 s/clip |
| 캐시 빌드 (32 worker) | 10k에 약 4.5시간 (streaming 속도가 병목) |

## Analysis (Opinion)

- **캐시**: 모델 입력 크기로 미리 줄인 무손실 캐시는 기존 full-res JPEG 캐시보다도 teacher를 덜 흔든다. 이후 모든 단계는 이 캐시를 쓴다.
- **slicing**: 층, KV group, head, MLP, vocab 재매핑, expert 결합까지 정확함을 확인했다. P2는 이 코드를 그대로 쓸 수 있다.
- **A2 detection 특성**: 정밀도는 높지만(~0.9) recall이 낮다(0.37). 놓치는 이유는 두 가지로 보인다.
  - 가린 객체: 참조 박스의 약 31%가 앞 객체에 절반 이상 가려져 있다. 이 박스는 참조 쪽 문제에 가깝다.
  - 작은 객체와 보행자: 가려지지 않은 작은 박스는 recall 0.40, 보행자는 0.06이다.
  - 크고 잘 보이는 객체는 0.80을 찾는다.
  - 즉 A2는 "눈에 띄는 주요 객체" detector처럼 동작한다. A2 라벨만으로 학습한 student는 이 상한을 물려받을 가능성이 높다.
- **obstacle 투영 라벨**: LiDAR 기반 autolabel이라 객체 목록은 더 완전하고 클래스도 있으며 teacher 비용이 들지 않는다. 대신 두 가지 손질이 필요하다.
  - 가린 객체를 걸러야 한다. 박스 z-buffer 근사(가시율 ≥ 0.5)로 할 수 있다.
  - 박스가 cuboid를 감싼 사각형이라 실제 보이는 영역보다 크다.
- **제안**: detection 라벨은 가림 필터를 적용한 obstacle 투영으로 하고, A2 detection은 평가 세트(약 1,000 clip, 약 2시간)에서만 실행해 teacher 대비 성능을 보고한다. 단, 원래 설계(A2 pseudo label)와 달라지므로 결정은 사용자가 한다.
- **진행 상황**: driving 라벨(CoC + meta-action)은 D5와 무관해 P1 라벨링을 이미 시작했다. 평가 세트는 궤적 8개를 포함한다. G0e(vocab coverage)는 teacher 출력이 필요해서 P1 이후에 한다.
