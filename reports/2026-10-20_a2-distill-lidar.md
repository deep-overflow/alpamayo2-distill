# Experiment: 증류 student에 LiDAR(range image) 입력 추가

- 날짜: 2026-10-20 · branch `a2-distill-lidar` · 계획 `plans/2026-10-20_a2-distill-lidar-range.md`
- 코드: `experiments/distill/` (`build_lidar_distill.py`, `lidar_lib.py`, `check_lidar_alignment.py`, `check_lidar_wiring.py`, `train_student.py --lidar`, `eval_student.py`, `analyze_p4.py`, `analyze_p4_pairs.py`, `diag_traj_forcing.py`)
- 산출물:
  - 게이트: `outputs/distill_lidar_l0a/`, `outputs/distill_lidar_l0b/`
  - 학습: `outputs/distill_s2_wide_final/`(카메라), `outputs/distill_s2_wide_lidar/`(LiDAR)
  - 평가: `outputs/distill_eval/<tag>/<set>/`, `outputs/distill_p4/`, `outputs/distill_diag_forcing/`
- 가중치: `/mnt/nvme1n1/ad_vla/outputs/chan/a2_distill/s2_wide/{distill_s2_wide_final,distill_s2_wide_lidar}/best`
- LiDAR 캐시: `/mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed/distill_a2_lidar/<set>/`

## Purpose

- **가설**: 카메라 전용 student(0.99B, ViT 0.6B + 나머지 0.47B)에 LiDAR range image 토큰만 추가하면 세 과제(궤적, reasoning, 2D detection)가 좋아진다.
- **비교 원칙**: 입력 외의 조건은 모두 같다. 시작 모델(`s2_wide/init`), 데이터(train 10k), A2 라벨, 손실, 학습 일정, 평가 코드가 같다.
- 카메라 student의 최종 P4 평가(A2, A1.5 대비)도 함께 정리한다.

## Setup

### 모델

| | 카메라 student `cam_s2` | LiDAR student `lidar_s2` |
|---|---|---|
| 구조 | C-wide: 14층, d 1280, 16Q/4KV, MLP 5632, expert d 512 | 같음 + RangeViT |
| 파라미터 | 0.991B (학습 0.575B) | 1.006B (학습 0.590B) |
| 추가 모듈 | — | RangeViT 15.3M: patch 8×16 → 16×64, 6층, d 384, 6 head, 2×2 merger → 256 토큰 → MLP(1280) |
| 입력 | 6카메라 × 4프레임 | 같음 + `LiDAR: <\|vision_start\|>` + 256 토큰 + `<\|vision_end\|>` (마지막 카메라 이미지 뒤, 프롬프트 +258 토큰) |

**LiDAR 입력**
- range image 4 × 128 × 1024.
  - 채널: log range, intensity, rig z, 유효 mask.
  - 행: 빔 0–127(위→아래).
  - 열: rig@t0 기준 방위각, 전방이 0°.
- **시간 정렬**
  - t0 이전에 끝난 마지막 회전만 쓴다(assert). t0에 가장 가까운 회전은 t0 이후 +20…+78 ms까지 측정되어 미래 정보가 섞이기 때문이다.
  - 점마다 egomotion으로 rig@t0 좌표계로 deskew한다.
- **positional embedding**
  - encoder 덧셈형: 방위각 sin/cos(k=1..8), 빔의 실제 고도각, patch별 Δt를 각각 다중 주파수로 넣고 MLP에 통과시킨다.
  - attention: 고도 축 RoPE와 방위 축 cyclic RoPE(열 64 = 열 0).
  - LLM: Qwen MRoPE video 격자 (1, 16, 64)를 쓴다.
  - Δt 임베딩은 사용자가 제안한 FPS 대응이다(카메라 30 fps, LiDAR 10 Hz).

### 학습 (두 모델 공통)

- 시작점 `s2_wide/init`(S1 2.81B → C-wide slicing). RangeViT는 무작위로 초기화한다.
- **데이터**
  - train 10k clip(전부 LiDAR 있음), 과제당 시퀀스 하나씩(driving, det).
  - dev: 카메라 238 clip, LiDAR 232 clip(LiDAR 캐시가 있는 clip만).
- **손실**: CE + top-64 KL(A2 logit) + expert flow matching(GT 미래, n_fm 4), 가중치 모두 1.
- **일정**: 최대 10 epoch(3,120 step), lr 1e-4, wd 0.05, warmup 100, global batch 32, patience 5(dev 평가는 100 step마다).
- **하드웨어**: Blackwell 4장(GPU 0–3), FSDP2.
- **두 실행의 차이 (계획 §7)**
  - LiDAR 실행부터 dev flow 손실의 노이즈를 clip별 seed로 고정했다. 카메라 실행은 무작위 노이즈였다.
  - 따라서 dev `fm`과 조기 종료 시점은 정확히 같은 조건이 아니다.

### 평가 (P4)

- **세트**: val_500(in-distribution, t0 5.1 s), ood_262(OOD-val). LiDAR student는 LiDAR가 있는 clip만 평가한다(val 482, ood 240).
- **student 추론**
  - free-running: CoC + meta-action을 T 0.6 / top-p 0.98로 `<|traj_future_start|>`까지 생성한다.
  - 궤적: expert denoise로 K=8개를 만든다.
  - detection: JSON을 T 0.1로 생성한다.
- **teacher**: A2의 같은 설정 출력(`label_teacher.py`, denoise K=8). A1.5는 궤적만 비교한다(`baseline_pred_*`).
- **지표**
  - 궤적: minADE@1/6/8, minFDE@8, avgADE (GT 대비, m).
  - detection: front-wide 투영 obstacle 박스 대비 P/R(IoU 0.5). 참조 박스가 있는 clip만 쓴다(val 436, ood 232).
  - reasoning: A2 대비 meta-action(종·횡·차선) 세 항목 일치율, CoC ROUGE-L.
- **paired 비교**: 두 student가 모두 가진 clip에서 clip별 차이를 구하고 bootstrap 95% CI를 붙인다(2,000회).

## Results (Facts)

### 게이트

| 게이트 | 결과 |
|---|---|
| L0a 인과성 | 사용한 회전은 모두 t0보다 22–81 ms 먼저 끝남. 점별 시각 ≤ t0 assert 통과 |
| L0a deskew | obstacle 3D 박스 안 점: raw 239,092 → deskew 250,939 (+5.0%), 12/12 clip에서 증가. 점 이동량 중앙값 0.82 m, 최대 13.5 m |
| L0b 연결 | LiDAR 없을 때 logit 차이 0.0. range 토큰 256개가 연속이고 격자는 t 고정, h 0–7, w 0–31. 뒤 텍스트 위치가 이어짐. RangeViT 모든 부분에 gradient 흐름. cyclic RoPE 열 64 = 열 0, 이음매 차이 / 이웃 차이 중앙값 = 1.00 |
| G0e vocab | prompt OOV 0 (평가 시퀀스 전부 `remap` assert 통과). report 세트 A2 출력의 student vocab 밖 토큰: val 2 / 64,787 (3.1e-5), ood 2 / 38,176 (5.2e-5) |

LiDAR 캐시 (`distill_a2_lidar/<set>/summary.json`)
- 샘플 수: train 10,000 (9.8 GB, 오류 0), dev 232, val 482, ood 240.
- range image 칸 중 유효 칸은 평균 86.5%이고, 회전당 점은 평균 30.4만 개다.
- train에서 회전은 t0보다 0–100 ms 먼저 끝난다(중앙값 47 ms).
- 가장 오래된 점은 t0보다 98–199 ms 이르다(평균 147 ms). 이 값이 Δt 임베딩의 범위다.

### 학습

| | 카메라 | LiDAR |
|---|---:|---:|
| 종료 | step 2200 (7.05 epoch), 조기 종료 | step 2000 (6.41 epoch), 조기 종료 |
| 시간 | 9.95 h | 10.34 h |
| 속도 | 16.3 s/step | 17.4 s/step |
| best step | 1700 | 1500 |
| dev loss (best) | 2.005 | 2.011 |
| ce_drv / kl_drv | 0.326 / 0.188 | 0.322 / 0.186 |
| ce_det / kl_det | 0.438 / 0.205 | 0.428 / 0.186 |
| fm | 0.848 (무작위 노이즈) | 0.889 (고정 노이즈) |

dev 곡선: `outputs/distill_p4/dev_curves_lidar_s2_vs_cam_s2.png`. 모든 손실 항목에서 두 곡선이 거의 겹친다.

### 궤적 (`outputs/distill_p4/summary.txt`)

| 세트 | 모델 | n | minADE@1 | minADE@6 | minADE@8 | minFDE@8 | avgADE |
|---|---|---:|---:|---:|---:|---:|---:|
| val_500 | A1.5 | 500 | 1.760 | 0.824 | 0.691 | 1.840 | 1.789 |
| val_500 | A2 | 500 | 2.017 | 1.158 | 1.082 | 3.213 | 1.975 |
| val_500 | cam_s2 | 500 | 3.653 | 2.385 | 2.257 | 6.344 | 3.726 |
| val_500 | lidar_s2 | 482 | 3.754 | 2.475 | 2.338 | 6.364 | 3.840 |
| ood_262 | A1.5 | 262 | 2.090 | 1.000 | 0.921 | 2.226 | 2.113 |
| ood_262 | A2 | 262 | 2.162 | 1.253 | 1.147 | 3.200 | 2.229 |
| ood_262 | cam_s2 | 262 | 4.554 | 3.039 | 2.874 | 7.546 | 4.475 |
| ood_262 | lidar_s2 | 240 | 4.710 | 3.139 | 2.966 | 7.714 | 4.587 |

paired 차이 (m, 95% CI):

| 비교 | 세트 | n | minADE@1 | minADE@6 | minFDE@8 |
|---|---|---:|---|---|---|
| cam_s2 − A2 | val | 500 | +1.636 [+1.340, +1.912] | +1.226 [+1.003, +1.439] | +3.131 [+2.403, +3.815] |
| cam_s2 − A2 | ood | 262 | +2.392 [+1.974, +2.806] | +1.786 [+1.439, +2.110] | +4.346 [+3.381, +5.245] |
| lidar_s2 − cam_s2 | val | 482 | +0.119 [−0.073, +0.313] | +0.100 [−0.064, +0.258] | +0.026 [−0.479, +0.516] |
| lidar_s2 − cam_s2 | ood | 240 | +0.218 [−0.228, +0.751] | +0.109 [−0.161, +0.421] | +0.169 [−0.668, +1.023] |

### detection (참조 박스 대비, IoU 0.5)

| 세트 | 모델 | P | R | 박스/clip | R 0–30 m | R 30–60 m | R 60 m+ | R vehicle | 종료율 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val (436 clip) | A2 | 0.888 | 0.359 | 2.13 | 0.441 | 0.261 | 0.136 | 0.452 | 1.000 |
| val | cam_s2 | 0.455 | 0.209 | 2.42 | 0.287 | 0.111 | 0.030 | 0.263 | 0.992 |
| val | lidar_s2 | 0.411 | 0.218 | 2.79 | 0.300 | 0.116 | 0.030 | 0.273 | 0.979 |
| ood (232 clip) | A2 | 0.911 | 0.311 | 2.68 | 0.385 | 0.195 | 0.222 | 0.426 | 1.000 |
| ood | cam_s2 | 0.492 | 0.179 | 2.84 | 0.231 | 0.099 | 0.037 | 0.255 | 0.989 |
| ood | lidar_s2 | 0.398 | 0.177 | 3.49 | 0.232 | 0.093 | 0.037 | 0.249 | 0.967 |

- 종료율은 최대 길이 전에 `<|im_end|>`로 끝난 비율이다.
- 보행자 recall은 세 모델 모두 0.01 수준이다(val 참조 460개).

### reasoning과 A2 일치도 (같은 clip, `outputs/distill_p4/pairs_lidar_s2_vs_cam_s2.json`)

| 세트 | 모델 | det F1 vs A2 | meta-action 3항목 일치 | CoC ROUGE-L |
|---|---|---:|---:|---:|
| val (482) | cam_s2 | 0.412 | 0.305 | 0.518 |
| val | lidar_s2 | 0.403 | 0.307 | 0.522 |
| ood (240) | cam_s2 | 0.415 | 0.267 | 0.424 |
| ood | lidar_s2 | 0.370 | 0.200 | 0.460 |

lidar_s2 − cam_s2, clip별 paired 차이 (95% CI):

| 항목 | val | ood |
|---|---|---|
| meta-action 3항목 일치 | +0.002 [−0.046, +0.048] | **−0.067 [−0.125, −0.008]** |
| CoC ROUGE-L | +0.004 [−0.021, +0.030] | **+0.036 [+0.006, +0.066]** |
| detection TP / clip | +0.048 [−0.030, +0.128] | −0.013 [−0.142, +0.112] |
| detection FP / clip | **+0.330 [+0.179, +0.500]** | **+0.659 [+0.422, +0.918]** |

### 궤적 격차 진단 (cam_s2, val 100 clip, `outputs/distill_diag_forcing/`)

| 입력 CoC | minADE@6 | minFDE@8 |
|---|---:|---:|
| student 자신의 CoC (free) | 2.231 | 5.968 |
| A2의 CoC를 강제 입력 (forced) | 1.966 | 5.001 |
| A2 자체 (teacher) | 1.499 | 4.047 |

- forced − free = −0.265 [−0.470, −0.074]
- forced − teacher = +0.468 [+0.014, +0.891]

### 지연 (샘플당 초, batch 1, HF eager, 단일 GPU)

| 세트 | 모델 | prefill | CoC 생성 | denoise | detection |
|---|---|---:|---:|---:|---:|
| val | cam_s2 | 2.70 | 0.65 | 0.24 | 4.74 |
| val | lidar_s2 | 2.41 | 0.57 | 0.21 | 4.49 |
| ood | cam_s2 | 3.64 | 1.01 | 0.33 | 6.84 |
| ood | lidar_s2 | 2.16 | 0.61 | 0.21 | 4.65 |

같은 모델도 세트마다 지연이 1.3–1.5배 달라졌다. 다른 작업과 GPU를 나눠 쓴 시점에 측정했다.

## Analysis (Opinion)

**1. LiDAR 입력은 이번 설정에서 이득이 없다.**
- 궤적: 모든 지표에서 lidar_s2가 약간 나쁘지만(+0.10 m minADE@6) CI가 0을 포함한다. 효과가 없거나 측정할 수 없을 만큼 작다.
- dev 손실(teacher-forced)도 거의 같다(2.011 vs 2.005). 가설은 지지되지 않는다.

**2. 주된 원인은 라벨이 LiDAR 정보를 요구하지 않는다는 점으로 본다.**
- reasoning과 detection 라벨은 카메라만 본 A2의 출력이다. LiDAR가 정답을 더 잘 맞히게 해도 "A2를 흉내 내는" 손실은 줄지 않는다.
- 궤적만 GT 라벨이다. 그런데 진단 결과 student 궤적의 약점은 지각보다 다른 데 있다.
  - 격차의 약 1/3은 student가 만든 CoC 품질에서 온다(forced − free = −0.27 m).
  - 나머지 2/3은 축소된 expert/VLM 특징에서 온다(forced − teacher = +0.47 m).
  - 0.47B 예산에서는 장면 기하보다 계획 용량이 병목일 가능성이 높다.
- 새 encoder(15M, 무작위 초기화)가 6 epoch 안에 쓸모 있는 특징을 배우기 어려웠을 수도 있다. LiDAR 경로에 사전학습이나 보조 손실이 없다.

**3. detection에서 LiDAR student가 박스를 더 많이 내고 정밀도가 떨어진다.**
- 박스 수: val 2.79 vs 2.42, ood 3.49 vs 2.84. FP가 유의하게 늘었고 TP는 같다(P 0.455 → 0.411, ood 0.492 → 0.398).
- 종료율도 낮다(0.979 vs 0.992). 박스 목록을 끝내지 못하고 반복하는 경우가 늘어난 것으로 보인다.
- teacher-forced dev ce_det는 오히려 낮다(0.428 vs 0.438). 라벨을 맞추는 능력과 자유 생성 시 멈추는 행동이 따로 노는 것이다.
- 원인 후보: 프롬프트가 258 토큰 길어진 것, LiDAR가 보이는 물체(A2 라벨에 없는 먼 물체나 가려진 물체)를 더 말하게 된 것. 확인하지는 않았다.

**4. ood에서는 섞인 신호가 나온다.**
- meta-action 일치는 −6.7%p로 유의하게 낮고, ROUGE-L은 +3.6%p로 유의하게 높다.
- 두 항목 모두 n 240에 다중 비교 보정을 하지 않았다. 방향도 서로 반대라 LiDAR의 실제 효과라고 보기 어렵다. 우연이나 학습 변동일 가능성을 배제할 수 없다.
- seed 반복이 없어서 학습 변동 폭을 모른다.

**5. 카메라 student 자체 (P3/P4 정리)**
- A2 대비 궤적: minADE@6이 val +1.23 m, ood +1.79 m 나쁘다.
- A2 대비 detection: recall이 0.21 vs 0.36이고 precision이 0.46 vs 0.89다. 먼 물체(30 m+)에서 특히 약하다.
- A2와의 reasoning 일치: meta 3항목 일치 0.30, ROUGE-L 0.52.
- 파라미터는 A2의 2.8%(0.99B / 35.8B)다. 다만 지연은 최적화하지 않은 HF 경로라 아직 의미 있는 비교가 아니다.

**제안하는 다음 단계** (효과가 큰 것부터)
1. **LiDAR가 필요한 라벨을 쓴다.** detection 라벨을 투영 obstacle GT(3D 박스 투영, 이미 `distill_refboxes`로 만들 수 있음)로 바꾸거나 섞는다. 그러면 LiDAR의 이점을 직접 잴 수 있다. 원래 G0c에서 준비한 경로다.
2. **LiDAR 경로를 사전학습한다.** range image → BEV 점유나 거리 예측 같은 보조 손실을 붙이거나, LiDAR 토큰만 먼저 정렬하는 짧은 단계를 둔다.
3. **궤적 병목을 해결한다.** 이것이 LiDAR보다 먼저다. expert 폭(d 512)을 늘리고 그만큼 VLM MLP를 줄이는 예산 재분배를 검토한다. student CoC의 exposure bias도 줄인다(scheduled sampling, 자기 생성 CoC로 expert 미세조정).
4. **재현성**: 두 student에 seed를 2개씩 더 돌려 학습 변동 폭을 확인한다. 각 약 10시간, GPU 0–3.
