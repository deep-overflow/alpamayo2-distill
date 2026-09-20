# Experiment: Alpamayo 2 Super vs Alpamayo 1.5 — OOD-val 262 open-loop

## Purpose

- Alpamayo 2 Super(34B = Qwen3-VL-32B + 2B expert)를 compression 트랙의 primary 롱테일 셋인
  **OOD-val 262 클립**에서 open-loop로 평가하고, 디스크에 이미 있는 Alpamayo-1.5-10B baseline
  (`baseline_pred_oodval`)과 클립 단위로 페어드 비교한다.
- 질문: 롱테일 주행에서 A2가 A1.5보다 (궤적 기준으로) 얼마나 나은가, 차이가 있다면 어디서 오는가
  (bucket / event cluster / horizon / 샘플 분산), 실패는 어떻게 생겼는가.
- 증류 트랙의 **teacher 기준선 측정**이며 방법 비교가 아니다.

## Setup

| 항목 | 값 |
|---|---|
| 모델 | `nvidia/Alpamayo2-Super` @ `00554695e729` (bf16, 71.6 GB) vs `nvidia/Alpamayo-1.5-10B` @ `7aba8293c099` (저장된 rows) |
| 데이터 | `eval_sets/ood_val.parquet`, 262 클립, t0 = 클립별 `event_start_timestamp`, GT = `ego_future_xyz` 64 step / 6.4 s |
| 입력 | A2: 카메라 6대 `[0,1,2,3,5,6]` × 4 프레임 (prompt 평균 4,580 tok). A1.5: 카메라 4대 `[0,1,2,6]` × 4 프레임 |
| 프레임 캐시 | A2용으로 7-cam ring을 HF에서 재스트리밍해 `pre_processed/ood_a2` 구축 (262/262, 2.64 GB, JPEG q95 = 기존 캐시와 동일). ego 텐서 262/262 bitwise 동일, 공유 카메라 프레임 픽셀 동일 |
| 프로토콜 | `run_baseline.py`와 동일 추정량: **CoC 롤아웃 1회**(temp 0.6, top_p 0.98, max 256 tok) → **그 CoC의 KV 캐시 위에서 K=8 디노이징**(seed = sha256 clip seed + k, batch 1) |
| 리덕션 | **minADE@6 / minFDE@6 primary** (8개 중 앞 6개, 표에 쓰는 리덕션), @8·@1·horizon 1.6/3.2/6.4 s 병기 |
| Diffusion | 각 모델의 기본값 (A2: euler 10 step) |
| 하드웨어 | RTX 5880 Ada 48 GB × 4 (GPU 4–7), replica 2개 × 카드 2장. 레이어 분할 [30, 34], expert layer i를 VLM layer i와 같은 카드에 배치. peak 33.6 / 36.2 GB |
| 비용 | A2 16.6 s/clip (생성 5.3 s + 디노이징 9.0 s), shard당 36 min. A1.5는 24 GB 카드 1장 |
| GT-CoC teacher-forced 조건 | 이번 패스에서 제외 (결정 사항) |

### 검증 게이트 (모두 통과)

| 게이트 | 결과 |
|---|---|
| 결정성: 같은 3클립을 다른 카드(6,7)에서 재실행 | CoC 문장 동일, max\|Δwaypoint\| = 0.0 |
| 배치 불변성: 3-GPU 분할 [19,22,23] vs 2-GPU 분할 [30,34] | CoC 문장 동일, max\|Δwaypoint\| = 0.0 |
| **릴리스 API parity**: 재구현한 샘플링 루프 vs `sample_trajectories_from_data` (같은 시드 주입) | CoC 문장 동일, max\|Δwaypoint\| = 0.00e+00 m (`check_native_parity.py`) |
| 캐시: ego 텐서 vs 기존 4-cam 캐시 | 262/262 차이 0 |
| 실행 중 | NaN 0, 에러 0, `<traj_future_start>` 미생성 0/262, CoC degenerate 0/262 |

## Results (Facts)

delta = A2 − A1.5, **음수면 A2가 GT에 더 가깝다**. CI는 paired bootstrap 10k (평균), p는 Wilcoxon.
n = 262.

### Overall

| metric | A1.5 mean / median | A2 mean / median | Δmean [95% CI] | Δmedian | A2 승률 | p |
|---|---|---|---|---:|---:|---:|
| **minADE@6** | 0.9995 / 0.5745 | 1.0351 / 0.5806 | **+0.036 [−0.037, +0.109]** | +0.035 | 45.0% | 0.100 |
| **minFDE@6** | 2.4959 / 1.1184 | 2.7075 / 1.2620 | +0.212 [−0.015, +0.436] | +0.142 | 41.2% | 0.019 |
| minADE@8 | 0.9214 / 0.5272 | 0.9440 / 0.5567 | +0.023 [−0.047, +0.093] | +0.013 | 47.3% | 0.216 |
| minFDE@8 | 2.2256 / 0.9836 | 2.2999 / 1.0277 | +0.074 [−0.138, +0.287] | +0.125 | 40.8% | 0.067 |
| ADE@1 (sample 0) | 2.0897 / 1.4440 | 2.1724 / 1.5048 | +0.083 [−0.049, +0.224] | +0.080 | 43.9% | 0.130 |
| avgADE (8개 평균) | 2.1125 / 1.6357 | 2.2300 / 1.6734 | **+0.118 [+0.031, +0.206]** | +0.084 | 40.5% | 0.002 |

### Horizon (minADE@6)

| horizon | A1.5 mean | A2 mean | Δmean [95% CI] |
|---|---:|---:|---|
| 1.6 s | 0.0650 | 0.0654 | +0.0004 [−0.004, +0.005] |
| 3.2 s | 0.2585 | 0.2625 | +0.004 [−0.015, +0.025] |
| 6.4 s | 0.9995 | 1.0351 | +0.036 [−0.037, +0.109] |

### Scenario bucket (minADE@6)

| bucket | n | A1.5 mean / median | A2 mean / median | Δmean [95% CI] | A2 승률 |
|---|---:|---|---|---|---:|
| cruise | 140 | 0.804 / 0.532 | 0.882 / 0.540 | +0.078 [−0.008, +0.172] | 45.7% |
| decel_stop | 51 | 1.327 / 0.430 | 1.219 / 0.450 | −0.108 [−0.286, +0.051] | 47.1% |
| accel | 42 | 0.985 / 0.670 | 1.115 / 0.755 | +0.130 [−0.084, +0.326] | 40.5% |
| turn | 29 | 1.388 / 1.030 | 1.334 / 0.851 | −0.054 [−0.293, +0.185] | 44.8% |

### OOD event cluster (minADE@6, n ≥ 9만)

| cluster | n | A1.5 mean | A2 mean | Δmean [95% CI] | A2 승률 |
|---|---:|---:|---:|---|---:|
| WORK_ZONES_TEMP_TRAFFIC_CONTROL | 118 | 0.874 | 0.952 | +0.078 [−0.010, +0.169] | 42.4% |
| PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY | 52 | 1.202 | 1.182 | −0.020 [−0.166, +0.114] | 50.0% |
| SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR | 50 | 1.023 | 1.122 | +0.099 [−0.134, +0.335] | 40.0% |
| COMPLEX_INTERSECTION_INTERACTION | 14 | 1.073 | 0.895 | −0.178 [−0.509, +0.124] | 57.1% |
| OTHER_LONGTAIL | 11 | 1.294 | 1.146 | −0.149 [−0.547, +0.151] | 54.5% |
| CYCLISTS_AND_MICROMOBILITY_COMPLEX | 9 | 1.060 | 0.916 | −0.143 [−0.638, +0.389] | 55.6% |

(n ≤ 5인 3개 cluster는 `metrics.json`에만 기록.)

### 샘플 분산과 종방향 오차

| 항목 | A1.5 mean | A2 mean | Δmean [95% CI] | p |
|---|---:|---:|---|---:|
| endpoint spread (8개 끝점의 RMS 반경, m) | 3.80 | 4.20 | **+0.40 [+0.23, +0.59]** | 4e-9 |
| 진행거리 std over 8 (m) | 3.61 | 3.90 | **+0.29 [+0.18, +0.41]** | 3e-8 |
| 진행거리 bias, 8개 평균 (모델 − 사람, m) | +0.41 | +0.72 | +0.32 [−0.03, +0.67] | 0.079 |
| best-of-6 샘플의 \|진행거리 오차\| (m) | 2.51 | 2.67 | +0.16 [−0.08, +0.40] | 0.095 |

참고: best-of-6 샘플의 \|진행거리 오차\| 평균(2.51 / 2.67 m)은 minFDE@6 평균(2.50 / 2.71 m)과 거의 같은 크기다.

### Tail (minADE@6 초과 비율)

| 임계 | A1.5 | A2 |
|---|---:|---:|
| > 1 m | 30.9% | 28.2% |
| > 2 m | 10.7% | 13.7% |
| > 4 m | 2.7% | 2.7% |

### CoC

| | A1.5 | A2 |
|---|---:|---:|
| degenerate rate | 0.0% | 0.0% |
| 평균 토큰 / 문자 수 | 17.1 / 77 | 15.1 / 71 |
| 두 모델의 CoC 문장이 완전히 같은 클립 | 0.8% (2/262) | |
| 상위 head 2단어 | keep distance 72, nudge to 37, nudge left 33 | nudge left 46, keep distance 40, slow down 18 |

reference CoC의 어휘(steer left 40, steer right 34, decelerate to 30 …)는 두 모델의 어휘와 달라
"첫 단어 일치율"(A1.5 6.9%, A2 11.1%)은 추론 품질 지표로 쓸 수 없다. judge 없이는 CoC 품질을 비교하지 않았다.

### NVIDIA 모델 카드 수치 (같은 1,434 "challenging" 샘플, 벤더 보고)

| | A1.5-10B | A2 Super |
|---|---:|---:|
| open-loop minADE_6 @ 6.4 s | 0.916 m | 0.911 m |
| closed-loop AlpaSim score (913 scenarios) | 1.37 ± 0.10 | 1.50 ± 0.13 |
| LingoQA Lingo-Judge | (카드에 없음) | 79.2 |

### 산출물

- `outputs/a2_oodval/`: `config.json`, `metrics.json`, `summary.txt`, rows(`alpamayo2_s{0,1}of2.json`, `pred_xy_k` 포함)
- `outputs/a2_oodval/plots/`: `scatter_minade6.png`, `ecdf_minade6.png`, `by_bucket.png`, `by_cluster.png`, `by_k_and_horizon.png`
- `outputs/a2_oodval/viz/all/<clip_id>.png` 262장 + `viz/index.md` (A2 worst 10, A1.5 worst 10, A2 최대 이득/손실 10)

## Analysis (Opinion)

- **OOD-val open-loop 궤적 정확도에서 A2 Super는 A1.5와 구분되지 않는다.** minADE@6 Δ = +0.036 m,
  CI [−0.037, +0.109]가 0을 포함하고 폭도 좁다. SAMPLING.md 기준 이 셋 크기의 MDE가 ~0.06–0.1 m이므로
  "차이가 있더라도 ±0.1 m 안"이라고 읽는 것이 정확하다. 3.4배 큰 모델이 open-loop에서 이득을 주지 않는다.
- 이것은 파이프라인 아티팩트가 아닐 가능성이 높다. (1) 릴리스 API와 bit 단위 parity, 배치 불변, 결정성이
  모두 확인됐고, (2) 1.6 s ADE가 두 모델에서 0.065 m로 같아 ego 상태·입력 조건화가 정상이며,
  (3) **NVIDIA 자신의 카드 수치도 open-loop는 0.916 vs 0.911 m로 동률**이다. 벤더가 주장하는 A2의 이득은
  closed-loop(AlpaSim 1.37 → 1.50)와 텍스트 과제 쪽이다.
- 방향성은 오히려 A2가 약간 불리하다: avgADE(+0.12 m, CI가 0 배제)와 minFDE@6(Wilcoxon p=0.019)에서
  A2가 나쁘고, 8개 샘플의 끝점 분산이 A2 쪽이 10% 크다(+0.40 m). 즉 A2의 expert는 같은 CoC 아래에서
  더 넓게 뿌리는데, 그 넓이가 best-of-6을 개선하지 못했다. 평균적인 단일 샘플은 A1.5가 GT에 더 가깝다.
- **오차는 두 모델 모두 종방향(속도 프로파일)이 지배한다.** best 샘플의 진행거리 오차가 FDE와 같은 크기다.
  클립별 그림에서도 승패는 "추론이 맞았는가"보다 "6개 샘플 중 하나가 사람의 감속/가속 정도를 맞췄는가"로
  갈린다 (예: `049a4497` — 두 모델 모두 CoC가 reference와 다르지만 A1.5의 한 샘플이 우연히 정지 거리를 맞춤;
  `682fed39` — A1.5의 CoC가 의미상 맞는데 궤적은 감속하지 않음). CoC–궤적 결합이 두 모델 모두 느슨하다.
- bucket/cluster별로는 어떤 CI도 0을 배제하지 않는다. decel_stop·turn·교차로·자전거 쪽에서 A2가,
  cruise·accel·work zone 쪽에서 A1.5가 나은 방향이지만 표본이 작아 주장할 수 없다.
- **증류 관점의 함의**: open-loop minADE를 teacher 품질의 대리 지표로 쓰면 A2를 teacher로 쓸 근거가
  이 셋에서는 나오지 않는다. A2가 실제로 나은 축(closed-loop, 추론/VQA)을 증류 목표와 평가 지표로 삼아야
  teacher 교체가 의미를 갖는다. 반대로 "A2의 궤적 분포를 A1.5 크기 student로 옮기기"는 open-loop 기준으로는
  얻을 것이 거의 없다.

### 한계

- A2는 카메라 6대, A1.5는 4대를 본다 (각 모델의 입력 계약; 제거 불가한 차이).
- 모델당 클립당 CoC 1회 추출이라 클립 단위 차이는 노이즈가 크다. 집계 페어드 통계만 읽을 것.
- A2의 학습 데이터가 클립 단위로 공개돼 있지 않아 OOD-val(공식 val) 누수를 배제할 수 없다.
  누수가 있었다면 A2에 유리한 방향인데도 이득이 없었다.
- CoC 품질은 비교하지 않았다 (GT-CoC teacher-forced 조건과 judge 모두 제외).
- A2-native 추정량(독립 CoC 8개)은 돌리지 않았다. CoC 다양성이 minADE@K를 낮추므로 그 설정에서는
  A2 수치가 더 낮게 나올 것이고, 공정 비교하려면 A1.5도 같은 방식으로 재실행해야 한다.

### Suggested next steps

1. **Closed-loop(AlpaSim)에서 A2 vs A1.5** — 벤더가 주장하는 이득이 있는 곳. compression 레포의 alpasim
   파이프라인 재사용 가능 여부(6-cam 렌더, 34B 서빙 비용)부터 확인.
2. **GT-CoC teacher-forced 조건 추가** (`minADE_tf`, `nll_gtcoc`) — "올바른 추론이 주어졌을 때 궤적"과
   추론 NLL로 A2의 추론 쪽 이득을 분리. rows가 클립 단위로 재개 가능해 롤아웃 재실행은 불필요.
3. **CoC judge** — (gt_coc, A1.5 gen_coc, A2 gen_coc) 262쌍이 이미 저장돼 있어 GPU 없이 가능.
4. A2-native 추정량을 두 모델 모두에 돌려 "추정량만으로 수치가 얼마나 움직이는가"를 고정.
