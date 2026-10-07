# Experiment: Alpamayo 2 Super의 dual-objective Taylor importance — Finding 1 재현

## Purpose

- compression 트랙 논문의 Finding 1이 **다른 규모·깊이의 모델**(Alpamayo 2 Super, 34B, 64층)에서도 나타나는지 검정한다.
  - Finding 1의 내용: Alpamayo 1.5에서 `I_traj`(flow-matching loss)와 `I_CoC`(CoC NLL)의 중요도는 trunk에서 같은 unit을 고르고 후반층에서 갈라진다. 그 사이에 한 번의 step(L22–23, 4.6×)이 있다.
- 토큰 타입 분해(gradient anatomy D1)도 함께 재현한다. 1.5에서는 두 점수가 vision → text 위치로 넘어갔고, 같은 위치로 제한하면 후반층 불일치가 대부분 사라졌다.
- 판정 기준 R1–R9와 무결성 게이트 G0a–e는 실행 전에 `plans/2026-09-27_a2-dual-importance.md`에 등록했다.

## Setup

| 항목 | 값 |
|---|---|
| 모델 | `nvidia/Alpamayo2-Super` @ `00554695e729` (VLM Qwen3-VL-32B 64층, Q 64 / MLP 25,600; expert 64층, Q 16 / MLP 6,144) |
| 비교 대상 | A1.5 기존 run: `importance_v2`(Fig. 1 원천), `gradanat_v1`(토큰 분해), `stepimp_fm_perstep_v2`(expert 스텝) |
| 클립 | `calib_100`, A1.5와 **같은 100클립**(official train, t0 = 5.1 s). A2용 7-cam 캐시를 새로 만들었고, ego 텐서는 기존 캐시와 100/100 bitwise 동일 |
| 입력 | A2: 카메라 6대 × 4프레임, prompt 4,580 토큰(vision 4,320) / A1.5: 카메라 4대, prompt 3,086 토큰 |
| 추정량 | A1.5 `run_importance.py`와 동일. 클립 시드 롤아웃 1회 → teacher-forced VLM forward → CoC NLL backward → expert 10스텝 FM(`t=(s+0.5)/10`, 스텝마다 새 noise, target `x1−ε`) → cache gradient로 VLM backward 1회. VLM gate는 `TypedUnitGates`(토큰 타입별), expert gate는 스텝별 기록 |
| 토큰 타입 | vision = image pad, hist = 이산 궤적 토큰, sink = 0번 위치, CoC = 생성 구간 전부, prompt text = 나머지 |
| 깊이 구간 | A1.5 구간을 상대 깊이로 옮김. trunk 0–38 / late 39–62(1.5: 0–21 / 22–34), 토큰 구간 11–30 / 48–60(1.5: 6–17 / 27–34). 마지막 층(`I_traj` ≡ 0)은 제외 |
| 하드웨어 | RTX 5880 Ada × 4 (GPU 4–7), replica 1개. 저장 텐서는 카드 여유 안에서 GPU에 두고 넘치는 것만 pinned host RAM으로 보냄(`GpuFirstOffload`) |
| 비용 | 100클립 50.9분(평균 29.6 s/clip). GPU peak 34.4 GB, host RSS 최대 59.4 GB, swap 증가 없음 |
| 코드 | `experiments/importance/{a2_grad_lib,run_importance_a2,analyze_importance_a2}.py` |

## Results (Facts)

### 무결성 게이트

| 게이트 | 결과 |
|---|---|
| G0a: 마지막 층 `I_traj` = 0 | A1.5·A2 모두 Q, MLP 최댓값 0.0 |
| G0b: expert 스텝별 합 = 누적 점수 (같은 pass) | Q 0.0, MLP 0.0 |
| G0c: 분석 코드가 A1.5 공개 수치 재현 | 34/34 |
| G0d: 다른 카드 쌍에서 3클립 재실행 | CoC 3/3 동일, 층별 Spearman 중앙값 ≥ 0.9986 (bitwise는 아님) |
| G0e: CoC 끝 토큰, NLL/FM 유한 | 100/100 |
| typed gate 항등식 (Σ types = plain gate) | 상대오차 2.0e-06 / 1.8e-06 (A1.5 6e-06) |
| 배치 불변성 (4-GPU hybrid vs 2-GPU offload) | CoC 3/3 동일, Spearman 중앙값 ≥ 0.998 |

**게이트 운영 중 달라진 점** (계획서에 실행 전 기록)
- **G0e 기준 수정.** A2는 생성 모드에서 `<|cot_end|>` 없이 `<|traj_future_start|>` 하나로 CoC를 닫는다. 그래서 G0e를 "마지막 생성 토큰이 `<|traj_future_start|>`"로 바꿨다. NLL에 들어가는 종료 토큰은 A1.5가 2개, A2가 1개다.
- **G0b 재측정.** 서로 다른 두 run을 비교하면 run 사이 비결정성(1e-3)과 구별되지 않았다. 그래서 같은 pass에 누적 gate와 스텝별 gate를 동시에 두고 다시 쟀다.
- **메모리 전략 변경.** `save_on_cpu`는 가중치까지 복사해 host RAM을 227 GB 썼고, 공유 서버가 swap을 쓰기 시작했다. 그래서 hybrid 전략으로 바꿨다.

### 사전 등록 판정

| # | 가설 | A1.5 | A2 | A2 값 |
|---|---|:-:|:-:|---|
| R1 | trunk 일치, late 불일치 | 통과 | **통과** | Q 0.873 → 0.537, MLP 0.687 → 0.378 |
| R2 | 단일 step 존재 | 통과 | **통과** | R² 0.72 / 0.81, step CI 하한 2.77× / 2.95×, τ 최빈 비율 0.58 / 1.00, step 있는 클립 98% / 99% |
| R3 | step이 같은 상대 깊이 (τ ∈ [36, 46]) | 통과 | **불통과** | τ = 51 (Q) / 50 (MLP), 상대 깊이 0.81 / 0.79 (1.5: 0.66) |
| R4 | Q late 불일치는 잡음 천장이 아님 | 통과 | **통과** (경계) | 천장 보정 late ρ 0.584 (기준 ≤ 0.6), 천장 몫 0.145 |
| R5 | expert MLP 질량 집중 | 통과 | **통과** | 하위 50% unit 몫: MLP 0.39%, Q 22.6% |
| R6 | expert 스텝 프로파일 U자 | 통과 | **통과** | argmin Q 7 / MLP 6, 첫/최소 4.83 / 2.43, 끝/최소 2.72 / 7.44 |
| R7 | `I_traj` peak이 `I_CoC`보다 앞 | 통과 | **불통과** | Q 45 = 45 (동률), MLP 45 < 54 |
| R8 | vision → text 전환이 step보다 앞 | **불통과** | 불통과 | 아래 주석 |
| R9 | late 불일치는 주로 토큰 구성 차이 | 통과 | **불통과** | Q vision 위치 보정 일치 0.714(≥ 0.70 충족), pooled 0.584보다 0.13 높음(≥ 0.2 미충족) |

**R8은 잘못 등록됐다.** 계획서에는 "A1.5 자기 판정: 통과"라고 적었지만, 이는 6–17층 값을 보고 쓴 것이다.
등록한 조건은 trunk(0–21층) vision 몫 > 0.5였고, 이 구간으로 계산하면 A1.5 Q `I_CoC`가 0.47로 조건을 넘지 못한다.
A1.5 자신이 떨어지는 기준이므로 R8은 재현 판정으로 읽을 수 없다. 아래 토큰 분해 표를 기술 통계로 대신 본다.

### 핵심 수치 (A1.5 → A2)

| 항목 | A1.5 Q | A2 Q | A1.5 MLP | A2 MLP |
|---|---:|---:|---:|---:|
| 순위 일치 ρ, trunk | 0.782 | 0.873 | 0.753 | 0.687 |
| 순위 일치 ρ, late | 0.335 | 0.537 | 0.339 | 0.378 |
| split-half 자기 일치 late: `I_traj` / `I_CoC` | 0.82 / 0.89 | 0.85 / 0.88 | 0.46 / 0.50 | 0.58 / 0.47 |
| 천장 보정 late ρ | 0.375 | 0.584 | 0.539 | 0.544 |
| step τ (상대 깊이) | 23 (0.66) | 51 (0.81) | 23 (0.66) | 50 (0.79) |
| step 크기 [95% CI] | 4.64× [3.85, 5.47] | 3.06× [2.77, 3.35] | 4.56× [3.37, 6.07] | 3.29× [2.95, 3.69] |
| 깊이 peak `I_traj` / `I_CoC` | L19 / L24 | L45 / L45 | L18 / L27 | L45 / L54 |
| 유지집합 겹침(keep ≈ 60%), late / chance | 0.676 / 0.594 | 0.753 / 0.594 | 0.688 / 0.601 | 0.700 / 0.601 |

### 토큰 분해 — 층 점수 중 vision 위치에서 온 가법 몫 (구간 평균)

| 점수 | 축 | A1.5 trunk → late | A2 trunk → late | A1.5 lo → hi (6–17 → 27–34) | A2 lo → hi (11–30 → 48–60) | hand-over 층 A1.5 / A2 |
|---|---|---|---|---|---|---|
| `I_traj` | Q | 0.58 → 0.34 | 0.47 → 0.55 | 0.72 → 0.35 | 0.50 → 0.50 | L19 / 없음 |
| `I_CoC` | Q | 0.47 → 0.04 | 0.69 → 0.45 | 0.63 → 0.02 | 0.73 → 0.35 | L17 / L55 |
| `I_traj` | MLP | 0.74 → 0.28 | 0.62 → 0.52 | 0.90 → 0.26 | 0.68 → 0.48 | L21 / 없음 |
| `I_CoC` | MLP | 0.59 → 0.04 | 0.90 → 0.77 | 0.75 → 0.02 | 0.92 → 0.75 | L18 / 없음 |

같은 위치로 제한한 late 일치(천장 보정, Q). 두 모델 모두 이 구간의 전 층을 썼다(A1.5 13/13층, A2 24/24층).

| 위치 | A1.5 | A2 |
|---|---:|---:|
| pooled (출하 점수) | 0.38 | 0.58 |
| vision | 0.79 | 0.71 |
| prompt text | 0.81 | 0.73 |
| 생성 CoC | 0.41 | 0.56 |
| ego-history | 0.94 | 0.92 |

### expert

| 항목 | A1.5 | A2 |
|---|---|---|
| 하위 50% unit의 1차 질량 몫: Q / MLP | 24.96% / 0.44% | 22.61% / 0.39% |
| 스텝 질량 Q (t = 0.05 … 0.95) | 0.356, 0.137, …, 최소 0.046, …, 0.067 | 0.250, 0.137, …, 최소 0.052, …, 0.141 |
| 스텝 질량 MLP | 0.226, 0.113, …, 최소 0.059, …, 0.149 | 0.117, 0.085, …, 최소 0.048, …, **0.359** |

### 사후 관찰 (사전 등록 아님)

FM loss가 expert 쪽에서 cache를 직접 읽는 양 `|k·dL/dk| + |v·dL/dv|`의 cache 층별 분포(추가 비용 없이 기록한 기술 통계)다.

| | A1.5 | A2 |
|---|---|---|
| 직접 읽기가 가장 큰 cache 층 | 22, 21, 26, 23 | 50, 49, 47 |
| 누적 50% 지점 층 (상대 깊이) | 24 (0.69) | 51 (0.81) |
| step 직전 세 층의 몫 (균등 분포면) | cache 21–23: 23.3% (8.3%) | cache 48–50: 19.9% (4.7%) |
| `I_traj` 층 합이 가장 가파르게 떨어지는 곳 (l ≥ 3, Q) | L21 → L23 (2.90×) | L49 → L51 (4.26×) |

A1.5 쪽 최댓값 21–22는 A1.5 port map이 **독립적으로** 잰 주 포트 21–23(`2026-09-20_gradient-anatomy`)과 겹친다.
반면 계획 단계에서 확인한 VLM 쪽 cache 점수(`kv_*`)는 L14–20에 몰려 포트의 대리 지표가 되지 못했다.

### 산출물

- `outputs/a2_importance_v1/`
  - `importance.npz`, `importance_perclip.npz`: compression 레포 키와 동일
  - `anatomy_perclip_{q,mlp}.npz`, `anatomy_direct.npz`, `step_importance*.npz`, `metrics.json`, `config_s0of1.json`, `perclip/`
- `outputs/a2_importance_analysis/`
  - `metrics.json`, `summary.txt`, `config.json`
  - `plots/`: rank_agreement, ratio_step, depth_traj, depth_coc, ceiling, token_share_traj, token_share_coc, same_token, expert_steps, expert_concentration

## Analysis (Opinion)

**재현되는 것: Finding 1의 골격.**
- 두 중요도는 A2에서도 trunk에서 같은 unit을 고르고(Q ρ 0.87), 후반에서 갈라지며(0.54), 그 사이에 단일 step이 있다(98–99% 클립, CI 하한 2.8×).
- Q의 후반 불일치는 여전히 잡음 천장으로 설명되지 않는다. 다만 판정 여유가 0.016으로 얇다.
- expert 쪽 두 성질은 거의 그대로 재현된다. MLP 1차 질량이 극도로 몰려 있고(0.39% vs 0.44%), 스텝 프로파일이 U자다.
- 따라서 "trunk 공유 + 후반 분기 + 한 번의 전환"이라는 서술은 규모 3.4배, 깊이 1.8배의 다른 모델로 옮겨 가도 유지된다.

**재현되지 않는 것: 위치와 토큰 서사.**
- step이 상대 깊이 0.66이 아니라 0.80에 있다(R3).
- A2의 후반층 중요도는 여전히 **vision 위치에서 상당 부분** 나온다. `I_CoC` MLP late vision 몫이 0.77로, A1.5의 0.04와 크게 다르다. 그래서 "두 점수가 17–24층에서 vision → text로 넘어간다"는 1.5의 서술과, "후반층은 기능적으로 언어 채널"이라는 읽기는 A2로 일반화되지 않는다.
- same-token 설명(R9)이 약해진 것도 같은 맥락이다. A2는 pooled 후반 불일치 자체가 덜 심해서(0.58), 토큰을 맞춰 회복할 몫이 작다.

**step 위치를 정하는 것은 상대 깊이가 아니라 expert가 cache를 읽는 곳으로 보인다.**
- 두 모델 모두 step 직전 세 층에 expert의 직접 cache 읽기가 균등 분포 대비 3–4배 몰려 있다.
- `I_traj`의 급락도 바로 그 층에서 일어난다(A1.5 L21→23, A2 L49→51).
- 이는 A1.5 리포트의 구조 설명과 맞는다. 그 설명은 "층 l의 unit은 m > l인 cache에만 쓸 수 있으므로, 주 포트 위의 unit은 `I_traj`를 거의 받지 못한다"는 것이다.
- 다만 이것은 사후 관찰이고, A2 port map(cache band별 seed backward)으로 검정한 결과가 아니다.
- 논문에서 쓸 수 있는 형태로 줄이면: **"변화점은 고정된 깊이가 아니라 action expert의 주 cache 판독 위치에 생긴다"**는 주장이다. A2가 그 사례를 하나 더하지만, 확정하려면 아래 1번이 필요하다.

**주의할 교락.**
- 두 모델은 규모, 깊이, 입력(6캠, 4,580 토큰), 학습 데이터, expert 폭(1536 vs 2048)이 동시에 다르다.
- 그래서 A2의 늦은 step과 후반 vision 의존이 **무엇 때문인지**는 이 실험으로 가를 수 없다.
- Q의 τ는 bootstrap 최빈 비율이 0.58로, A1.5(1.00)보다 덜 안정적이다. MLP(1.00)와 한 층 차이라 결론은 같다.

**증류 관점 (의견).**
- A2를 teacher로 쓸 때 두 목적이 갈라지는 구간은 L50 이후 약 13개 층이다. expert는 cache 47–50을 가장 많이 읽는다.
- 그래서 student의 cache를 teacher에 맞출 층을 고를 때, 균등 간격보다 이 판독 구간에 가중을 두는 설계를 검토할 만하다.

### Suggested next steps

1. **A2 port map** (gradient anatomy D2, cache band별 seed VLM backward).
   - "τ는 주 포트 위치"를 사전 등록 예측(τ ≈ 50)으로 검정한다. 클립당 backward가 약 6회 늘어나므로, 20–30클립이면 충분할 것이다.
2. **무작위 readout probe** (causal validation D).
   - 내용 없는 readout을 expert 문으로 흘려도 A2에서 같은 층(≈ 50)에 step이 생기는지 본다. 생긴다면 "출력 인터페이스의 성질"이 A2에서도 성립한다.
3. **후반층 기능 검정** (held-out band mask).
   - A2 후반층이 A1.5처럼 "언어 채널"(FM 불변, NLL만 변화)인지, vision 의존을 반영해 주행에도 관여하는지 확인한다.
4. **held-out 재현** (val100 / OOD100).
   - A1.5와 같이 다른 클립에서도 τ와 R1–R6이 유지되는지 확인한다.
