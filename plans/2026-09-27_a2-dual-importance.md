# Plan: Alpamayo 2 Super의 dual-objective Taylor importance — Finding 1 재현 (2026-09-27)

## 목적

compression 트랙 논문의 Finding 1은 Alpamayo 1.5에서 두 중요도가 **trunk에서는 같은 unit을,
후반층에서는 다른 unit을 고른다**는 관찰이다. `I_traj`는 flow-matching loss 기준이고 `I_CoC`는
CoC NLL 기준이다. 순위 일치도는 0–21층에서 약 0.78, 22–34층에서 약 0.33이고, L22–23에 4.6× step이 있다.
이 구조가 **다른 규모·다른 깊이의 모델(A2, 34B, 64층)**에서도 나타나는지 확인한다.
나타나면 "네트워크의 성질"이라는 논문 주장의 외부 타당성이 생기고, 나타나지 않으면 그 주장의 범위가 A1.5로 좁아진다.

측정 대상은 dense A2 하나다. 프루닝이나 평가는 하지 않는다.

## 재현 대상 — A1.5 기준값

아래 값은 모두 compression 레포의 저장 파일에서 다시 계산해 확인했다.
출처는 `importance_v2`(Fig. 1의 원천, `calib_100`, Blackwell), `stepimp_fm_perstep_v2`, `figures/fig3_why_differs_stats.json`이다.

| 항목 | Q head | MLP channel | 계산 규칙 |
|---|---:|---:|---|
| 층내 순위 일치 ρ, trunk (0–21) | 0.782 | 0.753 | 100클립 평균 점수 두 벡터의 층별 Spearman, 구간 평균 |
| 층내 순위 일치 ρ, late (22–34) | 0.335 | 0.339 | 같은 규칙. L35는 `I_traj` ≡ 0이라 제외 |
| `log(ΣI_traj/ΣI_CoC)` 2-level step fit: τ | 23 | 23 | `fig_why_differs.step_fit`, 층 0–34 |
| step 크기 (early/late) [bootstrap 95% CI] | 4.64× [3.85, 5.47] | 4.56× [3.37, 6.07] | 클립 bootstrap 2,000회 |
| fit R² / bootstrap τ 최빈 비율 / 개별 클립 step 비율 | 0.86 / 1.00 / 1.00 | 0.93 / 0.98 / 0.99 | |
| 천장 보정(disattenuated) late ρ / 천장이 설명하는 하락 몫 | 0.37 / 0.03 | 0.54 / 0.44 | split-half 50회 |
| 깊이 프로파일 peak: `I_traj` vs `I_CoC` | L19 vs L24 | L18 vs L27 | 층 합, 각자 최대로 정규화 |
| 마지막 층 `I_traj` | 정확히 0 | 정확히 0 | 구조적 (L35의 출력은 cache에 닿지 않음) |
| expert: 하위 50% unit의 1차 질량 몫 | 24.96% | 0.44% | 층내 오름차순 누적, 층 평균 |
| expert: 스텝 질량 프로파일 (t = 0.05 … 0.95) | 첫 0.356 · 최소 0.046(step 6) · 끝 0.067 | 첫 0.226 · 최소 0.059(step 6) · 끝 0.149 | 스텝별 `Σ|g_s|`, 합 = 1 |
| **토큰 분해** (`gradanat_v1`, 같은 100클립, Ada): vision 가법 몫 `I_traj` 6–17층 → 27–34층 | 0.72 → 0.35 | 0.90 → 0.26 | 층별 가법 몫 `Σ_u G_type,u·sign(G_full,u)` / 층 점수, 구간 평균 |
| 같은 구간, `I_CoC` | 0.63 → 0.02 | 0.75 → 0.02 | 〃 |
| hand-over 층 (vision 몫이 처음 0.5 아래로): `I_traj` / `I_CoC` | L19 / L17 | L21 / L18 | τ(23)보다 앞 |
| same-token 일치 (late, 천장 보정): pooled → vision → prompt → CoC | 0.38 → 0.79 → 0.81 → 0.41 | 0.54 → 0.72 → 0.74 → 0.38 | `finding1_numbers.py` 표 D |

## 사전 등록 판정 (실행 전 고정)

A2의 층 수는 64이고 `I_traj` ≡ 0인 마지막 층(L63)을 뺀 63개 층으로 계산한다.
A1.5 구간을 상대 깊이로 옮기면 경계 22/36 × 64 = 39.1이 되므로, **trunk는 0–38, late는 39–62**로 둔다.
A1.5의 τ/LAST = 23/35 = 0.657이고, A2의 예측 τ ≈ 0.657 × 63 = 41.4다.

| # | 가설 | "재현"으로 판정하는 조건 | A1.5 자기 판정 |
|---|---|---|---|
| R1 | trunk 일치, late 불일치 | 두 축 모두 trunk ρ ≥ 0.6이고, late ρ ≤ trunk ρ − 0.2 | 통과 (0.78→0.34, 0.75→0.34) |
| R2 | 단일 step이 존재 | 두 축 모두 R² ≥ 0.7, bootstrap step CI 하한 > 1.5×, τ 최빈 비율 ≥ 0.5, 개별 클립 step 비율 ≥ 0.9 | 통과 |
| R3 | step이 같은 상대 깊이 | τ ∈ [36, 46] (상대 깊이 0.657 ± 0.08). 밖이면 "다른 위치"로 보고 | 통과 (23/35) |
| R4 | Q의 late 불일치는 잡음 천장이 아님 | Q disattenuated late ρ ≤ 0.6이고 천장 몫 ≤ 0.25. MLP는 보고만 함 | 통과 (0.37, 0.03) |
| R5 | expert MLP 질량 집중 | expert MLP 하위 50% 몫 < 5%이고 Q > 15% | 통과 (0.44%, 24.96%) |
| R6 | expert 스텝 프로파일 U자 | 축마다 argmin ∈ {2,…,7}이고 첫/최소 ≥ 1.3, 끝/최소 ≥ 1.3 | 통과 (Q 7.68/1.45, MLP 3.85/2.54) |
| R7 | `I_traj` peak이 `I_CoC` peak보다 앞 | 두 축 모두 peak_traj < peak_CoC | 통과 (19<24, 18<27) |
| R8 | 두 점수가 vision → text로 넘어가고, 그 전환이 step보다 앞 | 두 축·두 점수 모두 trunk vision 몫 > 0.5, late vision 몫 < 0.5, hand-over 층 < τ | 통과 (17–21 < 23) |
| R9 | late 불일치는 주로 토큰 구성 차이 | Q late 천장 보정 일치가 vision 위치에서 ≥ 0.70이고 pooled보다 ≥ 0.2 높음 | 통과 (0.79 vs 0.38) |

토큰 분해 구간은 A1.5의 6–17 / 27–34를 상대 깊이로 옮긴 **11–30 / 48–60**으로 둔다.
토큰 타입은 A1.5 `compute_spans`와 같은 규칙을 쓴다.
- vision = image pad 토큰
- hist = 이산 궤적 토큰(`[151669, 155669)`)
- sink = 0번 위치
- CoC = 생성 구간
- prompt text = 나머지(카메라 ID·프레임 번호 라벨, vision start/end 포함)

R2 불통과("step 없음")와 R3 불통과("다른 깊이")는 둘 다 결론이다. 결과 표에 그대로 기록한다.
R3는 상대 깊이 가설만 검정한다. A1.5 리포트의 다른 설명("expert의 cache 포트 위치가 step을 정한다")을
검정하려면 port map이 필요하므로 이번 범위 밖이다. 이 프로파일은 A1.5에서 L14–20에 몰려 포트 위치(21–23)의
대리 지표가 되지 못했다. 그래서 cache gradient 프로파일(`kv_*`)은 기술 통계로만 보고한다.

### 무결성 게이트 (하나라도 실패하면 결과를 해석하지 않고 멈춘다)

| # | 조건 |
|---|---|
| G0a | 모든 클립에서 `traj_vlm_q[63]`, `traj_vlm_mlp[63]`가 정확히 0 (gate·cache 배선 확인) |
| G0b | 스텝별로 기록한 expert signed gradient의 합의 절댓값이, 같은 run에서 누적한 출하 점수와 상대오차 1e-5 이내로 같음 |
| G0c | 분석 코드가 `importance_v2` / `stepimp_fm_perstep_v2`에서 위 A1.5 표를 재현 (결정적 값은 정확히, bootstrap·split 값은 ±0.01) |
| G0d | 3클립을 다른 카드 조합에서 재실행했을 때 층별 Spearman 중앙값 ≥ 0.99 (A1.5 `run_importance`도 bitwise는 아니었음, 유지집합 0.997) |
| G0e | 자기 CoC NLL과 FM loss가 유한하고, CoC 구간이 `<\|cot_end\|><\|traj_future_start\|>`로 끝남을 토큰 단위로 확인 |

**G0e 수정 (스모크 후, 2026-09-27).** A2는 생성 모드에서 prompt가 `assistant\n`으로 끝나고, CoC 텍스트 뒤에 `<|traj_future_start|>` 하나만 낸다. `<|cot_start|>`와 `<|cot_end|>`는 없다(calib 3클립을 토큰 단위로 확인). CoC 구간 규칙은 A1.5와 같은 "생성한 토큰 전부"로 두고, G0e는 "마지막 생성 토큰이 `<|traj_future_start|>`이고 `<|cot_end|>`가 없음"으로 바꾼다. 따라서 NLL이 포함하는 종료 토큰이 A1.5는 2개(`cot_end`, `traj_future_start`), A2는 1개다.

## 설계

### 추정량 — A1.5 `run_importance.py`를 그대로 이식

클립마다 다음을 한다.

1. **롤아웃.** `seed = sha256(f"42:{clip_id}")[:4]`로 no-grad 롤아웃 1회를 한다(temperature 0.6, top_p 0.98, max 256). 결과로 `seq_tf` = prompt + 생성 CoC(`<|traj_future_start|>`까지)를 얻는다.
2. **teacher-forced forward.** gradient를 켜고 `seq_tf`로 VLM forward를 한 번 하며 KV cache를 만든다. 가중치는 고정이다. 곱셈 gate(값 1.0)를 `o_proj` 입력(Q head별)과 `down_proj` 입력(MLP channel별)에 둔다.
3. **CoC NLL.** 생성 위치의 NLL로 backward(`retain_graph`)를 해 `I_CoC`를 얻는다.
4. **trajectory FM.** 두 단계로 한다.
   - expert 10스텝의 FM MSE를 detached cache leaf 위에서 backward한다. `t = (s+0.5)/10`, 스텝마다 새 noise, target `x1 − ε`이고, A2의 학습 규약 `x_t = t·x1 + (1−t)·ε`과 같음을 확인했다.
   - 누적된 cache gradient를 seed로 VLM backward를 한 번 해 `I_traj`를 얻는다.
5. **점수.** `|dL/dg|`를 클립 평균한다. KV 그룹 점수(`|k ⊙ dL/dk|`)도 같이 기록한다.

A2에서 달라지는 것은 다음뿐이다.
- 모듈 경로: `model.expert.expert.layers`, `model.expert.action_in_proj` 등.
- 64층, Q 64 / MLP 25600, expert Q 16 / MLP 6144.
- 카메라 6대 프롬프트(평균 4,580 토큰).
- 다중 GPU 배치: gate를 각 층의 카드에 만든다.

A1.5 라이브러리(`analysis_lib`, `prune_lib`)는 `alpamayo1_5`를 import하므로 가져다 쓸 수 없다.
필요한 함수(`UnitGates`, `coc_nll`, `kv_group_scores`, `expert_fm_grads`, cache get/set, `gt_actions`)를
수식 변경 없이 옮기고, 출처를 docstring에 적는다.

**토큰 분해 (gradient anatomy의 D1).** VLM gate는 A1.5 `TypedUnitGates`와 같은 방식으로 만든다.
(토큰 타입 × unit)마다 gate가 있고 custom autograd Function이라 (T, U) 텐서를 저장하지 않는다.
gradient가 위치에 대한 합이므로 타입별 gradient의 합이 출하 점수와 정확히 같다(A1.5에서 6e-06).
그래서 **한 run에서** 출하 점수와 토큰 분해가 함께 나오고, backward 횟수(CE 1 + FM 1)도 늘지 않는다.
A1.5 anatomy의 D2/D3(cache band·위치 타입별 추가 backward 11회)는 이번 범위 밖이다.
cache band 분해는 R3의 대안 설명(포트 위치)을 검정하는 도구라, 결과를 보고 후속으로 정한다.
FM이 cache를 직접 읽는 양 `|k·dL/dk| + |v·dL/dv|`는 (cache 층 × 위치 타입)으로 추가 비용 없이 기록한다.

**추가 기록 (비용 거의 0).** expert gate gradient를 스텝마다 비우며 signed 값을 기록한다.
그러면 `stepimp_fm_perstep_v2`와 같은 per-step 배열을 얻는다. 출하 점수 `|Σ_s g_s|`는 그 합으로 복원한다(G0b).
VLM의 스텝별 프로파일은 스텝마다 VLM backward가 필요해 10배 비싸므로 제외한다.

### 클립

`calib_100`을 쓴다. A1.5 Fig. 1과 **같은 100클립**(official train, t0 = 5.1 s)이다.
A2용 7-cam 캐시를 HF에서 스트리밍해 만들고(`build_cache_a2.py`에 `--manifest`/`--cache-name` 추가,
기본값 불변), ego 텐서가 기존 `calib` 캐시와 bitwise 같은지 확인한다.

### 메모리 전략 — stage 0에서 측정으로 결정

A1.5 기준으로 보정한 추정치는 다음과 같다.
- 공식은 저장 activation ≈ 층당·토큰당 약 390 KB다. 이 공식으로 A1.5를 계산하면 23 GB가 나오고 실측은 약 20 GB였다.
- A2에 적용하면 4,600 토큰 × 64층 ≈ **110–115 GB**이고, 가중치 67 GB를 더하면 **약 180 GB**다.
- Ada 4장(약 188 GB)으로는 경계선이다.

gradient checkpointing은 쓰지 않는다. transformers가 checkpointing 중에 `use_cache`를 끄고,
재계산이 KV cache에 다시 append하기 때문이다(A1.5 runner의 기록과 같은 이유).

| 전략 | 방법 | 예상 |
|---|---|---|
| S-A | 카드 4장(4–7), offload 없음 | 맞으면 가장 빠름 (추정 25–35 s/clip) |
| S-B | 카드 2장 + `torch.autograd.graph.save_on_cpu` (저장 activation을 host RAM으로) | GPU는 추론 수준, host RAM 프로세스당 약 110 GB, 추정 45–70 s/clip |

결정 규칙은 다음과 같다. 첫 5클립 중 prompt가 가장 긴 클립 1개로 S-B, S-A 순서로 측정한다.
S-A가 모든 카드에 2 GB 이상 여유를 남기면 S-A(replica 1개)로 간다.
아니면 S-B로 간다. host RAM이 허용 한도 안이면 replica 2개(4–5, 6–7), 아니면 1개다.
두 전략이 모두 돌면 같은 클립의 점수를 비교해 G0d에 준하는 일치를 확인한다.

**stage 0–2 결과와 전략 변경 (2026-09-27, 실행 전 기록).**
- S-B(`save_on_cpu`, 2장)는 동작했다. 첫 클립을 뺀 정상 상태가 약 51 s/clip, GPU peak 36.6/39.5 GB였다. 그러나 **host RAM이 프로세스당 227 GB**였다. 원인은 `save_on_cpu`가 가중치(67 GB)까지 복사하는 것이다.
- verify run(243 GB) 중에는 **host가 swap을 쓰기 시작했고**(20 GB), FM backward가 1,033 s로 느려졌다. 공유 서버에서 계속 쓸 방식이 아니다.
- 그래서 **hybrid**(`a2_grad_lib.GpuFirstOffload`)로 바꿨다. 카드 4장(4–7)을 쓰고, 파라미터는 복사하지 않는다. 저장 텐서는 각 카드의 여유(사용 가능량 − 12 GB) 안에서는 GPU에 두고, 넘치는 것만 pinned host RAM으로 보낸다.
- hybrid 측정 결과: 저장 텐서가 GPU에 74.8 GB, host에 48.2 GB 들어갔다. host RSS 70.7 GB, GPU peak 34.2–34.7 GB, 첫 클립 72 s였고 swap은 늘지 않았다.
- S-A(offload 없음)는 저장 텐서 약 123 GB가 카드 여유 약 75 GB를 넘으므로 시험하지 않았다.
- **G0b는 계획의 "같은 run" 조건대로 다시 쟀다.** 스텝마다 비우는 expert gate 옆에, 비우지 않는 두 번째 gate를 같은 pass에 두었다. 두 run 사이의 비교(1e-3)는 비결정성(`I_traj` 자체가 run 간 5e-3–9e-3)과 구별되지 않아 판정에서 제외했다.

### 산출물 (저장소 규칙)

- `outputs/a2_importance/`
  - `config.json`, `metrics.json`(클립별 coc_len, prompt_len, fm_loss, peak GB, s/clip)
  - `importance.npz`, `importance_perclip.npz`: 키가 A1.5와 동일(`{coc,traj}_{vlm_q,vlm_mlp,kv_k,kv_v,exp_q,exp_mlp}`)해서 compression 레포의 논문 스크립트가 그대로 읽을 수 있다. per-clip 약 1.5 GB.
  - `anatomy_perclip_{q,mlp}.npz`: 타입별 signed gradient. 키는 `gradanat_v1`과 동일(`ce`/`fm_full`, `ce_type`/`fm_type`)하다. MLP는 클립당 65 MB, 전체 약 6.5 GB. `direct`는 (클립 × cache 층 × 위치 타입)이다.
  - `step_importance.npz`: expert per-step, 키는 `stepimp_*`와 동일
- `outputs/a2_importance_analysis/`: `metrics.json`, `summary.txt`, `plots/`
  - 상대 깊이 x축에 두 모델을 겹친 그림 4종: 순위 일치, `I_traj/I_CoC` 비율과 step fit, 정규화 깊이 프로파일, 천장
  - expert 스텝 프로파일, expert 질량 집중 곡선
- 리포트: `reports/2026-09-2x_a2-dual-importance.md` (사실/해석 분리, 한국어)

### 구현 파일 (`experiments/importance/`, 신규)

| 파일 | 역할 |
|---|---|
| `a2_grad_lib.py` | A1.5 `prune_lib`/`analysis_lib`에서 옮긴 gate·NLL·FM gradient 함수의 A2판(다중 GPU gate, expert 스텝별 기록), no-grad 롤아웃, teacher-forced forward |
| `run_importance_a2.py` | runner. `--strategy {4gpu,offload}`, `--shard/--n-shards`, `--limit`, 재개 가능, 10클립마다 저장 |
| `analyze_importance_a2.py` | A1.5와 A2에 **같은 함수**로 R1–R7과 G0a–c를 계산해 표와 그림을 만든다 |
| `experiments/oodval/build_cache_a2.py` | `--manifest`/`--cache-name` 인자만 추가 (기존 동작 불변) |

## 단계와 비용

| 단계 | 내용 | 예상 |
|---|---|---|
| 0 | 7-cam `calib_100` 캐시 (스트리밍 16 worker) + ego bitwise 검증 | 약 15–20분 |
| 1 | 구현 + ruff, G0c(분석 코드로 A1.5 표 재현) | 반나절 |
| 2 | 메모리 스모크 1클립 × S-B/S-A → 전략 결정. G0a/G0b/G0e | 약 30분 |
| 3 | G0d: 3클립 재실행 | 약 10분 |
| 4 | 본 실행 100클립. 처음 몇 클립과 GPU·RAM 사용을 확인하고 주기적으로 보고 | 약 1–2시간 |
| 5 | 분석, 그림, 리포트 | 약 1시간 |

## 위험

- **메모리.** 두 전략 모두 안 되면(S-B에서 host RAM 부족 등) 클립 수를 줄이지 않는다. 대신 replica를 1개로 줄이고 시간을 늘린다.
- **공유 서버 host RAM.** S-B는 pinned 메모리를 프로세스당 약 110 GB 쓴다. 현재 가용은 476 GB다. 한도를 넘지 않게 replica 수로 조절한다.
- **MLP 신뢰도.** A1.5도 late MLP의 split-half 신뢰도가 0.46–0.49로 낮았다. A2는 층당 channel이 25,600개라 더 낮을 수 있다. 그래서 R4는 Q에만 판정을 걸었다.
- **모델 간 차이의 교락.** 규모(34B), 깊이(64층), 입력(6캠, 4,580 토큰), 학습 데이터가 동시에 다르다.
  "재현된다 / 안 된다"는 말할 수 있다. 하지만 **무엇 때문에** 다른지는 이 실험 하나로 가를 수 없다.
- **학습 데이터 누수.** A2 학습셋이 클립 단위로 공개되지 않아 `calib_100`(official train)이 A2 학습에 쓰였을 수 있다.
  A1.5도 같은 조건(train 클립)이었으므로 비교 조건은 같다.

## 결정 사항 (2026-09-27 승인)

1. **범위**: Finding 1 핵심 + expert 스텝·질량 집중 + **토큰 타입 분해(D1)**. R8·R9와 토큰 분해 표를 추가했다.
2. **클립**: `calib_100`만.
3. **자원**: Ada 4–7, host RAM 약 250 GB까지(offload 전략이면 replica 2개 가능).
4. 브랜치 `a2-dual-importance`. 커밋과 push는 요청 시에만 한다.

## 결과 (2026-09-27)

`reports/2026-09-27_a2-dual-importance.md`.
- 통과: R1, R2, R4(경계), R5, R6. 불통과: R3(τ = 50/51, 상대 깊이 0.80), R7(Q peak 동률), R9.
- R8은 A1.5 자신이 등록 조건을 넘지 못해 판정에서 제외했다. 계획서의 "A1.5 통과" 표기가 6–17층 값에 근거한 오기였다.
