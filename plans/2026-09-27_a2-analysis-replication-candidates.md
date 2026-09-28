# Alpamayo 1.5 분석 → Alpamayo 2 재현 후보 정리 (2026-09-27)

출처: `alpamayo-model-compression`의 `CLAUDE.md`(79a278b, 2026-09-21), `paper/2026-08-26_contributions.md`,
`paper/2026-09-22_dual-expert-method-narrative.md`, `plans/2026-09-24_token-share-replication.md`,
`reports/{archive,evaluation}/`의 분석 리포트 요약부. A1.5 수치는 리포트에서 그대로 옮겼고,
OOD-val 수치 몇 개는 저장된 rows(`baseline_pred_oodval`, `outputs/a2_oodval`)에서 다시 계산했다.
실행 계획서가 아니라 후보 목록이다. 고른 항목은 별도 계획서에서 게이트를 사전 등록한다.

## 0. 비교 전제 — 두 모델의 구조 차이

| | Alpamayo 1.5-10B | Alpamayo 2 Super-34B | 분석에 주는 영향 |
|---|---|---|---|
| VLM text | 36층, hidden 4096, Q 32 / KV 8, MLP 12288 | **64층**, hidden 5120, **Q 64** / KV 8, MLP 25600 | 층 번호는 **상대 깊이(l/N)**로 비교. KV 그룹당 Q가 4 → 8 |
| expert | 36층, hidden 2048, Q 16, MLP 8256 | **64층**, hidden **1536**, Q 16, MLP 6144 | expert가 상대적으로 얇다(파라미터 비중 20.6% → 약 7%) |
| 인터페이스 | expert 층 i가 VLM cache 층 i를 읽음, head_dim 128 × KV 8 | 동일 | "마지막 층은 FM에 안 보인다"는 구조 논증이 그대로 적용됨 |
| 입력 | 카메라 4대 × 4프레임, prompt 3,086 tok (vision 2,880) | 카메라 **6대**(후방 좌·우 추가) × 4프레임, prompt 평균 4,580 tok, 카메라 ID·프레임 번호 텍스트 라벨 | span 계산을 새로 짜야 함. 카메라별 분석이 6개로 늘어남 |
| 출력 | CoC 1문장 → 궤적. meta-action 토큰 **없음**(CoC 머리 어절로 대체) | CoC → (선택) **meta-action 블록**(Longitudinal / Lateral / Lane) → 궤적. VQA·grounding 과제도 있음 | 추론–행동 일관성을 실제 meta-action으로 잴 수 있음 |
| diffusion | flow matching, euler 10 step | 동일 | 스텝 축 분석이 그대로 이식됨 |
| 하드웨어 | 48 GB 카드 1장, gradient pass peak 42 GB | 추론만 해도 48 GB 카드 2장(peak 33.6 / 36.2 GB), 16.6 s/clip | **gradient 분석은 3–4장 또는 A100 80 GB 2장이 필요할 것으로 추정** (weights 67 GB + activation 60–100 GB 추정, 스모크로 확인 필요) |

### A1.5 랜드마크를 A2 깊이로 옮긴 사전 예측 (×64/36 ≈ 1.78)

| A1.5 관찰 | A1.5 층 | 상대 깊이 | A2에서 기대하는 층(상대 깊이 가설) |
|---|---|---|---|
| `I_traj`–`I_CoC` 순위 변화점, 4.6× step | L22–23 | 0.61–0.64 | L39–41 |
| attention sink band (sink head 181개 전부) | L7–16 | 0.19–0.44 | L12–28 |
| 행동 채널 / 언어 채널 (pathway stage 2) | L0–17 / L18–35 | 0–0.49 / 0.5–1 | L0–30 / L32–63 |
| gradient의 vision → text 전환 | L17–24 | 0.47–0.67 | L30–43 |
| expert의 주 cache 포트 (`I_traj`의 35–41%) | cache 21–23 | 0.58–0.64 | cache 37–41 |
| FM에 안 보이는 층 (`I_traj` ≡ 0) | L35 | 마지막 | **L63 (구조적으로 확정)** |

A2가 가장 날카로운 검증이 되는 지점이 있다. gradient anatomy 리포트는 변화점을 "VLM 배선의 ramp ×
expert의 cache 21–23 포트 집중"으로 설명한다. 그렇다면 A2의 변화점은 상대 깊이가 아니라 **A2 expert의
포트가 몰리는 곳**에 생겨야 한다. 두 가설이 다른 층을 가리키면 A2가 둘을 가른다.

## 1. 후보 목록

비용은 A2 기준 추정이다. 추론은 2-GPU replica 1개가 rollout 5.3 s, 디노이징 샘플당 1.1 s이고, GPU 4–7에서 replica 2개를 쓴다.

### Tier 0 — GPU 불필요, 이미 저장된 rows로 바로 가능

| # | 분석 (A1.5 출처) | A1.5 결과 (재현 대상) | A2에서 묻는 것 | 비용 |
|---|---|---|---|---|
| 0-1 | CoC–행동 일관성 (`coc_action_consistency.py`, 2026-09-08) | CoC가 한 문장 템플릿이고 머리 어절이 곧 meta-action(1,000 rollout에서 36종, 상위 15종이 90.1%). 저장된 waypoint가 없어 GT bucket으로 대체했으므로 "추론 정확도"만 잼 | 두 모델 모두 `pred_xy_k`가 저장돼 있어 **정확한 self-consistency**(CoC 머리 ↔ 자기 궤적의 bucket)를 잴 수 있다. OOD-val 머리 어휘는 A1.5 26종(상위 15종 93.5%), A2 39종(81.3%)으로 A2가 더 다양함 | 수 시간(코드), GPU 0 |
| 0-2 | 샘플 전형성 (selftraj anchor, 2026-09-10) | 단일 draw의 GT 거리 1.76 m vs minADE@6 0.824 m, 8 draw 범위 2.65 m | 이미 일부 확인: A2 endpoint spread +0.40 m, avgADE +0.12 m. horizon별·bucket별로 확장 | 1시간 |
| 0-3 | 상황별 실패 (`analyze_failure_by_situation.py`) | arm별 실패가 감속·정지·회전 같은 상황에 몰리는지 | A2와 A1.5의 실패 상황 분포가 같은가 | 1시간 |

### Tier 1 — 추론만 필요(gradient 없음), 논문 C2(기전) 주장과 직결, 비용 낮음

| # | 분석 (A1.5 출처) | A1.5 결과 (재현 대상) | A2에서 묻는 것 | A2 비용 | 포팅 |
|---|---|---|---|---|---|
| 1-1 | **Pathway map stage 1**: expert ← cache span knockout (2026-08-25) | 생성 CoC의 토큰당 인과 손상이 prompt text의 **43×**, vision의 16×. CoC 차단 시 ΔminADE 중앙값 +0.134 m, sink 차단 −0.0001(인과적으로 0), 개별로 유의한 카메라는 front-tele뿐. attention 질량은 vision 72.6 / text 16.7 / hist 3.9 / CoC 2.3 / sink 1.1% | CoC 의존이 34B에서도 같은 크기인가. 후방 카메라 2대를 expert가 쓰는가. sink가 여전히 0인가 | 50클립 × K=8, 약 14조건 → replica 2개로 **약 1시간** | 쉬움. 러너가 이미 expert attention mask를 직접 만든다 |
| 1-2 | **Pathway map stage 2**: VLM 내부 edge × 층 band knockout (2026-08-25) | action-only 0셀 : language-only **14셀**(한 방향 해리). 행동 채널 L0–17, 언어 채널 L18–35. CoC←vision 차단 시 NLL +155%·궤적 −0.5%, cross-frame 차단 시 궤적 +64%(cross-camera는 n.s.), VLM 안에서 sink 제거 시 궤적 +59% | 한 방향 해리와 초반/후반 채널 분리가 64층에서도 같은 **상대 깊이**에 나타나는가 | 8 edge × 5 band + baseline, 50클립 → **약 3–4시간** | 중간. Qwen3-VL 계열이라 mask 주입 방식이 같다. band를 상대 깊이로 재정의 |
| 1-3 | **Attention 통계** Phase A (archive 2026-07-20) | VLM의 3-band 구조(L0–6 text, L7–16 sink, L17–35 vision). sink mass > 0.5인 head 181개(15.7%)가 전부 L7–16에 있음 | 3-band가 있는가, sink band가 L12–28에 오는가. KV당 Q 8개에서도 sink head가 한 band에 몰리는가 | CoC query만 eager, 50클립 **약 15분** | 쉬움 |
| 1-4 | **추론 단계 프로파일링** (archive 2026-07-21) | FLOPs 비중: ViT 18 / prefill 76 / decode 0.4 / expert 5.5%. latency 비중: decode 32–38%, expert 25–28%. decode와 expert는 dispatch-bound | A2의 16.6 s가 어느 단계에서 나오는가. 증류의 동기 수치가 된다 | 24클립 **약 15분** | 쉬움. 단 2-GPU 파이프라인이라 카드 간 전송이 섞인다. 80 GB 카드 1장이면 더 깨끗하다 |
| 1-5 | **GT-CoC teacher-forced 조건** (`run_baseline.py`의 TF 경로) | OOD-val에서 A1.5는 TF minADE@6 0.966 vs rollout 0.9995(Δmean −0.034, 중앙값 −0.003) — 정답 추론을 줘도 궤적이 거의 안 좋아짐. `nll_gtcoc` 3.14 | A2는 정답 CoC를 받으면 궤적이 더 좋아지는가(CoC–궤적 결합이 더 강한가). A2 추론 품질의 NLL 판독도 겸함 | 262클립, rollout 재사용 → **약 30분** | 쉬움. A2의 CoC 앞뒤 special token 배치만 토큰 단위로 확인 |
| 1-6 | **Cache-use map** 주변분포만 (2026-08-29) | expert는 모든 층·스텝에서 cache를 읽음(질량 0.976–0.959). cache가 없으면 65% 클립이 10 m 넘게 발산. 가장 아픈 곳은 L11–15의 마지막 두 스텝(3.5 m), 후반 스텝이 더 비쌈 | 의존 층 band와 스텝 프로파일이 같은 모양인가 | 전체 격자(64층 × 10스텝 = 640셀)는 클립당 약 11분이라 과함. 층 주변분포 + 스텝 주변분포(K=4)만 → 50클립 **약 3시간** | 중간 |

### Tier 2 — gradient 필요, 논문 Finding 1(핵심 그림)의 일반화 검증, 무겁고 메모리 위험

| # | 분석 (A1.5 출처) | A1.5 결과 (재현 대상) | A2에서 묻는 것 | A2 비용 |
|---|---|---|---|---|
| 2-1 | **Dual-objective Taylor importance** (`run_importance.py`, Finding 1 / Fig. 1) — 한 번 돌리면 아래가 모두 나옴 | 층내 순위 일치: trunk 약 0.8, 후반 0.33–0.41. 변화점 L22–23, 4.6× step(서로소 추출·n·OOD 무관, val100·OOD100에서 재현). L35의 `I_traj` ≡ 0. **expert 스텝 질량은 U자, VLM은 단조 감소.** expert 1차 질량 중 하위 50% 유닛의 몫이 MLP 0.44% vs Q 24.96% | 변화점이 있는가, 어디에 있는가(상대 깊이 L39–41인가, A2 포트 위치인가). U자 스텝 프로파일과 expert MLP 집중도가 재현되는가 | 7-cam `calib_100` 캐시 약 10분 + **메모리 스모크 필수**. 되면 100클립 약 2–3시간(4장 1 replica, 추정) |
| 2-2 | **Gradient anatomy** + 무작위 readout probe (2026-09-20/21) | 17–24층에서 두 점수 모두 vision → text 토큰으로 전환(MLP vision 몫 `I_traj` 0.90→0.26, `I_CoC` 0.75→0.02). 같은 토큰으로 제한하면 후반도 일치(천장 보정 ρ: pooled 0.38 vs vision 0.79 / prompt 0.81). `I_traj`의 84–89%가 cache ≥22로 들어오고 주 포트는 cache 21–23. **무작위 readout도 같은 프로파일(0.98/0.96)과 5.4× step을 만듦 → 과제가 아니라 출력 인터페이스의 성질** | 인터페이스 설명이 A2에서도 성립하는가. probe가 A2 변화점을 사전에 예측하는가(가장 강한 검증) | 2-1 인프라 위에서 수 시간 + cache 포트 지도 64층분 |
| 2-3 | **인과 검증**: held-out band mask (2026-09-21) | 후반층(22–34)의 19–40%를 빼도 FM ±1%, minADE 불변, NLL만 움직임 → 후반은 기능적으로 언어 채널. trunk에서는 합집합이 압도적(FM +4.8% dual vs +17.0% CoC-only vs +26.1% traj-only) | 후반 = 언어 채널, trunk = 공유라는 기능 지도가 A2에서도 성립하는가 | 2-1의 점수가 있어야 함. mask 추론 수 시간 |

### Tier 3 — 지금은 비추천 (이유)

| 분석 | 이유 |
|---|---|
| dual vs 단일 기준 프루닝 비교(C1), 캘리브레이션 추출·크기·층화 연구 | 분석이 아니라 **방법** 재현이다. A2를 실제로 자르고 개루프·폐루프로 평가해야 하며, 폐루프 없이는 이 레포의 결론 규칙상 판정할 수 없다(개루프 ≠ 주행) |
| 프루닝된 모델 분석(cache shift, pruned VLM internals, stream error, RAC) | 프루닝된 A2 arm이 먼저 있어야 한다 |
| 실효 계획 지평, 폐루프 전반 | AlpaSim에 A2를 올려야 한다(6-cam 렌더, 34B 서빙). 그 자체로 별도 과제 |
| J-lens | 사전 토큰당 backward가 필요해 2-1보다 무겁고, 논문 서사에서 비중이 줄었다 |
| cache-use map 전체 격자 | 640셀이라 과함. 1-6의 주변분포로 대체 |

## 2. 권장 순서

1. **Tier 0** (GPU 불필요, 반나절): 0-1 self-consistency와 0-2 전형성. A2 rows가 이미 있어서 바로 된다.
2. **Tier 1 묶음** (포팅 약 1–2일 + GPU 약 6시간): 1-4 → 1-3 → 1-1 → 1-5 → 1-2 순서.
   공통 인프라(A2 span 계산, 2-GPU 배치, cache knockout)를 한 번 만들면 뒤의 것들이 싸진다.
   평가셋은 이미 만든 7-cam OOD-val 캐시에서 50클립을 쓴다. in-distribution이 필요하면 val 50클립 7-cam 캐시를 약 10분에 만든다.
3. **판정 지점**: Tier 1에서 A2가 같은 기전(CoC 의존, 한 방향 해리, 깊이 채널 분리)을 보이면
   Tier 2로 간다. Finding 1의 일반화는 논문에서 가장 가치가 크지만, gradient 메모리 스모크를 통과해야 한다.
4. **Tier 2** (메모리 스모크 → 2-1 → 2-2 → 2-3): 2-2의 무작위 readout probe로 A2 변화점을
   **먼저 예측해 등록한 뒤** 2-1로 확인하면, 사후 설명이 아니라 사전 예측의 검증이 된다.

## 3. 공통 주의점 (A1.5 트랙에서 배운 것)

- 층은 상대 깊이로 비교한다. 예측은 표 0의 두 가설(상대 깊이 vs 포트 위치)로 **실행 전에** 등록한다.
- 한 번에 edge 하나만 막는 knockout은 대체 가능한 경로를 과소평가한다. nested window(`run_pathway2.py --cuts`)를 쓴다.
- 1차 점수의 후반층 질량은 기능이 아니다(2026-09-21). 점수 깊이 프로파일만으로 기능을 주장하지 않는다.
- 부분 seed VLM backward는 bf16 반올림 수준까지만 선형이다. 정확해야 하는 분해는 full-seed backward 한 번에서 읽는다.
- 모든 비교는 한 GPU 아키텍처 안에서 한다(Ada). 모델 간 비교는 클립으로만 짝지어진다.
- A2의 학습 데이터 누수 가능성(val) 때문에, 분석 대상 클립이 학습에 쓰였는지는 배제할 수 없다. 기전 분석에는 영향이 작지만 기록해 둔다.
