# Plan: Alpamayo 2 Super 가지치기 + 증류 → 소형 멀티태스크 student

- 작성 2026-10-15 · 상태: **승인됨 (2026-10-15), P0 완료, P1 진행 중**
- branch(승인 후 생성): `a2-prune-distill`
- 코드 위치: `experiments/distill/` (A2 `src/`는 수정하지 않음)

## 1. 목적

Alpamayo 2 Super(35.8B)를 직접 가지치기해 **ViT는 그대로 두고 나머지를 0.5B 이하로** 줄인 student를 만든다. 이 student가 카메라 입력만으로 세 과제를 수행하도록 학습한다.

1. front-wide 카메라의 2D object detection (텍스트 JSON bbox)
2. reasoning: CoC + meta-action
3. trajectory prediction: flow-matching expert, 6.4 s, 64 waypoint

detection과 reasoning 라벨은 원본 A2가 만든 pseudo label이고, trajectory 라벨은 데이터셋의 GT ego future이다.

목표는 **가장 성능이 좋은 모델 하나**를 만드는 것이다. ablation은 방법 선택에 필요한 최소한(G2 게이트)만 한다.

## 2. 확정된 결정 (2026-10-14~15 Q&A)

| # | 항목 | 결정 |
|---|---|---|
| D1 | 목적 | 최고 성능 모델 하나. ablation은 최소한 |
| D2 | student 생성 방식 | A2를 직접 구조적 가지치기한 뒤 증류 |
| D3 | 0.5B 범위 | ViT(0.60B)는 유지. 텍스트 디코더층, embed, lm_head, expert, action proj 합계 ≤ 0.5B (전체 약 1.0B) |
| D4 | reasoning 라벨 | A2 `meta_action` 과제의 추론 모드 출력 (CoC + meta-action, 미래 정보 없음) |
| D5 | detection 라벨 | P0에서 A2의 다객체 detection을 3D obstacle 투영 박스와 비교해 검증한 뒤, **사용자가 결정** |
| D6 | detection 대상 | front-wide 1대(cam 1), 출력은 A2 grounding과 같은 텍스트 JSON (0–1000 정규화) |
| D7 | 학습 데이터 | train split에서 LiDAR가 있는 **서로 다른 clip 10k**, clip당 t0 1개, 기동 bucket 층화 |
| D8 | LiDAR | 필터로만 사용. 포인트클라우드는 받지 않고, 후속 작업을 위해 clip ID만 맞춘다 |
| D9 | GPU | Blackwell 0–3만 사용. Ada 4–7은 사용하지 않음 |

## 3. 확인한 사실 (코드·데이터)

- **A2 파라미터** (safetensors header 기준, 총 35.81B): 텍스트 디코더 64층 31.21B, `embed_tokens`와 `lm_head` 각 0.80B(tie 안 됨, vocab 155,776), ViT 0.596B(blocks 411.5M, deepstack merger 134.5M, merger 44.8M, 기타 4.5M), expert 2.42B(64층, hidden 1536, 16Q/8KV, MLP 6144).
- **expert와 VLM의 결합**: `ExpertModel.forward`가 `past_key_values=vlm_outputs.past_key_values`를 그대로 받는다. 즉 expert의 i번째 층이 VLM i번째 층의 KV cache를 읽는다. 이 때문에 가지치기에는 두 가지 제약이 생긴다.
  - (a) 깊이를 줄일 때는 두 tower에서 **같은 층 index**를 제거해야 한다.
  - (b) KV head(개수 × 128)는 두 tower가 **공유**해야 한다.
  - 설정에 `cache_layer_indices` 필드가 있지만 release 코드에서는 쓰이지 않는다.
- **A2 detection 기능**: 공식 과제가 아니다. `vqa` 경로의 grounding(단일 객체 지목, 기본 카메라 cam 1)만 있다.
- **이미지 입력**: A2 processor는 `max_pixels=196,608`, patch 16, merge 2를 쓴다. 1080×1920 이미지는 320×576으로 줄어들고 이미지당 180 토큰이 된다. 6 cam × 4 frame이면 4,320 토큰이다.
- **데이터 가용성**: LiDAR는 전체 306,152 clip 중 298,326개에 있고, `obstacle.offline` 라벨이 있는 clip과 집합이 정확히 같다. train split에서 LiDAR가 있는 clip은 149,774개(1,576 chunk)이다.
  - 로컬에는 camera 4대 × 350 chunk, egomotion 621 chunk, obstacle 95 chunk가 있다. rear-left/right 카메라와 LiDAR는 없다.
  - 기존 7cam streaming 빌드는 262 clip에 19.1분(16 worker)이 걸렸다.
- **자원**: `/mnt/nvme1n1` 여유 168 GB(사용률 98%), `/` 여유 88 GB. A2 teacher 처리량은 GPU 2장 1쌍 기준 CoC 생성 5.3 s/clip, 전체 16.6 s/clip(oodval 측정).

## 4. 설계

### 4.1 데이터

- **학습 manifest (`distill_train_10k`)**
  - 조건: train split, LiDAR 있음, 6 cam [0,1,2,3,5,6], egomotion 있음.
  - 제외: calib_100(가지치기·probe용), 모든 평가 세트 clip.
  - chunk당 최대 10 clip으로 다양성을 확보한다.
  - 후보 clip마다 t0 grid(2.0–13.0 s, 0.1 s 간격)에서 bucket을 계산한다. bucket은 {turn_left, turn_right, decel_stop, accel, cruise}이며, A1.5 `eval_lib.bucket` 규칙을 import해 그대로 쓴다.
  - 가장 부족한 bucket부터 채우는 방식으로 clip당 t0 하나를 고른다. 기존 `make_train_set.py`와 같은 방식이다.
- **평가 세트**
  - dev(체크포인트 선택용): `val_official_238`.
  - report: `val_500`과 OOD-val 262. trajectory는 전 clip에서 평가한다. detection 지표는 LiDAR clip(각 482, 240)에서만 낸다.
- **캐시 (`pre_processed/distill_a2/<set>/`)**
  - `build_cache_a2.py`를 확장해 7cam을 streaming한 뒤, **A2 processor와 같은 resize 함수로 320×576까지 줄여** JPEG q95로 저장한다.
  - 저장하는 카메라는 driving profile 6대 × 4 frame이고, clip당 약 1.2 MB이다. 10.7k clip이면 약 13 GB이다.
- **2D 참조 박스 (`project_obstacles.py`)**
  - `obstacle.offline`(rig 좌표 3D cuboid)을 front-wide의 t0 frame 시각으로 보간한다.
  - `sensor_extrinsics`로 카메라 좌표로 옮기고, f-theta intrinsics(`physical_ai_av.utils.camera_models`)로 꼭짓점 8개를 투영한 뒤 외접 사각형을 만들어 이미지에 clip한다.
  - 필터: 카메라 앞쪽, 박스 면적의 50% 이상이 이미지 안, 높이 12 px 이상(320×576 기준), 거리 80 m 이하.
  - 클래스는 vehicle / pedestrian / cyclist / other로 묶는다.
  - obstacle 라벨은 clip별로 HF에서 streaming해 `labels/obstacle_perclip/`에 저장한다(약 0.6 MB/clip, 약 7 GB).
  - 한계: LiDAR 가시성 검사가 없어서 가려진 객체도 포함된다. 따라서 recall은 하한값이다.

### 4.2 Teacher pseudo label (A2, Blackwell 2쌍: (0,1), (2,3))

- **driving 시퀀스**: `build_text_task_messages(task="meta_action")`로 CoC, meta-action, `<|traj_future_start|>`를 생성한다. 생성 설정은 oodval과 같은 release 기본값과 seed를 쓴다.
  - 저장 항목: 텍스트, 토큰 id, 생성 토큰별 **top-64 logits**(fp16).
  - teacher 궤적(8회 denoising)은 평가 세트에서만 생성한다. 학습에는 GT 궤적을 쓴다.
- **detection 시퀀스**: P0에서 고른 prompt로 grounding 경로(driving 6cam profile)를 실행한다. JSON 텍스트, 파싱한 박스, top-64 logits를 저장한다.
- clip 단위로 재시작이 가능하고 2쌍에 shard한다. 메모리가 허용하면 batch 생성으로 처리량을 높인다.

### 4.3 Student 구조와 가지치기

**구조 제약**
- head_dim 128, MRoPE, Qwen3 블록 구조는 유지한다.
- 깊이: VLM과 expert에서 같은 층 index를 함께 제거한다.
- KV group: 두 tower가 같은 group을 유지한다. Q head는 남긴 group 안에서 고른다.
- 잔차 폭 d를 줄일 때 ViT merger와 deepstack merger의 출력 projection 행도 같은 mask로 자른다.
- ViT block은 동결하고 merger 출력층만 학습한다.

**vocab 가지치기**: 모든 prompt, teacher 출력(train + dev), special token, 숫자·JSON 기호에서 실제로 쓰인 토큰만 남긴다(예상 V′ 약 10–30k). d=1024에서 vocab을 그대로 두면 embed와 lm_head만 319M으로 예산의 64%를 차지하기 때문이다. lm_head는 tie하지 않는다.

**중요도** (calib: 학습 세트 중 128 clip, 세 과제 혼합, forward 위주)
- 잔차 채널: 모든 RMSNorm 출력의 평균 |x| (Minitron 방식).
- attention head와 KV group: head별 출력 norm. 기존 dual Taylor 점수(I_traj, I_CoC; `outputs/a2_importance_v1`)와 교차 확인한다.
- MLP 채널: 평균 |act(gate)·up|.
- 깊이: Block Influence(1 − cos(h_in, h_out))로 후보를 고르고, 상위 후보는 층 하나를 뺐을 때의 과제 손실로 확인한다. expert read port 대역(A2 L47–50)이 남는지 보고한다.

**2단계 일정** (기본값)

| 단계 | 구조 | 디코더 + embed/head | expert | 합계(ViT 제외) |
|---|---|---:|---:|---:|
| A2 | L=64, d=5120, 64Q/8KV, m=25,600, V=155,776 | 32.80B | 2.42B | 35.22B |
| S1 | L=32, d=2048, 16Q/8KV, m=8,192; expert d=768, 8Q, m=3,072; V′≈32k | 약 2.14B | 약 0.33B | 약 2.5B |
| S2 후보 C-deep | L=28, d=896, 8Q/4KV, m=3,584; expert d=384, 4Q, m=1,536 | 390M | 73M | 463M |
| S2 후보 C-balanced | L=20, d=1024, 16Q/4KV, m=4,096; expert d=384, 4Q, m=1,536 | 406M | 52M | 458M |
| S2 후보 C-wide | L=14, d=1280, 16Q/4KV, m=5,120; expert d=512, 4Q, m=2,048 | 428M | 60M | 488M |

- S2 수치는 V′=24k를 가정했다. 최종 V′가 정해지면 다시 계산한다.
- ViT 쪽은 merger 출력이 d로 줄어들어 약 0.52B가 된다.

### 4.4 학습

- **clip당 시퀀스 2개**
  - (a) driving: 6cam×4f + 과거 궤적 → CoC + meta-action + `<|traj_future_start|>` → expert flow matching(GT future).
  - (b) detection: 6cam×4f + 질문 → JSON.
- **손실**: L = CE(driving 텍스트) + CE(det 텍스트) + λ_kd·KL_top64(teacher‖student) + λ_fm·FM(GT).
  - 기본값 λ_kd = λ_fm = 1이다. 첫 100 step의 손실 크기를 보고 한 번만 조정한다.
  - teacher top-k는 V′로 다시 매핑하고 정규화한다.
- **최적화**: AdamW(β 0.9/0.95, wd 0.05), cosine, warmup 200 step, bf16, global batch 32.
  - S1: lr 5e-5, 3 epoch, FSDP 4장.
  - S2: lr 1e-4, 최대 10 epoch, dev 기준 early stop, DDP 4장.
- **메모리**: KV cache로 tower를 결합하는 구조라 HF gradient checkpointing과 함께 쓸 수 없다. S1은 checkpointing 없이 들어가는지 G2a에서 확인한다. 안 되면 두 tower를 한 번에 forward하는 joint forward(블록 attention mask)를 구현한다.

### 4.5 평가 (report 세트)

- **trajectory**: minADE@1/@6, avgADE, minFDE. oodval과 같은 프로토콜(CoC 1회 × denoising 8회)이다.
  - 비교 대상: A2(OOD-val은 기존 결과, val_500은 새로 실행), A1.5, 등속 baseline.
  - student − A2 차이에 paired bootstrap CI를 붙인다.
- **detection**: 투영 참조 박스 대비 precision, recall, AP@0.5(클래스 무관 + 묶은 클래스별), 거리 구간별 recall, A2 pseudo label과의 F1@0.5. A2 teacher 자체 점수도 같은 표에 넣는다.
- **reasoning**: A2 대비 meta-action 일치율(필드별, 세 필드 모두), CoC ROUGE-L, 형식 유효율(JSON과 meta-action 파싱).
- **효율**: 부품별 파라미터 수, Blackwell 1장에서의 prefill / 생성 / denoise 지연, 최대 메모리.

## 5. 단계와 게이트

**P0. 데이터와 실현 가능성** (예상 1–2일, 대부분 streaming 대기)
- G0a (캐시 충실도): pre-resize JPEG 경로와 원본 경로를 비교한다. 20 clip에서 `pixel_values` 차이, teacher-forced CoC 분포 KL, 고정 seed 궤적 ADE 차이가 seed 간 차이보다 작아야 한다.
- G0b (slicing 정확성): 모두 남기는 mask로 slicing하면 teacher와 logits가 **완전히 같아야** 한다. 부분 mask를 쓰면 파라미터 수가 §4.3 공식과 일치해야 한다.
- G0c (투영 확인): 20 clip의 front-wide에 참조 박스를 overlay한 그림을 저장하고 육안으로 확인한다. 시각 정렬 오차는 50 ms 미만이어야 한다.
- **G0d (A2 detection probe)**: calib_100 중 obstacle 라벨이 있는 96 clip에 prompt 3종을 적용한다. 어느 카메라와 frame 좌표인지 overlay로 확인하고, precision/recall@0.5/0.3, 클래스별·거리별 결과, 출력 길이, 처리 시간을 측정한다.
  - → **결과를 보고하면 사용자가 detection 라벨 소스를 결정한다(D5).**
- G0e (vocab coverage): prompt의 OOV는 0이어야 한다. report 세트 teacher 출력의 OOV 비율을 보고한다.

**P1. Teacher 라벨링**: train 10k + dev + report, 약 11k clip. 처리량은 P0에서 측정한다(예상 15–25시간).

**P2. 가지치기와 S1**: 중요도 계산 → S1 slicing → KD 3 epoch.
- G2a: S1이 GPU 메모리에 들어가는지 확인한다.
- G2b(방법 선택, 짧은 실행 각 1 epoch, dev 손실 기준):
  - S2 후보 3개(S1에서 가지치기) 중 하나를 고른다.
  - 고른 구조로 S1 경유와 A2에서 바로 0.5B로 가는 one-shot을 비교해 더 나은 쪽을 택한다.

**P3. S2 본학습**: G2b에서 선택한 설정으로 최대 10 epoch을 돌리고 dev 기준 early stop한다.

**P4. 평가와 보고**: §4.5 지표를 내고 `reports/YYYY-MM-DD_a2-prune-distill.md`를 작성한다(Facts와 Analysis 분리).

## 6. 산출물

| 경로 | 내용 |
|---|---|
| `experiments/distill/` | `make_distill_sets.py`, `build_cache_distill.py`, `project_obstacles.py`, `det_probe_a2.py`, `label_teacher.py`, `vocab_prune.py`, `slice_lib.py`, `train_student.py`, `eval_student.py`, `analyze_distill.py` |
| `outputs/distill_sets/` | manifest, `config.json`, `summary.txt`, 분포 plot |
| `pre_processed/distill_a2/{train10k,dev,val500}/` | 320×576 캐시. OOD-val은 기존 `ood_a2`를 재사용 |
| `outputs/distill_p0_detprobe/` | G0d 결과와 overlay |
| `outputs/distill_teacher/` | clip별 teacher 출력, top-k logits |
| `outputs/distill_s1/`, `outputs/distill_s2_<arch>/`, `outputs/distill_final/` | config, metrics(step별), summary, plot, 최종·best 가중치만 보관 |
| `outputs/distill_eval/` | §4.5 지표 |

예상 디스크 사용량은 50 GB 미만이다(캐시 약 15 GB, obstacle 약 7 GB, teacher 출력 약 3 GB, 체크포인트 약 15 GB). optimizer 상태는 실행 중에만 보관한다.

## 7. 리스크

1. **압축 비율**: 디코더를 약 64배 압축한다. 10k clip이 주는 감독 토큰은 약 2–3M이라 회복이 부족할 수 있다. 완화책은 top-64 KL로 밀도 높은 신호를 주는 것과 2단계 일정이다. 그래도 부족하면 clip당 t0를 늘리는 방안을 제안한다(teacher 비용 증가).
2. **A2 detection 품질**: 단일 객체 grounding만 학습했을 수 있다 → G0d에서 판단한다.
3. **참조 박스 오차**: 가려진 객체가 포함되어 detection recall은 하한값이다. 모든 모델에 같은 조건이라 비교에는 유효하다.
4. **RMSNorm 채널 제거**: RMS 분모가 바뀐다. KD로 회복된다고 가정한다.
5. **KV cache와 checkpointing 비호환**: S1 메모리 문제 → G2a에서 확인하고, 필요하면 joint forward로 대응한다.
6. **자원**: teacher(2장)와 학습이 Blackwell 0–3을 나눠 써야 해서 단계를 순차로 진행한다. 디스크 여유가 168 GB뿐이다.

## 8. 승인 요청: 바꾸고 싶으면 알려 주세요

- 2단계 가지치기(A2 → 약 2.5B → ≤0.5B)와 G2b의 방법 선택 게이트
- ViT block 동결, vocab 가지치기, lm_head tie 안 함
- 손실 구성(CE + top-64 KL + GT flow matching). teacher 궤적은 증류하지 않음
- 평가 세트(dev `val_official_238`, report `val_500` + OOD-val 262)와 등속 baseline
- G0d 이후 detection 라벨 소스는 그때 다시 묻는다

## 9. 변경 기록 (승인 후)

| 날짜 | 항목 | 계획 | 변경 | 이유 |
|---|---|---|---|---|
| 10-15 | 캐시 형식 | 320×576 JPEG q95 | 320×576 **무손실 WebP** | G0a KL 기준 실패(3.4배 > 3배). 미리 정한 대체안을 적용했고, 무손실은 두 기준 모두 통과 |
| 10-15 | D5 detection 라벨 | P0 후 결정 | **A2 pseudo label만** (prompt `p2_release_style`) | 사용자 결정. G0d에서 P 0.90 / R 0.37 |
| 10-15 | GPU | Blackwell 0–3 | **Blackwell 1–3** | 사용자 요청("0번 지피유는 사용지 말아줘"). teacher는 3장에 복제본 하나 |
| 10-15 | P1 생성 방식 | clip별 seed, 1개씩 | train과 det는 **batch 8**, RNG는 batch당 seed 하나(첫 clip의 seed) | GPU 3장으로 처리량 확보. 평가 세트 driving(궤적 포함)은 clip별 seed 유지 |
| 10-15 | P1 순서 | 세트 순서 | train driving → train det → 평가 driving(+궤적) → 평가 det | P2·P3에 필요한 train 라벨을 먼저 만들기 위함 |
| 10-15 | S1 구조 | 16Q / 8KV | **16Q / 4KV** (group당 Q 4개) | head는 group 안에서만 줄일 수 있어서, C-balanced·C-wide(group당 4개)를 S1에서 자르려면 S1도 group당 4개가 필요. 약 2.4B |
| 10-15 | vocab 가지치기 | 코퍼스 토큰 | 필수 id(Qwen special, history 궤적 1000개, marker) + prompt + teacher 생성 토큰 + teacher top-8 후보. student에서는 future 궤적 id를 제거(`future_vocab_size=0`) | 생성·KD에 필요한 토큰만 유지 |
