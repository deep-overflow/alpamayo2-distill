# Plan: 카메라 + LiDAR(range image) 입력 student

- 작성 2026-10-20 · 상태: **승인됨 (2026-10-20)**
- branch(승인 후 생성): `a2-distill-lidar`. 먼저 `a2-prune-distill`에 남은 수정 사항을 커밋한다.
- 코드 위치: `experiments/distill/` (A2 `src/`는 수정하지 않음)

## 1. 목적

카메라 전용 최종 student(C-wide, `distill_s2_wide_final`)에 **LiDAR 입력만 추가**했을 때 세 과제(궤적, reasoning, detection)가 좋아지는지 본다.

- 입력 외의 조건은 모두 같게 둔다: 시작 모델, 데이터, 라벨, 손실, 학습 일정, 평가.
- 비교 대상: 카메라 student, LiDAR student, A2, A1.5(궤적만). 같은 평가 코드로 잰다.

## 2. 확정된 결정 (2026-10-20 Q&A)

| # | 항목 | 결정 |
|---|---|---|
| L1 | detection 라벨 | A2 pseudo label 유지 (카메라 student와 동일) |
| L2 | 융합 방식 | LiDAR range image를 토큰으로 바꿔 프롬프트에 추가. RGB 6카메라 × 4프레임은 그대로 유지 |
| L3 | encoder | 전용 소형 range-ViT를 새로 학습 (약 13M, 0.5B 예산 안: 466M + 13M ≈ 479M) |
| L4 | positional embedding | encoder 안: 주기적 방위각(차량 전방 = 0°, sin/cos 다중 주파수 + cyclic RoPE) + 빔의 실제 고도각 + **시간 임베딩**(patch별 Δt = t0 − 측정 시각, sin/cos 다중 주파수). LLM: range 토큰을 (시간, 고도 행, 방위 열) 격자로 두고 Qwen MRoPE 사용 |
| L5 | 해상도 | 128(빔) × 1024(방위) → patch 16×64 → 2×2 merge → **256 토큰** |
| L6 | 회전 수 | 1개 (아래 §3의 시간 정렬 규칙에 따름) |
| L7 | 학습 조건 | 카메라 S2와 동일: `s2_wide/init` + LiDAR 경로(무작위 초기화), 최대 10 epoch, lr 1e-4, warmup 100, patience 5, global batch 32, GPU 0–3 |
| L8 | P4 평가 | LiDAR 작업과 병행. 카메라 student 평가를 먼저 끝낸다 |

## 3. 확인한 사실과 시간 정렬 (2026-10-20 측정)

**LiDAR 데이터**
- `lidar_top_360fov`, clip당 parquet 하나(약 355 MB, 회전 199개).
- **회전 하나가 row group 하나**이고 약 1.4 MB다 → 필요한 회전만 범위 읽기로 받을 수 있다(clip당 약 11초).
- 점 속성: xyz(float32), intensity(uint8), **점별 시각**(µs, t0와 같은 시계), `model_element`(uint32 2개, 빔/발사 인덱스로 추정).
- 회전당 점 약 27.6만 개, 수직 −25.5° ~ +14.4°(약 128 링), 수평 360°(링당 약 2,150점), 거리 중앙값 17 m / p99 95 m.
- 로컬 사본이 없다. venv에 Draco 디코더가 없어서 **DracoPy 설치가 필요**하다(승인 요청).

**FPS와 시간 정렬** (6 clip 표본)

| 센서 | 주기 | t0 대비 위치 |
|---|---|---|
| 카메라 | 30 fps | 캐시 4프레임 = −317 / −217 / −117 / −17 ms. t0 프레임은 항상 t0 이전(−33 ~ 0 ms) |
| LiDAR | 10 Hz, 회전당 100 ms | t0에 가장 가까운 회전은 **t0 이후 +20 ~ +78 ms까지 측정** (미래 정보) |
| LiDAR | — | t0 이전에 끝난 마지막 회전은 −180 ~ −22 ms 구간 |

**규칙**
1. **인과성**: t0 **이전에 끝난 마지막 회전**(spin_end ≤ t0)만 쓴다. 모든 점의 시각이 t0 이하인지 검사(assert)한다.
2. **deskew**: 각 점을 측정 시각의 차량 자세에서 **t0 시점 차량 좌표계(rig@t0)**로 옮긴다(egomotion 보간 사용). 이 좌표계는 궤적과 카메라 t0 프레임이 쓰는 것과 같다. 보정하지 않으면 20 m/s에서 최대 3.6 m가 어긋난다.
3. **시간 positional embedding** (사용자 제안, 2026-10-20): 다른 차량의 움직임(최대 약 0.18초분)은 보정할 수 없다. 그래서 patch별 Δt를 encoder의 시간 임베딩으로 넣어 모델이 각 방위 구간의 측정 시점을 알게 한다. 회전은 방위를 따라 100 ms에 걸쳐 측정되므로 patch마다 Δt가 다르다. 처음 계획했던 "나이" 입력 채널은 이것으로 대체한다.
   - LLM의 MRoPE 시간축은 바꾸지 않는다. A2는 카메라 프레임 시간을 MRoPE가 아닌 텍스트 라벨("frame 0–3")로 표현하므로, 맞춰 볼 시간 기준이 없다.
- 대안(채택하지 않음): t0를 걸친 현재 회전의 t0 이전 부분과 이전 회전을 방위별로 합쳐 더 최신으로 만드는 방법. 구현이 복잡하고 회전 경계에서 불연속이 생겨 제외했다.

## 4. 설계

### 4.1 LiDAR 캐시 (`build_lidar_distill.py`)

세트: train_10k, dev_238, val_500, ood_262 (약 11k 샘플). 샘플마다 다음을 한다.

1. zip 디렉터리와 parquet footer를 범위 읽기로 받고, 규칙 1에 맞는 회전의 row group만 받는다.
2. Draco를 디코딩하고, `sensor_extrinsics['lidar_top_360fov']`로 rig 좌표계로 옮긴다.
3. 규칙 2로 deskew한다.
4. **range image (4 × 128 × 1024)**를 만든다.
   - 행: 빔 인덱스(`model_element[:, 0]`, 0–127, 위→아래. 값별 고도 편차 0.005°로 확인됨).
   - 열: rig@t0 기준 방위각. 전방이 0°이고, [−180°, 180°)를 1024 구간으로 나눈다.
   - 채널: log(range), intensity/255, 높이 z(rig), 유효 mask. 한 칸에 점이 여러 개면 가장 가까운 점을 쓴다.
   - 시간 임베딩용으로 칸별 Δt(ms)를 별도 배열로 저장한다. 입력 채널이 아니라 encoder의 positional embedding에 쓴다.
5. 빔별 평균 고도각(positional embedding 입력)을 함께 저장한다.

- 저장: float16 npz, 샘플당 약 1.3 MB, 총 약 14 GB → `pre_processed/distill_a2_lidar/<set>/`.
- 저장: float16 npz(4채널 + Δt), 샘플당 약 1.3 MB.
- 예상 시간: 32 worker로 약 1.5–2시간. CPU와 네트워크를 쓴다.

### 4.2 모델 (`lidar_lib.py`)

**RangeViT**
- patch는 8행 × 16열이고, 128×1024 입력에서 16×64 = 1,024 patch가 된다.
- 6층, d=384, 6 head이고, patch merger는 2×2 MLP다.
- 출력은 8×32 = 256 토큰이고, MLP로 LLM hidden(1280)에 맞춘다. 총 약 13M 파라미터다.

**positional embedding (L4)**
- 덧셈형 임베딩: patch 중심의 방위각 φ에 대해 sin/cos(kφ), k=1..8을 쓴다. 주기 함수라 ±180° 이음매가 이어진다. 빔 고도각 θ와 patch별 Δt(시간)는 다중 주파수 sin/cos로 넣는다. 셋을 MLP에 통과시켜 patch 임베딩에 더한다.
- attention 안: 2D RoPE를 쓴다. 방위 축은 주파수를 2πk/64로 둔 **cyclic RoPE**라 열 64가 열 0과 같아진다. 고도 축은 일반 RoPE다.
- LLM 단계: range 토큰을 Qwen의 video 격자(t=1, h=16, w=64, merge 2 → 8×32)로 취급한다. `get_rope_index`가 (t, h, w) MRoPE 위치를 계산한다. ±180° 연속성은 encoder 임베딩이 담당한다.

**프롬프트 연결**
- 카메라 이미지 뒤에 `LiDAR: <|vision_start|>` + `<|video_pad|>` × 256 + `<|vision_end|>`를 넣는다. 이 id들은 student vocab의 필수 id에 이미 들어 있다.
- forward wrapper가 하는 일:
  - placeholder 자리의 임베딩을 range 토큰으로 바꾼다.
  - position_ids와 rope_deltas를 `get_rope_index(..., video_grid_thw=range 격자)`로 계산해 넘긴다. expert는 rope_deltas를 쓴다.
  - 카메라 이미지는 기존 경로 그대로 처리한다.

### 4.3 학습

- `train_student.py`에 `--lidar` 옵션을 추가한다. 데이터셋에 range image를 붙이고, Step이 wrapper를 쓴다.
- 나머지는 L7 그대로다. ViT 공유와 시퀀스별 backward도 그대로 쓴다.
- 체크포인트에는 RangeViT 가중치와 설정이 포함된다.

### 4.4 평가 (P4, 두 student 공통)

1. **평가 세트 teacher 라벨**: val_500, ood_262에 대해 A2의 driving(궤적 K=8 포함)과 detection을 생성한다. GPU 0–3에서 약 2–3시간 걸린다.
2. **`eval_student.py`**: student가 직접 CoC + meta-action을 생성하고(vocab 재매핑을 처리하는 자체 decode), expert로 궤적 K=8을 만들고, detection JSON을 생성한다.
3. **지표**
   - 궤적: minADE@1/6/8, minFDE vs GT. A2와 A1.5(`baseline_pred_indist`, `baseline_pred_oodval`)와 비교한다.
   - detection: 투영 obstacle 박스 기준 P/R@0.5·0.3, 거리·클래스별. A2 라벨과의 F1도 본다.
   - reasoning: A2 대비 meta-action 일치율, CoC ROUGE-L.
   - 효율: 파라미터, 지연, 메모리.
   - 카메라 vs LiDAR 차이에는 paired bootstrap CI를 붙인다.

## 5. 게이트

- **L0a 데이터**
  - (i) 모든 점의 시각이 t0 이하인지 확인한다.
  - (ii) deskew 전후의 LiDAR 점을 front-wide t0 영상에 투영해 overlay 10장을 그린다. 정량 지표로는 obstacle 3D 박스 안에 들어오는 점의 비율이 deskew 후에 늘어나는지 본다.
  - (iii) range image를 시각화하고 빔 순서를 확인한다.
- **L0b 연결**
  - (i) LiDAR placeholder가 없을 때 wrapper의 출력이 기존 forward와 같아야 한다.
  - (ii) range 토큰 위치가 올바른 (t, h, w) 격자이고, 뒤 텍스트 위치가 일관되게 밀리는지 확인한다.
  - (iii) RangeViT로 gradient가 흐르는지 확인한다.
  - (iv) FSDP smoke 학습을 돌린다.

## 6. 순서와 시간

1. `a2-prune-distill` 커밋 → `a2-distill-lidar` branch를 만들고 DracoPy를 설치한다.
2. 병행 작업
   - LiDAR 캐시 (CPU, 약 2시간)
   - 평가 세트 teacher 라벨 (GPU 0–3, 약 2–3시간)
   - 코드 작성 (`lidar_lib`, wrapper, `eval_student`)
3. 카메라 student P4 평가 (GPU, 약 1–2시간) → L0 게이트
4. LiDAR student 학습 (GPU 0–3, 최대 약 10–15시간)
5. LiDAR student 평가 → 보고서 `reports/YYYY-MM-DD_a2-distill-lidar.md`

## 7. 리스크

- **teacher 한계**: A2는 LiDAR를 보지 못한다. reasoning·detection 라벨은 카메라 기준이라 LiDAR의 이점은 주로 궤적(GT 라벨)과 A2 라벨을 더 잘 맞추는 쪽에서 나타날 것이다. detection recall이 A2 수준을 넘기는 어렵다.
- **시간차**: LiDAR는 카메라 t0 프레임보다 평균 약 0.1초 오래됐다. 움직이는 객체는 어긋날 수 있다. 시간 positional embedding으로 완화한다.
- **dev 손실 노이즈**: flow 손실 평가가 무작위라 dev 손실이 흔들린다. 이번 실행부터 dev 평가의 노이즈를 고정한다(카메라 S2와 조기 종료 조건이 미세하게 달라진다는 점을 보고서에 적는다).
- **디스크**: `/mnt` 여유 약 99 GB에 LiDAR 캐시 14 GB와 체크포인트 수 GB가 더해진다.

## 8. 변경 기록 (승인 후)

| 날짜 | 항목 | 계획 | 변경 | 이유 |
|---|---|---|---|---|
| 10-20 | FPS 대응 | "나이" 입력 채널 | encoder의 **시간 positional embedding**(patch별 Δt) | 사용자 제안. 시간은 위치 정보 성격이라 임베딩이 더 자연스러움. LLM MRoPE 시간축은 카메라도 실제 시각을 쓰지 않아 변경하지 않음 |
| 10-20 | 빔 인덱스 | 추정 | `model_element[:, 0]` = 빔 0–127 (위→아래), `[:, 1]` = 방위 발사 인덱스 0–3598 (0.1°) | 확인 완료 |
| 10-20 | RangeViT 크기 | 약 13M | 15.3M (pos_mlp와 hidden 1280 출력 MLP 포함) | 예산 466M + 15M = 481M로 0.5B 안 |
| 10-20 | dev 평가 노이즈 | — | LiDAR 실행만 clip별 고정 seed | 계획 §7대로. 카메라 실행과 dev `fm`, 조기 종료 조건이 미세하게 다름 |
| 10-20 | 결과 | — | 보고서 `reports/2026-10-20_a2-distill-lidar.md` | 세 과제 모두 유의한 이득 없음. LiDAR student는 detection FP가 늘어남 |
