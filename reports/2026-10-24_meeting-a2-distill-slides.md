# 미팅 발표 자료: Alpamayo 2 압축과 LiDAR 입력 추가

- 대상: 외부 협업자. 배경 슬라이드는 따로 있으므로 여기서는 1–7번만 정리한다.
- 슬라이드는 영어로 만들고, 이 문서는 내용 정리용이다.
- 수치 출처: `reports/2026-10-22_a2-distill-summary.html`, `reports/2026-10-20_a2-distill-lidar.md`, `outputs/distill_*`

## 슬라이드 순서

| # | 슬라이드 | 핵심 내용 |
|---|---|---|
| 1 | Alpamayo 2 structured pruning | 중요도 측정, 자르는 규칙, A2 → S1 |
| 2 | Mid-student (S1) training | KD 손실, 3 epoch, dev 2.252 |
| 3 | Mid-student pruning | S1 중요도 재측정, 후보 3개와 one-shot 비교 |
| 4 | Final student training | 7 epoch, dev 2.005 |
| 5 | Final student with LiDAR | range image 토큰, 위치 임베딩, 시간 정렬, dev 2.011 |
| 6a | Results: student vs teacher | 궤적, detection, reasoning |
| 6b | Results: camera vs LiDAR | 같은 clip paired 차이와 CI |
| 6c | Why the gap / why no LiDAR gain | 진단 실험, 라벨의 한계 |
| 7 | Takeaways & next steps | 결론 3개, 다음 실험 |

---

## 1. Alpamayo 2 structured pruning (A2 → S1)

**핵심 메시지**: 측정한 중요도에 따라 층, head, 채널을 통째로 제거한다. 결과는 더 작은 dense 모델이다.

**폭 중요도**
- train clip 128개(시나리오별로 고르게)에 forward hook을 걸어 activation 통계를 모은다.
- 세 과제(driving reasoning, detection, expert flow matching)로 각각 측정한다.

| 단위 | 점수 |
|---|---|
| query head | head별 attention 출력의 L2 norm |
| MLP 채널 | \|act(gate) · up\| |
| hidden 채널 | RMSNorm 출력 크기를 남길 층들에서 합산 |

**깊이 중요도 (drop-one)**
- 층을 하나씩 건너뛰고, driving CE, detection CE, flow 손실의 상대 증가율을 잰다.
- A2에서는 층 0과 6을 빼면 driving CE가 +12–13 nats 올라 손실이 폭증한다.
- 중간층(8–36)은 대부분 손실 변화가 0.01 미만이다.
- Block Influence(1 − cos)로는 이를 잡지 못한다. 몇몇 거대한 residual 채널 때문에 중간층이 항등 함수처럼 보이고, 결국 층 6을 버리는 쪽으로 골랐다.

**구조 제약**
- expert의 i번째 층은 VLM i번째 층의 KV cache를 읽는다. 그래서 층과 KV group은 두 tower에서 함께 자른다.
- 0–2층은 항상 남긴다. deepstack 시각 특징이 이 세 층에 들어가기 때문이다.
- head는 KV group 안에서만 줄일 수 있다.

| | A2 | S1 |
|---|---:|---:|
| 층 수 | 64 | 32 |
| VLM hidden / MLP | 5120 / 25,600 | 2048 / 8192 |
| query / KV head | 64 / 8 | 16 / 4 |
| expert hidden / MLP | 1536 / 6144 | 768 / 3072 |
| vocab | 155,776 | 4,885 |
| 파라미터 (ViT 제외) | ≈ 35.2B | 2.27B |

**발표 메모**: vocab은 prompt, teacher 출력, teacher의 top-8 후보에 실제로 나온 토큰만 남겼다. 그대로 두면 0.5B 예산에서 embedding과 LM head가 대부분을 차지한다.

---

## 2. Mid-student (S1) training

**핵심 메시지**: 두 번째로 자르기 전에, knowledge distillation으로 2.81B 중간 모델의 성능을 회복한다.

- **손실**: teacher 토큰에 대한 CE + top-64 logit KL + GT 궤적에 대한 flow matching.
  - clip마다 두 시퀀스를 쓴다: driving(CoC + meta-action)과 detection(JSON 박스).
- **데이터**: train 10k clip.
  - 궤적: GT
  - reasoning, detection: Alpamayo 2 pseudo label
- **학습**: 3 epoch, lr 5e-5, GPU 4장(FSDP), 11.6시간. ViT는 고정한다.
- **결과**: dev loss **2.252**

| ce_drv | kl_drv | ce_det | kl_det | fm |
|---:|---:|---:|---:|---:|
| 0.389 | 0.241 | 0.486 | 0.229 | 0.907 |

---

## 3. Mid-student pruning (S1 → S2)

**핵심 메시지**: 학습된 S1에서 중요도를 다시 재고, 최종 구조를 고른다.

**같은 절차를 학습된 S1에 다시 적용**한다. 학습 후에는 중요도가 바뀌기 때문이다.
- S1 drop-one 결과
  - 앞 5개 층(A2의 0, 1, 2, 6, 7층)과 마지막 몇 층이 중요하다.
  - 가운데 층이 가장 덜 중요하다.
- 최종적으로 남은 A2 층: 0, 1, 2, 6, 7, 51, 54, 57–63.

**후보 비교**: 같은 예산(약 0.47B)의 후보 3개와 A2에서 바로 자른 one-shot을 각각 1 epoch 학습했다.

| 후보 | 층 | hidden | 경로 | dev loss |
|---|---:|---:|---|---:|
| **Wide** | 14 | 1280 | A2 → S1 → S2 | **2.497** |
| Balanced | 20 | 1024 | A2 → S1 → S2 | 2.532 |
| Deep | 28 | 896 | A2 → S1 → S2 | 2.568 |
| Balanced | 20 | 1024 | A2 → S2 (one-shot) | 4.639 |

- **두 번에 나눠 자르기가 one-shot보다 훨씬 낫다.** 같은 구조에서 2.53 vs 4.64다.
- **이 예산에서는 넓고 얕은 구조가 가장 좋다.** 다만 구조 간 차이(0.07)는 경로에 따른 차이(2.1)에 비하면 작다.

**발표 메모**: 두 번에 나눠 자르는 쪽은 S1 학습에 계산을 더 썼다. 그래서 계산량까지 맞춘 비교는 아니다.

---

## 4. Final student training

**핵심 메시지**: Wide 구조(0.99B)를 수렴할 때까지 학습한다.

- **구조**: 14층, hidden 1280, 16 Q / 4 KV head, MLP 5632, expert hidden 512.
  - ViT 제외 0.47B, 전체 0.99B, 학습 파라미터 0.58B.
- **학습**: 자른 S1 가중치에서 시작하고, 손실은 S1과 같다.
  - 최대 10 epoch, lr 1e-4, global batch 32, dev loss 기준 조기 종료.
  - 7 epoch에서 멈췄고, best는 5.4 epoch다. GPU 4장으로 약 10시간 걸렸다.
- **결과**: dev loss **2.005**. 1 epoch 후의 2.497에서 내려갔다.
  - S1의 2.252보다도 낮다. S1이 배운 것을 이어받았고, 더 오래 학습했기 때문으로 본다.

| ce_drv | kl_drv | ce_det | kl_det | fm |
|---:|---:|---:|---:|---:|
| 0.326 | 0.188 | 0.438 | 0.205 | 0.848 |

---

## 5. Final student with LiDAR

**핵심 메시지**: LiDAR를 range image 토큰 256개로 추가하고, 나머지는 바꾸지 않는다.

**입력**
- range image 128 × 1024, 채널 4개: 거리, intensity, 높이, 유효 mask.
- 전용 RangeViT(15M, 무작위 초기화)가 토큰 256개로 바꾼다.
- 이 토큰은 카메라 이미지 뒤에 넣는다.

**위치 임베딩**
- 주기적 방위각: 차량 전방 = 0°이고 ±180°에서 이어진다(sin/cos + cyclic RoPE).
- 빔의 실제 고도각.
- patch별 시간 offset. LiDAR는 10 Hz로 돌고 카메라는 30 fps라서, 방위 구간마다 측정 시각이 다르다.

**인과성과 정렬**
- t0에 가장 가까운 회전은 t0 이후 20–80 ms까지 측정된다(미래 정보). 그래서 t0 이전에 끝난 마지막 회전만 쓴다.
- 점마다 t0 시점의 차량 자세로 motion compensation(deskew)한다. 이렇게 하면 3D 박스 안에 들어오는 점이 5% 늘어난다.

**통제된 비교**: 시작 모델, 데이터, 라벨, 손실, 일정이 4번 슬라이드와 같다.

**결과**: 6.4 epoch(10.3시간)에서 멈췄다. dev loss는 2.011로, 카메라 student의 2.005와 거의 같다.

---

## 6a. Results: student vs teacher

**핵심 메시지**: 0.99B 모델 하나로 세 과제를 모두 수행한다. 궤적 오차는 teacher의 약 2배다.

**궤적** (minADE@6, m, 같은 clip)

| 모델 | 파라미터 | in-dist (482) | OOD (240) |
|---|---:|---:|---:|
| Alpamayo 1.5 (참고) | — | 0.820 | 1.022 |
| Alpamayo 2 (teacher) | 35.8B | 1.157 | 1.256 |
| student, 카메라 | 0.99B | 2.375 | 3.031 |
| student, 카메라 + LiDAR | 1.01B | 2.475 | 3.139 |

**detection** (in-dist val, IoU 0.5)

| 모델 | P | R | R 0–30 m | R 30–60 m |
|---|---:|---:|---:|---:|
| teacher | 0.888 | 0.375 | 0.454 | 0.277 |
| student, 카메라 | 0.456 | 0.218 | 0.295 | 0.119 |
| student, LiDAR | 0.410 | 0.227 | 0.308 | 0.122 |

**reasoning** (카메라 student와 teacher의 일치도)
- meta-action 세 항목 모두 일치: 0.30. 항목별로는 종 0.50, 횡 0.65, 차선 0.76.
- CoC ROUGE-L: 0.52.

**발표 메모**: detection 기준 박스는 데이터셋의 3D 자동 라벨을 front 카메라에 투영한 것이다. 가시성과 LiDAR 가림으로 거른다. 사람이 단 라벨이 아니며, teacher도 이 기준에서 recall이 0.38에 그친다.

---

## 6b. Results: camera vs camera + LiDAR

**핵심 메시지**: LiDAR로 측정 가능한 이득은 없다. LiDAR 모델은 틀린 박스(FP)를 더 많이 낸다.

같은 clip에서 짝지은 LiDAR − 카메라 차이 (95% bootstrap CI):

| 지표 | in-dist (482) | OOD (240) |
|---|---|---|
| minADE@6 (m) | +0.10 [−0.06, +0.26] | +0.11 [−0.16, +0.42] |
| detection TP / clip | +0.04 [−0.03, +0.12] | −0.00 [−0.13, +0.12] |
| detection FP / clip | **+0.33 [+0.18, +0.50]** | **+0.66 [+0.42, +0.91]** |
| meta-action 일치 | +0.00 [−0.05, +0.05] | −0.07 [−0.13, −0.01] |
| CoC ROUGE-L | +0.00 [−0.02, +0.03] | +0.04 [+0.01, +0.07] |

- OOD의 reasoning 두 지표는 서로 반대 방향이다. seed가 하나뿐이고 다중 비교 보정을 하지 않았으므로 LiDAR의 효과로 해석하지 않는다.

---

## 6c. Why the gap / why no LiDAR gain

**핵심 메시지**: 병목은 지각이 아니라 계획 용량과 supervision이다.

**궤적 격차 분해**: teacher의 CoC를 student에 강제로 넣어 본다(val 100 clip).

| 사용한 reasoning | minADE@6 (m) |
|---|---:|
| student 자신의 CoC | 2.23 |
| teacher CoC 강제 입력 | 1.97 |
| teacher 자체 | 1.50 |

- 격차의 약 1/3은 student의 reasoning에서 온다: −0.27 m [−0.47, −0.07].
- 약 2/3은 작아진 action expert와 특징에서 온다: +0.47 m [+0.01, +0.89].

**LiDAR가 도움이 되지 않은 이유**
- reasoning과 detection 라벨은 카메라만 본 teacher가 만들었다. 그 라벨을 따라 하는 데는 LiDAR가 필요 없다.
- 궤적은 GT를 쓰지만, 병목은 장면 기하가 아니라 용량이다.
- 새 encoder(15M)는 처음부터 학습되고, 보조 손실 없이 약 6 epoch만 학습했다.

---

## 7. Takeaways & next steps

**결론**
1. 0.99B student(teacher 파라미터의 2.8%)가 궤적, reasoning, detection을 한 모델에서 유지한다.
   - 궤적 오차는 약 2배다.
   - detection recall은 teacher의 0.6배다.
2. 학습 손실 기반의 깊이 중요도를 쓴 2단계 structured pruning이, student의 정확한 구조보다 훨씬 중요하다.
3. supervision이 필요로 하지 않으면, 센서를 추가해도 도움이 되지 않는다.

**다음 단계**
- detection을 teacher의 카메라 박스 대신 투영한 3D 자동 라벨로 학습한다. 평가 쪽 라벨은 준비됐고, 두 student를 다시 학습한다.
- LiDAR 경로를 사전학습하거나, 기하 보조 손실을 붙인다.
- 예산을 action expert 쪽으로 옮기고, student reasoning의 exposure bias를 줄인다.
- seed를 반복해 학습 변동 폭을 확인한다.

---

## 부록 A. 중요도 측정 세부

**폭**
- 두 tower의 모든 decoder 층에 forward hook을 건다. 데이터는 train clip 128개다.
- 측정값

  | 단위 | 측정값 |
  |---|---|
  | query head | attention 출력의 L2 norm |
  | MLP 채널 | \|act(gate) · up\| |
  | hidden 채널 | RMSNorm 출력 크기 |

- 각 점수를 자신의 평균으로 나눠 세 과제가 같은 비중이 되게 한다.
- VLM은 전체 토큰과 생성 토큰 두 기준을 함께 쓴다.

**깊이**
- 층 i를 VLM과 expert 양쪽에서 건너뛴다. 둘을 함께 빼는 이유는 expert가 층마다 VLM의 KV cache를 읽기 때문이다.
- 점수는 16 clip에서 잰 driving CE, detection CE, flow 손실의 상대 증가율 평균이다.

**선택 규칙**

| 대상 | 규칙 |
|---|---|
| 층 | 0–2층은 고정하고, 나머지는 drop-one 점수가 높은 순서로 고른다 |
| KV group | 소속 query head 점수의 합(VLM + expert)으로 고른다 |
| query head | 남긴 group 안에서 고른다 |
| MLP 채널 | 층마다 점수가 높은 채널을 고른다 |
| hidden | 남긴 층의 점수 합 + final norm 점수로 고른다 |

**측정 횟수**: A2에서 한 번(S1을 만들 때), 학습된 S1에서 한 번(S2 후보를 만들 때) 측정했다.

## 부록 B. 학습 세부
- teacher 라벨: Alpamayo 2가 train 10k clip에 대해 만든 CoC, meta-action, detection JSON.
- S1: KD 3 epoch, lr 5e-5, GPU 4장, 11.6시간.
- 최종 student: KD 최대 10 epoch, lr 1e-4, global batch 32, dev loss 조기 종료. 7 epoch, GPU 4장, 약 10시간.
- ViT는 고정했고, 학습 파라미터는 0.58B다.

## 부록 C. 평가 세부
- open-loop 평가만 했다.
- 궤적: 6.4초 GT 미래 대비, 샘플 6개 또는 8개 중 최선값.
- detection 기준 박스: 데이터셋의 3D 장애물 자동 라벨을 front-wide 카메라에 투영한다. 필터는 다음과 같다.
  - 박스의 50% 이상이 영상 안
  - 높이 12 px 이상
  - 거리 80 m 이하
  - 3D 박스 안 LiDAR 점 3개 이상
  - 차량, 보행자, 자전거만
- 평가 clip 중 46개는 데이터셋에 LiDAR가 없다. 카메라와 LiDAR 비교는 모두 LiDAR가 있는 482 / 240 clip에서 했다.
- 한계
  - seed가 하나뿐이다.
  - reasoning과 detection 라벨이 teacher의 pseudo label이다.
  - detection 기준이 사람이 단 라벨이 아니라 자동 라벨이다.
  - 지연은 아직 최적화하지 않았다.

## 예상 질문
- **A1.5가 A2보다 궤적이 좋은 이유**: 이전 OOD 평가에서도 같은 경향이었다.
- **detection 기준이 정답 라벨인가**: 사람이 단 라벨이 아니라 자동 라벨이라는 점을 먼저 밝힌다.
- **LiDAR가 효과 없다는 결론의 범위**: "A2 라벨로 증류할 때"라는 조건을 붙여 말한다.
