# Autopilot 결과 (3~6단계)

시작 2026-09-24T11:50:28 · 경과 13.62 h · 작업 24/24 완료

## 확정 설정 (3단계 결과)

`--model efficientnet_b0 --lr 0.0006 --weight-decay 0.01 --batch-size 64 --epochs 150`

## 3단계 탐색 (fold 0/1)

| 실행 | fold 수 | window acc | macro-F1 | k3 | trial |
|---|---:|---:|---:|---:|---:|
| `hp_lr0.0006_wd0.01_f1` | 1 | 91.84 | 91.78 | 95.88 | 100.0 |
| `hp_lr0.0003_wd0.01_f1` | 1 | 91.58 | 91.50 | 94.71 | 100.0 |
| `hp_lr0.0006_wd0.01_f0` | 1 | 91.32 | 91.30 | 96.18 | 100.0 |
| `hp_lr0.0003_wd0.05_f1` | 1 | 91.18 | 91.07 | 95.29 | 100.0 |
| `hp_b32_f0` | 1 | 91.05 | 91.04 | 96.18 | 100.0 |
| `hp_lr0.0003_wd0.01_f0` | 1 | 91.05 | 91.04 | 95.88 | 100.0 |
| `hp_lr0.0003_wd0.05_f0` | 1 | 91.05 | 91.03 | 96.47 | 100.0 |
| `hp_lr0.001_wd0.001_f0` | 1 | 90.92 | 90.92 | 96.47 | 100.0 |
| `hp_lr0.001_wd0.05_f0` | 1 | 90.79 | 90.79 | 96.32 | 100.0 |
| `hp_b128_f0` | 1 | 90.79 | 90.78 | 97.06 | 100.0 |
| `hp_lr0.0003_wd0.001_f0` | 1 | 90.53 | 90.51 | 97.79 | 100.0 |
| `hp_ep250_f0` | 1 | 90.39 | 90.35 | 96.03 | 100.0 |
| `hp_lr0.0006_wd0.05_f0` | 1 | 90.26 | 90.26 | 96.18 | 100.0 |
| `hp_ep400_f0` | 1 | 90.13 | 90.12 | 96.18 | 100.0 |
| `hp_lr0.001_wd0.01_f0` | 1 | 90.00 | 90.01 | 96.18 | 100.0 |
| `hp_lr0.0006_wd0.001_f0` | 1 | 90.00 | 89.97 | 96.32 | 100.0 |

## 확정 설정 5-fold

| 실행 | fold 수 | window acc | macro-F1 | k3 | trial |
|---|---:|---:|---:|---:|---:|
| `final_base` | 5 | 91.47 ±1.67 | 91.46 | 95.76 | 100.0 |

## 4단계 ablation (확정 설정에서 하나씩 제거)

| 제거한 것 | macro-F1 | 기여도 |
|---|---:|---:|
| `abl_no_augment` | 91.83 | -0.37%p |
| `abl_hop150` | 90.25 | +1.21%p |
| `abl_no_mixup` | 91.24 | +0.21%p |
| `abl_no_cwtnorm` | 91.58 | -0.12%p |
| `abl_no_ema` | 91.35 | +0.10%p |

기여도는 확정 설정(91.46)에서 뺀 값이다. 양수면 그 요소가 도움이 됐다는 뜻.

## 5단계 seed 재확인

| 실행 | fold 수 | window acc | macro-F1 | k3 | trial |
|---|---:|---:|---:|---:|---:|
| `final_seed2026` | 5 | 91.50 ±1.17 | 91.47 | 96.21 | 100.0 |
| `final_base` | 5 | 91.47 ±1.67 | 91.46 | 95.76 | 100.0 |
| `final_seed123` | 5 | 90.42 ±1.17 | 90.39 | 95.85 | 100.0 |

## 6단계 동결 기록

- 동결 시각: 2026-09-25T01:14:11 (test는 이 기록 뒤에 처음 열렸다)
- 선택된 실행: `abl_no_augment`
- 검증 macro-F1: 91.83
- 교체 기준: ablation이 확정 설정을 0.3%p 넘게 앞설 때만 교체
- 후보에서 제외된 seed 실행: `final_seed123`, `final_seed2026`
- 5-fold 실행 순위: `abl_no_augment` 91.83, `abl_no_cwtnorm` 91.58, `final_seed2026` 91.47, `final_base` 91.46, `abl_no_ema` 91.35, `abl_no_mixup` 91.24, `final_seed123` 90.39, `abl_hop150` 90.25

## 최종 test 결과 (보류했던 50개 시행, 1회 평가)

| 판정 단위 | 값 |
|---|---:|
| 윈도우 1개 (300 ms) | 94.11% |
| macro-F1 | 94.10 |
| 연속 3개 (600 ms) | 98.24% |
| 연속 5개 (900 ms) | 99.87% |
| 시행 전체 (3 s) | 100.00% |

혼동행렬 (행=정답 A~E):
```
A [190, 0, 0, 0, 0]
B [1, 179, 3, 1, 6]
C [0, 1, 173, 7, 9]
D [0, 3, 4, 179, 4]
E [0, 9, 3, 5, 173]
```

## 실행 로그

```
11:50:28 시작: model=efficientnet_b0 예산=24.0h (6단계 몫 1.0h 별도 확보)
11:50:28 RUN  hp_lr0.0006_wd0.01_f0 (efficientnet_b0, fold=0, 예상 0.31h)
12:05:34 OK   hp_lr0.0006_wd0.01_f0: macro-F1 0.9130 acc 0.9132 (0.25h)
12:05:34 RUN  hp_lr0.001_wd0.01_f0 (efficientnet_b0, fold=0, 예상 0.31h)
12:20:32 OK   hp_lr0.001_wd0.01_f0: macro-F1 0.9001 acc 0.9000 (0.25h)
12:20:32 RUN  hp_lr0.0003_wd0.01_f0 (efficientnet_b0, fold=0, 예상 0.31h)
12:35:30 OK   hp_lr0.0003_wd0.01_f0: macro-F1 0.9104 acc 0.9105 (0.25h)
12:35:30 RUN  hp_lr0.0006_wd0.05_f0 (efficientnet_b0, fold=0, 예상 0.31h)
12:50:32 OK   hp_lr0.0006_wd0.05_f0: macro-F1 0.9026 acc 0.9026 (0.25h)
12:50:32 RUN  hp_lr0.0006_wd0.001_f0 (efficientnet_b0, fold=0, 예상 0.31h)
13:05:29 OK   hp_lr0.0006_wd0.001_f0: macro-F1 0.8997 acc 0.9000 (0.25h)
13:05:29 RUN  hp_lr0.001_wd0.05_f0 (efficientnet_b0, fold=0, 예상 0.31h)
13:20:26 OK   hp_lr0.001_wd0.05_f0: macro-F1 0.9079 acc 0.9079 (0.25h)
13:20:26 RUN  hp_lr0.001_wd0.001_f0 (efficientnet_b0, fold=0, 예상 0.31h)
13:35:23 OK   hp_lr0.001_wd0.001_f0: macro-F1 0.9092 acc 0.9092 (0.25h)
13:35:23 RUN  hp_lr0.0003_wd0.05_f0 (efficientnet_b0, fold=0, 예상 0.31h)
13:50:31 OK   hp_lr0.0003_wd0.05_f0: macro-F1 0.9103 acc 0.9105 (0.25h)
13:50:31 RUN  hp_lr0.0003_wd0.001_f0 (efficientnet_b0, fold=0, 예상 0.31h)
14:05:34 OK   hp_lr0.0003_wd0.001_f0: macro-F1 0.9051 acc 0.9053 (0.25h)
14:05:34 3a fold0 순위: lr0.0006/wd0.01=0.9130, lr0.0003/wd0.01=0.9104, lr0.0003/wd0.05=0.9103
14:05:34 RUN  hp_lr0.0006_wd0.01_f1 (efficientnet_b0, fold=1, 예상 0.31h)
14:20:31 OK   hp_lr0.0006_wd0.01_f1: macro-F1 0.9178 acc 0.9184 (0.25h)
14:20:31 RUN  hp_lr0.0003_wd0.01_f1 (efficientnet_b0, fold=1, 예상 0.31h)
14:35:30 OK   hp_lr0.0003_wd0.01_f1: macro-F1 0.9150 acc 0.9158 (0.25h)
14:35:30 RUN  hp_lr0.0003_wd0.05_f1 (efficientnet_b0, fold=1, 예상 0.31h)
14:50:27 OK   hp_lr0.0003_wd0.05_f1: macro-F1 0.9107 acc 0.9118 (0.25h)
14:50:27 3a 확정: lr=0.0006 wd=0.01 (fold0+1 평균 macro-F1 0.9154)
14:50:27 RUN  hp_b32_f0 (efficientnet_b0, fold=0, 예상 0.31h)
15:15:39 OK   hp_b32_f0: macro-F1 0.9104 acc 0.9105 (0.42h)
15:15:39 RUN  hp_b128_f0 (efficientnet_b0, fold=0, 예상 0.41h)
15:27:41 OK   hp_b128_f0: macro-F1 0.9078 acc 0.9079 (0.20h)
15:27:41 3b 확정: batch=64 유지 (lr 0.0006)
15:27:41 RUN  hp_ep250_f0 (efficientnet_b0, fold=0, 예상 0.53h)
15:52:25 OK   hp_ep250_f0: macro-F1 0.9035 acc 0.9039 (0.41h)
15:52:25 RUN  hp_ep400_f0 (efficientnet_b0, fold=0, 예상 0.81h)
16:31:39 OK   hp_ep400_f0: macro-F1 0.9012 acc 0.9013 (0.65h)
16:31:39 3c 확정: epochs=150
16:31:39 RUN  final_base (efficientnet_b0, fold=all, 예상 1.45h)
17:45:08 OK   final_base: macro-F1 0.9146 acc 0.9147 (1.22h)
17:45:08 RUN  abl_no_augment (efficientnet_b0, fold=all, 예상 1.44h)
18:57:28 OK   abl_no_augment: macro-F1 0.9183 acc 0.9184 (1.21h)
18:57:28 RUN  abl_hop150 (efficientnet_b0, fold=all, 예상 0.51h)
19:25:20 OK   abl_hop150: macro-F1 0.9025 acc 0.9029 (0.46h)
19:25:20 RUN  abl_no_mixup (efficientnet_b0, fold=all, 예상 1.49h)
20:38:15 OK   abl_no_mixup: macro-F1 0.9124 acc 0.9126 (1.22h)
20:38:15 RUN  abl_no_cwtnorm (efficientnet_b0, fold=all, 예상 1.46h)
21:51:46 OK   abl_no_cwtnorm: macro-F1 0.9158 acc 0.9161 (1.23h)
21:51:46 RUN  abl_no_ema (efficientnet_b0, fold=all, 예상 1.44h)
22:55:23 OK   abl_no_ema: macro-F1 0.9135 acc 0.9137 (1.06h)
22:55:23 RUN  final_seed123 (efficientnet_b0, fold=all, 예상 1.34h)
00:04:42 OK   final_seed123: macro-F1 0.9039 acc 0.9042 (1.16h)
00:04:42 RUN  final_seed2026 (efficientnet_b0, fold=all, 예상 1.35h)
01:14:11 OK   final_seed2026: macro-F1 0.9147 acc 0.9150 (1.16h)
01:14:11 확정 설정 교체: abl_no_augment가 final_base를 +0.37%p 앞섬 (기준 0.3%p)
01:14:11 동결: abl_no_augment (검증 macro-F1 0.9183) -> refit 시작
01:27:19 refit 완료 -> test 평가 1회
01:27:27 TEST window 94.11% / k3 98.24% / trial 100.00%
```
