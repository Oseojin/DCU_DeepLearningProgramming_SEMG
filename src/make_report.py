"""results/ 에 저장된 검증·test 결과에서 비교표와 혼동행렬 그림을 만든다.

학습도 평가도 하지 않고 JSON/CSV를 읽기만 한다. numpy와 matplotlib만 필요하다(GPU 불필요).

  python make_report.py                       # ../results 를 읽고 같은 폴더에 쓴다
  python make_report.py --results-dir DIR     # 다른 결과 폴더

생성물 (모두 results/ 아래)
  model_comparison.md / .csv      모델별 Accuracy·Precision·Recall·F1 (5-fold 평균 ± 표준편차)
  per_class_metrics.csv           모델별·클래스별 precision/recall/F1 (5-fold 합산 혼동행렬 기준)
  confusion_summary.md            클래스별 재현율, 상위 오분류 쌍, 대응표본 t 통계량
  window_position_accuracy.csv    윈도우 위치(0~18)별 정확도
  confusion_matrix/*.png          모델별 혼동행렬, 전체 비교 그림, 최종 test 혼동행렬
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASSES = ["A", "B", "C", "D", "E"]

# (run 폴더, 표 이름, 그림 제목, 파일명)
MODELS = [
    ("baseline_densenet161", "DenseNet161 (논문 설정)", "DenseNet161 (paper setting)", "densenet161_paper"),
    ("cand_densenet161_newrecipe", "DenseNet161 (새 학습 레시피)", "DenseNet161 (new recipe)", "densenet161_newrecipe"),
    ("cand_densenet121_cwt", "DenseNet121-CWT", "DenseNet121-CWT", "densenet121_cwt"),
    ("cand_densenet_s_cwt", "DenseNet-S-CWT", "DenseNet-S-CWT", "densenet_s_cwt"),
    ("cand_densenet_xs_cwt", "DenseNet-XS-CWT", "DenseNet-XS-CWT", "densenet_xs_cwt"),
    ("cand_efficientnet_b0", "EfficientNet-B0", "EfficientNet-B0", "efficientnet_b0"),
]
REFERENCE = "cand_efficientnet_b0"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prf(cm):
    """혼동행렬(행=정답, 열=예측)에서 클래스별 precision, recall, F1."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision, recall, f1


def load_run(runs_dir, run):
    directory = runs_dir / run
    summary = read_json(directory / "summary.json")
    folds = [read_json(directory / f"fold_{i}" / "metrics.json") for i in range(5)]
    seconds = 0.0
    for i in range(5):
        with (directory / f"fold_{i}" / "history.csv").open(encoding="utf-8") as file:
            seconds += sum(float(row["seconds"]) for row in csv.DictReader(file))
    matrices = [np.array(f["validation"]["window"]["confusion_matrix"]) for f in folds]
    per_fold = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for cm in matrices:
        precision, recall, f1 = prf(cm)
        per_fold["accuracy"].append(np.trace(cm) / cm.sum())
        per_fold["precision"].append(precision.mean())
        per_fold["recall"].append(recall.mean())
        per_fold["f1"].append(f1.mean())
    positions = np.array([[f["validation"]["by_window_position"][str(p)] for p in range(19)]
                          for f in folds]).mean(axis=0)
    return {
        "run": run, "summary": summary, "parameters": folds[0]["parameters"],
        "hours": seconds / 3600, "matrices": matrices, "pooled": sum(matrices),
        "per_fold": {k: np.array(v) for k, v in per_fold.items()}, "positions": positions,
        "best_epochs": [f["best_epoch"] for f in folds],
    }


def mean_std(values):
    return float(np.mean(values)) * 100, float(np.std(values, ddof=1)) * 100


def draw_cm(ax, cm, title, fontsize=8):
    cm = np.asarray(cm)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    image = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1f}%)", ha="center", va="center",
                    fontsize=fontsize, color="white" if pct[i, j] > 55 else "black")
    ax.set_xticks(range(5), CLASSES)
    ax.set_yticks(range(5), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    return image


def save_single_cm(cm, title, path):
    figure, ax = plt.subplots(figsize=(5.6, 4.8))
    image = draw_cm(ax, cm, title)
    figure.colorbar(image, ax=ax, label="row-normalised (%)", fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def top_confusions(cm, count=4):
    cm = np.asarray(cm)
    pairs = [(int(cm[i, j]), f"{CLASSES[i]}→{CLASSES[j]}") for i in range(5) for j in range(5) if i != j]
    return sorted(pairs, reverse=True)[:count]


def paired_t(a, b):
    diff = np.asarray(a) - np.asarray(b)
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    return diff.mean() * 100, (diff.mean() / se if se > 0 else float("inf")), int((diff > 0).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()
    results = args.results_dir
    runs_dir, figures = results / "runs", results / "confusion_matrix"
    figures.mkdir(parents=True, exist_ok=True)

    runs = {run: load_run(runs_dir, run) for run, *_ in MODELS}
    names = {run: label for run, label, *_ in MODELS}

    # ---------------------------------------------------------------- 비교표
    header = ["모델", "파라미터", "Accuracy", "Precision", "Recall", "F1-score",
              "k=3 (600 ms)", "k=5 (900 ms)", "시행 단위 (3 s)", "학습 시간"]
    rows, csv_rows = [], []
    for run, label, *_ in MODELS:
        r = runs[run]
        cells = {k: mean_std(r["per_fold"][k]) for k in ("accuracy", "precision", "recall", "f1")}
        k3, k5 = r["summary"]["consecutive"]["k3"], r["summary"]["consecutive"]["k5"]
        trial = r["summary"]["trial"]["accuracy"]["mean"] * 100
        fmt = lambda m: f"{m[0]:.2f} ±{m[1]:.2f}"  # noqa: E731
        rows.append([label, f"{r['parameters']:,}", fmt(cells["accuracy"]), fmt(cells["precision"]),
                     fmt(cells["recall"]), fmt(cells["f1"]),
                     f"{k3['mean'] * 100:.2f}", f"{k5['mean'] * 100:.2f}", f"{trial:.0f}%",
                     f"{r['hours']:.2f} h"])
        csv_rows.append({
            "run": run, "model": label, "parameters": r["parameters"],
            "accuracy_mean": cells["accuracy"][0], "accuracy_std": cells["accuracy"][1],
            "precision_mean": cells["precision"][0], "precision_std": cells["precision"][1],
            "recall_mean": cells["recall"][0], "recall_std": cells["recall"][1],
            "f1_mean": cells["f1"][0], "f1_std": cells["f1"][1],
            "k3": k3["mean"] * 100, "k5": k5["mean"] * 100, "trial": trial, "train_hours": r["hours"],
        })
    lines = ["# 모델별 성능 비교 (5-fold 교차검증, 검증 윈도우 760개 × 5 fold = 3,800개)", "",
             "Accuracy/Precision/Recall/F1은 윈도우 단위, fold별로 계산한 macro 평균의 5-fold 평균 ± 표본표준편차(%).", "",
             "| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    (results / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (results / "model_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    # ------------------------------------------------- 클래스별 지표, 혼동 요약
    with (results / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["run", "class", "precision", "recall", "f1", "support"])
        for run, *_ in MODELS:
            precision, recall, f1 = prf(runs[run]["pooled"])
            for i, name in enumerate(CLASSES):
                writer.writerow([run, name, f"{precision[i] * 100:.2f}", f"{recall[i] * 100:.2f}",
                                 f"{f1[i] * 100:.2f}", int(runs[run]["pooled"][i].sum())])

    summary = ["# 혼동행렬 요약 (5-fold 합산, 행=정답, 열=예측, 클래스당 760개)", ""]
    for run, label, *_ in MODELS:
        pooled = runs[run]["pooled"]
        _, recall, _ = prf(pooled)
        order = np.argsort(recall)
        summary += [f"## {label} (`{run}`)", "", "```",
                    *[f"{CLASSES[i]} {pooled[i].tolist()}  recall {recall[i] * 100:.1f}%" for i in range(5)],
                    "```",
                    f"- 가장 잘 분류: {CLASSES[order[-1]]} ({recall[order[-1]] * 100:.1f}%), "
                    f"가장 약한 클래스: {CLASSES[order[0]]} ({recall[order[0]] * 100:.1f}%)",
                    "- 상위 오분류: " + ", ".join(f"{pair} {n}회" for n, pair in top_confusions(pooled)), ""]
    reference = runs[REFERENCE]["per_fold"]["accuracy"]
    summary += ["## 대응표본 t 검정 (같은 fold를 쌍으로, df=4, 양측 5% 임계 t=2.776)", "",
                f"기준: {names[REFERENCE]} 대비 각 모델. 차이 = EfficientNet-B0 − 비교 모델 (%p).", "",
                "| 비교 모델 | 정확도 차이 | t | EfficientNet-B0가 이긴 fold |", "|---|---:|---:|---:|"]
    for run, label, *_ in MODELS:
        if run != REFERENCE:
            diff, t, wins = paired_t(reference, runs[run]["per_fold"]["accuracy"])
            summary.append(f"| {label} | {diff:+.2f} | {t:.2f} | {wins}/5 |")
    (results / "confusion_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    with (results / "window_position_accuracy.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["position", "start_ms"] + [run for run, *_ in MODELS])
        for p in range(19):
            writer.writerow([p, p * 150] + [f"{runs[run]['positions'][p] * 100:.2f}" for run, *_ in MODELS])

    # ---------------------------------------------------------------- 그림
    for run, _, title, stem in MODELS:
        r = runs[run]
        acc = np.trace(r["pooled"]) / r["pooled"].sum() * 100
        save_single_cm(r["pooled"], f"{title}\n5-fold pooled, window acc {acc:.2f}%",
                       figures / f"{stem}.png")
    figure, axes = plt.subplots(2, 3, figsize=(15, 9.6))
    for ax, (run, _, title, _) in zip(axes.ravel(), MODELS):
        pooled = runs[run]["pooled"]
        draw_cm(ax, pooled, f"{title}\nacc {np.trace(pooled) / pooled.sum() * 100:.2f}%", fontsize=7.5)
    figure.suptitle("Confusion matrices, 5-fold pooled (rows: true, columns: predicted)", fontsize=12)
    figure.tight_layout()
    figure.savefig(figures / "all_models.png", dpi=130)
    plt.close(figure)

    test_path = results / "final" / "test_metrics.json"
    if test_path.exists():
        test = read_json(test_path)["metrics"]
        cm = np.array(test["window"]["confusion_matrix"])
        precision, recall, f1 = prf(cm)
        acc = np.trace(cm) / cm.sum() * 100
        save_single_cm(cm, f"EfficientNet-B0 final (held-out test, 50 trials)\nwindow acc {acc:.2f}%",
                       figures / "final_test_efficientnet_b0.png")
        final = ["# 최종 모델 test 결과 (보류했던 50개 시행, 950개 윈도우, 1회 평가)", "",
                 "| 지표 | 값 |", "|---|---:|",
                 f"| Accuracy (윈도우 1개, 300 ms) | {acc:.2f}% |",
                 f"| Precision (macro) | {precision.mean() * 100:.2f}% |",
                 f"| Recall (macro) | {recall.mean() * 100:.2f}% |",
                 f"| F1-score (macro) | {f1.mean() * 100:.2f}% |",
                 f"| 연속 3개 (600 ms) | {test['consecutive']['k3'] * 100:.2f}% |",
                 f"| 연속 5개 (900 ms) | {test['consecutive']['k5'] * 100:.2f}% |",
                 f"| 시행 전체 (3 s) | {test['trial']['accuracy'] * 100:.2f}% |", "",
                 "클래스별 (행=정답):", "", "| 클래스 | precision | recall | F1 | 혼동행렬 행 |", "|---|---:|---:|---:|---|"]
        final += [f"| {CLASSES[i]} | {precision[i] * 100:.2f} | {recall[i] * 100:.2f} | {f1[i] * 100:.2f} | "
                  f"{cm[i].tolist()} |" for i in range(5)]
        final += ["", "상위 오분류: " + ", ".join(f"{pair} {n}회" for n, pair in top_confusions(cm, 5))]
        (results / "final_test.md").write_text("\n".join(final) + "\n", encoding="utf-8")

    print((results / "model_comparison.md").read_text(encoding="utf-8"))
    print(f"Saved tables and {len(MODELS) + 2} figures under {results}")


if __name__ == "__main__":
    main()
