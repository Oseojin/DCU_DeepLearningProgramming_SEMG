"""runs/ 아래의 완료된 실행들을 한 표로 비교한다. 학습도 평가도 하지 않고 읽기만 한다."""

import argparse
import json
import statistics
from pathlib import Path


def load(run_dir):
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    if not summary_path.exists() or not config_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    folds = summary["folds"]
    return {
        "run": run_dir.name,
        "model": config["model"],
        "folds": len(folds),
        "params": folds[0]["parameters"],
        "train_windows": folds[0].get("train_windows"),
        "best_epoch": statistics.median(row["best_epoch"] for row in folds),
        "window": summary["window"]["accuracy"],
        "macro_f1": summary["window"]["macro_f1"],
        "k3": summary["consecutive"]["k3"],
        "trial": summary["trial"]["accuracy"],
        "setup": (f"e{config['epochs']} b{config['batch_size']} lr{config['lr']:g} "
                  f"wd{config['weight_decay']:g} hop{config['train_hop']} "
                  f"{config['augment']} mix{config['mixup']:g}"),
    }


def cell(stat):
    mean = f"{stat['mean']:.2%}"
    return mean if stat["std"] is None else f"{mean} ±{stat['std']:.2%}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--sort", choices=["window", "macro_f1", "k3", "trial", "run"],
                        default="macro_f1")
    args = parser.parse_args()
    rows = [row for row in (load(d) for d in sorted(args.runs_dir.iterdir()) if d.is_dir()) if row]
    if not rows:
        raise SystemExit(f"No completed run (summary.json) under {args.runs_dir}")
    if args.sort != "run":
        rows.sort(key=lambda r: r[args.sort]["mean"], reverse=True)

    header = ("run", "model", "params", "n", "ep", "window acc", "macro-F1", "k3", "trial acc")
    widths = [max(len(header[0]), *(len(r["run"]) for r in rows)),
              max(len(header[1]), *(len(r["model"]) for r in rows)),
              11, 2, 4, 16, 16, 16, 16]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(
            (r["run"], r["model"], f"{r['params']:,}", r["folds"], int(r["best_epoch"]),
             cell(r["window"]), cell(r["macro_f1"]), cell(r["k3"]), cell(r["trial"])), widths)))
    print("\nep = fold별 최저 검증손실 epoch의 중앙값 (refit에서 쓰는 값)")
    print("k3 = 연속 3윈도우(600 ms) 확률 평균 기준 정확도\n")
    for r in rows:
        print(f"{r['run']}: {r['setup']}, train_windows={r['train_windows']}")


if __name__ == "__main__":
    main()
