"""과제 제출용 단일 진입점. 저장소 루트에서 실행한다.

  python main.py report            저장된 결과(results/)로 비교표·혼동행렬을 다시 만든다. 수 초, GPU 불필요.
  python main.py eval              동봉된 최종 체크포인트로 보류 test 50개 시행을 평가한다. CPU 가능, 수 분.
  python main.py train             6개 모델의 5-fold 교차검증을 처음부터 다시 돌린다. GPU 약 13시간.
  python main.py final             최종 모델: 교차검증 -> development 전체 재학습 -> test 1회 평가. GPU 약 1.5시간.

  python main.py train --smoke     동작 확인용. EfficientNet-B0를 1 epoch, fold 0만 학습한다(수치는 의미 없음).
  python main.py train --dry-run   실행할 명령만 출력한다(final도 동일).

실제 작업은 src/ 의 스크립트가 한다. 이 파일은 논문 설정 기준선 -> 후보 5개 순서와 인자를
README에 적힌 그대로 호출할 뿐이다. 결과는 src/runs/ 에 쌓이며(git 제외), 같은 이름의 실행이
이미 끝나 있으면 건너뛴다.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
RUNS = SRC / "runs"
CHECKPOINT = ROOT / "checkpoints" / "efficientnet_b0_final.pt"
SHIPPED_TEST = ROOT / "results" / "final" / "test_metrics.json"
SHIPPED_SPLITS = ROOT / "results" / "splits.json"
DATA_DIR = ROOT / "data"

# 논문 설정(기준선). 증강·정규화는 전부 끄고, 논문이 값을 밝히지 않은 dropout만 0.2로 둔다.
PAPER_SETTING = [
    "--model", "densenet161", "--fold", "all", "--epochs", "45", "--batch-size", "16",
    "--lr", "1e-3", "--min-lr", "1e-3", "--warmup-epochs", "0", "--weight-decay", "0",
    "--dropout", "0.2", "--drop-rate", "0", "--label-smoothing", "0", "--mixup", "0",
    "--ema-decay", "0", "--augment", "none", "--train-hop", "150", "--cwt-norm", "none",
]
# 새 학습 레시피(train.py 기본값: 150 epoch, batch 64, AdamW lr 6e-4 / wd 1e-2, 강한 증강, mixup 0.2,
# label smoothing 0.1, EMA 0.999, 학습 hop 50, log1p CWT 정규화)로 같은 분할에서 비교하는 후보 5개.
CANDIDATES = [
    ("cand_densenet_s_cwt", "densenet_s_cwt"),
    ("cand_densenet121_cwt", "densenet121_cwt"),
    ("cand_densenet_xs_cwt", "densenet_xs_cwt"),
    ("cand_efficientnet_b0", "efficientnet_b0"),
    ("cand_densenet161_newrecipe", "densenet161"),
]
BASELINE = "baseline_densenet161"


def run(script, arguments, dry_run=False):
    command = [sys.executable, script, *arguments]
    print("$ " + " ".join(f'"{c}"' if " " in c else c for c in command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=SRC, check=True)


def finished(name):
    return (RUNS / name / "summary.json").exists()


def guard(name):
    """끝난 실행은 건너뛰고, 중간에 끊긴 실행이 있으면 지우도록 안내한다."""
    if finished(name):
        print(f"[skip] runs/{name} 은(는) 이미 완료되었습니다.")
        return False
    if (RUNS / name).exists():
        raise SystemExit(f"src/runs/{name} 이(가) 중간에 끊긴 채 남아 있습니다. 지우고 다시 실행하세요.")
    return True


def common(args):
    return ["--device", args.device]


def command_report(args):
    run("make_report.py", [], args.dry_run)


def command_train(args):
    if args.smoke:
        name = "smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        run("train.py", ["--model", "efficientnet_b0", "--fold", "0", "--epochs", "1",
                         "--train-hop", "150", "--run-dir", f"runs/{name}", *common(args)], args.dry_run)
        return
    if guard(BASELINE):
        run("train.py", [*PAPER_SETTING, "--run-dir", f"runs/{BASELINE}", *common(args)], args.dry_run)
    manifest = f"runs/{BASELINE}/splits.json"       # 이후 모든 실행이 같은 분할을 쓴다.
    for name, model in CANDIDATES:
        if guard(name):
            run("train.py", ["--model", model, "--fold", "all", "--manifest", manifest,
                             "--run-dir", f"runs/{name}", *common(args)], args.dry_run)
    if not args.dry_run:
        run("report.py", [])


def command_final(args):
    # 확정 설정 = EfficientNet-B0 + train.py 기본값 + 증강 끔(results/autopilot/REPORT.md의 6단계 동결 기록).
    cv = "final_cv"
    manifest = (f"runs/{BASELINE}/splits.json" if finished(BASELINE) else str(SHIPPED_SPLITS))
    if guard(cv):
        run("train.py", ["--model", "efficientnet_b0", "--fold", "all", "--augment", "none",
                         "--manifest", manifest, "--data-dir", str(DATA_DIR),
                         "--run-dir", f"runs/{cv}", *common(args)], args.dry_run)
    if not (RUNS / cv / "refit" / "final.pt").exists():
        run("refit.py", ["--cv-run", f"runs/{cv}", "--data-dir", str(DATA_DIR), *common(args)],
            args.dry_run)
    out = RUNS / cv / "refit" / "test_metrics.json"
    if out.exists():
        print("[skip] test는 이미 평가되었습니다(한 번만 열도록 evaluate.py가 거부합니다).")
    else:
        run("evaluate.py", ["--checkpoint", f"runs/{cv}/refit/final.pt",
                            "--data-dir", str(DATA_DIR), *common(args)], args.dry_run)


def command_eval(args):
    if not CHECKPOINT.exists():
        raise SystemExit(f"체크포인트가 없습니다: {CHECKPOINT}")
    output = RUNS / f"reproduced_test_{datetime.now():%Y%m%d_%H%M%S}.json"
    run("evaluate.py", ["--checkpoint", str(CHECKPOINT), "--data-dir", str(DATA_DIR),
                        "--output", str(output), *common(args)], args.dry_run)
    if args.dry_run:
        return
    new = json.loads(output.read_text(encoding="utf-8"))["metrics"]
    old = json.loads(SHIPPED_TEST.read_text(encoding="utf-8"))["metrics"]
    print("\n지표                  이번 평가    저장된 결과(results/final)")
    rows = [("window accuracy", new["window"]["accuracy"], old["window"]["accuracy"]),
            ("window macro-F1", new["window"]["macro_f1"], old["window"]["macro_f1"]),
            ("k=3 (600 ms)", new["consecutive"]["k3"], old["consecutive"]["k3"]),
            ("k=5 (900 ms)", new["consecutive"]["k5"], old["consecutive"]["k5"]),
            ("trial (3 s)", new["trial"]["accuracy"], old["trial"]["accuracy"])]
    for label, a, b in rows:
        print(f"{label:<20} {a:>9.2%}    {b:>9.2%}")
    same = new["window"]["confusion_matrix"] == old["window"]["confusion_matrix"]
    print("혼동행렬:", "저장된 결과와 완전히 일치" if same else
          "저장된 결과와 다름(GPU fp16 자동혼합정밀 vs CPU fp32의 수치 차이일 수 있음)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=["report", "eval", "train", "final"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="명령만 출력하고 실행하지 않는다")
    parser.add_argument("--smoke", action="store_true", help="train: 1 epoch, fold 0만 (동작 확인용)")
    args = parser.parse_args()
    {"report": command_report, "eval": command_eval,
     "train": command_train, "final": command_final}[args.command](args)


if __name__ == "__main__":
    main()
