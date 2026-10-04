"""3~6단계를 무인으로 수행한다.

  python autopilot.py --dry-run          계획과 예상 시간만 출력
  python autopilot.py                    실행 (기본 예산 24시간)
  python autopilot.py --budget-hours 10  예산 축소

앞 단계의 결과를 읽어 다음 단계의 설정을 스스로 정한다. 남은 예산이 모자라면
우선순위가 낮은 작업을 건너뛰되, 최종 test 평가 몫은 처음부터 떼어 둔다.
중단 후 다시 실행하면 완료된 작업을 건너뛰고 이어서 진행한다.

test 집합은 6단계에서 정확히 한 번만 열린다. 그 시점의 설정을 state.json에
먼저 기록(동결)한 뒤 평가하며, 이미 평가했으면 다시 실행하지 않는다.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
WORK = RUNS / "_autopilot"
STATE_PATH = WORK / "state.json"
STATUS_PATH = WORK / "status.txt"
REPORT_PATH = WORK / "REPORT.md"

# 5-fold 기준선 실행에서 실측한 값. 첫 작업 이후에는 실제 측정치로 대체된다.
DEFAULT_SEC_PER_EPOCH_FOLD = {"efficientnet_b0": 5.8, "densenet_xs_cwt": 6.0,
                              "densenet_s_cwt": 11.3, "densenet121_cwt": 14.1,
                              "densenet161": 20.5}
PROCESS_OVERHEAD_SEC = 120      # CWT 사전계산 등 학습 외 고정 비용
SAFETY = 1.15
REFIT_RESERVE_HOURS = 1.0
# ablation이 확정 설정을 이 폭 이상 앞설 때만 최종 설정을 바꾼다. 잡음으로 뒤집히지 않게 한다.
SWITCH_MARGIN = 0.003

BASE = {"epochs": 150, "batch-size": 64, "lr": 6e-4, "weight-decay": 1e-2,
        "train-hop": 50, "cwt-norm": "log_minmax", "augment": "strong",
        "mixup": 0.2, "label-smoothing": 0.1, "ema-decay": 0.999,
        "dropout": 0.3, "drop-rate": 0.1, "warmup-epochs": 10}

# 3단계 탐색 공간. 앞쪽이 우선순위가 높고, 예산이 모자라면 뒤쪽부터 잘린다.
HP_GRID = [(6e-4, 1e-2), (1e-3, 1e-2), (3e-4, 1e-2), (6e-4, 5e-2), (6e-4, 1e-3),
           (1e-3, 5e-2), (1e-3, 1e-3), (3e-4, 5e-2), (3e-4, 1e-3)]
BATCH_OPTIONS = (32, 128)       # 64는 격자에서 이미 확인됨
EPOCH_OPTIONS = (250, 400)      # 150은 격자에서 이미 확인됨
TOP_K_REFINE = 3                # fold 1로 재확인할 상위 조합 수

# 4단계 ablation: 이름 -> 끄는 설정. 기대 효과가 큰 순서.
ABLATIONS = [("abl_no_augment", {"augment": "none"}),
             ("abl_hop150", {"train-hop": 150}),
             ("abl_no_mixup", {"mixup": 0.0}),
             ("abl_no_cwtnorm", {"cwt-norm": "none"}),
             ("abl_no_ema", {"ema-decay": 0.0})]


# --------------------------------------------------------------------------- 상태

def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"schema": 1, "started": datetime.now().isoformat(timespec="seconds"),
            "jobs": {}, "decisions": [], "frozen": None, "test": None}


def save_state(state):
    WORK.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def log(state, message):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)
    state.setdefault("decisions", []).append(f"{stamp} {message}")


# --------------------------------------------------------------------------- 시간

class Budget:
    def __init__(self, hours, reserve_hours):
        self.deadline = time.monotonic() + hours * 3600
        self.reserve = reserve_hours * 3600
        self.start = time.monotonic()

    def remaining(self, with_reserve=True):
        left = self.deadline - time.monotonic()
        return left - self.reserve if with_reserve else left

    def elapsed_hours(self):
        return (time.monotonic() - self.start) / 3600


def windows_per_trial(hop):
    return (3000 - 300) // hop + 1


def estimate_seconds(state, model, epochs, folds, hop):
    per = state.get("measured", {}).get(model) or DEFAULT_SEC_PER_EPOCH_FOLD.get(model, 12.0)
    ratio = windows_per_trial(hop) / windows_per_trial(50)
    return per * epochs * folds * ratio * SAFETY + PROCESS_OVERHEAD_SEC


def record_measurement(state, model, run_dir, hop):
    """완료된 실행에서 epoch-fold당 실제 초를 뽑아 추정치를 갱신한다."""
    import csv
    seconds, count = 0.0, 0
    for path in sorted(run_dir.glob("fold_*/history.csv")):
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        seconds += sum(float(r["seconds"]) for r in rows)
        count += len(rows)
    if not count:
        return
    normalized = (seconds / count) * (windows_per_trial(50) / windows_per_trial(hop))
    previous = state.setdefault("measured", {}).get(model)
    state["measured"][model] = normalized if previous is None else (previous + normalized) / 2


# --------------------------------------------------------------------------- 실행

def cli_args(overrides):
    out = []
    for key, value in overrides.items():
        out += [f"--{key}", str(value)]
    return out


def read_summary(run_dir):
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    s = json.loads(path.read_text(encoding="utf-8"))
    return {"macro_f1": s["window"]["macro_f1"]["mean"],
            "accuracy": s["window"]["accuracy"]["mean"],
            "accuracy_std": s["window"]["accuracy"]["std"],
            "macro_f1_std": s["window"]["macro_f1"]["std"],
            "k3": s["consecutive"]["k3"]["mean"],
            "trial": s["trial"]["accuracy"]["mean"],
            "folds": len(s["folds"]),
            "best_epochs": [f["best_epoch"] for f in s["folds"]],
            "loss": statistics.mean(f["validation"]["loss"] for f in s["folds"])}


def run_job(state, budget, name, model, overrides, fold, manifest, mandatory=False):
    """하나의 train.py 실행. 반환: 결과 dict 또는 None(건너뜀/실패)."""
    done = state["jobs"].get(name)
    if done and done.get("status") == "ok":
        return done["result"]
    run_dir = RUNS / name
    if run_dir.exists() and (run_dir / "summary.json").exists():
        result = read_summary(run_dir)
        state["jobs"][name] = {"status": "ok", "result": result, "config": overrides,
                               "model": model, "fold": fold, "reused": True}
        save_state(state)
        return result

    folds = 5 if fold == "all" else 1
    need = estimate_seconds(state, model, int(overrides.get("epochs", BASE["epochs"])),
                            folds, int(overrides.get("train-hop", BASE["train-hop"])))
    if not mandatory and need > budget.remaining():
        log(state, f"SKIP {name}: 예상 {need/3600:.2f}h > 남은 예산 {budget.remaining()/3600:.2f}h")
        state["jobs"][name] = {"status": "skipped_budget", "estimate_hours": need / 3600}
        save_state(state)
        return None

    if run_dir.exists():
        run_dir.rename(run_dir.with_name(run_dir.name + ".incomplete-" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    WORK.mkdir(parents=True, exist_ok=True)
    log_path = WORK / f"{name}.log"
    command = [sys.executable, str(HERE / "train.py"), "--model", model, "--fold", fold,
               "--manifest", str(manifest), "--run-dir", str(run_dir)] + cli_args(overrides)
    log(state, f"RUN  {name} ({model}, fold={fold}, 예상 {need/3600:.2f}h)")
    save_state(state)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode
    spent = (time.monotonic() - started) / 3600
    result = read_summary(run_dir)
    if result is None:
        log(state, f"FAIL {name}: exit {code}, {spent:.2f}h 소모. 로그 {log_path.name}")
        state["jobs"][name] = {"status": "failed", "exit": code, "hours": spent}
        save_state(state)
        return None
    record_measurement(state, model, run_dir, int(overrides.get("train-hop", BASE["train-hop"])))
    log(state, f"OK   {name}: macro-F1 {result['macro_f1']:.4f} acc {result['accuracy']:.4f} ({spent:.2f}h)")
    state["jobs"][name] = {"status": "ok", "result": result, "config": overrides,
                           "model": model, "fold": fold, "hours": spent}
    save_state(state)
    write_status(state, budget)
    return result


# --------------------------------------------------------------------------- 단계

def phase3(state, budget, model, manifest):
    """3단계. fold 0 -> 상위 3개를 fold 1로 재확인 -> batch -> epochs."""
    scores = {}
    for lr, wd in HP_GRID:
        name = f"hp_lr{lr:g}_wd{wd:g}_f0"
        overrides = {**BASE, "lr": lr, "weight-decay": wd}
        r = run_job(state, budget, name, model, overrides, "0", manifest,
                    mandatory=(lr, wd) == HP_GRID[0])
        if r:
            scores[(lr, wd)] = r["macro_f1"]
    if not scores:
        raise SystemExit("3단계에서 성공한 실행이 없습니다.")

    top = sorted(scores, key=scores.get, reverse=True)[:TOP_K_REFINE]
    log(state, "3a fold0 순위: " + ", ".join(f"lr{lr:g}/wd{wd:g}={scores[(lr,wd)]:.4f}" for lr, wd in top))
    combined = {}
    for lr, wd in top:
        name = f"hp_lr{lr:g}_wd{wd:g}_f1"
        r = run_job(state, budget, name, model, {**BASE, "lr": lr, "weight-decay": wd}, "1", manifest)
        combined[(lr, wd)] = (scores[(lr, wd)] + r["macro_f1"]) / 2 if r else scores[(lr, wd)]
    lr, wd = max(combined, key=combined.get)
    log(state, f"3a 확정: lr={lr:g} wd={wd:g} (fold0+1 평균 macro-F1 {combined[(lr,wd)]:.4f})")

    # 3b. batch size. 학습률은 선형 스케일.
    best_batch, best_score = 64, scores[(lr, wd)]
    for batch in BATCH_OPTIONS:
        name = f"hp_b{batch}_f0"
        r = run_job(state, budget, name, model,
                    {**BASE, "lr": lr * batch / 64, "weight-decay": wd, "batch-size": batch},
                    "0", manifest)
        if r and r["macro_f1"] > best_score:
            best_batch, best_score = batch, r["macro_f1"]
    if best_batch != 64:
        lr = lr * best_batch / 64
        log(state, f"3b 확정: batch={best_batch}, lr을 {lr:g}로 선형 스케일")
    else:
        log(state, f"3b 확정: batch=64 유지 (lr {lr:g})")

    # 3c. epoch 길이. 검증 정확도가 끝까지 오르고 있었으므로 연장을 확인한다.
    best_epochs, best_score = BASE["epochs"], best_score
    for epochs in EPOCH_OPTIONS:
        name = f"hp_ep{epochs}_f0"
        r = run_job(state, budget, name, model,
                    {**BASE, "lr": lr, "weight-decay": wd, "batch-size": best_batch,
                     "epochs": epochs, "warmup-epochs": max(10, epochs // 15)}, "0", manifest)
        if r and r["macro_f1"] > best_score:
            best_epochs, best_score = epochs, r["macro_f1"]
    log(state, f"3c 확정: epochs={best_epochs}")

    final = {**BASE, "lr": lr, "weight-decay": wd, "batch-size": best_batch,
             "epochs": best_epochs, "warmup-epochs": max(10, best_epochs // 15)}
    state["chosen"] = {"model": model, **{k: v for k, v in final.items()}}
    save_state(state)
    return final


def phase_reference(state, budget, model, final, manifest):
    return run_job(state, budget, "final_base", model, final, "all", manifest, mandatory=True)


def phase4(state, budget, model, final, manifest):
    for name, patch in ABLATIONS:
        run_job(state, budget, name, model, {**final, **patch}, "all", manifest)


def phase5(state, budget, model, final, manifest):
    for seed in (123, 2026):
        run_job(state, budget, f"final_seed{seed}", model, {**final, "seed": seed}, "all", manifest)


def choose_final_run(state, notes=None):
    """동결 대상 5-fold 실행을 고른다. 반환 (이름, fold_all dict) 또는 (None, {}).

    후보는 서로 다른 설정인 확정 설정과 ablation뿐이다. seed 실행은 설정이 같으므로
    여기서 고르면 검증 잡음을 고르는 것이 되어 제외한다. ablation은 SWITCH_MARGIN을
    넘게 앞설 때만 확정 설정을 대체한다.
    """
    say = notes.append if notes is not None else (lambda _: None)
    fold_all = {n: j for n, j in state["jobs"].items()
                if j.get("status") == "ok" and j.get("fold") == "all"}
    if not fold_all:
        return None, {}
    score = lambda n: fold_all[n]["result"]["macro_f1"]
    ablations = {n: j for n, j in fold_all.items() if n.startswith("abl_")}
    if "final_base" not in fold_all:
        best = max(fold_all, key=score)
        say(f"final_base가 없어 완료된 실행 중 최고인 {best}를 사용합니다.")
        return best, fold_all
    best = "final_base"
    if ablations:
        challenger = max(ablations, key=score)
        gap = (score(challenger) - score(best)) * 100
        if score(challenger) > score(best) + SWITCH_MARGIN:
            say(f"확정 설정 교체: {challenger}가 final_base를 {gap:+.2f}%p 앞섬 "
                f"(기준 {SWITCH_MARGIN*100:.1f}%p)")
            best = challenger
        else:
            say(f"확정 설정 유지: 최고 ablation {challenger}와의 차이가 {gap:+.2f}%p로 기준 미만")
    return best, fold_all


def phase6(state, model, final):
    """6단계. 설정을 동결 기록한 뒤 test를 정확히 한 번 연다."""
    if state.get("test"):
        log(state, "6단계 생략: test는 이미 평가되었습니다.")
        return
    notes = []
    best, fold_all = choose_final_run(state, notes)
    for note in notes:
        log(state, note)
    if best is None:
        log(state, "6단계 중단: 완료된 5-fold 실행이 없습니다.")
        return
    score = lambda n: fold_all[n]["result"]["macro_f1"]
    state["frozen"] = {"run": best, "model": model, "config": fold_all[best]["config"],
                       "validation_macro_f1": score(best), "switch_margin": SWITCH_MARGIN,
                       "ranking": sorted(((n, score(n)) for n in fold_all), key=lambda x: -x[1]),
                       "seed_runs_excluded": sorted(n for n in fold_all if n.startswith("final_seed")),
                       "frozen_at": datetime.now().isoformat(timespec="seconds")}
    save_state(state)
    log(state, f"동결: {best} (검증 macro-F1 {score(best):.4f}) -> refit 시작")

    run_dir = RUNS / best
    refit_dir = run_dir / "refit"
    if not (refit_dir / "final.pt").exists():
        if refit_dir.exists():
            refit_dir.rename(refit_dir.with_name("refit.incomplete-" + datetime.now().strftime("%H%M%S")))
        with (WORK / "refit.log").open("w", encoding="utf-8") as handle:
            code = subprocess.run([sys.executable, str(HERE / "refit.py"), "--cv-run", str(run_dir)],
                                  stdout=handle, stderr=subprocess.STDOUT).returncode
        if not (refit_dir / "final.pt").exists():
            log(state, f"refit 실패 (exit {code}). test는 열지 않았습니다.")
            save_state(state)
            return
    log(state, "refit 완료 -> test 평가 1회")
    with (WORK / "evaluate.log").open("w", encoding="utf-8") as handle:
        subprocess.run([sys.executable, str(HERE / "evaluate.py"),
                        "--checkpoint", str(refit_dir / "final.pt")],
                       stdout=handle, stderr=subprocess.STDOUT)
    result_path = refit_dir / "test_metrics.json"
    if not result_path.exists():
        log(state, "test 평가 실패. evaluate.log를 확인하세요.")
        save_state(state)
        return
    m = json.loads(result_path.read_text(encoding="utf-8"))["metrics"]
    state["test"] = {"run": best, "path": str(result_path),
                     "window_accuracy": m["window"]["accuracy"], "macro_f1": m["window"]["macro_f1"],
                     "k3": m["consecutive"]["k3"], "k5": m["consecutive"]["k5"],
                     "trial_accuracy": m["trial"]["accuracy"],
                     "confusion_matrix": m["window"]["confusion_matrix"]}
    log(state, f"TEST window {m['window']['accuracy']:.2%} / k3 {m['consecutive']['k3']:.2%} "
               f"/ trial {m['trial']['accuracy']:.2%}")
    save_state(state)


# --------------------------------------------------------------------------- 출력

def write_status(state, budget):
    lines = [f"autopilot  경과 {budget.elapsed_hours():.2f}h  남은 예산 {budget.remaining(False)/3600:.2f}h",
             f"갱신 {datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    for name, job in state["jobs"].items():
        if job.get("status") == "ok":
            r = job["result"]
            lines.append(f"  {name:<28} ok       macro-F1 {r['macro_f1']:.4f}  acc {r['accuracy']:.4f}  fold={job.get('fold')}")
        else:
            lines.append(f"  {name:<28} {job.get('status')}")
    if state.get("chosen"):
        c = state["chosen"]
        lines += ["", f"  확정 설정: lr={c['lr']:g} wd={c['weight-decay']:g} batch={c['batch-size']} epochs={c['epochs']}"]
    if state.get("test"):
        t = state["test"]
        lines += ["", f"  TEST: window {t['window_accuracy']:.2%}  k3 {t['k3']:.2%}  trial {t['trial_accuracy']:.2%}"]
    WORK.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_report(state, budget):
    ok = {n: j for n, j in state["jobs"].items() if j.get("status") == "ok"}
    out = ["# Autopilot 결과 (3~6단계)", "",
           f"시작 {state['started']} · 경과 {budget.elapsed_hours():.2f} h · "
           f"작업 {len(ok)}/{len(state['jobs'])} 완료", ""]

    c = state.get("chosen")
    if c:
        out += ["## 확정 설정 (3단계 결과)", "",
                f"`--model {c['model']} --lr {c['lr']:g} --weight-decay {c['weight-decay']:g} "
                f"--batch-size {c['batch-size']} --epochs {c['epochs']}`", ""]

    def table(names, title):
        rows = [(n, ok[n]["result"]) for n in names if n in ok]
        if not rows:
            return []
        body = ["## " + title, "",
                "| 실행 | fold 수 | window acc | macro-F1 | k3 | trial |", "|---|---:|---:|---:|---:|---:|"]
        for n, r in sorted(rows, key=lambda x: -x[1]["macro_f1"]):
            std = f" ±{r['accuracy_std']*100:.2f}" if r["accuracy_std"] else ""
            body.append(f"| `{n}` | {r['folds']} | {r['accuracy']*100:.2f}{std} | "
                        f"{r['macro_f1']*100:.2f} | {r['k3']*100:.2f} | {r['trial']*100:.1f} |")
        return body + [""]

    out += table([n for n in ok if n.startswith("hp_")], "3단계 탐색 (fold 0/1)")
    out += table(["final_base"], "확정 설정 5-fold")

    abl = [n for n, _ in ABLATIONS if n in ok]
    if abl and "final_base" in ok:
        base = ok["final_base"]["result"]["macro_f1"]
        out += ["## 4단계 ablation (확정 설정에서 하나씩 제거)", "",
                "| 제거한 것 | macro-F1 | 기여도 |", "|---|---:|---:|"]
        for n in abl:
            v = ok[n]["result"]["macro_f1"]
            out.append(f"| `{n}` | {v*100:.2f} | {(base-v)*100:+.2f}%p |")
        out += ["", f"기여도는 확정 설정({base*100:.2f})에서 뺀 값이다. 양수면 그 요소가 도움이 됐다는 뜻.", ""]

    seeds = [n for n in ok if n.startswith("final_seed")]
    if seeds:
        out += table(seeds + ["final_base"], "5단계 seed 재확인")

    if state.get("frozen"):
        f = state["frozen"]
        out += ["## 6단계 동결 기록", "",
                f"- 동결 시각: {f['frozen_at']} (test는 이 기록 뒤에 처음 열렸다)",
                f"- 선택된 실행: `{f['run']}`",
                f"- 검증 macro-F1: {f['validation_macro_f1']*100:.2f}",
                f"- 교체 기준: ablation이 확정 설정을 {f.get('switch_margin', 0)*100:.1f}%p 넘게 앞설 때만 교체",
                "- 후보에서 제외된 seed 실행: " + (", ".join(f"`{n}`" for n in f.get("seed_runs_excluded", [])) or "없음"),
                "- 5-fold 실행 순위: " + ", ".join(f"`{n}` {v*100:.2f}" for n, v in f["ranking"]), ""]
    if state.get("test"):
        t = state["test"]
        out += ["## 최종 test 결과 (보류했던 50개 시행, 1회 평가)", "",
                "| 판정 단위 | 값 |", "|---|---:|",
                f"| 윈도우 1개 (300 ms) | {t['window_accuracy']*100:.2f}% |",
                f"| macro-F1 | {t['macro_f1']*100:.2f} |",
                f"| 연속 3개 (600 ms) | {t['k3']*100:.2f}% |",
                f"| 연속 5개 (900 ms) | {t['k5']*100:.2f}% |",
                f"| 시행 전체 (3 s) | {t['trial_accuracy']*100:.2f}% |", "",
                "혼동행렬 (행=정답 A~E):", "```"]
        for i, row in enumerate(t["confusion_matrix"]):
            out.append(f"{'ABCDE'[i]} {row}")
        out += ["```", ""]

    skipped = [n for n, j in state["jobs"].items() if j.get("status") != "ok"]
    if skipped:
        out += ["## 수행되지 않은 작업", ""]
        for n in skipped:
            j = state["jobs"][n]
            note = f" (예상 {j['estimate_hours']:.2f}h)" if "estimate_hours" in j else ""
            out.append(f"- `{n}`: {j.get('status')}{note}")
        out.append("")
    out += ["## 실행 로그", "", "```"] + state.get("decisions", []) + ["```", ""]
    REPORT_PATH.write_text("\n".join(out), encoding="utf-8")
    print(f"\n보고서: {REPORT_PATH}")


def plan_preview(state, model, budget_hours):
    per = DEFAULT_SEC_PER_EPOCH_FOLD.get(model, 12.0)
    def h(epochs, folds, hop=50):
        return (per * epochs * folds * windows_per_trial(hop) / windows_per_trial(50) * SAFETY
                + PROCESS_OVERHEAD_SEC) / 3600
    ep = max(EPOCH_OPTIONS) if EPOCH_OPTIONS else BASE["epochs"]
    rows = [(f"3a lr×wd 격자, fold 0 ({len(HP_GRID)}회)", len(HP_GRID) * h(BASE["epochs"], 1)),
            (f"3a 상위 {TOP_K_REFINE}개 fold 1 재확인", TOP_K_REFINE * h(BASE["epochs"], 1)),
            (f"3b batch {'/'.join(map(str, BATCH_OPTIONS))}, fold 0", len(BATCH_OPTIONS) * h(BASE["epochs"], 1)),
            (f"3c epochs {'/'.join(map(str, EPOCH_OPTIONS))}, fold 0", sum(h(e, 1) for e in EPOCH_OPTIONS)),
            ("확정 설정 5-fold (1회)", h(ep, 5)),
            (f"4단계 ablation {len(ABLATIONS)}개 (5-fold)", (len(ABLATIONS) - 1) * h(ep, 5) + h(ep, 5, 150)),
            ("5단계 seed 2개 (5-fold)", 2 * h(ep, 5)),
            ("6단계 refit + test", REFIT_RESERVE_HOURS)]
    print(f"\n계획 ({model}, 예산 {budget_hours} h)\n")
    total = 0.0
    for label, hours in rows:
        total += hours
        print(f"  {label:<38} {hours:6.2f} h   (누적 {total:6.2f} h)")
    print(f"\n  합계 {total:.2f} h — 3c가 가장 긴 {ep} epoch을 고른다고 가정한 최악값입니다.")
    print(f"  더 짧은 epoch이 이기면 4·5단계 비용이 그만큼 줄어듭니다.")
    print(f"  예산을 넘는 작업은 우선순위가 낮은 것부터 자동으로 건너뜁니다.")
    print(f"  6단계 몫 {REFIT_RESERVE_HOURS:.1f} h는 처음부터 떼어 두므로 test 평가는 보장됩니다.")
    print("  epoch당 시간은 첫 작업이 끝나면 실측치로 대체되어 추정이 정확해집니다.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--budget-hours", type=float, default=24.0)
    parser.add_argument("--manifest", type=Path, default=RUNS / "baseline_densenet161" / "splits.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-test", action="store_true", help="6단계를 건너뛴다")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-seeds", action="store_true")
    args = parser.parse_args()

    state = load_state()
    if args.dry_run:
        plan_preview(state, args.model, args.budget_hours)
        return
    if not args.manifest.exists():
        raise SystemExit(f"manifest를 찾을 수 없습니다: {args.manifest}")

    budget = Budget(args.budget_hours, 0.0 if args.no_test else REFIT_RESERVE_HOURS)
    log(state, f"시작: model={args.model} 예산={args.budget_hours}h "
               f"(6단계 몫 {budget.reserve/3600:.1f}h 별도 확보)")
    try:
        final = phase3(state, budget, args.model, args.manifest)
        phase_reference(state, budget, args.model, final, args.manifest)
        if not args.skip_ablation:
            phase4(state, budget, args.model, final, args.manifest)
        if not args.skip_seeds:
            phase5(state, budget, args.model, final, args.manifest)
        if not args.no_test:
            budget.reserve = 0.0
            phase6(state, args.model, final)
    except KeyboardInterrupt:
        log(state, "사용자 중단. 다시 실행하면 이어서 진행합니다.")
    finally:
        save_state(state)
        write_status(state, budget)
        write_report(state, budget)


if __name__ == "__main__":
    main()
