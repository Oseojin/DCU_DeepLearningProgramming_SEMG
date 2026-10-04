"""검증 fold를 학습한다. 보류된 test 집합은 이 파일에서 절대 읽지 않는다."""

import argparse
import copy
import csv
import json
import math
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from augment import DEFAULT_AUGMENT, STRONG_AUGMENT
from data import (DEFAULT_DATA, EVAL_HOP, PREPROCESSING, build_dataset,
                  make_manifest, validate_manifest)
from engine import (EarlyStopping, evaluate, learning_rate_at, seed_everything,
                    select_device, train_epoch, write_json)
from models import MODEL_NAMES, decay_groups, make_model

DIRECTORY = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default="densenet_s_cwt")
    parser.add_argument("--data-dir", type=Path, help=f"Default: {DEFAULT_DATA}")
    parser.add_argument("--manifest", type=Path, help="Reuse a saved splits.json")
    parser.add_argument("--run-dir", type=Path, help="New output directory; must not already exist")
    parser.add_argument("--fold", choices=["0", "1", "2", "3", "4", "all"], default="0")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.3, help="Classifier dropout")
    parser.add_argument("--drop-rate", type=float, default=0.1, help="DenseNet dense-layer dropout")
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.2, help="mixup alpha; 0 disables")
    parser.add_argument("--augment", choices=["none", "strong"], default="strong")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="0 disables weight EMA")
    parser.add_argument("--train-hop", type=int, default=50,
                        help=f"Training window hop in samples. Evaluation always uses {EVAL_HOP}.")
    parser.add_argument("--cwt-norm", choices=["none", "log_minmax"], default="log_minmax")
    parser.add_argument("--tta", action="store_true", help="Average over small time shifts at evaluation")
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=60)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42, help="Model/shuffle seed, offset by fold")
    parser.add_argument("--split-seed", type=int, default=42, help="Data split seed")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--refilter", action="store_true",
                        help="Reapply legacy filters; default is already-filtered CSV")
    args = parser.parse_args()
    for key in ["epochs", "batch_size", "patience", "min_epochs", "cpu_threads", "train_hop"]:
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    for key in ["num_workers", "warmup_epochs", "seed", "split_seed"]:
        if getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} must be nonnegative")
    for key in ["lr", "min_lr", "weight_decay", "dropout", "drop_rate", "min_delta",
                "label_smoothing", "mixup", "ema_decay"]:
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and nonnegative")
    if args.lr <= 0 or args.min_lr > args.lr:
        parser.error("Require lr > 0 and min-lr <= lr")
    if max(args.dropout, args.drop_rate, args.label_smoothing, args.ema_decay) >= 1:
        parser.error("dropout, drop-rate, label-smoothing and ema-decay must be < 1")
    if args.train_hop > 300:
        parser.error("--train-hop must not exceed the 300-sample window length")
    if max(args.seed + 4, args.split_seed) >= 2**32:
        parser.error("Seeds must fit in the NumPy 32-bit range (allow four fold offsets)")
    return args


def loader_for(dataset, config, device, shuffle, seed):
    return DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed), num_workers=config["num_workers"],
        pin_memory=device.type == "cuda", drop_last=False,
    )


def checkpoint_payload(model, config, manifest, epoch, fold, metrics=None):
    return {
        "schema_version": 2,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": config, "manifest": manifest, "epoch": epoch, "fold": fold,
        "validation": metrics,
    }


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_fold(config, manifest, fold_index, run_dir, device):
    directory = run_dir / f"fold_{fold_index}"
    directory.mkdir()
    seed = config["seed"] + fold_index
    seed_everything(seed)
    split = manifest["folds"][fold_index]
    print(f"Preparing fold {fold_index}: {len(split['train'])} train / "
          f"{len(split['validation'])} validation trials", flush=True)
    train_data = build_dataset(split["train"], config["data_dir"], config["refilter"],
                               hop=config["train_hop"], cwt_norm=config["cwt_norm"])
    val_data = build_dataset(split["validation"], config["data_dir"], config["refilter"],
                             hop=EVAL_HOP, cwt_norm=config["cwt_norm"])
    train_loader = loader_for(train_data, config, device, True, seed)
    val_loader = loader_for(val_data, config, device, False, seed)

    model = make_model(config["model"], config["dropout"], config["drop_rate"]).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"model={config['model']} parameters={parameters:,} device={device} "
          f"train_windows={len(train_data)} (hop={config['train_hop']}) "
          f"validation_windows={len(val_data)} (hop={EVAL_HOP})", flush=True)

    optimizer = torch.optim.AdamW(decay_groups(model, config["weight_decay"]), lr=config["lr"])
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    # checkpoint 선택은 label smoothing이 없는 순수 CE로 한다(비교 가능성 유지).
    eval_criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"] and device.type == "cuda")
    augment = STRONG_AUGMENT if config["augment"] == "strong" else DEFAULT_AUGMENT
    ema = (AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(config["ema_decay"]),
                         use_buffers=True) if config["ema_decay"] > 0 else None)
    tta_shifts = (-20, 20) if config["tta"] else ()
    stopping = (EarlyStopping(config["patience"],
                              min(config["min_epochs"], config["epochs"]), config["min_delta"])
                if config["early_stopping"] else None)
    print(f"augment={config['augment']} mixup={config['mixup']} "
          f"label_smoothing={config['label_smoothing']} ema={config['ema_decay']} "
          f"cwt_norm={config['cwt_norm']} early_stopping="
          f"{'on' if stopping else 'off'} epochs={config['epochs']}", flush=True)

    stop_reason = "epochs_completed"
    best_loss, best_epoch, best_metrics, best_source = float("inf"), 0, None, "raw"
    fields = ["epoch", "lr", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
              "val_macro_f1", "val_trial_accuracy", "val_k3", "ema_loss", "ema_accuracy",
              "seconds"]
    with (directory / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(config["epochs"]):
            start = time.perf_counter()
            lr = learning_rate_at(epoch, config["epochs"], config["lr"],
                                  config["warmup_epochs"], config["min_lr"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            print(f"Epoch {epoch + 1}/{config['epochs']} lr={lr:.3g}", flush=True)
            training = train_epoch(model, train_loader, optimizer, criterion, scaler, device,
                                   config["amp"], augment, config["mixup"], ema)
            validation = evaluate(model, val_loader, eval_criterion, device, config["amp"], tta_shifts)
            ema_validation = (evaluate(ema.module, val_loader, eval_criterion, device,
                                       config["amp"], tta_shifts) if ema is not None else None)
            writer.writerow({
                "epoch": epoch + 1, "lr": lr, "train_loss": training["loss"],
                "train_accuracy": training["accuracy"], "val_loss": validation["loss"],
                "val_accuracy": validation["window"]["accuracy"],
                "val_macro_f1": validation["window"]["macro_f1"],
                "val_trial_accuracy": validation["trial"]["accuracy"],
                "val_k3": validation["consecutive"]["k3"],
                "ema_loss": "" if ema_validation is None else ema_validation["loss"],
                "ema_accuracy": "" if ema_validation is None else ema_validation["window"]["accuracy"],
                "seconds": time.perf_counter() - start,
            })
            file.flush()
            print(f"  val_loss={validation['loss']:.5f} "
                  f"val_accuracy={validation['window']['accuracy']:.2%} "
                  f"k3={validation['consecutive']['k3']:.2%} "
                  f"trial={validation['trial']['accuracy']:.2%}"
                  + ("" if ema_validation is None else
                     f" | ema_loss={ema_validation['loss']:.5f} "
                     f"ema_accuracy={ema_validation['window']['accuracy']:.2%}"), flush=True)
            # raw와 EMA 중 검증 손실이 낮은 쪽을 저장한다.
            for source, metrics, weights in (("raw", validation, model),
                                             ("ema", ema_validation, None if ema is None else ema.module)):
                if metrics is not None and metrics["loss"] < best_loss:
                    best_loss, best_epoch, best_metrics, best_source = (
                        metrics["loss"], epoch + 1, metrics, source
                    )
                    save_checkpoint(directory / "best.pt", checkpoint_payload(
                        weights, config, manifest, best_epoch, fold_index, best_metrics
                    ))
            if stopping is not None and stopping.update(best_loss, epoch + 1):
                stop_reason = "early_stopping"
                print(f"Early stopping; best epoch={best_epoch}", flush=True)
                break
    summary = {"fold": fold_index, "parameters": parameters, "best_epoch": best_epoch,
               "best_source": best_source, "completed_epochs": epoch + 1,
               "early_stopping_enabled": config["early_stopping"], "stop_reason": stop_reason,
               "train_windows": len(train_data), "validation_windows": len(val_data),
               "validation": best_metrics}
    write_json(directory / "metrics.json", summary)
    print(f"Fold {fold_index} finished: {stop_reason}; completed={epoch + 1}/{config['epochs']}; "
          f"best epoch={best_epoch} ({best_source}), val_loss={best_loss:.5f}", flush=True)
    return summary


def main():
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    device = select_device(args.device)
    if device.type == "cpu":
        print("Using CPU. CUDA training needs a CUDA-enabled PyTorch installation.", flush=True)
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        root = (args.data_dir or Path(manifest["data_root"])).resolve()
    else:
        root = (args.data_dir or DEFAULT_DATA).resolve()
        manifest = make_manifest(root, args.split_seed)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"data_dir": str(root), "split_seed": manifest["split_seed"],
                   "preprocessing": {**PREPROCESSING, "refilter": args.refilter,
                                     "train_hop": args.train_hop, "cwt_norm": args.cwt_norm},
                   "augmentation": copy.deepcopy(
                       STRONG_AUGMENT if args.augment == "strong" else DEFAULT_AUGMENT),
                   "torch_version": str(torch.__version__),
                   "torchvision_version": str(torchvision.__version__),
                   "effective_device": str(device),
                   "effective_amp": args.amp and device.type == "cuda"})
    run_dir = args.run_dir or DIRECTORY / "runs" / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + args.model + "_" + uuid.uuid4().hex[:6]
    )
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "splits.json", manifest)
    indices = range(5) if args.fold == "all" else [int(args.fold)]
    summaries = [train_fold(config, manifest, index, run_dir, device) for index in indices]
    aggregate = {"folds": summaries, "test_evaluated": False}
    for level in ["window", "trial"]:
        aggregate[level] = {}
        for metric in ["accuracy", "macro_f1", "weighted_f1"]:
            values = [s["validation"][level][metric] for s in summaries]
            aggregate[level][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
    aggregate["consecutive"] = {
        key: {"mean": float(np.mean([s["validation"]["consecutive"][key] for s in summaries])),
              "std": (float(np.std([s["validation"]["consecutive"][key] for s in summaries], ddof=1))
                      if len(summaries) > 1 else None)}
        for key in summaries[0]["validation"]["consecutive"]
    }
    write_json(run_dir / "summary.json", aggregate)
    print(f"Saved: {run_dir}\nTest set remains held out.", flush=True)


if __name__ == "__main__":
    main()
