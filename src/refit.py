"""5-fold 교차검증이 끝난 뒤, 선택된 설정으로 development 전체를 재학습한다."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from augment import DEFAULT_AUGMENT, STRONG_AUGMENT
from data import build_dataset, validate_manifest
from engine import (learning_rate_at, seed_everything, select_device, train_epoch, write_json)
from evaluate import expected_preprocessing
from models import decay_groups, make_model
from train import checkpoint_payload, loader_for, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-run", type=Path, required=True, help="Completed --fold all run")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    config = json.loads((args.cv_run / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.cv_run / "splits.json").read_text(encoding="utf-8"))
    summary = json.loads((args.cv_run / "summary.json").read_text(encoding="utf-8"))
    if sorted(row["fold"] for row in summary["folds"]) != list(range(5)):
        raise ValueError("Refit requires all five completed validation folds")
    validate_manifest(manifest)
    if config["preprocessing"] != expected_preprocessing(config):
        raise ValueError("CV preprocessing differs from this code")
    config["epochs"] = int(statistics.median(row["best_epoch"] for row in summary["folds"]))
    config["fold"] = "refit"
    config["cv_run"] = str(args.cv_run.resolve())
    config["data_dir"] = str((args.data_dir or Path(config["data_dir"])).resolve())
    config["device"] = args.device
    device = select_device(args.device)
    config["effective_device"] = str(device)
    config["effective_amp"] = config["amp"] and device.type == "cuda"
    directory = (args.run_dir or args.cv_run / "refit").resolve()
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", config)
    write_json(directory / "splits.json", manifest)
    torch.set_num_threads(config["cpu_threads"])
    seed_everything(config["seed"])
    dataset = build_dataset(manifest["development"], config["data_dir"], config["refilter"],
                            hop=config["train_hop"], cwt_norm=config["cwt_norm"])
    loader = loader_for(dataset, config, device, True, config["seed"])
    model = make_model(config["model"], config["dropout"], config["drop_rate"]).to(device)
    optimizer = torch.optim.AdamW(decay_groups(model, config["weight_decay"]), lr=config["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"] and device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    augment = STRONG_AUGMENT if config["augment"] == "strong" else DEFAULT_AUGMENT
    ema = (AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(config["ema_decay"]),
                         use_buffers=True) if config["ema_decay"] > 0 else None)
    print(f"Refit {len(dataset)} windows for {config['epochs']} epochs (CV median); "
          f"no validation, no early stopping.", flush=True)
    with (directory / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "lr", "loss", "accuracy"])
        writer.writeheader()
        for epoch in range(config["epochs"]):
            lr = learning_rate_at(epoch, config["epochs"], config["lr"],
                                  config["warmup_epochs"], config["min_lr"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            print(f"Epoch {epoch + 1}/{config['epochs']} lr={lr:.3g}", flush=True)
            metrics = train_epoch(model, loader, optimizer, criterion, scaler, device,
                                  config["amp"], augment, config["mixup"], ema)
            writer.writerow({"epoch": epoch + 1, "lr": lr, **metrics})
            file.flush()
    # 교차검증에서 EMA가 더 나았다면 refit도 EMA 가중치를 쓴다.
    sources = [row.get("best_source", "raw") for row in summary["folds"]]
    weights = ema.module if (ema is not None and sources.count("ema") > len(sources) / 2) else model
    config["refit_weights"] = "ema" if weights is not model else "raw"
    save_checkpoint(directory / "final.pt",
                    checkpoint_payload(weights, config, manifest, config["epochs"], None))
    print(f"Saved {directory / 'final.pt'} ({config['refit_weights']} weights); "
          f"test set remains held out.", flush=True)


if __name__ == "__main__":
    main()
