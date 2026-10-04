"""checkpoint에 저장된 분할과 전처리를 그대로 사용하는 최종 test 평가."""

import argparse
from pathlib import Path

import torch
from torch import nn

from data import EVAL_HOP, PREPROCESSING, build_dataset, validate_manifest
from engine import evaluate, select_device, write_json
from models import make_model
from train import loader_for


def expected_preprocessing(config):
    return {**PREPROCESSING, "refilter": config["refilter"],
            "train_hop": config["train_hop"], "cwt_norm": config["cwt_norm"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path,
                        help="Override CSV location; file hashes must still match")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, help="New JSON path, defaults beside checkpoint")
    args = parser.parse_args()
    output = args.output or args.checkpoint.with_name("test_metrics.json")
    if output.exists():
        raise FileExistsError(f"Result exists; choose a new --output: {output}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["schema_version"] != 2:
        raise ValueError("Unsupported checkpoint version")
    config, manifest = checkpoint["config"], checkpoint["manifest"]
    validate_manifest(manifest)
    if config["preprocessing"] != expected_preprocessing(config):
        raise ValueError("Checkpoint preprocessing differs from this code")
    torch.set_num_threads(config["cpu_threads"])
    device = select_device(args.device)
    root = args.data_dir or Path(config["data_dir"])
    data = build_dataset(manifest["test"], root, config["refilter"],
                         hop=EVAL_HOP, cwt_norm=config["cwt_norm"])
    loader = loader_for(data, config, device, False, config["seed"])
    model = make_model(config["model"], config["dropout"], config["drop_rate"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    metrics = evaluate(model, loader, nn.CrossEntropyLoss(), device, config["amp"],
                       (-20, 20) if config["tta"] else ())
    for row in metrics["trial_predictions"]:
        row["path"] = manifest["test"][row["trial_id"]]["path"]
    result = {"checkpoint": str(args.checkpoint.resolve()), "epoch": checkpoint["epoch"],
              "fold": checkpoint["fold"], "split": "held_out_test", "metrics": metrics}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(f"Test window accuracy={metrics['window']['accuracy']:.2%}, "
          f"macro-F1={metrics['window']['macro_f1']:.4f}, "
          f"k3={metrics['consecutive']['k3']:.2%}, "
          f"trial accuracy={metrics['trial']['accuracy']:.2%}\nSaved: {output.resolve()}")


if __name__ == "__main__":
    main()
