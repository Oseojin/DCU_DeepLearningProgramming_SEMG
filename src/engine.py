"""train.py / evaluate.py가 공유하는 학습·평가 유틸리티."""

import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from augment import apply_augmentations, mixup, mixup_loss
from data import EVAL_HOP


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(request):
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Install a CUDA-enabled PyTorch build or use --device cpu."
        )
    return torch.device("cuda" if request == "auto" and torch.cuda.is_available() else
                        "cpu" if request == "auto" else request)


def autocast_context(device, amp):
    return torch.autocast("cuda", dtype=torch.float16) if amp and device.type == "cuda" else nullcontext()


def learning_rate_at(epoch, epochs, base_lr, warmup_epochs=5, min_lr=1e-6):
    """0부터 세는 epoch. warmup 끝에서 최대, 마지막 epoch에서 최소가 된다."""
    warmup = min(warmup_epochs, max(0, epochs - 1))
    if warmup and epoch < warmup:
        return base_lr * (0.1 + 0.9 * epoch / max(1, warmup - 1))
    remaining = epochs - warmup
    progress = (epoch - warmup) / max(1, remaining - 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


class EarlyStopping:
    def __init__(self, patience=30, min_epochs=60, min_delta=1e-4):
        self.patience, self.min_epochs, self.min_delta = patience, min_epochs, min_delta
        self.reference_loss = float("inf")
        self.bad_epochs = 0

    def update(self, loss, epoch):
        if loss < self.reference_loss - self.min_delta:
            self.reference_loss, self.bad_epochs = loss, 0
        else:
            self.bad_epochs += 1
        return epoch >= self.min_epochs and self.bad_epochs >= self.patience


def train_epoch(model, loader, optimizer, criterion, scaler, device,
                amp=True, augment=None, mixup_alpha=0.0, ema=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch, (x, y, _, _) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        if augment is not None:
            x = apply_augmentations(x, augment)
        x, y_a, y_b, lam = mixup(x, y, mixup_alpha)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            logits = model(x)
            loss = mixup_loss(criterion, logits, y_a, y_b, lam)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update_parameters(model)
        total_loss += loss.detach().item() * len(y)
        # mixup 적용 시 학습 정확도는 주 라벨 기준의 근사값이다.
        correct += (logits.argmax(1) == y_a).sum().item()
        total += len(y)
        if batch == 1 or batch % 20 == 0 or batch == len(loader):
            print(f"  batch {batch}/{len(loader)} loss={total_loss / total:.5f}", flush=True)
    return {"loss": total_loss / total, "accuracy": correct / total}


def classification_metrics(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=list(range(5)),
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=list(range(5)),
                                      average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(5))).tolist(),
    }


def _consecutive_accuracy(probabilities, labels, trial_ids, starts, k):
    """연속 k개 윈도우의 확률을 평균했을 때의 정확도. k=1은 단일 윈도우 성능."""
    correct = count = 0
    for trial_id in np.unique(trial_ids):
        selected = trial_ids == trial_id
        order = np.argsort(starts[selected])
        probs = probabilities[selected][order]
        truth = int(labels[selected][0])
        for i in range(len(probs) - k + 1):
            correct += int(probs[i:i + k].mean(axis=0).argmax() == truth)
            count += 1
    return correct / count if count else float("nan")


@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp=True, tta_shifts=()):
    model.eval()
    probabilities, labels, trial_ids, starts = [], [], [], []
    total_loss = 0.0
    for x, y, ids, offs in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        with autocast_context(device, amp):
            logits = model(x)
            loss = criterion(logits, y)
            probs = logits.float().softmax(1)
            for shift in tta_shifts:
                probs = probs + model(torch.roll(x, shift, dims=3)).float().softmax(1)
        probs = probs / (1 + len(tta_shifts))
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation output")
        total_loss += loss.item() * len(y)
        probabilities.append(probs.cpu().numpy())
        labels.append(y.cpu().numpy())
        trial_ids.append(ids.numpy())
        starts.append(offs.numpy())
    probabilities = np.concatenate(probabilities)
    labels = np.concatenate(labels)
    trial_ids = np.concatenate(trial_ids)
    starts = np.concatenate(starts)

    trial_labels, trial_predictions, trial_rows = [], [], []
    for trial_id in np.unique(trial_ids):
        selected = trial_ids == trial_id
        true_labels = np.unique(labels[selected])
        if len(true_labels) != 1:
            raise ValueError("A trial contains inconsistent subject labels")
        mean_probability = probabilities[selected].mean(axis=0)
        truth, prediction = int(true_labels[0]), int(mean_probability.argmax())
        trial_labels.append(truth)
        trial_predictions.append(prediction)
        trial_rows.append({"trial_id": int(trial_id), "label": truth, "prediction": prediction,
                           "probabilities": mean_probability.tolist(),
                           "windows": int(selected.sum())})

    predictions = probabilities.argmax(1)
    # 윈도우 위치(행동 구간)별 정확도. grasp/rotate/stop 중 어디가 약한지 본다.
    position = starts // EVAL_HOP
    by_position = {
        int(p): float((predictions[position == p] == labels[position == p]).mean())
        for p in np.unique(position)
    }
    return {
        "loss": total_loss / len(labels), "windows": len(labels), "trials": len(trial_rows),
        "window": classification_metrics(labels, predictions),
        "trial": classification_metrics(trial_labels, trial_predictions),
        "consecutive": {
            f"k{k}": _consecutive_accuracy(probabilities, labels, trial_ids, starts, k)
            for k in (1, 3, 5)
        },
        "by_window_position": by_position,
        "trial_predictions": trial_rows,
    }
