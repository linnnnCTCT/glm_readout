"""Linear probe using explicit train/validation/test embedding splits."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear probe with explicit split embeddings.")
    parser.add_argument("--train-embeddings", type=str, required=True)
    parser.add_argument("--val-embeddings", type=str, required=True)
    parser.add_argument("--test-embeddings", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--task", type=str, default="auto", choices=["auto", "classification", "regression"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def load_payload(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    embeddings = payload["embeddings"].float()
    labels = payload.get("labels")
    if labels is None:
        raise ValueError(f"Embeddings file does not contain labels: {path}")
    labels = labels.squeeze()
    if labels.ndim != 1:
        raise ValueError(f"Expected labels [N], got {tuple(labels.shape)} from {path}")
    return embeddings, labels


def infer_task(labels: torch.Tensor) -> str:
    if labels.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        return "classification"
    if labels.ndim == 1 and torch.allclose(labels, labels.round()):
        return "classification"
    return "regression"


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == labels).float().mean().item())


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    total_samples = 0
    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            batch_size = embeddings.shape[0]
            total_loss += float(loss.item()) * batch_size
            if task == "classification":
                total_metric += accuracy(outputs, labels) * batch_size
            else:
                mse = float(((outputs.squeeze(-1) - labels) ** 2).mean().item())
                total_metric += mse * batch_size
            total_samples += batch_size

    denom = max(total_samples, 1)
    metrics = {"loss": total_loss / denom}
    if task == "classification":
        metrics["accuracy"] = total_metric / denom
    else:
        metrics["mse"] = total_metric / denom
    return metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    x_train, y_train = load_payload(args.train_embeddings)
    x_val, y_val = load_payload(args.val_embeddings)
    x_test, y_test = load_payload(args.test_embeddings)

    if x_train.shape[1] != x_val.shape[1] or x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Embedding dims must match across train/validation/test.")

    task = args.task if args.task != "auto" else infer_task(y_train)
    emb_dim = x_train.shape[1]

    if task == "classification":
        y_train = y_train.long()
        y_val = y_val.long()
        y_test = y_test.long()
        num_classes = int(max(y_train.max(), y_val.max(), y_test.max()).item()) + 1
        model = nn.Linear(emb_dim, num_classes)
        criterion = nn.CrossEntropyLoss()
        select_higher_is_better = True
        select_metric_name = "accuracy"
    else:
        y_train = y_train.float()
        y_val = y_val.float()
        y_test = y_test.float()
        model = nn.Linear(emb_dim, 1)
        criterion = nn.MSELoss()
        select_higher_is_better = False
        select_metric_name = "mse"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=min(args.batch_size, len(x_train)),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=min(args.batch_size, max(len(x_val), 1)),
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=min(args.batch_size, max(len(x_test), 1)),
        shuffle=False,
    )

    best_epoch = 0
    best_metric = None
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, val_loader, task=task, criterion=criterion, device=device)
        metric_value = val_metrics[select_metric_name]
        is_better = False
        if best_metric is None:
            is_better = True
        elif select_higher_is_better:
            is_better = metric_value > best_metric
        else:
            is_better = metric_value < best_metric

        if is_better:
            best_metric = metric_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    train_metrics = evaluate(model, train_loader, task=task, criterion=criterion, device=device)
    val_metrics = evaluate(model, val_loader, task=task, criterion=criterion, device=device)
    test_metrics = evaluate(model, test_loader, task=task, criterion=criterion, device=device)

    result: dict[str, Any] = {
        "task": task,
        "embedding_dim": emb_dim,
        "num_train": int(x_train.shape[0]),
        "num_validation": int(x_val.shape[0]),
        "num_test": int(x_test.shape[0]),
        "best_epoch": best_epoch,
        "selection_metric": select_metric_name,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    print(json.dumps(result, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
