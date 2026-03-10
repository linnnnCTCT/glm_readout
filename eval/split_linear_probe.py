"""Probe evaluation using explicit train/validation/test embedding splits."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe with explicit split embeddings.")
    parser.add_argument("--train-embeddings", type=str, required=True)
    parser.add_argument("--val-embeddings", type=str, required=True)
    parser.add_argument("--test-embeddings", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--task", type=str, default="auto", choices=["auto", "classification", "regression"])
    parser.add_argument("--classifier", type=str, default="linear", choices=["linear", "xgboost"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)

    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=20)
    parser.add_argument("--xgb-tree-method", type=str, default="hist")
    parser.add_argument("--xgb-device", type=str, default="auto")
    parser.add_argument("--xgb-n-jobs", type=int, default=0)
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


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def summarize_scalar(values: list[float | int | None]) -> dict[str, float] | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    arr = np.asarray(filtered, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def summarize_metric_dict(metric_dicts: list[dict[str, float | None]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for metrics in metric_dicts for key in metrics})
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        stats = summarize_scalar([metrics.get(key) for metrics in metric_dicts])
        if stats is not None:
            summary[key] = stats
    return summary


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    loss_value: float,
) -> dict[str, float | None]:
    num_classes = probabilities.shape[1]
    metrics: dict[str, float | None] = {
        "loss": float(loss_value),
        "accuracy": float(accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }

    if num_classes == 2:
        positive_prob = probabilities[:, 1]
        metrics["f1"] = float(f1_score(labels, predictions, average="binary", zero_division=0))
        try:
            metrics["auroc"] = float(roc_auc_score(labels, positive_prob))
        except ValueError:
            metrics["auroc"] = None
        try:
            metrics["auprc"] = float(average_precision_score(labels, positive_prob))
        except ValueError:
            metrics["auprc"] = None
        return metrics

    metrics["f1"] = float(f1_score(labels, predictions, average="macro", zero_division=0))
    classes = np.arange(num_classes)
    labels_binarized = label_binarize(labels, classes=classes)
    try:
        metrics["auroc"] = float(
            roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["auroc"] = None
    try:
        metrics["auprc"] = float(
            average_precision_score(labels_binarized, probabilities, average="macro")
        )
    except ValueError:
        metrics["auprc"] = None
    return metrics


def regression_metrics(labels: np.ndarray, predictions: np.ndarray, loss_value: float) -> dict[str, float]:
    mse = float(np.mean((predictions - labels) ** 2))
    return {
        "loss": float(loss_value),
        "mse": mse,
    }


def evaluate_linear_classification(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float | None]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            logits = model(embeddings)
            loss = criterion(logits, labels)
            batch_size = embeddings.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    probabilities = torch.softmax(logits, dim=-1).numpy()
    predictions = probabilities.argmax(axis=1)
    loss_value = total_loss / max(total_samples, 1)
    return classification_metrics(labels, probabilities, predictions, loss_value=loss_value)


def evaluate_linear_regression(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            outputs = model(embeddings).squeeze(-1)
            loss = criterion(outputs, labels)
            batch_size = embeddings.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            all_predictions.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())

    predictions = torch.cat(all_predictions, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    loss_value = total_loss / max(total_samples, 1)
    return regression_metrics(labels, predictions, loss_value=loss_value)


def evaluate_linear(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float | None]:
    if task == "classification":
        return evaluate_linear_classification(model, loader, criterion=criterion, device=device)
    return evaluate_linear_regression(model, loader, criterion=criterion, device=device)


def run_linear_probe(
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    task: str,
    args: argparse.Namespace,
    seed: int,
    emb_dim: int,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            if task == "regression":
                outputs = outputs.squeeze(-1)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_linear(model, val_loader, task=task, criterion=criterion, device=device)
        metric_value = val_metrics[select_metric_name]
        is_better = False
        if best_metric is None:
            is_better = True
        elif select_higher_is_better:
            is_better = float(metric_value) > float(best_metric)
        else:
            is_better = float(metric_value) < float(best_metric)

        if is_better:
            best_metric = float(metric_value)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    train_metrics = evaluate_linear(model, train_loader, task=task, criterion=criterion, device=device)
    val_metrics = evaluate_linear(model, val_loader, task=task, criterion=criterion, device=device)
    test_metrics = evaluate_linear(model, test_loader, task=task, criterion=criterion, device=device)

    return {
        "classifier": "linear",
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_metric": select_metric_name,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def resolve_xgb_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_xgb_classifier(args: argparse.Namespace, seed: int, num_classes: int) -> Any:
    if xgb is None:
        raise ImportError("xgboost is required for --classifier xgboost")

    device = resolve_xgb_device(args.xgb_device)
    eval_metric = "logloss" if num_classes == 2 else "mlogloss"
    params: dict[str, Any] = {
        "n_estimators": args.xgb_n_estimators,
        "learning_rate": args.xgb_learning_rate,
        "max_depth": args.xgb_max_depth,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample_bytree,
        "reg_lambda": args.xgb_reg_lambda,
        "random_state": seed,
        "tree_method": args.xgb_tree_method,
        "device": device,
        "eval_metric": eval_metric,
    }
    if args.xgb_n_jobs > 0:
        params["n_jobs"] = args.xgb_n_jobs
    if args.xgb_early_stopping_rounds > 0:
        params["early_stopping_rounds"] = args.xgb_early_stopping_rounds

    if num_classes == 2:
        params["objective"] = "binary:logistic"
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = num_classes
    return xgb.XGBClassifier(**params)


def evaluate_xgboost_classification(
    model: Any,
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> dict[str, float | None]:
    probabilities = model.predict_proba(features)
    if probabilities.ndim == 1:
        probabilities = np.stack([1.0 - probabilities, probabilities], axis=1)
    predictions = probabilities.argmax(axis=1)
    loss_value = float(log_loss(labels, probabilities, labels=list(range(num_classes))))
    return classification_metrics(labels, probabilities, predictions, loss_value=loss_value)


def run_xgboost_probe(
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    task: str,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    if task != "classification":
        raise ValueError("XGBoost probe currently supports classification only.")

    y_train = y_train.long()
    y_val = y_val.long()
    y_test = y_test.long()
    num_classes = int(max(y_train.max(), y_val.max(), y_test.max()).item()) + 1

    x_train_np = x_train.numpy()
    x_val_np = x_val.numpy()
    x_test_np = x_test.numpy()
    y_train_np = y_train.numpy()
    y_val_np = y_val.numpy()
    y_test_np = y_test.numpy()

    model = build_xgb_classifier(args, seed=seed, num_classes=num_classes)
    model.fit(
        x_train_np,
        y_train_np,
        eval_set=[(x_val_np, y_val_np)],
        verbose=False,
    )

    best_iteration = getattr(model, "best_iteration", None)
    selection_metric = "logloss" if args.xgb_early_stopping_rounds > 0 else "final_model"

    train_metrics = evaluate_xgboost_classification(model, x_train_np, y_train_np, num_classes=num_classes)
    val_metrics = evaluate_xgboost_classification(model, x_val_np, y_val_np, num_classes=num_classes)
    test_metrics = evaluate_xgboost_classification(model, x_test_np, y_test_np, num_classes=num_classes)

    result: dict[str, Any] = {
        "classifier": "xgboost",
        "seed": seed,
        "selection_metric": selection_metric,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    if best_iteration is not None:
        result["best_iteration"] = int(best_iteration)
    return result


def run_single_probe(
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    task: str,
    args: argparse.Namespace,
    seed: int,
    emb_dim: int,
) -> dict[str, Any]:
    set_global_seed(seed)
    if args.classifier == "linear":
        return run_linear_probe(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            task=task,
            args=args,
            seed=seed,
            emb_dim=emb_dim,
        )
    return run_xgboost_probe(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        task=task,
        args=args,
        seed=seed,
    )


def build_summary(
    *,
    runs: list[dict[str, Any]],
    task: str,
    classifier: str,
    emb_dim: int,
    num_train: int,
    num_validation: int,
    num_test: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "task": task,
        "classifier": classifier,
        "embedding_dim": emb_dim,
        "num_train": num_train,
        "num_validation": num_validation,
        "num_test": num_test,
        "num_seeds": len(runs),
        "seeds": [int(run["seed"]) for run in runs],
        "selection_metric": runs[0]["selection_metric"],
        "seed_results": runs,
        "summary": {
            "train_metrics": summarize_metric_dict([run["train_metrics"] for run in runs]),
            "validation_metrics": summarize_metric_dict([run["validation_metrics"] for run in runs]),
            "test_metrics": summarize_metric_dict([run["test_metrics"] for run in runs]),
        },
    }

    best_epoch_stats = summarize_scalar([safe_float(run.get("best_epoch")) for run in runs])
    if best_epoch_stats is not None:
        summary["summary"]["best_epoch"] = best_epoch_stats

    best_iteration_stats = summarize_scalar([safe_float(run.get("best_iteration")) for run in runs])
    if best_iteration_stats is not None:
        summary["summary"]["best_iteration"] = best_iteration_stats

    return summary


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds else [args.seed]

    x_train, y_train = load_payload(args.train_embeddings)
    x_val, y_val = load_payload(args.val_embeddings)
    x_test, y_test = load_payload(args.test_embeddings)

    if x_train.shape[1] != x_val.shape[1] or x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Embedding dims must match across train/validation/test.")

    task = args.task if args.task != "auto" else infer_task(y_train)
    emb_dim = int(x_train.shape[1])

    if task == "classification":
        y_train = y_train.long()
        y_val = y_val.long()
        y_test = y_test.long()
    else:
        y_train = y_train.float()
        y_val = y_val.float()
        y_test = y_test.float()

    runs = [
        run_single_probe(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            task=task,
            args=args,
            seed=seed,
            emb_dim=emb_dim,
        )
        for seed in seeds
    ]

    if len(runs) == 1:
        result: dict[str, Any] = {
            "task": task,
            "classifier": args.classifier,
            "embedding_dim": emb_dim,
            "num_train": int(x_train.shape[0]),
            "num_validation": int(x_val.shape[0]),
            "num_test": int(x_test.shape[0]),
            **runs[0],
        }
    else:
        result = build_summary(
            runs=runs,
            task=task,
            classifier=args.classifier,
            emb_dim=emb_dim,
            num_train=int(x_train.shape[0]),
            num_validation=int(x_val.shape[0]),
            num_test=int(x_test.shape[0]),
        )

    print(json.dumps(result, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
