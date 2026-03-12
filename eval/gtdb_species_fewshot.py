"""Few-shot 13k-way species classification and k-means clustering on exported embeddings."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    completeness_score,
    f1_score,
    matthews_corrcoef,
    normalized_mutual_info_score,
)
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class EmbeddingPayload:
    embeddings: torch.Tensor
    labels: torch.Tensor
    ids: list[str]
    genome_ids: list[str]
    label_to_id: dict[str, int]
    id_to_label: list[str]


@dataclass
class FeatureTransform:
    l2_normalize: bool
    mean: torch.Tensor | None
    scale: torch.Tensor | None

    @classmethod
    def fit(
        cls,
        embeddings: torch.Tensor,
        *,
        l2_normalize: bool,
        standardize: bool,
    ) -> "FeatureTransform":
        with torch.no_grad():
            work = embeddings.float()
            if l2_normalize:
                work = F.normalize(work, p=2, dim=1)
            mean = None
            scale = None
            if standardize:
                mean = work.mean(dim=0)
                scale = work.std(dim=0, unbiased=False)
                scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
        return cls(l2_normalize=l2_normalize, mean=mean, scale=scale)

    def apply(self, embeddings: torch.Tensor) -> torch.Tensor:
        work = embeddings.float()
        if self.l2_normalize:
            work = F.normalize(work, p=2, dim=1)
        if self.mean is not None and self.scale is not None:
            work = (work - self.mean) / self.scale
        return work


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run nested few-shot 13k-way classification and k-means clustering on GTDB embeddings."
    )
    parser.add_argument("--train-embeddings", type=str, required=True)
    parser.add_argument("--test-embeddings", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10, 20, 50])
    parser.add_argument("--base-shot", type=int, default=50)
    parser.add_argument("--selection-seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--disable-l2-normalize", action="store_true")
    parser.add_argument("--disable-standardize", action="store_true")

    parser.add_argument("--auc-num-classes", type=int, default=1024)
    parser.add_argument("--auc-samples-per-class", type=int, default=8)

    parser.add_argument("--cluster-batch-size", type=int, default=8192)
    parser.add_argument("--cluster-fit-passes", type=int, default=1)
    parser.add_argument("--cluster-n-init", type=int, default=3)
    parser.add_argument("--cluster-max-no-improvement", type=int, default=20)
    parser.add_argument("--cluster-reassignment-ratio", type=float, default=0.01)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_embedding_payload(path: str | Path) -> EmbeddingPayload:
    payload = torch.load(path, map_location="cpu")
    embeddings = payload["embeddings"].float()
    labels = payload["labels"].long().view(-1)
    ids = list(payload.get("ids", [f"sample_{index:08d}" for index in range(len(labels))]))
    genome_ids = list(payload.get("genome_ids", [str(label.item()) for label in labels]))
    raw_label_to_id = payload.get("label_to_id")
    raw_id_to_label = payload.get("id_to_label")

    if raw_label_to_id is None:
        unique_labels = sorted({int(label.item()) for label in labels})
        label_to_id = {str(label): label for label in unique_labels}
        id_to_label = [str(label) for label in unique_labels]
    else:
        label_to_id = {str(label): int(index) for label, index in raw_label_to_id.items()}
        if raw_id_to_label is None:
            id_to_label = [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]
        else:
            id_to_label = [str(item) for item in raw_id_to_label]

    return EmbeddingPayload(
        embeddings=embeddings,
        labels=labels,
        ids=ids,
        genome_ids=genome_ids,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
    )


def validate_payloads(train: EmbeddingPayload, test: EmbeddingPayload) -> None:
    if train.embeddings.ndim != 2 or test.embeddings.ndim != 2:
        raise ValueError("Embeddings must be rank-2 tensors [N, D].")
    if train.embeddings.shape[1] != test.embeddings.shape[1]:
        raise ValueError("Train/test embedding dimensions do not match.")
    if train.label_to_id != test.label_to_id:
        raise ValueError("Train/test label mappings differ. Export test embeddings with the train label map.")


def build_selection_plan(
    labels: torch.Tensor,
    ids: list[str],
    *,
    base_shot: int,
    selection_seed: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    label_array = labels.numpy()
    rng = np.random.default_rng(selection_seed)
    per_class_indices: dict[int, np.ndarray] = {}
    per_class_ids: dict[str, list[str]] = {}
    for label_id in sorted(np.unique(label_array)):
        class_indices = np.where(label_array == label_id)[0]
        if class_indices.shape[0] < base_shot:
            raise ValueError(
                f"Label {label_id} has only {class_indices.shape[0]} samples; need at least {base_shot}."
            )
        shuffled = np.array(class_indices, copy=True)
        rng.shuffle(shuffled)
        selected = shuffled[:base_shot]
        per_class_indices[int(label_id)] = selected
        per_class_ids[str(label_id)] = [ids[index] for index in selected]
    summary = {
        "selection_seed": selection_seed,
        "base_shot": base_shot,
        "num_classes": len(per_class_indices),
        "sample_ids_by_label": per_class_ids,
    }
    return per_class_indices, summary


def materialize_shot_indices(per_class_indices: dict[int, np.ndarray], shot: int) -> np.ndarray:
    selected = [indices[:shot] for _, indices in sorted(per_class_indices.items())]
    return np.concatenate(selected, axis=0)


def train_linear_probe(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    num_classes: int,
    feature_transform: FeatureTransform,
    emb_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> nn.Module:
    model = nn.Linear(emb_dim, num_classes)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    transformed_train = feature_transform.apply(train_embeddings)
    train_loader = DataLoader(
        TensorDataset(transformed_train, train_labels.long()),
        batch_size=min(args.batch_size, len(train_labels)),
        shuffle=True,
        num_workers=args.num_workers,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_loss = None

    for _ in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=args.use_amp and device.type == "cuda"):
                logits = model(batch_embeddings)
                loss = criterion(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = batch_labels.shape[0]
            epoch_loss += float(loss.item()) * batch_size
            epoch_samples += batch_size

        mean_loss = epoch_loss / max(epoch_samples, 1)
        if best_loss is None or mean_loss < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_full_metrics(
    model: nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    feature_transform: FeatureTransform,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    predictions: list[np.ndarray] = []
    references: list[np.ndarray] = []
    negative_log_likelihood = 0.0
    total_samples = 0

    with torch.no_grad():
        for start in range(0, embeddings.shape[0], batch_size):
            end = min(start + batch_size, embeddings.shape[0])
            batch_embeddings = feature_transform.apply(embeddings[start:end]).to(device, non_blocking=True)
            batch_labels = labels[start:end].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                logits = model(batch_embeddings)
                log_probs = torch.log_softmax(logits, dim=1)

            batch_predictions = log_probs.argmax(dim=1)
            gathered = log_probs.gather(1, batch_labels.unsqueeze(1)).squeeze(1)
            negative_log_likelihood += float((-gathered).sum().item())
            total_samples += batch_labels.shape[0]
            predictions.append(batch_predictions.cpu().numpy())
            references.append(batch_labels.cpu().numpy())

    y_true = np.concatenate(references, axis=0)
    y_pred = np.concatenate(predictions, axis=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "log_loss": float(negative_log_likelihood / max(total_samples, 1)),
    }


def estimate_sampled_auroc(
    model: nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    feature_transform: FeatureTransform,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    auc_num_classes: int,
    auc_samples_per_class: int,
    seed: int,
) -> dict[str, Any]:
    label_array = labels.numpy()
    unique_classes = np.unique(label_array)
    rng = np.random.default_rng(seed)

    if unique_classes.shape[0] > auc_num_classes:
        selected_classes = np.sort(rng.choice(unique_classes, size=auc_num_classes, replace=False))
    else:
        selected_classes = np.sort(unique_classes)

    selected_indices: list[int] = []
    for class_id in selected_classes:
        class_indices = np.where(label_array == class_id)[0]
        if class_indices.shape[0] > auc_samples_per_class:
            class_indices = rng.choice(class_indices, size=auc_samples_per_class, replace=False)
        selected_indices.extend(int(index) for index in class_indices)

    selected_indices = sorted(selected_indices)
    subset_labels = labels[selected_indices]
    class_id_to_local = {int(class_id): index for index, class_id in enumerate(selected_classes.tolist())}
    local_labels = np.asarray([class_id_to_local[int(label.item())] for label in subset_labels], dtype=np.int64)

    model_cpu = model.cpu()
    weight = model_cpu.weight.detach()[selected_classes]
    bias = model_cpu.bias.detach()[selected_classes]

    logits_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(selected_indices), batch_size):
            batch_indices = selected_indices[start : start + batch_size]
            batch_embeddings = feature_transform.apply(embeddings[batch_indices])
            batch_device = batch_embeddings.to(device, non_blocking=True)
            batch_weight = weight.to(device, non_blocking=True)
            batch_bias = bias.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                batch_logits = batch_device @ batch_weight.t() + batch_bias
            logits_chunks.append(batch_logits.float().cpu())

    model.to(device)
    sampled_logits = torch.cat(logits_chunks, dim=0).numpy()

    auroc_values: list[float] = []
    for local_class_id, class_id in enumerate(selected_classes.tolist()):
        binary_labels = (local_labels == local_class_id).astype(np.int64)
        positives = int(binary_labels.sum())
        negatives = int(binary_labels.shape[0] - positives)
        if positives == 0 or negatives == 0:
            continue

        scores = sampled_logits[:, local_class_id]
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, scores.shape[0] + 1, dtype=np.float64)
        positive_ranks = ranks[binary_labels == 1]
        auc = (positive_ranks.sum() - positives * (positives + 1) / 2.0) / (positives * negatives)
        if np.isfinite(auc):
            auroc_values.append(float(auc))

    result: dict[str, Any] = {
        "num_selected_classes": int(selected_classes.shape[0]),
        "samples_per_class": int(auc_samples_per_class),
        "num_selected_samples": int(len(selected_indices)),
        "method": "sampled_classwise_ovr",
        "seed": int(seed),
        "auroc_macro_ovr": None,
    }
    if auroc_values:
        result["auroc_macro_ovr"] = float(np.mean(auroc_values))
    return result


def iter_embedding_batches(embeddings: torch.Tensor, batch_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + batch_size, embeddings.shape[0])) for start in range(0, embeddings.shape[0], batch_size)]


def compute_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cluster_counters: dict[int, Counter[int]] = defaultdict(Counter)
    for cluster_id, label_id in zip(y_pred.tolist(), y_true.tolist()):
        cluster_counters[int(cluster_id)][int(label_id)] += 1
    correct = sum(max(counter.values()) for counter in cluster_counters.values() if counter)
    return float(correct / max(len(y_true), 1))


def run_minibatch_kmeans(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_clusters: int,
    batch_size: int,
    fit_passes: int,
    n_init: int,
    max_no_improvement: int,
    reassignment_ratio: float,
    l2_normalize: bool,
    seed: int,
) -> dict[str, Any]:
    cluster_embeddings = embeddings.float()
    if l2_normalize:
        cluster_embeddings = F.normalize(cluster_embeddings, p=2, dim=1)
    if cluster_embeddings.shape[0] < num_clusters:
        raise ValueError(
            f"Need at least as many samples as clusters: num_samples={cluster_embeddings.shape[0]} "
            f"< num_clusters={num_clusters}"
        )

    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=batch_size,
        n_init=n_init,
        max_no_improvement=max_no_improvement,
        reassignment_ratio=reassignment_ratio,
        random_state=seed,
    )

    init_end = min(max(batch_size, num_clusters), cluster_embeddings.shape[0])
    kmeans.partial_fit(cluster_embeddings[:init_end].numpy())

    for fit_pass in range(fit_passes):
        start_offset = 0 if fit_pass > 0 else init_end
        for start in range(start_offset, cluster_embeddings.shape[0], batch_size):
            end = min(start + batch_size, cluster_embeddings.shape[0])
            kmeans.partial_fit(cluster_embeddings[start:end].numpy())

    predictions = np.empty(cluster_embeddings.shape[0], dtype=np.int32)
    for start, end in iter_embedding_batches(cluster_embeddings, batch_size):
        predictions[start:end] = kmeans.predict(cluster_embeddings[start:end].numpy())

    y_true = labels.numpy()
    return {
        "algorithm": "MiniBatchKMeans",
        "num_clusters": int(num_clusters),
        "num_samples": int(cluster_embeddings.shape[0]),
        "purity": compute_purity(y_true, predictions),
        "completeness": float(completeness_score(y_true, predictions)),
        "ari": float(adjusted_rand_score(y_true, predictions)),
        "nmi": float(normalized_mutual_info_score(y_true, predictions)),
        "fit_passes": int(fit_passes),
        "batch_size": int(batch_size),
        "n_init": int(n_init),
        "max_no_improvement": int(max_no_improvement),
        "reassignment_ratio": float(reassignment_ratio),
        "l2_normalize": bool(l2_normalize),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)
    train_payload = load_embedding_payload(args.train_embeddings)
    test_payload = load_embedding_payload(args.test_embeddings)
    validate_payloads(train_payload, test_payload)

    shots = sorted(set(args.shots))
    if not shots:
        raise ValueError("--shots cannot be empty.")
    if shots[-1] != args.base_shot:
        raise ValueError(f"Largest shot must equal --base-shot={args.base_shot}.")

    per_class_indices, selection_summary = build_selection_plan(
        train_payload.labels,
        train_payload.ids,
        base_shot=args.base_shot,
        selection_seed=args.selection_seed,
    )
    torch.save(selection_summary, output_dir / "fewshot_selection.pt")

    l2_normalize = not args.disable_l2_normalize
    standardize = not args.disable_standardize
    num_classes = len(train_payload.id_to_label)
    emb_dim = int(train_payload.embeddings.shape[1])

    classification_results: dict[str, Any] = {}
    for shot in shots:
        shot_indices = materialize_shot_indices(per_class_indices, shot)
        shot_indices_tensor = torch.as_tensor(shot_indices, dtype=torch.long)
        shot_embeddings = train_payload.embeddings[shot_indices_tensor]
        shot_labels = train_payload.labels[shot_indices_tensor]

        feature_transform = FeatureTransform.fit(
            shot_embeddings,
            l2_normalize=l2_normalize,
            standardize=standardize,
        )
        model = train_linear_probe(
            shot_embeddings,
            shot_labels,
            num_classes=num_classes,
            feature_transform=feature_transform,
            emb_dim=emb_dim,
            args=args,
            device=device,
        )

        metrics = evaluate_full_metrics(
            model,
            test_payload.embeddings,
            test_payload.labels,
            feature_transform=feature_transform,
            batch_size=args.eval_batch_size,
            device=device,
            use_amp=args.use_amp,
        )
        auc_metrics = estimate_sampled_auroc(
            model,
            test_payload.embeddings,
            test_payload.labels,
            feature_transform=feature_transform,
            batch_size=args.eval_batch_size,
            device=device,
            use_amp=args.use_amp,
            auc_num_classes=args.auc_num_classes,
            auc_samples_per_class=args.auc_samples_per_class,
            seed=args.seed + shot,
        )
        classification_results[str(shot)] = {
            "num_train_samples": int(shot_indices.shape[0]),
            "per_class_shot": int(shot),
            "metrics": metrics,
            "auc": auc_metrics,
        }

    clustering_results = run_minibatch_kmeans(
        test_payload.embeddings,
        test_payload.labels,
        num_clusters=num_classes,
        batch_size=args.cluster_batch_size,
        fit_passes=args.cluster_fit_passes,
        n_init=args.cluster_n_init,
        max_no_improvement=args.cluster_max_no_improvement,
        reassignment_ratio=args.cluster_reassignment_ratio,
        l2_normalize=l2_normalize,
        seed=args.seed,
    )

    result = {
        "train_embeddings": str(Path(args.train_embeddings).resolve()),
        "test_embeddings": str(Path(args.test_embeddings).resolve()),
        "num_classes": int(num_classes),
        "embedding_dim": int(emb_dim),
        "shots": shots,
        "selection": {
            "base_shot": int(args.base_shot),
            "selection_seed": int(args.selection_seed),
            "nested_subsets": True,
            "selection_plan_path": str((output_dir / "fewshot_selection.pt").resolve()),
        },
        "classification": classification_results,
        "clustering": clustering_results,
    }

    output_path = output_dir / "results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
