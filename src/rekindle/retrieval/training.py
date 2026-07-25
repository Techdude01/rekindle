"""Contrastive two-tower training with strict negative and validation rules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as functional
from torch.utils.data import DataLoader, TensorDataset

from rekindle.retrieval.model import TwoTowerRetriever, select_device
from rekindle.retrieval.sequences import SequenceExamples


@dataclass(frozen=True)
class TrainingResult:
    """Measured result of one reproducible retriever training run."""

    best_epoch: int
    validation_sampled_recall_at_100: float
    validation_full_catalog_recall_at_100: float
    device: str


class NegativeSampler:
    """Mix uniform and popularity-weighted negatives while excluding prior positives."""

    def __init__(self, user_item_sequences: list[list[int]], item_count: int, seed: int) -> None:
        self.user_item_sequences = user_item_sequences
        self.item_count = item_count
        all_items = np.concatenate(
            [np.asarray(sequence, dtype=np.int32) for sequence in user_item_sequences if sequence]
        )
        popularity = np.bincount(all_items, minlength=item_count).astype(np.float64) ** 0.75
        self.popularity_probabilities = popularity / popularity.sum()
        self.random = np.random.default_rng(seed)

    def sample(
        self,
        user_indices: np.ndarray,
        prior_event_counts: np.ndarray,
        targets: np.ndarray,
        negative_count: int = 4,
        uniform_negative_count: int = 2,
    ) -> np.ndarray:
        """Return valid uniform and popularity-weighted negatives per target."""
        if negative_count < 1 or not 0 <= uniform_negative_count <= negative_count:
            raise ValueError("Negative sampler counts must be non-negative and compatible")
        batch_size = len(targets)
        candidates = np.concatenate(
            (
                self.random.integers(
                    self.item_count,
                    size=(batch_size, uniform_negative_count),
                    dtype=np.int32,
                ),
                self.random.choice(
                    self.item_count,
                    size=(batch_size, negative_count - uniform_negative_count),
                    replace=True,
                    p=self.popularity_probabilities,
                ).astype(np.int32),
            ),
            axis=1,
        )
        for row_index, (user_index, position, target) in enumerate(
            zip(user_indices, prior_event_counts, targets, strict=True)
        ):
            excluded = set(self.user_item_sequences[int(user_index)][: int(position)])
            excluded.add(int(target))
            if len(excluded) >= self.item_count:
                raise RuntimeError("No unseen item is available for negative sampling")
            for negative_index in range(candidates.shape[1]):
                attempts = 0
                while candidates[row_index, negative_index] in excluded:
                    if negative_index < uniform_negative_count:
                        candidate = self.random.integers(self.item_count, dtype=np.int32)
                    else:
                        candidate = np.int32(
                            self.random.choice(
                                self.item_count,
                                p=self.popularity_probabilities,
                            )
                        )
                    candidates[row_index, negative_index] = candidate
                    attempts += 1
                    if attempts > 16:
                        # A tiny catalogue can have no remaining positive-mass item under
                        # popularity sampling. Retain a valid learning signal instead of
                        # looping forever; realistic catalogues almost never take this path.
                        while candidates[row_index, negative_index] in excluded:
                            candidates[row_index, negative_index] = self.random.integers(
                                self.item_count,
                                dtype=np.int32,
                            )
                        break
        return candidates


def mask_seen_in_batch_items(
    logits: torch.Tensor,
    user_indices: np.ndarray,
    prior_event_counts: np.ndarray,
    candidate_targets: np.ndarray,
    user_item_sequences: list[list[int]],
) -> torch.Tensor:
    """Mask in-batch targets that are invalid negatives for another user's context."""
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for row_index, (user_index, position) in enumerate(
        zip(user_indices, prior_event_counts, strict=True)
    ):
        excluded = set(user_item_sequences[int(user_index)][: int(position)])
        excluded.add(int(candidate_targets[row_index]))
        invalid_columns = np.isin(candidate_targets, list(excluded))
        invalid_columns[row_index] = False  # Keep this row's diagonal positive label.
        mask[row_index] = torch.from_numpy(invalid_columns).to(logits.device)
    return logits.masked_fill(mask, -torch.inf)


def train_retriever(
    train_examples: SequenceExamples,
    validation_examples: SequenceExamples,
    retrieval_config: dict[str, Any],
    seed: int,
    output_directory: Path,
) -> TrainingResult:
    """Train a two-tower model and early-stop on sampled validation Recall@100."""
    if train_examples.item_ids != validation_examples.item_ids:
        raise ValueError("Training and validation item vocabularies must match")
    if train_examples.user_ids != validation_examples.user_ids:
        raise ValueError("Training and validation user vocabularies must match")
    if retrieval_config["sampled_negatives_per_positive"] != 4:
        raise ValueError("Training requires exactly two uniform and two popularity negatives")

    torch.manual_seed(seed)
    device = select_device()
    model = TwoTowerRetriever(
        user_count=len(train_examples.user_ids),
        item_count=len(train_examples.item_ids),
        embedding_dim=retrieval_config["embedding_dim"],
        history_recency_decay=retrieval_config["history_recency_decay"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=retrieval_config["learning_rate"],
        weight_decay=retrieval_config["weight_decay"],
    )
    dataset = TensorDataset(
        torch.from_numpy(train_examples.user_indices),
        torch.from_numpy(train_examples.histories),
        torch.from_numpy(train_examples.targets),
        torch.from_numpy(train_examples.prior_event_counts),
    )
    loader = DataLoader(
        dataset,
        batch_size=retrieval_config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    sampler = NegativeSampler(
        train_examples.user_item_sequences,
        item_count=len(train_examples.item_ids),
        seed=seed,
    )

    best_recall = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    for epoch in range(1, retrieval_config["max_epochs"] + 1):
        model.train()
        for user_indices, histories, targets, prior_event_counts in loader:
            batch_user_indices = user_indices.numpy()
            batch_prior_event_counts = prior_event_counts.numpy()
            batch_targets = targets.numpy()
            extra_negatives = sampler.sample(
                batch_user_indices,
                batch_prior_event_counts,
                batch_targets,
            )
            user_indices = user_indices.to(device)
            histories = histories.to(device)
            targets = targets.to(device)
            extra_negatives_tensor = torch.from_numpy(extra_negatives).to(device)

            query_vectors = model.encode_users(user_indices, histories)
            positive_vectors = model.encode_items(targets)
            in_batch_logits = query_vectors @ positive_vectors.T
            in_batch_logits = mask_seen_in_batch_items(
                in_batch_logits,
                batch_user_indices,
                batch_prior_event_counts,
                batch_targets,
                train_examples.user_item_sequences,
            )
            negative_vectors = model.encode_items(extra_negatives_tensor.reshape(-1)).reshape(
                len(targets),
                -1,
                positive_vectors.shape[-1],
            )
            extra_logits = torch.einsum("bd,bnd->bn", query_vectors, negative_vectors)
            logits = torch.cat((in_batch_logits, extra_logits), dim=1)
            logits = logits / retrieval_config["temperature"]
            labels = torch.arange(len(targets), device=device)
            loss = functional.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        recall = sampled_recall_at_k(
            model,
            validation_examples,
            k=retrieval_config["primary_recall_k"],
            batch_size=retrieval_config["evaluation_batch_size"],
            device=device,
            negative_count=retrieval_config["evaluation_sampled_negatives"],
            seed=seed + 1,
        )
        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= retrieval_config["early_stopping_patience"]:
                break

    if best_state is None:
        raise RuntimeError("Retriever training did not produce a model state")
    model.load_state_dict(best_state)
    validation_full_catalog_recall = recall_at_k(
        model,
        validation_examples,
        k=retrieval_config["primary_recall_k"],
        batch_size=retrieval_config["evaluation_batch_size"],
        device=device,
        example_indices=_validation_subset_indices(
            validation_examples.count,
            retrieval_config["full_catalog_evaluation_examples"],
            seed=seed + 2,
        ),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "user_count": len(train_examples.user_ids),
            "item_count": len(train_examples.item_ids),
            "embedding_dim": retrieval_config["embedding_dim"],
            "history_recency_decay": retrieval_config["history_recency_decay"],
        },
        output_directory / "model.pt",
    )
    result = TrainingResult(
        best_epoch=best_epoch,
        validation_sampled_recall_at_100=best_recall,
        validation_full_catalog_recall_at_100=validation_full_catalog_recall,
        device=str(device),
    )
    (output_directory / "training-result.json").write_text(
        json.dumps(result.__dict__, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _validation_subset_indices(count: int, maximum: int, seed: int) -> np.ndarray:
    """Choose a fixed validation subset for a tractable exact-catalog check."""
    if maximum < 1:
        raise ValueError("full_catalog_evaluation_examples must be positive")
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    random = np.random.default_rng(seed)
    return np.sort(random.choice(count, size=maximum, replace=False))


@torch.no_grad()
def sampled_recall_at_k(
    model: TwoTowerRetriever,
    examples: SequenceExamples,
    k: int,
    batch_size: int,
    device: torch.device,
    negative_count: int,
    seed: int,
) -> float:
    """Score targets against a fixed, seen-item-filtered sampled candidate set."""
    if examples.count == 0:
        return 0.0
    if k > negative_count + 1:
        raise ValueError("Recall K cannot exceed the sampled candidate count")
    model.eval()
    sampler = NegativeSampler(examples.user_item_sequences, len(examples.item_ids), seed)
    hits = 0
    for start in range(0, examples.count, batch_size):
        stop = min(start + batch_size, examples.count)
        targets = examples.targets[start:stop]
        negatives = sampler.sample(
            examples.user_indices[start:stop],
            examples.prior_event_counts[start:stop],
            targets,
            negative_count=negative_count,
            uniform_negative_count=negative_count // 2,
        )
        candidate_indices = np.concatenate((targets[:, None], negatives), axis=1)
        user_indices = torch.from_numpy(examples.user_indices[start:stop]).to(device)
        histories = torch.from_numpy(examples.histories[start:stop]).to(device)
        candidates = torch.from_numpy(candidate_indices).to(device)
        candidate_vectors = model.encode_items(candidates.reshape(-1)).reshape(
            len(targets),
            candidate_indices.shape[1],
            -1,
        )
        scores = torch.einsum(
            "bd,bcd->bc",
            model.encode_users(user_indices, histories),
            candidate_vectors,
        )
        top_columns = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()
        hits += int(np.sum(np.any(top_columns == 0, axis=1)))
    return hits / examples.count


@torch.no_grad()
def recall_at_k(
    model: TwoTowerRetriever,
    examples: SequenceExamples,
    k: int,
    batch_size: int,
    device: torch.device,
    example_indices: np.ndarray | None = None,
) -> float:
    """Compute exact-catalog Recall@K after filtering every prior user interaction."""
    if examples.count == 0:
        return 0.0
    selected = (
        np.arange(examples.count, dtype=np.int64) if example_indices is None else example_indices
    )
    if len(selected) == 0:
        return 0.0
    model.eval()
    item_indices = torch.arange(len(examples.item_ids), device=device)
    item_vectors = model.encode_items(item_indices)
    hits = 0
    for start in range(0, len(selected), batch_size):
        indices = selected[start : start + batch_size]
        user_indices = torch.from_numpy(examples.user_indices[indices]).to(device)
        histories = torch.from_numpy(examples.histories[indices]).to(device)
        targets = examples.targets[indices]
        scores = model.encode_users(user_indices, histories) @ item_vectors.T
        for local_index, (user_index, position) in enumerate(
            zip(
                examples.user_indices[indices],
                examples.prior_event_counts[indices],
                strict=True,
            )
        ):
            seen = examples.user_item_sequences[int(user_index)][: int(position)]
            if seen:
                scores[local_index, seen] = -torch.inf
        top_items = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()
        hits += sum(target in row for target, row in zip(targets, top_items, strict=True))
    return hits / len(selected)
