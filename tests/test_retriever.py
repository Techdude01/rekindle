import numpy as np
import torch

from rekindle.retrieval.model import TwoTowerRetriever
from rekindle.retrieval.sequences import SequenceExamples
from rekindle.retrieval.training import (
    NegativeSampler,
    mask_seen_in_batch_items,
    train_retriever,
)


def test_two_tower_encodes_padded_histories_to_unit_vectors() -> None:
    model = TwoTowerRetriever(
        user_count=3,
        item_count=5,
        embedding_dim=8,
        history_recency_decay=0.8,
    )

    user_vectors = model.encode_users(
        torch.tensor([0, 1]),
        torch.tensor([[-1, -1, 1, 2], [-1, 0, 1, 3]]),
    )
    item_vectors = model.encode_items(torch.tensor([0, 1, 2]))

    assert user_vectors.shape == (2, 8)
    assert item_vectors.shape == (3, 8)
    assert torch.allclose(user_vectors.norm(dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(item_vectors.norm(dim=1), torch.ones(3), atol=1e-5)


def test_negative_sampler_excludes_prior_and_target_items() -> None:
    sampler = NegativeSampler([[0, 1, 2]], item_count=5, seed=42)

    negatives = sampler.sample(
        user_indices=np.array([0], dtype=np.int32),
        prior_event_counts=np.array([2], dtype=np.int32),
        targets=np.array([2], dtype=np.int32),
    )

    assert set(negatives[0]).issubset({3, 4})


def test_in_batch_mask_keeps_only_the_diagonal_copy_of_a_positive() -> None:
    logits = torch.zeros((2, 2))

    masked = mask_seen_in_batch_items(
        logits,
        user_indices=np.array([0, 1], dtype=np.int32),
        prior_event_counts=np.array([2, 2], dtype=np.int32),
        candidate_targets=np.array([2, 2], dtype=np.int32),
        user_item_sequences=[[0, 1, 2], [1, 3, 2]],
    )

    assert torch.isfinite(torch.diagonal(masked)).all()
    assert torch.isneginf(masked[0, 1])
    assert torch.isneginf(masked[1, 0])


def test_retriever_training_writes_a_checkpoint_and_metrics(tmp_path) -> None:
    examples = SequenceExamples(
        user_indices=np.array([0, 0, 1, 1], dtype=np.int32),
        histories=np.array(
            [[-1, -1, 0, 1], [-1, 0, 1, 2], [-1, -1, 1, 2], [-1, 1, 2, 3]],
            dtype=np.int32,
        ),
        targets=np.array([2, 3, 3, 4], dtype=np.int32),
        prior_event_counts=np.array([2, 3, 2, 3], dtype=np.int32),
        user_ids=["u1", "u2"],
        item_ids=["i1", "i2", "i3", "i4", "i5"],
        user_item_sequences=[[0, 1, 2, 3], [1, 2, 3, 4]],
    )
    config = {
        "embedding_dim": 8,
        "batch_size": 4,
        "max_epochs": 1,
        "early_stopping_patience": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "temperature": 0.07,
        "history_recency_decay": 0.8,
        "evaluation_batch_size": 2,
        "sampled_negatives_per_positive": 4,
        "evaluation_sampled_negatives": 3,
        "full_catalog_evaluation_examples": 2,
        "primary_recall_k": 2,
    }

    progress: list[str] = []
    result = train_retriever(
        examples,
        examples,
        config,
        seed=42,
        output_directory=tmp_path,
        progress_callback=progress.append,
    )

    assert result.best_epoch == 1
    assert 0.0 <= result.validation_sampled_recall_at_100 <= 1.0
    assert 0.0 <= result.validation_full_catalog_recall_at_100 <= 1.0
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "training-result.json").exists()
    assert any(message.startswith("Device:") for message in progress)
    assert any(message.startswith("Epoch 1:") for message in progress)
