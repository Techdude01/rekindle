"""Load a trained retriever and retrieve exact warm-catalog candidates offline."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from rekindle.retrieval.model import TwoTowerRetriever, select_device
from rekindle.retrieval.sequences import SequenceExamples

ProgressCallback = Callable[[str], None]


def load_retriever(path: Path, device: torch.device | None = None) -> TwoTowerRetriever:
    """Restore a saved retriever checkpoint for local offline evaluation."""
    selected_device = device or select_device()
    checkpoint = torch.load(path, map_location=selected_device)
    model = TwoTowerRetriever(
        user_count=int(checkpoint["user_count"]),
        item_count=int(checkpoint["item_count"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        history_recency_decay=float(checkpoint["history_recency_decay"]),
        use_user_embedding=bool(checkpoint.get("use_user_embedding", True)),
    ).to(selected_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def retrieve_top_k(
    model: TwoTowerRetriever,
    examples: SequenceExamples,
    example_indices: np.ndarray,
    k: int,
    batch_size: int,
    device: torch.device | None = None,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Return exact top-K unseen item indices for each chosen replay event."""
    if k < 1 or k > len(examples.item_ids):
        raise ValueError("k must be between one and the warm catalog size")
    selected_device = device or next(model.parameters()).device
    item_indices = torch.arange(len(examples.item_ids), device=selected_device)
    item_vectors = model.encode_items(item_indices)
    candidate_rows: list[np.ndarray] = []
    total_batches = (len(example_indices) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(example_indices), batch_size), start=1):
        indices = example_indices[start : start + batch_size]
        user_indices = torch.from_numpy(examples.user_indices[indices]).to(selected_device)
        histories = torch.from_numpy(examples.histories[indices]).to(selected_device)
        scores = model.encode_users(user_indices, histories) @ item_vectors.T
        for row_index, (user_index, position) in enumerate(
            zip(
                examples.user_indices[indices],
                examples.prior_event_counts[indices],
                strict=True,
            )
        ):
            seen = examples.user_item_sequences[int(user_index)][: int(position)]
            if seen:
                scores[row_index, seen] = -torch.inf
        candidate_rows.append(torch.topk(scores, k=k, dim=1).indices.cpu().numpy())
        if progress_callback is not None:
            progress_callback(
                f"Neural retrieval | batch {batch_number:,}/{total_batches:,} "
                f"({batch_number / total_batches:.0%})."
            )
    return np.concatenate(candidate_rows, axis=0)
