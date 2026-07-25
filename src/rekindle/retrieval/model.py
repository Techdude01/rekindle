"""The warm-catalog, history-aware two-tower retrieval model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class TwoTowerRetriever(nn.Module):
    """Combine durable user IDs with a recency-weighted item-history context."""

    def __init__(
        self,
        user_count: int,
        item_count: int,
        embedding_dim: int,
        history_recency_decay: float,
    ) -> None:
        super().__init__()
        if not 0 < history_recency_decay <= 1:
            raise ValueError("history_recency_decay must be in (0, 1]")
        self.history_recency_decay = history_recency_decay
        self.user_embeddings = nn.Embedding(user_count, embedding_dim)
        self.history_item_embeddings = nn.Embedding(item_count, embedding_dim)
        self.target_item_embeddings = nn.Embedding(item_count, embedding_dim)
        self.user_projection = nn.Linear(embedding_dim * 2, embedding_dim)
        self.user_normalization = nn.LayerNorm(embedding_dim)
        self.item_normalization = nn.LayerNorm(embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use stable embedding initialization for dot-product contrastive training."""
        nn.init.normal_(self.user_embeddings.weight, std=0.02)
        nn.init.normal_(self.history_item_embeddings.weight, std=0.02)
        nn.init.normal_(self.target_item_embeddings.weight, std=0.02)
        nn.init.xavier_uniform_(self.user_projection.weight)
        nn.init.zeros_(self.user_projection.bias)

    def encode_users(self, user_indices: torch.Tensor, histories: torch.Tensor) -> torch.Tensor:
        """Encode IDs and right-aligned, padded item histories into unit query vectors."""
        valid_history = histories >= 0
        embedded_history = self.history_item_embeddings(histories.clamp_min(0))
        history_size = histories.shape[1]
        positions = torch.arange(history_size, device=histories.device, dtype=torch.float32)
        weights = self.history_recency_decay ** (history_size - 1 - positions)
        weights = weights.unsqueeze(0).unsqueeze(-1) * valid_history.unsqueeze(-1)
        pooled_history = (embedded_history * weights).sum(dim=1)
        pooled_history = pooled_history / weights.sum(dim=1).clamp_min(1e-8)

        stable_preferences = self.user_embeddings(user_indices)
        query = self.user_projection(torch.cat((stable_preferences, pooled_history), dim=-1))
        query = self.user_normalization(query)
        return functional.normalize(query, dim=-1)

    def encode_items(self, item_indices: torch.Tensor) -> torch.Tensor:
        """Encode warm-catalog product IDs into unit candidate vectors."""
        candidates = self.item_normalization(self.target_item_embeddings(item_indices))
        return functional.normalize(candidates, dim=-1)


def select_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, with CPU as the portable fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
