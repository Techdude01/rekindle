"""Local, single-request latency measurements for the frozen recommendation stages."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import torch

from rekindle.evaluation.metrics import fixed_subset_indices
from rekindle.ranking.model import RANKER_FEATURES
from rekindle.retrieval.inference import load_retriever
from rekindle.retrieval.model import TwoTowerRetriever, select_device
from rekindle.retrieval.sequences import (
    SequenceExamples,
    build_sequence_examples,
    load_sequence_events,
)

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class LatencySummary:
    """Single-worker latency distribution for one local serving component."""

    requests: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    queries_per_second: float


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Comparable exact/approximate retrieval and ranking benchmark results."""

    benchmark_queries: int
    warmup_queries: int
    catalog_items: int
    candidate_count: int
    average_ranker_candidates: float
    query_device: str
    faiss_threads: int
    query_encoding_and_transfer: LatencySummary
    exact_faiss_retrieval: LatencySummary
    hnsw_retrieval: LatencySummary
    ranker_scoring: LatencySummary
    exact_target_recall_at_200: float
    hnsw_target_recall_at_200: float
    hnsw_exact_candidate_overlap_at_200: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready result."""
        return asdict(self)


def run_latency_benchmark(
    project_root: Path,
    config: dict,
    progress_callback: ProgressCallback | None = None,
) -> BenchmarkMetrics:
    """Measure frozen-model local latency without changing model or index artifacts."""
    faiss = _import_faiss()
    benchmark_config = config["benchmark"]
    query_count = benchmark_config["query_count"]
    warmup_queries = benchmark_config["warmup_queries"]
    candidate_count = config["retrieval"]["candidate_count"]
    faiss.omp_set_num_threads(benchmark_config["faiss_threads"])

    examples = _load_test_examples(project_root, config)
    total_queries = query_count + warmup_queries
    selected_indices = fixed_subset_indices(
        examples.count, total_queries, seed=config["project"]["seed"] + 5
    )
    if len(selected_indices) < total_queries:
        raise ValueError("Benchmark needs more replay examples than are available")
    warmup_indices = selected_indices[:warmup_queries]
    measured_indices = selected_indices[warmup_queries:]
    device = select_device()
    _report(progress_callback, f"Loading frozen history-only retriever on {device.type}.")
    retriever = load_retriever(project_root / "artifacts/retriever-history-only/model.pt", device)

    _report(progress_callback, "Warming and measuring per-request query encoding.")
    _encode_query_vectors(retriever, examples, warmup_indices, device, record_latency=False)
    query_vectors, encoding_latency_ms = _encode_query_vectors(
        retriever, examples, measured_indices, device, record_latency=True
    )
    seen_items = _seen_items(examples, measured_indices)
    targets = examples.targets[measured_indices]
    item_vectors = _item_vectors(retriever, examples, device)
    exact_index = faiss.IndexFlatIP(item_vectors.shape[1])
    exact_index.add(item_vectors)
    hnsw_index = faiss.IndexHNSWFlat(
        item_vectors.shape[1], config["retrieval"]["hnsw_m"], faiss.METRIC_INNER_PRODUCT
    )
    hnsw_index.hnsw.efSearch = config["retrieval"]["hnsw_ef_search"]
    hnsw_index.add(item_vectors)

    _report(progress_callback, "Warming exact FAISS and HNSW indexes.")
    _retrieve_with_faiss(
        exact_index, query_vectors[:warmup_queries], seen_items[:warmup_queries], candidate_count
    )
    _retrieve_with_faiss(
        hnsw_index, query_vectors[:warmup_queries], seen_items[:warmup_queries], candidate_count
    )
    _report(progress_callback, "Measuring exact FAISS retrieval.")
    exact_candidates, exact_latency_ms = _retrieve_with_faiss(
        exact_index, query_vectors, seen_items, candidate_count, record_latency=True
    )
    _report(progress_callback, "Measuring HNSW retrieval.")
    hnsw_candidates, hnsw_latency_ms = _retrieve_with_faiss(
        hnsw_index, query_vectors, seen_items, candidate_count, record_latency=True
    )

    ranker_inputs = _load_ranker_inputs(
        project_root / "artifacts/ranker-crossfit/fold-3/candidates.parquet",
        query_count,
        seed=config["project"]["seed"] + 6,
    )
    ranker = lgb.Booster(model_file=str(project_root / "artifacts/ranker/model.txt"))
    _report(progress_callback, "Warming and measuring LambdaRank scoring.")
    _score_ranker(ranker, ranker_inputs[:warmup_queries], record_latency=False)
    ranker_latency_ms = _score_ranker(ranker, ranker_inputs, record_latency=True)

    metrics = BenchmarkMetrics(
        benchmark_queries=query_count,
        warmup_queries=warmup_queries,
        catalog_items=len(examples.item_ids),
        candidate_count=candidate_count,
        average_ranker_candidates=float(np.mean([len(values) for values in ranker_inputs])),
        query_device=device.type,
        faiss_threads=benchmark_config["faiss_threads"],
        query_encoding_and_transfer=_latency_summary(encoding_latency_ms),
        exact_faiss_retrieval=_latency_summary(exact_latency_ms),
        hnsw_retrieval=_latency_summary(hnsw_latency_ms),
        ranker_scoring=_latency_summary(ranker_latency_ms),
        exact_target_recall_at_200=_target_recall(exact_candidates, targets),
        hnsw_target_recall_at_200=_target_recall(hnsw_candidates, targets),
        hnsw_exact_candidate_overlap_at_200=_candidate_overlap(exact_candidates, hnsw_candidates),
    )
    output_path = project_root / "artifacts/benchmarks/latency.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics.to_dict(), indent=2) + "\n", encoding="utf-8")
    return metrics


def _load_test_examples(project_root: Path, config: dict) -> SequenceExamples:
    """Build the same replay-safe warm test queries used for final evaluation."""
    events = load_sequence_events(
        project_root / config["data"]["prepared_interactions_path"], target_split="test"
    )
    return build_sequence_examples(
        events,
        target_split="test",
        history_size=config["split"]["replay_history_size"],
        min_prior_interactions=config["split"]["min_prior_interactions"],
    )


@torch.no_grad()
def _encode_query_vectors(
    model: TwoTowerRetriever,
    examples: SequenceExamples,
    example_indices: np.ndarray,
    device: torch.device,
    record_latency: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode one online-style request at a time, including transfer to CPU FAISS."""
    vectors: list[np.ndarray] = []
    timings: list[float] = []
    for example_index in example_indices:
        _synchronize(device)
        started = time.perf_counter_ns()
        user_indices = torch.from_numpy(examples.user_indices[[example_index]]).to(device)
        histories = torch.from_numpy(examples.histories[[example_index]]).to(device)
        vector = model.encode_users(user_indices, histories).cpu().numpy()
        _synchronize(device)
        if record_latency:
            timings.append((time.perf_counter_ns() - started) / 1_000_000)
        vectors.append(np.ascontiguousarray(vector.astype(np.float32, copy=False)))
    return np.concatenate(vectors, axis=0), np.asarray(timings, dtype=np.float64)


@torch.no_grad()
def _item_vectors(
    model: TwoTowerRetriever, examples: SequenceExamples, device: torch.device
) -> np.ndarray:
    """Materialize the frozen item tower once; index construction is not timed."""
    item_indices = torch.arange(len(examples.item_ids), device=device)
    vectors = model.encode_items(item_indices).cpu().numpy()
    _synchronize(device)
    return np.ascontiguousarray(vectors.astype(np.float32, copy=False))


def _seen_items(examples: SequenceExamples, example_indices: np.ndarray) -> list[set[int]]:
    """Return every item preceding each target so benchmark candidates cannot repeat history."""
    return [
        set(
            examples.user_item_sequences[int(examples.user_indices[example_index])][
                : int(examples.prior_event_counts[example_index])
            ]
        )
        for example_index in example_indices
    ]


def _retrieve_with_faiss(
    index: object,
    query_vectors: np.ndarray,
    seen_items: list[set[int]],
    candidate_count: int,
    record_latency: bool = False,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Search FAISS and filter every previously seen product before returning top-K."""
    candidates: list[np.ndarray] = []
    timings: list[float] = []
    for vector, seen in zip(query_vectors, seen_items, strict=True):
        raw_count = min(index.ntotal, candidate_count + len(seen))
        started = time.perf_counter_ns()
        _, raw_indices = index.search(vector.reshape(1, -1), raw_count)
        filtered = np.fromiter(
            (item_index for item_index in raw_indices[0] if item_index not in seen),
            dtype=np.int64,
            count=candidate_count,
        )
        if len(filtered) != candidate_count:
            raise ValueError("FAISS search did not return enough unseen candidates")
        if record_latency:
            timings.append((time.perf_counter_ns() - started) / 1_000_000)
        candidates.append(filtered)
    return candidates, np.asarray(timings, dtype=np.float64)


def _load_ranker_inputs(path: Path, query_count: int, seed: int) -> list[np.ndarray]:
    """Select real, variable-size candidate sets from the held-out ranker fold."""
    candidates = pl.read_parquet(path).sort("query_id")
    groups = candidates.partition_by("query_id", maintain_order=True)
    selected = fixed_subset_indices(len(groups), query_count, seed)
    if len(selected) < query_count:
        raise ValueError("Ranker benchmark needs more candidate groups than are available")
    return [
        groups[int(index)].select(RANKER_FEATURES).cast(pl.Float32).to_numpy() for index in selected
    ]


def _score_ranker(
    ranker: lgb.Booster, inputs: list[np.ndarray], record_latency: bool
) -> np.ndarray:
    """Measure one top-K ranking request per actual candidate set."""
    timings: list[float] = []
    for features in inputs:
        started = time.perf_counter_ns()
        ranker.predict(features)
        if record_latency:
            timings.append((time.perf_counter_ns() - started) / 1_000_000)
    return np.asarray(timings, dtype=np.float64)


def _latency_summary(timings_ms: np.ndarray) -> LatencySummary:
    """Summarize measured component latency without hiding tail behavior."""
    if len(timings_ms) == 0:
        raise ValueError("Latency summary needs at least one recorded request")
    total_seconds = float(timings_ms.sum()) / 1_000
    return LatencySummary(
        requests=len(timings_ms),
        mean_ms=float(timings_ms.mean()),
        p50_ms=float(np.quantile(timings_ms, 0.5)),
        p95_ms=float(np.quantile(timings_ms, 0.95)),
        queries_per_second=float(len(timings_ms) / total_seconds),
    )


def _target_recall(candidates: list[np.ndarray], targets: np.ndarray) -> float:
    """Compute candidate recall on the benchmark's held-out next-item targets."""
    return float(np.mean([target in row for target, row in zip(targets, candidates, strict=True)]))


def _candidate_overlap(exact: list[np.ndarray], approximate: list[np.ndarray]) -> float:
    """Measure the fraction of exact candidates preserved by the HNSW index."""
    return float(
        np.mean(
            [
                len(set(approximate_row).intersection(exact_row)) / len(exact_row)
                for exact_row, approximate_row in zip(exact, approximate, strict=True)
            ]
        )
    )


def _synchronize(device: torch.device) -> None:
    """Ensure accelerator work belongs to the request being timed."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _import_faiss() -> object:
    """Load the optional local FAISS dependency only for this benchmark command."""
    try:
        import faiss
    except ModuleNotFoundError as error:
        raise RuntimeError("Install Rekindle's model dependencies to run benchmarks") from error
    return faiss


def _report(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)
