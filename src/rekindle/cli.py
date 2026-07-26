"""Command-line entry points for reproducible pipeline stages."""

import json
from pathlib import Path

import typer
from rich.console import Console

from rekindle.baselines.data import load_warm_events
from rekindle.baselines.item_cf import ItemItemCosine
from rekindle.baselines.popularity import TimeDecayedPopularity
from rekindle.config import load_config
from rekindle.data.prepare import prepare_dataset
from rekindle.evaluation.baselines import item_cf_recall_at_k, popularity_recall_at_k
from rekindle.evaluation.metrics import fixed_subset_indices
from rekindle.retrieval.sequences import (
    SequenceExamples,
    build_sequence_examples,
    load_sequence_events,
)
from rekindle.retrieval.training import train_retriever as train_retriever_model

app = typer.Typer(no_args_is_help=True, help="Rekindle's reproducible recommendation pipeline.")
console = Console()
CONFIG_OPTION = typer.Option(Path("config/base.yaml"), exists=True, readable=True)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict:
    return load_config(path)


@app.command("prepare-data")
def prepare_data(
    config: Path = CONFIG_OPTION,
) -> None:
    """Extract canonical event and metadata Parquet tables from local Amazon JSONL files."""
    manifest = prepare_dataset(_load(config), _project_root())
    console.print(f"[green]Prepared dataset.[/green] Manifest: {manifest}")


@app.command("fit-baselines")
def fit_baselines(
    config: Path = CONFIG_OPTION,
) -> None:
    """Fit validation-ready popularity and item–item CF artifacts from warm training events."""
    settings = _load(config)
    project_root = _project_root()
    interactions_path = project_root / settings["data"]["prepared_interactions_path"]
    if not interactions_path.exists():
        raise typer.BadParameter("Run prepare-data before fitting baselines.")

    events = load_warm_events(interactions_path, split="train")
    artifact_directory = project_root / "artifacts/baselines"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    for half_life_days in settings["baselines"]["popularity_half_lives_days"]:
        model = TimeDecayedPopularity.fit(events, half_life_days=half_life_days)
        model.save(artifact_directory / f"popularity-{half_life_days}d.json")

    item_cf = ItemItemCosine.fit(
        events,
        recency_decay=settings["baselines"]["item_cf_recency_decay"],
    )
    item_cf.save(artifact_directory / "item-cosine")
    console.print(
        "[green]Fitted baselines.[/green] "
        f"{events.height:,} warm training events, {len(item_cf.item_ids):,} candidate items."
    )


@app.command("build-sequences")
def build_sequences(
    config: Path = CONFIG_OPTION,
) -> None:
    """Build strict chronological next-item examples for retrieval training and validation."""
    settings = _load(config)
    project_root = _project_root()
    interactions_path = project_root / settings["data"]["prepared_interactions_path"]
    if not interactions_path.exists():
        raise typer.BadParameter("Run prepare-data before building retrieval sequences.")

    sequence_directory = project_root / "artifacts/sequences"
    for split in ("train", "validation"):
        examples = build_sequence_examples(
            load_sequence_events(interactions_path, target_split=split),
            target_split=split,
            history_size=settings["split"]["replay_history_size"],
            min_prior_interactions=settings["split"]["min_prior_interactions"],
        )
        examples.save(sequence_directory / split)
        console.print(f"[green]Built {split} examples:[/green] {examples.count:,}")


@app.command("train-retriever")
def train_retriever(
    config: Path = CONFIG_OPTION,
    history_only: bool = typer.Option(
        False,
        "--history-only",
        help="Ablate sparse user-ID embeddings; encode only the prior item history.",
    ),
    output_name: str = typer.Option(
        "retriever",
        "--output-name",
        help="Ignored-artifact directory name under artifacts/.",
    ),
) -> None:
    """Train the two-tower retriever with early stopping on validation Recall@100."""
    settings = _load(config)
    sequence_directory = _project_root() / "artifacts/sequences"
    train_directory = sequence_directory / "train"
    validation_directory = sequence_directory / "validation"
    if not train_directory.exists() or not validation_directory.exists():
        raise typer.BadParameter("Run build-sequences before training the retriever.")
    if Path(output_name).name != output_name:
        raise typer.BadParameter("output-name must be a single directory name.")
    console.print("[cyan]Loading training and validation sequence artifacts...[/cyan]")
    result = train_retriever_model(
        SequenceExamples.load(train_directory),
        SequenceExamples.load(validation_directory),
        settings["retrieval"],
        seed=settings["project"]["seed"],
        output_directory=_project_root() / "artifacts" / output_name,
        progress_callback=console.print,
        use_user_embedding=not history_only,
    )
    console.print(
        "[green]Trained retriever.[/green] "
        f"Best epoch: {result.best_epoch}; exact-catalog validation Recall@100 "
        f"(1,000-event selection subset): {result.validation_full_catalog_recall_at_100:.4f}; "
        f"sampled diagnostic Recall@100: {result.validation_sampled_recall_at_100:.4f}; "
        f"variant: {result.model_variant}; device: {result.device}."
    )


@app.command("evaluate-baselines")
def evaluate_baselines(
    config: Path = CONFIG_OPTION,
) -> None:
    """Compare training-only baselines on the retriever's fixed exact validation subset."""
    settings = _load(config)
    project_root = _project_root()
    validation_directory = project_root / "artifacts/sequences/validation"
    baseline_directory = project_root / "artifacts/baselines"
    if not validation_directory.exists() or not baseline_directory.exists():
        raise typer.BadParameter("Run build-sequences and fit-baselines before evaluation.")

    examples = SequenceExamples.load(validation_directory)
    selection_indices = fixed_subset_indices(
        examples.count,
        settings["retrieval"]["full_catalog_evaluation_examples"],
        seed=settings["project"]["seed"] + 2,
    )
    recall_k = settings["retrieval"]["primary_recall_k"]
    results: list[dict[str, str | float | int]] = []
    for half_life_days in settings["baselines"]["popularity_half_lives_days"]:
        model = TimeDecayedPopularity.load(
            baseline_directory / f"popularity-{half_life_days}d.json"
        )
        name = f"popularity_{half_life_days}d"
        console.print(f"[cyan]Evaluating {name}...[/cyan]")
        recall = popularity_recall_at_k(
            model,
            examples,
            selection_indices,
            k=recall_k,
            progress_callback=console.print,
        )
        results.append(
            {"name": name, "recall_at_100": recall, "evaluated_examples": len(selection_indices)}
        )

    console.print("[cyan]Evaluating item_item_cosine...[/cyan]")
    item_cf = ItemItemCosine.load(baseline_directory / "item-cosine")
    item_cf_recall = item_cf_recall_at_k(
        item_cf,
        examples,
        selection_indices,
        k=recall_k,
        progress_callback=console.print,
    )
    results.append(
        {
            "name": "item_item_cosine",
            "recall_at_100": item_cf_recall,
            "evaluated_examples": len(selection_indices),
        }
    )
    output_path = project_root / "artifacts/evaluation/validation-baselines.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    console.print("[green]Baseline Recall@100 results:[/green]")
    for result in results:
        console.print(f"  {result['name']}: {result['recall_at_100']:.4f}")


@app.command("generate-ranker-data")
def generate_ranker_data() -> None:
    """Generate cross-fitted ranking candidates (implemented in milestone two)."""
    raise typer.Exit("Ranker data generation is not implemented yet. Complete retrieval first.")


@app.command("train-ranker")
def train_ranker() -> None:
    """Train the LightGBM LambdaRank reranker (implemented in milestone two)."""
    raise typer.Exit("The ranker is not implemented yet. Complete ranker data generation first.")


@app.command("evaluate")
def evaluate() -> None:
    """Run sequential replay evaluation (implemented in milestone three)."""
    raise typer.Exit("Evaluation is not implemented yet. Complete model training first.")


@app.command("benchmark")
def benchmark() -> None:
    """Run local retrieval/ranking latency benchmarks (implemented in milestone three)."""
    raise typer.Exit("Benchmarking is not implemented yet. Complete model training first.")


@app.command("recommend")
def recommend(user_id: str = typer.Option(..., help="A locally prepared user identifier.")) -> None:
    """Produce a local top-10 recommendation demo (implemented in milestone three)."""
    raise typer.Exit(f"Recommendation demo is not implemented yet for user {user_id!r}.")
