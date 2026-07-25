"""Command-line entry points for reproducible pipeline stages."""

from pathlib import Path

import typer
from rich.console import Console

from rekindle.config import load_config
from rekindle.data.prepare import prepare_dataset

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


@app.command("train-retriever")
def train_retriever() -> None:
    """Train the two-tower retrieval model (implemented in milestone two)."""
    raise typer.Exit("The retrieval model is not implemented yet. Complete prepare-data first.")


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
