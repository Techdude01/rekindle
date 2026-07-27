from typer.testing import CliRunner

from rekindle.cli import app


def test_cli_exposes_pipeline_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "prepare-data" in result.output
    assert "fit-baselines" in result.output
    assert "build-sequences" in result.output
    assert "train-retriever" in result.output
    assert "evaluate-baselines" in result.output
    assert "evaluate-candidate-union" in result.output
