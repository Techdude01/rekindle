from pathlib import Path

import pytest

from rekindle.config import ConfigError, load_config


def test_loads_project_configuration() -> None:
    config = load_config(Path("config/base.yaml"))

    assert config["project"]["name"] == "rekindle"
    assert config["retrieval"]["candidate_count"] == 200


def test_rejects_missing_required_section(tmp_path: Path) -> None:
    config_path = tmp_path / "incomplete.yaml"
    config_path.write_text("project: {}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Missing required"):
        load_config(config_path)
