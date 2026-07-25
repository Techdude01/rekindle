"""Configuration loading with a deliberately small, inspectable contract."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a required project setting is absent or malformed."""


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file and validate the top-level project sections."""
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ConfigError(f"Expected a mapping at the root of {path}")

    required_sections = {"project", "data", "split", "retrieval", "ranking", "tracking"}
    missing = required_sections.difference(config)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ConfigError(f"Missing required configuration section(s): {joined}")

    malformed = [name for name in required_sections if not isinstance(config[name], Mapping)]
    if malformed:
        joined = ", ".join(sorted(malformed))
        raise ConfigError(f"Configuration section(s) must be mappings: {joined}")

    return config
