# src/drl/utils/config_loader.py

import yaml
from pathlib import Path

def load_config(config_path: str | Path) -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path (str | Path): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as file:
        return yaml.safe_load(file)
