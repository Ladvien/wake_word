from pathlib import Path

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Get project root (parent of wake_word directory)
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))


class ConfigManager:
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = CONFIG_PATH
        self.config_path = Path(config_path)
        self.project_root = PROJECT_ROOT
        self.config = self._load_and_validate_config()
        self._resolve_paths()

    def _load_and_validate_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def _resolve_paths(self):
        paths_config = self.config.get("paths", {})
        for key, path_str in paths_config.items():
            if isinstance(path_str, str) and not Path(path_str).is_absolute():
                paths_config[key] = str(self.project_root / path_str)

    def get(self, key_path: str, default=None):
        keys = key_path.split(".")
        value = self.config
        for key in keys:
            value = value.get(key, default) if isinstance(value, dict) else default
        return value

    def update(self, key_path: str, value):
        keys = key_path.split(".")
        config = self.config
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = value
