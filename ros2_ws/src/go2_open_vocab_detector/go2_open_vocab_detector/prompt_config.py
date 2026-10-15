"""Prompt configuration loading without ROS runtime dependencies."""

from pathlib import Path

import yaml


def load_prompt_classes(path: str) -> list[str]:
    """Load and validate a plain ``prompts: [...]`` YAML file."""
    prompt_path = Path(path).expanduser()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"prompt classes file not found: {prompt_path}")
    data = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("prompts"), list):
        raise ValueError("prompt classes file must contain a 'prompts' list")
    prompts = [str(item).strip() for item in data["prompts"]]
    if not prompts or any(not item for item in prompts):
        raise ValueError("prompt classes file contains an empty prompt")
    return prompts
