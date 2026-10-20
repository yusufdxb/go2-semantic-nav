"""Tests for prompt-class file loading."""

from pathlib import Path

import pytest
from go2_open_vocab_detector.prompt_config import load_prompt_classes


def test_load_prompt_classes(tmp_path: Path):
    path = tmp_path / "prompts.yaml"
    path.write_text("prompts:\n  - chair\n  - red table\n", encoding="utf-8")
    assert load_prompt_classes(str(path)) == ["chair", "red table"]


@pytest.mark.parametrize("content", ["{}\n", "prompts: nope\n", "prompts: []\n"])
def test_invalid_prompt_classes_fail(content: str, tmp_path: Path):
    path = tmp_path / "prompts.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompt_classes(str(path))


def test_missing_prompt_classes_file_fails(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_prompt_classes(str(tmp_path / "missing.yaml"))
