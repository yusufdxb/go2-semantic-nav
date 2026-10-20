"""Launch parameter regression checks; skip when the ROS underlay is absent."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("launch_ros")
from launch import LaunchContext


@pytest.fixture
def launch_module(monkeypatch):
    path = Path(__file__).parents[1] / "launch" / "scene_graph.launch.py"
    spec = importlib.util.spec_from_file_location("fusion_mapping_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "Node", lambda **kwargs: SimpleNamespace(**kwargs))
    return module


def context(tmp_path, **overrides):
    params = tmp_path / "custom.yaml"
    params.write_text(
        yaml.safe_dump(
            {
                "go2_scene_graph": {
                    "ros__parameters": {
                        "map_frame": "world",
                        "detections_topic": "/custom/detections",
                        "use_sim_time": True,
                        "snapshot_dir": "/tmp/example-snapshots",
                    }
                }
            }
        )
    )
    values = {
        "params_file": str(params),
        "fusion_mapping": "false",
        "map_frame": "",
        "detections_topic": "",
        "snapshot_dir": "",
        "use_sim_time": "",
        "fusion_topic": "/fusion/objects",
        "fused_detections_topic": "/semantic/fused",
        "log_level": "info",
    }
    values.update(overrides)
    ctx = LaunchContext()
    ctx.launch_configurations.update(values)
    return ctx


def test_disabled_preserves_original_yaml_parameters(launch_module, tmp_path):
    nodes = launch_module._nodes(context(tmp_path))
    assert len(nodes) == 1 and nodes[0].parameters[1] == {}


def test_enabled_inherits_frame_clock_and_original_input(launch_module, tmp_path):
    nodes = launch_module._nodes(context(tmp_path, fusion_mapping="true"))
    mapper, adapter = nodes
    assert mapper.parameters[1] == {"detections_topic": "/semantic/fused"}
    assert adapter.parameters[0] == {
        "map_frame": "world",
        "semantic_topic": "/custom/detections",
        "use_sim_time": True,
        "fusion_topic": "/fusion/objects",
        "output_topic": "/semantic/fused",
    }


def test_explicit_overrides_align_adapter_and_mapper(launch_module, tmp_path):
    mapper, adapter = launch_module._nodes(
        context(
            tmp_path,
            fusion_mapping="true",
            map_frame="global",
            detections_topic="/camera/semantic",
            use_sim_time="false",
            snapshot_dir="/tmp/another-snapshot-output",
        )
    )
    assert mapper.parameters[1] == {
        "map_frame": "global",
        "detections_topic": "/semantic/fused",
        "use_sim_time": False,
        "snapshot_dir": "/tmp/another-snapshot-output",
    }
    assert adapter.parameters[0]["map_frame"] == "global"
    assert adapter.parameters[0]["semantic_topic"] == "/camera/semantic"
    assert not adapter.parameters[0]["use_sim_time"]


def test_invalid_boolean_is_rejected(launch_module, tmp_path):
    with pytest.raises(ValueError, match="fusion_mapping"):
        launch_module._nodes(context(tmp_path, fusion_mapping="typo"))
