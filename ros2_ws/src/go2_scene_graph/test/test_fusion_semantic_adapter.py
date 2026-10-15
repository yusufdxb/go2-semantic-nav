"""Offline contract checks; no ROS runtime or validated hardware is implied."""

from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
from go2_scene_graph.fusion_semantic_adapter import (
    FusionPairError,
    FusionSemanticJoin,
    bridge_pair,
)


def point(x=0.0, y=0.0, z=0.0):
    return NS(x=x, y=y, z=z)


def pair(stamp=10, frame="odom"):
    detection = NS(
        label="chair",
        score=0.8,
        embedding=[1.0, 0.0, 0.0],
        embedding_dim=3,
        centroid_3d_camera=point(99, 99, 99),
        centroid_3d_map=point(88, 88, 88),
        centroid_3d_map_valid=True,
        dimensions_xyz=point(2, 3, 4),
        depth_median_m=99,
        tracklet_id="source-track",
        bbox_2d=NS(token="original-box"),
        mask_roi=NS(token="original-roi"),
        mask_rle=[2, 5],
    )
    semantic = NS(
        header=NS(stamp=NS(sec=stamp, nanosec=5), frame_id="camera"),
        source_frame_id="camera",
        detections=[detection],
        detector_backend="detector",
        encoder_backend="encoder",
        segmenter_backend="segmenter",
        latency_backproject_ms=3.0,
    )
    obj = NS(
        label="chair",
        detector_score=0.8,
        status=0,
        valid=True,
        track_id=4,
        support_points=20,
        position=point(1, 2, 3),
        covariance=[0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01],
    )
    fusion = NS(
        header=NS(stamp=NS(sec=stamp, nanosec=5), frame_id=frame),
        camera_frame="camera",
        status=0,
        objects=[obj],
    )
    return semantic, fusion


def test_fused_position_and_identity_preserved_original_depth_discarded():
    semantic, fusion = pair()
    output = bridge_pair(semantic, fusion)
    det = output.detections[0]
    assert output.header.frame_id == output.source_frame_id == "odom"
    assert output.header.stamp == semantic.header.stamp
    assert vars(det.centroid_3d_camera) == {"x": 1, "y": 2, "z": 3}
    assert not det.centroid_3d_map_valid
    assert vars(det.centroid_3d_map) == vars(point())
    assert vars(det.dimensions_xyz) == vars(point()) and det.depth_median_m == 0
    assert det.embedding == semantic.detections[0].embedding
    assert det.label == "chair" and det.tracklet_id == "source-track"
    assert det.mask_rle == [2, 5] and det.bbox_2d.token == "original-box"
    assert output.encoder_backend == "encoder" and output.latency_backproject_ms == -1
    det.embedding[0] = 7
    assert semantic.detections[0].embedding[0] == 1
    assert semantic.detections[0].depth_median_m == 99


def test_map_centroid_valid_only_in_configured_map_frame():
    semantic, fusion = pair(frame="world")
    assert not bridge_pair(semantic, fusion).detections[0].centroid_3d_map_valid
    result = bridge_pair(semantic, fusion, map_frame="world").detections[0]
    assert result.centroid_3d_map_valid and vars(result.centroid_3d_map) == vars(point(1, 2, 3))


@pytest.mark.parametrize(
    "change",
    [
        lambda s, f: setattr(f.header.stamp, "nanosec", 6),
        lambda s, f: setattr(f, "camera_frame", "other_camera"),
        lambda s, f: setattr(s.header, "frame_id", "wrong"),
        lambda s, f: setattr(f.header, "frame_id", ""),
        lambda s, f: setattr(f, "status", 1),
        lambda s, f: f.objects.clear(),
        lambda s, f: setattr(f.objects[0], "label", "table"),
        lambda s, f: setattr(f.objects[0], "detector_score", 0.7),
        lambda s, f: setattr(s.detections[0], "score", float("nan")),
        lambda s, f: setattr(f.objects[0], "status", 6),
        lambda s, f: setattr(f.objects[0], "valid", False),
        lambda s, f: setattr(f.objects[0], "status", 255),
        lambda s, f: setattr(f.objects[0], "track_id", 0),
        lambda s, f: setattr(f.objects[0], "support_points", 0),
        lambda s, f: setattr(f.objects[0].position, "x", float("nan")),
        lambda s, f: setattr(f.objects[0], "covariance", [0.0]),
        lambda s, f: f.objects[0].covariance.__setitem__(0, float("inf")),
        lambda s, f: f.objects[0].covariance.__setitem__(1, 0.1),
        lambda s, f: f.objects[0].covariance.__setitem__(0, -0.1),
        lambda s, f: setattr(f.objects[0], "covariance", [1, 2, 0, 2, 1, 0, 0, 0, 1]),
        lambda s, f: setattr(s.detections[0], "embedding_dim", 4),
        lambda s, f: setattr(s.detections[0], "embedding", [0, 0, 0]),
        lambda s, f: setattr(s.detections[0], "embedding", [float("nan"), 0, 1]),
    ],
)
def test_malformed_pairs_rejected(change):
    semantic, fusion = pair()
    change(semantic, fusion)
    with pytest.raises(FusionPairError):
        bridge_pair(semantic, fusion)


def test_invalid_object_omitted_without_detector_geometry_fallback():
    semantic, fusion = pair()
    fusion.objects[0].status, fusion.objects[0].valid = 7, False
    fusion.objects[0].position.x = float("nan")
    assert bridge_pair(semantic, fusion).detections == []


def test_partial_valid_frame_retains_only_matching_valid_object():
    semantic, fusion = pair()
    semantic.detections.append(deepcopy(semantic.detections[0]))
    fusion.objects.append(deepcopy(fusion.objects[0]))
    fusion.objects[0].valid, fusion.objects[0].status = False, 6
    semantic.detections[1].label = fusion.objects[1].label = "desk"
    assert [d.label for d in bridge_pair(semantic, fusion).detections] == ["desk"]


@pytest.mark.parametrize("fusion_first", [False, True])
def test_join_both_arrival_orders_and_replay_rejection(fusion_first):
    semantic, fusion = pair()
    join = FusionSemanticJoin()
    first, second = (
        ((join.add_fusion, fusion), (join.add_semantic, semantic))
        if fusion_first
        else ((join.add_semantic, semantic), (join.add_fusion, fusion))
    )
    assert first[0](first[1]) is None
    assert len(second[0](second[1]).detections) == 1
    assert not join.fusion and not join.semantic
    with pytest.raises(FusionPairError, match="Duplicate"):
        join.add_semantic(semantic)


def test_join_exact_match_does_not_pair_nearby_or_other_camera():
    semantic, fusion = pair()
    join = FusionSemanticJoin()
    assert join.add_semantic(semantic) is None
    fusion.header.stamp.nanosec += 1
    assert join.add_fusion(fusion) is None
    fusion.header.stamp.nanosec -= 1
    fusion.camera_frame = "other_camera"
    assert join.add_fusion(fusion) is None
    assert len(join.semantic) == 1 and len(join.fusion) == 2


def test_expiry_uses_receive_clock_and_duplicate_does_not_extend():
    now = [0.0]
    join = FusionSemanticJoin(clock=lambda: now[0], expiry_s=1)
    semantic, fusion = pair()
    join.add_semantic(semantic)
    now[0] = 0.9
    join.add_semantic(semantic)
    now[0] = 1.0
    assert join.add_fusion(fusion) is None
    assert not join.semantic


def test_cache_eviction_is_bounded_and_cannot_pair_evicted_frame():
    join = FusionSemanticJoin(capacity=2)
    for stamp in [10, 11, 12]:
        join.add_semantic(pair(stamp)[0])
    assert len(join.semantic) == 2
    assert join.add_fusion(pair(10)[1]) is None
    assert join.add_fusion(pair(12)[1]) is not None
    with pytest.raises(FusionPairError, match="out-of-order"):
        join.add_fusion(pair(11)[1])


def test_failed_pair_consumed_and_never_replaced_by_a_later_copy():
    join = FusionSemanticJoin()
    semantic, fusion = pair()
    join.add_semantic(semantic)
    fusion.objects[0].position.x = float("nan")
    with pytest.raises(FusionPairError):
        join.add_fusion(fusion)
    assert not join.semantic and not join.fusion
    with pytest.raises(FusionPairError, match="Duplicate"):
        join.add_fusion(pair()[1])


def test_cache_capacity_and_embedding_limits():
    semantic, fusion = pair()
    join = FusionSemanticJoin(max_embedding_dim=2)
    with pytest.raises(FusionPairError, match="Embedding capacity"):
        join.add_semantic(semantic)
    assert not join.semantic
    with pytest.raises(FusionPairError, match="capacity"):
        bridge_pair(semantic, fusion, max_detections=0)


@pytest.mark.parametrize("capacity,expiry", [(0, 1), (1001, 1), (1, 0), (1, float("nan"))])
def test_invalid_cache_parameters(capacity, expiry):
    with pytest.raises(ValueError):
        FusionSemanticJoin(capacity=capacity, expiry_s=expiry)
