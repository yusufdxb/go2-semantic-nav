"""RViz MarkerArray generation for the scene graph."""

from __future__ import annotations

import hashlib
from typing import Iterable

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from .graph_store import TrackedObject


def _label_color(label: str) -> ColorRGBA:
    """Deterministic pleasant color per label."""
    h = hashlib.md5(label.encode("utf-8")).digest()
    r = (h[0] / 255.0) * 0.6 + 0.3
    g = (h[1] / 255.0) * 0.6 + 0.3
    b = (h[2] / 255.0) * 0.6 + 0.3
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=0.9)


def build_object_markers(
    objects: Iterable[TrackedObject],
    map_frame: str,
    stamp,
    min_observations: int = 3,
) -> MarkerArray:
    """Emit one sphere + one label text marker per object in the map frame."""
    arr = MarkerArray()
    idx = 0

    # A delete-all marker prepended so stale markers from last tick vanish.
    clear = Marker()
    clear.header.frame_id = map_frame
    clear.header.stamp = stamp
    clear.action = Marker.DELETEALL
    clear.ns = "go2_scene_graph/objects"
    arr.markers.append(clear)

    for obj in objects:
        if obj.observation_count < min_observations:
            continue

        sphere = Marker()
        sphere.header.frame_id = map_frame
        sphere.header.stamp = stamp
        sphere.ns = "go2_scene_graph/objects"
        sphere.id = idx
        idx += 1
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position = Point(
            x=float(obj.position_map[0]),
            y=float(obj.position_map[1]),
            z=float(obj.position_map[2]),
        )
        sphere.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        # Use a conservative sphere sized by the smaller horizontal extent.
        size = max(0.15, float(min(obj.dimensions[0], obj.dimensions[1])))
        sphere.scale = Vector3(x=size, y=size, z=size)
        sphere.color = _label_color(obj.label)
        sphere.lifetime = Duration(sec=0, nanosec=int(2e8))  # 200 ms
        arr.markers.append(sphere)

        text = Marker()
        text.header.frame_id = map_frame
        text.header.stamp = stamp
        text.ns = "go2_scene_graph/labels"
        text.id = idx
        idx += 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position = Point(
            x=float(obj.position_map[0]),
            y=float(obj.position_map[1]),
            z=float(obj.position_map[2] + size * 0.8),
        )
        text.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        text.scale = Vector3(x=0.0, y=0.0, z=0.15)
        text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text.text = f"{obj.label} [{obj.confidence:.2f}]"
        text.lifetime = Duration(sec=0, nanosec=int(2e8))
        arr.markers.append(text)

    return arr
