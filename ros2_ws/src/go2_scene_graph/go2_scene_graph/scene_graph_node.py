"""go2_scene_graph ROS 2 node.

Subscribes to SemanticDetectionArray, transforms detections into the map frame via TF,
updates an in-memory SceneGraphStore, and periodically publishes:

- /semantic/scene_graph (SceneGraph)
- /semantic/object_markers (MarkerArray)

Also serves /semantic/query_objects for text-grounded lookup without navigation.
"""

from __future__ import annotations

import traceback
from typing import Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseStamped, Vector3
from rclpy.duration import Duration as RclDuration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_geometry_msgs import do_transform_point  # noqa: F401  (registers PointStamped hook)
from geometry_msgs.msg import PointStamped

from go2_semantic_msgs.msg import (
    GroundingCandidate,
    SceneGraph,
    SceneGraphEdge,
    SemanticDetectionArray,
    SemanticObject,
)
from go2_semantic_msgs.srv import QueryObjects

from .graph_store import AssociationParams, SceneGraphStore, cosine_similarity
from .marker_viz import build_object_markers
from .spatial_relations import RelationParams, compute_edges


def _qos_reliable_depth5() -> QoSProfile:
    q = QoSProfile(depth=5)
    q.reliability = ReliabilityPolicy.RELIABLE
    return q


def _time_to_msg(ns: int) -> TimeMsg:
    sec = ns // 1_000_000_000
    nsec = ns % 1_000_000_000
    return TimeMsg(sec=int(sec), nanosec=int(nsec))


class SceneGraphNode(Node):
    def __init__(self) -> None:
        super().__init__("go2_scene_graph")

        # --- parameters ---
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_link_frame", "base_link")
        self.declare_parameter("detections_topic", "/semantic/detections")
        self.declare_parameter("scene_graph_topic", "/semantic/scene_graph")
        self.declare_parameter("object_markers_topic", "/semantic/object_markers")
        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("tf_lookup_timeout_s", 0.1)
        self.declare_parameter("use_precomputed_map_centroid", True)

        # Association params
        self.declare_parameter("assoc_max_dist_m", 0.5)
        self.declare_parameter("assoc_embed_threshold", 0.80)
        self.declare_parameter("pose_ema_alpha", 0.4)
        self.declare_parameter("dims_ema_alpha", 0.2)
        self.declare_parameter("embed_ema_alpha", 0.2)
        self.declare_parameter("object_ttl_s", 45.0)
        self.declare_parameter("min_observations_to_publish", 3)
        self.declare_parameter("merge_on_label_match_bonus", 0.05)

        # Relation params
        self.declare_parameter("near_threshold_m", 1.2)
        self.declare_parameter("lateral_cone_deg", 35.0)
        self.declare_parameter("vertical_threshold_m", 0.15)
        self.declare_parameter("on_vertical_slack_m", 0.10)
        self.declare_parameter("max_edges", 200)

        self._store = SceneGraphStore(self._load_association_params())
        self._relation_params = self._load_relation_params()

        # --- TF ---
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # --- IO ---
        self._pub_graph = self.create_publisher(
            SceneGraph,
            self.get_parameter("scene_graph_topic").value,
            _qos_reliable_depth5(),
        )
        self._pub_markers = self.create_publisher(
            __import__("visualization_msgs.msg", fromlist=["MarkerArray"]).MarkerArray,
            self.get_parameter("object_markers_topic").value,
            _qos_reliable_depth5(),
        )
        self._sub = self.create_subscription(
            SemanticDetectionArray,
            self.get_parameter("detections_topic").value,
            self._on_detections,
            _qos_reliable_depth5(),
        )

        # --- Service: QueryObjects ---
        self._srv = self.create_service(
            QueryObjects, "/semantic/query_objects", self._on_query_objects
        )

        # --- Publish timer ---
        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self._pub_timer = self.create_timer(1.0 / max(rate_hz, 0.1), self._publish_snapshot)

        self.get_logger().info(
            f"go2_scene_graph ready: map_frame={self.get_parameter('map_frame').value}, "
            f"publish_rate={rate_hz} Hz"
        )

    # ------------------------------------------------------------- params

    def _load_association_params(self) -> AssociationParams:
        return AssociationParams(
            assoc_max_dist_m=float(self.get_parameter("assoc_max_dist_m").value),
            assoc_embed_threshold=float(self.get_parameter("assoc_embed_threshold").value),
            pose_ema_alpha=float(self.get_parameter("pose_ema_alpha").value),
            dims_ema_alpha=float(self.get_parameter("dims_ema_alpha").value),
            embed_ema_alpha=float(self.get_parameter("embed_ema_alpha").value),
            object_ttl_s=float(self.get_parameter("object_ttl_s").value),
            min_observations_to_publish=int(self.get_parameter("min_observations_to_publish").value),
            merge_on_label_match_bonus=float(self.get_parameter("merge_on_label_match_bonus").value),
        )

    def _load_relation_params(self) -> RelationParams:
        return RelationParams(
            near_threshold_m=float(self.get_parameter("near_threshold_m").value),
            lateral_cone_deg=float(self.get_parameter("lateral_cone_deg").value),
            vertical_threshold_m=float(self.get_parameter("vertical_threshold_m").value),
            on_vertical_slack_m=float(self.get_parameter("on_vertical_slack_m").value),
            max_edges=int(self.get_parameter("max_edges").value),
        )

    # -------------------------------------------------------------- TF helper

    def _transform_camera_point_to_map(
        self,
        point_xyz: np.ndarray,
        source_frame: str,
        stamp: TimeMsg,
    ) -> Optional[np.ndarray]:
        map_frame = str(self.get_parameter("map_frame").value)
        timeout_s = float(self.get_parameter("tf_lookup_timeout_s").value)

        if source_frame == map_frame:
            return point_xyz

        ps = PointStamped()
        ps.header.frame_id = source_frame
        ps.header.stamp = stamp
        ps.point.x = float(point_xyz[0])
        ps.point.y = float(point_xyz[1])
        ps.point.z = float(point_xyz[2])
        try:
            transform = self._tf_buffer.lookup_transform(
                map_frame,
                source_frame,
                stamp,
                timeout=RclDuration(seconds=timeout_s),
            )
            transformed = do_transform_point(ps, transform)
            return np.array(
                [transformed.point.x, transformed.point.y, transformed.point.z],
                dtype=np.float32,
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"TF {source_frame}→{map_frame} failed at stamp {stamp.sec}.{stamp.nanosec}: {exc}",
                throttle_duration_sec=5.0,
            )
            return None

    # --------------------------------------------------------- callback

    def _on_detections(self, msg: SemanticDetectionArray) -> None:
        if not msg.detections:
            return

        # Refresh params on each message (cheap).
        self._store.update_params(**vars(self._load_association_params()))
        self._relation_params = self._load_relation_params()

        use_precomputed = bool(self.get_parameter("use_precomputed_map_centroid").value)
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        source_frame = msg.source_frame_id or msg.header.frame_id

        ingested = 0
        for det in msg.detections:
            if det.embedding_dim <= 0 or len(det.embedding) != det.embedding_dim:
                self.get_logger().debug(
                    f"Skipping detection with bad embedding dim: {det.embedding_dim} vs {len(det.embedding)}"
                )
                continue

            if use_precomputed and det.centroid_3d_map_valid:
                p_map = np.array(
                    [det.centroid_3d_map.x, det.centroid_3d_map.y, det.centroid_3d_map.z],
                    dtype=np.float32,
                )
            else:
                p_cam = np.array(
                    [det.centroid_3d_camera.x, det.centroid_3d_camera.y, det.centroid_3d_camera.z],
                    dtype=np.float32,
                )
                if not np.isfinite(p_cam).all():
                    continue
                p_map = self._transform_camera_point_to_map(p_cam, source_frame, msg.header.stamp)
                if p_map is None:
                    continue

            dims = np.array(
                [det.dimensions_xyz.x, det.dimensions_xyz.y, det.dimensions_xyz.z],
                dtype=np.float32,
            )
            embedding = np.asarray(det.embedding, dtype=np.float32)

            try:
                self._store.ingest(
                    label=det.label,
                    score=float(det.score),
                    position_map=p_map,
                    dimensions=dims,
                    embedding=embedding,
                    depth_median_m=float(det.depth_median_m),
                    stamp_ns=stamp_ns,
                )
                ingested += 1
            except Exception as exc:
                self.get_logger().warn(f"ingest failed: {exc}")
                self.get_logger().debug(traceback.format_exc())

        if ingested > 0:
            self._store.prune_stale(now_ns=self.get_clock().now().nanoseconds)
            self.get_logger().debug(
                f"scene graph: +{ingested} detections, {len(self._store)} objects"
            )

    # --------------------------------------------------------- snapshot

    def _publish_snapshot(self) -> None:
        from visualization_msgs.msg import MarkerArray  # noqa: F401

        now = self.get_clock().now().to_msg()
        map_frame = str(self.get_parameter("map_frame").value)

        publishable = self._store.objects_with_min_observations()
        msg = SceneGraph()
        msg.header.frame_id = map_frame
        msg.header.stamp = now

        for obj in publishable:
            so = SemanticObject()
            so.object_id = obj.object_id
            so.label = obj.label
            so.confidence = float(obj.confidence)
            so.pose = PoseStamped()
            so.pose.header.frame_id = map_frame
            so.pose.header.stamp = _time_to_msg(obj.last_seen_ns)
            so.pose.pose.position.x = float(obj.position_map[0])
            so.pose.pose.position.y = float(obj.position_map[1])
            so.pose.pose.position.z = float(obj.position_map[2])
            so.pose.pose.orientation.w = 1.0
            so.dimensions_xyz = Vector3(
                x=float(obj.dimensions[0]),
                y=float(obj.dimensions[1]),
                z=float(obj.dimensions[2]),
            )
            so.embedding = obj.embedding.astype(np.float32).tolist()
            so.embedding_dim = int(obj.embedding_dim)
            so.observation_count = int(obj.observation_count)
            so.first_seen = _time_to_msg(obj.first_seen_ns)
            so.last_seen = _time_to_msg(obj.last_seen_ns)
            so.last_depth_median_m = float(obj.last_depth_median_m)
            # Label history: emit current label first, then up to 4 others by cumulative score.
            ranked = sorted(obj.label_history.items(), key=lambda kv: kv[1], reverse=True)[:5]
            so.label_history = [k for k, _ in ranked]
            so.label_history_scores = [float(v) for _, v in ranked]
            msg.nodes.append(so)

        # Edges
        edges = compute_edges(publishable, self._relation_params)
        for e in edges:
            em = SceneGraphEdge()
            em.source_object_id = e["source_id"]
            em.target_object_id = e["target_id"]
            em.relation_type = e["relation_type"]
            em.distance_m = float(e["distance_m"])
            em.relation_score = float(e["relation_score"])
            msg.edges.append(em)

        msg.total_objects = len(publishable)
        msg.total_edges = len(edges)
        if publishable:
            confs = [o.confidence for o in publishable]
            msg.min_object_confidence = float(min(confs))
            msg.max_object_confidence = float(max(confs))
            msg.median_object_confidence = float(np.median(confs))
        else:
            msg.min_object_confidence = 0.0
            msg.max_object_confidence = 0.0
            msg.median_object_confidence = 0.0

        self._pub_graph.publish(msg)

        markers = build_object_markers(
            publishable,
            map_frame=map_frame,
            stamp=now,
            min_observations=int(self.get_parameter("min_observations_to_publish").value),
        )
        self._pub_markers.publish(markers)

    # ----------------------------------------------------------- service

    def _on_query_objects(self, request, response):
        """Text-only ranking; this node cannot encode text itself. Callers are expected
        to provide already-tokenizable CLIP text; our language-grounding node owns the
        text encoder and calls this service after scoring. In this minimal service
        endpoint, we perform a lexical-only match, returning objects whose label
        contains the query substring. CLIP scoring lives in go2_language_grounding.
        """
        response.success = True
        response.message = "lexical-match only; use go2_language_grounding for CLIP scoring"
        q = (request.text_query or "").strip().lower()
        response.parsed_target_noun = q
        response.parsed_attribute = ""
        response.parsed_relation = ""
        response.parsed_reference_noun = ""
        response.parse_latency_ms = 0.0
        response.score_latency_ms = 0.0
        response.total_latency_ms = 0.0

        matches = self._store.filter_by_label(q)
        matches.sort(key=lambda o: (o.confidence, o.observation_count), reverse=True)
        for obj in matches[: max(1, int(request.top_k))]:
            cand = GroundingCandidate()
            cand.object_id = obj.object_id
            cand.label = obj.label
            cand.score = float(obj.confidence)
            cand.score_clip = 0.0
            cand.score_label = 1.0
            cand.score_spatial = 0.0
            ps = PoseStamped()
            ps.header.frame_id = str(self.get_parameter("map_frame").value)
            ps.header.stamp = _time_to_msg(obj.last_seen_ns)
            ps.pose.position.x = float(obj.position_map[0])
            ps.pose.position.y = float(obj.position_map[1])
            ps.pose.position.z = float(obj.position_map[2])
            ps.pose.orientation.w = 1.0
            cand.object_pose = ps
            response.candidates.append(cand)
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SceneGraphNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
