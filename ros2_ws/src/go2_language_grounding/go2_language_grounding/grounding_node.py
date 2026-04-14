"""go2_language_grounding ROS 2 node.

Exposes:
- action server `/semantic/ground_and_navigate`
- publisher  `/goal_pose` (PoseStamped in map frame)
- publisher  `/semantic/grounding_viz` (MarkerArray)

Internally maintains a cache of the latest SceneGraph and the latest global costmap.
On each action goal:
  1. parse the text query
  2. encode target text with CLIP (via go2_open_vocab_detector.backends)
  3. score graph nodes and pick chosen object
  4. resolve spatial relation if any
  5. sample a reachable stand-off pose
  6. publish /goal_pose (unless dry_run)
  7. feedback + result
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.duration import Duration as RclDuration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from go2_semantic_msgs.action import GroundAndNavigate
from go2_semantic_msgs.msg import (
    GroundingCandidate,
    SceneGraph,
    SemanticObject,
)

from .goal_sampler import SamplerParams, sample_stand_off
from .query_parser import ParsedQuery, parse
from .scoring import ScoreWeights, combine, cosine_similarity, lexical_label_match


def _qos_reliable_depth10() -> QoSProfile:
    q = QoSProfile(depth=10)
    q.reliability = ReliabilityPolicy.RELIABLE
    return q


class GroundingNode(Node):
    def __init__(self) -> None:
        super().__init__("go2_language_grounding")

        # --- parameters ---
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("scene_graph_topic", "/semantic/scene_graph")
        self.declare_parameter("costmap_topic", "/global_costmap/costmap")
        self.declare_parameter("goal_pose_topic", "/goal_pose")
        self.declare_parameter("grounding_viz_topic", "/semantic/grounding_viz")

        self.declare_parameter("encoder_backend", "openclip_vit_b16")
        self.declare_parameter("device", "cuda:0")

        self.declare_parameter("score_weight_clip", 0.7)
        self.declare_parameter("score_weight_label", 0.2)
        self.declare_parameter("score_weight_spatial", 0.1)

        self.declare_parameter("stand_off_m", 0.9)
        self.declare_parameter("goal_ring_samples", 12)
        self.declare_parameter("goal_min_stand_off_m", 0.4)
        self.declare_parameter("goal_max_stand_off_m", 2.0)
        self.declare_parameter("costmap_free_max_cost", 50)
        self.declare_parameter("retry_with_wider_ring", True)

        self.declare_parameter("use_costmap_gate", True)
        self.declare_parameter("min_scene_graph_objects", 1)
        self.declare_parameter("max_action_duration_s", 60.0)

        # Rejection thresholds — see RESULTS.md §"Honest negative results" for the v1→v2 rationale.
        self.declare_parameter("reject_absolute_floor", 0.15)
        self.declare_parameter("reject_label_floor", 0.40)
        self.declare_parameter("reject_clip_floor", 0.22)
        self.declare_parameter("reject_margin_min", 0.0,
                               ParameterDescriptor(description=(
                                   "If >0, require top-1.score - top-2.score >= this margin. "
                                   "Stricter than the label/clip floors but cuts recall when "
                                   "multiple legitimate candidates cluster together."
                               )))

        # --- state ---
        self._latest_scene_graph: Optional[SceneGraph] = None
        self._latest_costmap: Optional[OccupancyGrid] = None
        self._encoder = None
        self._text_cache: dict[str, np.ndarray] = {}

        # --- pub/sub ---
        self._pub_goal = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("goal_pose_topic").value),
            _qos_reliable_depth10(),
        )
        from visualization_msgs.msg import MarkerArray  # noqa: F401
        self._pub_viz = self.create_publisher(
            __import__("visualization_msgs.msg", fromlist=["MarkerArray"]).MarkerArray,
            str(self.get_parameter("grounding_viz_topic").value),
            _qos_reliable_depth10(),
        )

        self._sub_sg = self.create_subscription(
            SceneGraph,
            str(self.get_parameter("scene_graph_topic").value),
            self._on_scene_graph,
            _qos_reliable_depth10(),
        )
        self._sub_cm = self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter("costmap_topic").value),
            self._on_costmap,
            QoSProfile(depth=1),
        )

        # --- action server ---
        self._action = ActionServer(
            self,
            GroundAndNavigate,
            "/semantic/ground_and_navigate",
            execute_callback=self._execute_action,
            goal_callback=self._on_goal_request,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
        )

        # Lazy encoder load
        try:
            self._load_encoder()
        except Exception as exc:
            self.get_logger().warn(
                f"Encoder load failed; grounding will fall back to lexical-only: {exc}"
            )

        self.get_logger().info(
            "go2_language_grounding ready: "
            f"encoder={self.get_parameter('encoder_backend').value}, "
            f"scene_graph_topic={self.get_parameter('scene_graph_topic').value}"
        )

    # --------------------------------------------------------------- setup

    def _load_encoder(self) -> None:
        from go2_open_vocab_detector.backends import make_encoder

        self._encoder = make_encoder(str(self.get_parameter("encoder_backend").value))
        self._encoder.load(device=str(self.get_parameter("device").value))

    # ---------------------------------------------------------- subscribers

    def _on_scene_graph(self, msg: SceneGraph) -> None:
        self._latest_scene_graph = msg

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        self._latest_costmap = msg

    # ---------------------------------------------------------- encoding

    def _encode_text_cached(self, text: str) -> Optional[np.ndarray]:
        if self._encoder is None or not text:
            return None
        if text in self._text_cache:
            return self._text_cache[text]
        try:
            emb = self._encoder.encode_text([text])[0]
            self._text_cache[text] = emb
            return emb
        except Exception as exc:
            self.get_logger().warn(f"encode_text failed for {text!r}: {exc}")
            return None

    # ---------------------------------------------------------- scoring

    def _score_candidates(
        self,
        query: ParsedQuery,
        graph: SceneGraph,
        weights: ScoreWeights,
    ) -> list[GroundingCandidate]:
        """Score each scene-graph node against the parsed query.

        When an attribute is present (e.g., "red chair"), we use a **relative-prompt**
        disambiguation trick: compare cosine("{attribute} {noun}", embedding) to
        cosine("{noun}", embedding). Higher delta → attribute is more aligned with
        that object. This is the community-standard recipe and avoids the failure
        mode where all chairs score similarly on plain "chair" embedding.
        """
        attr_noun = (query.attribute + " " + query.target_noun).strip()
        plain_noun = query.target_noun.strip() or query.raw.strip()
        text_for_clip_full = attr_noun or plain_noun

        full_emb = self._encode_text_cached(text_for_clip_full)
        plain_emb = (
            self._encode_text_cached(plain_noun)
            if query.attribute and plain_noun and plain_noun != text_for_clip_full
            else None
        )

        out: list[GroundingCandidate] = []
        for node in graph.nodes:
            obj_emb = np.asarray(node.embedding, dtype=np.float32)
            if obj_emb.size == 0:
                continue

            sim_full = cosine_similarity(obj_emb, full_emb) if full_emb is not None else 0.0

            # Attribute-aware score: if we have both "red chair" and "chair" encodings,
            # reward the delta so the red one wins over a non-red chair.
            if plain_emb is not None:
                sim_plain = cosine_similarity(obj_emb, plain_emb)
                # Map [−0.2, +0.2] delta to [0, 1], clipped.
                delta = sim_full - sim_plain
                attr_bonus = max(0.0, min(1.0, (delta + 0.05) / 0.15))
                # Blend base similarity with attribute bonus.
                clip_score = 0.6 * sim_full + 0.4 * attr_bonus
            else:
                clip_score = sim_full

            label_score = lexical_label_match(query.target_noun, node.label)

            # Spatial score: neutral at this stage. _apply_relation_filter may upgrade/downgrade.
            spatial_score = 0.5
            total = combine(clip_score, label_score, spatial_score, weights)

            cand = GroundingCandidate()
            cand.object_id = node.object_id
            cand.label = node.label
            cand.score = float(total)
            cand.score_clip = float(clip_score)
            cand.score_label = float(label_score)
            cand.score_spatial = float(spatial_score)
            cand.object_pose = node.pose
            out.append(cand)

        out.sort(key=lambda c: c.score, reverse=True)
        return out

    def _apply_relation_filter(
        self,
        candidates: list[GroundingCandidate],
        query: ParsedQuery,
        graph: SceneGraph,
    ) -> list[GroundingCandidate]:
        if query.relation is None:
            return candidates

        # Find a reference object (by lexical + CLIP on reference phrase).
        ref_attr_noun = (query.reference_attribute + " " + query.reference_noun).strip()
        if not ref_attr_noun:
            return candidates
        ref_emb = self._encode_text_cached(ref_attr_noun)

        best_ref: Optional[SemanticObject] = None
        best_ref_score = -1.0
        for node in graph.nodes:
            lex = lexical_label_match(query.reference_noun, node.label)
            clip = (
                cosine_similarity(np.asarray(node.embedding, dtype=np.float32), ref_emb)
                if ref_emb is not None
                else 0.0
            )
            s = 0.6 * clip + 0.4 * lex
            if s > best_ref_score:
                best_ref_score = s
                best_ref = node
        if best_ref is None:
            return candidates

        # Prefer edges of the requested type from the graph if available.
        rel = query.relation
        allowed_sources = {
            e.source_object_id
            for e in graph.edges
            if e.relation_type == rel and e.target_object_id == best_ref.object_id
        }
        if allowed_sources:
            filtered = [c for c in candidates if c.object_id in allowed_sources]
            if filtered:
                for c in filtered:
                    c.score_spatial = 1.0
                    c.score = combine(
                        c.score_clip, c.score_label, c.score_spatial, self._current_weights()
                    )
                return filtered

        # Fallback: no spatial edges match — do not hard-filter, but penalise spatial score.
        for c in candidates:
            c.score_spatial = 0.2
            c.score = combine(c.score_clip, c.score_label, c.score_spatial, self._current_weights())
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def _current_weights(self) -> ScoreWeights:
        return ScoreWeights(
            clip=float(self.get_parameter("score_weight_clip").value),
            label=float(self.get_parameter("score_weight_label").value),
            spatial=float(self.get_parameter("score_weight_spatial").value),
        )

    # ---------------------------------------------------------- action

    def _on_goal_request(self, goal_request) -> GoalResponse:
        if self._latest_scene_graph is None or not self._latest_scene_graph.nodes:
            self.get_logger().warn("goal rejected: scene graph not yet available")
            return GoalResponse.REJECT
        if not goal_request.text_query.strip():
            self.get_logger().warn("goal rejected: empty text_query")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _publish_feedback(self, goal_handle, state: str, candidates: list[GroundingCandidate], graph_count: int, distance: float = -1.0) -> None:
        fb = GroundAndNavigate.Feedback()
        fb.state = state
        fb.grounding_confidence = candidates[0].score if candidates else 0.0
        fb.scene_graph_object_count = graph_count
        fb.current_target_id = candidates[0].object_id if candidates else ""
        fb.current_candidates = list(candidates[:5])
        fb.distance_to_goal_m = float(distance)
        goal_handle.publish_feedback(fb)

    def _execute_action(self, goal_handle):
        start_ns = time.time_ns()
        req: GroundAndNavigate.Goal = goal_handle.request
        graph = self._latest_scene_graph
        if graph is None or not graph.nodes:
            return self._fail_result("GROUNDING_FAILED", "no scene graph available", start_ns)

        parsed = parse(req.text_query)
        if req.preferred_relation:
            parsed.relation = req.preferred_relation
        weights = self._current_weights()

        self._publish_feedback(goal_handle, "PARSING", [], len(graph.nodes))

        candidates = self._score_candidates(parsed, graph, weights)
        if not candidates:
            return self._fail_result("GROUNDING_FAILED", "no candidates in scene graph", start_ns)

        candidates = self._apply_relation_filter(candidates, parsed, graph)
        self._publish_feedback(goal_handle, "SCORING", candidates, len(graph.nodes))

        # Three-layer rejection (v2 + optional margin):
        #   1. absolute floor — the total composite must clear a low bar
        #   2. label-OR-clip floor — weak on BOTH dimensions → refuse
        #   3. (optional) margin floor — top-1 must beat top-2 by a configurable delta
        # See RESULTS.md §"Honest negative results" for the v1→v2 rationale.
        top = candidates[0]
        absolute_floor = float(self.get_parameter("reject_absolute_floor").value)
        label_floor = float(self.get_parameter("reject_label_floor").value)
        clip_floor = float(self.get_parameter("reject_clip_floor").value)
        margin_min = float(self.get_parameter("reject_margin_min").value)

        if top.score < absolute_floor:
            return self._fail_result(
                "GROUNDING_FAILED",
                f"top candidate score {top.score:.3f} below absolute floor {absolute_floor:.2f}",
                start_ns,
                final_candidates=candidates[:5],
            )
        if top.score_label < label_floor and top.score_clip < clip_floor:
            return self._fail_result(
                "GROUNDING_FAILED",
                (f"top candidate weak on both label ({top.score_label:.2f}<{label_floor:.2f}) "
                 f"and CLIP ({top.score_clip:.2f}<{clip_floor:.2f}); refusing to guess"),
                start_ns,
                final_candidates=candidates[:5],
            )
        if margin_min > 0.0 and len(candidates) >= 2:
            margin = top.score - candidates[1].score
            if margin < margin_min:
                return self._fail_result(
                    "GROUNDING_FAILED",
                    (f"top-1/top-2 margin {margin:.3f} below required {margin_min:.3f} "
                     "— cannot confidently disambiguate"),
                    start_ns,
                    final_candidates=candidates[:5],
                )

        # Sample a reachable stand-off pose.
        self._publish_feedback(goal_handle, "SAMPLING_GOAL", candidates, len(graph.nodes))
        target_xyz = np.array(
            [
                top.object_pose.pose.position.x,
                top.object_pose.pose.position.y,
                top.object_pose.pose.position.z,
            ],
            dtype=np.float32,
        )
        costmap = self._latest_costmap if bool(self.get_parameter("use_costmap_gate").value) else None

        sampler_params = SamplerParams(
            stand_off_m=float(req.stand_off_m) if req.stand_off_m > 0.01 else float(self.get_parameter("stand_off_m").value),
            goal_ring_samples=int(self.get_parameter("goal_ring_samples").value),
            goal_min_stand_off_m=float(self.get_parameter("goal_min_stand_off_m").value),
            goal_max_stand_off_m=float(self.get_parameter("goal_max_stand_off_m").value),
            costmap_free_max_cost=int(self.get_parameter("costmap_free_max_cost").value),
            retry_with_wider_ring=bool(self.get_parameter("retry_with_wider_ring").value),
        )
        goal_pose = sample_stand_off(
            target_xyz=target_xyz,
            costmap=costmap,
            params=sampler_params,
            map_frame=str(self.get_parameter("map_frame").value),
        )
        if goal_pose is None:
            return self._fail_result(
                "COSTMAP_UNREACHABLE",
                "no reachable pose in stand-off ring",
                start_ns,
                final_candidates=candidates[:5],
                chosen=top,
            )

        # Publish (unless dry_run)
        if not req.dry_run:
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            self._pub_goal.publish(goal_pose)

        total_s = (time.time_ns() - start_ns) / 1e9

        result = GroundAndNavigate.Result()
        result.success = True
        result.message = "goal published" if not req.dry_run else "dry-run: goal computed, not published"
        result.final_goal = goal_pose
        result.chosen_object_id = top.object_id
        result.chosen_object_label = top.label
        result.grounding_score = float(top.score)
        result.terminal_state = "REACHED" if req.dry_run else "NAVIGATING"
        result.total_latency_s = float(total_s)

        self._publish_feedback(goal_handle, result.terminal_state, candidates, len(graph.nodes))
        goal_handle.succeed()
        return result

    def _fail_result(
        self,
        terminal_state: str,
        message: str,
        start_ns: int,
        final_candidates: Optional[list[GroundingCandidate]] = None,
        chosen: Optional[GroundingCandidate] = None,
    ) -> GroundAndNavigate.Result:
        result = GroundAndNavigate.Result()
        result.success = False
        result.message = message
        result.chosen_object_id = chosen.object_id if chosen else ""
        result.chosen_object_label = chosen.label if chosen else ""
        result.grounding_score = chosen.score if chosen else 0.0
        result.terminal_state = terminal_state
        result.total_latency_s = (time.time_ns() - start_ns) / 1e9
        # final_goal left as default PoseStamped
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GroundingNode()
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
