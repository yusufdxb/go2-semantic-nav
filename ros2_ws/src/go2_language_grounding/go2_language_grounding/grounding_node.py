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
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from go2_semantic_msgs.action import GroundAndNavigate
from go2_semantic_msgs.msg import (
    GroundingCandidate,
    SceneGraph,
    SemanticObject,
)
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from .goal_sampler import SamplerParams, sample_stand_off
from .query_parser import ParsedQuery, parse
from .safety_gates import frame_error, freshness_error, publication_error
from .scoring import ScoreWeights, combine, cosine_similarity, lexical_label_match


def _qos_reliable_depth10() -> QoSProfile:
    q = QoSProfile(depth=10)
    q.reliability = ReliabilityPolicy.RELIABLE
    return q


class GroundingNode(Node):
    def __init__(self, *, parameter_overrides=None) -> None:
        super().__init__(
            "go2_language_grounding", parameter_overrides=parameter_overrides or []
        )

        # --- parameters ---
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("scene_graph_topic", "/semantic/scene_graph")
        self.declare_parameter("costmap_topic", "/global_costmap/costmap")
        self.declare_parameter("goal_pose_topic", "/goal_pose")
        self.declare_parameter("grounding_viz_topic", "/semantic/grounding_viz")
        self.declare_parameter("navigation_backend", "nav2_action")
        self.declare_parameter("navigate_to_pose_action", "/navigate_to_pose")
        self.declare_parameter("navigation_server_timeout_s", 3.0)
        self.declare_parameter("navigation_cancel_timeout_s", 2.0)

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
        self.declare_parameter("allow_goal_publication", False)
        self.declare_parameter("min_scene_graph_objects", 1)
        self.declare_parameter("max_action_duration_s", 60.0)
        self.declare_parameter("max_scene_graph_age_s", 3.0)
        self.declare_parameter("max_costmap_age_s", 3.0)

        # Rejection thresholds — see RESULTS.md §"Honest negative results" for the v1→v2 rationale.
        self.declare_parameter("reject_absolute_floor", 0.15)
        self.declare_parameter("reject_label_floor", 0.40)
        self.declare_parameter("reject_clip_floor", 0.30)
        self.declare_parameter("reject_margin_min", 0.0,
                               ParameterDescriptor(description=(
                                   "If >0, require top-1.score - top-2.score >= this margin. "
                                   "Stricter than the label/clip floors but cuts recall when "
                                   "multiple legitimate candidates cluster together."
                               )))

        # --- state ---
        self._latest_scene_graph: Optional[SceneGraph] = None
        self._latest_costmap: Optional[OccupancyGrid] = None
        self._scene_graph_received_ns = 0
        self._costmap_received_ns = 0
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
        self._nav_client = ActionClient(
            self,
            NavigateToPose,
            str(self.get_parameter("navigate_to_pose_action").value),
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
        self._scene_graph_received_ns = self.get_clock().now().nanoseconds

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        self._latest_costmap = msg
        self._costmap_received_ns = self.get_clock().now().nanoseconds

    def _runtime_gate_error(self) -> Optional[str]:
        """Validate cached inputs using receipt time and frame contracts."""
        now_ns = self.get_clock().now().nanoseconds
        graph_error = freshness_error(
            now_ns=now_ns,
            received_ns=self._scene_graph_received_ns,
            max_age_s=float(self.get_parameter("max_scene_graph_age_s").value),
            label="scene graph",
        )
        if graph_error:
            return graph_error
        if self._latest_scene_graph is None:
            return "scene graph not received"
        minimum_objects = max(1, int(self.get_parameter("min_scene_graph_objects").value))
        object_count = len(self._latest_scene_graph.nodes)
        if object_count < minimum_objects:
            return f"scene graph has {object_count} objects; requires {minimum_objects}"

        map_frame = str(self.get_parameter("map_frame").value)
        graph_frame_error = frame_error(
            actual=self._latest_scene_graph.header.frame_id,
            expected=map_frame,
            label="scene graph",
        )
        if graph_frame_error:
            return graph_frame_error

        if not bool(self.get_parameter("use_costmap_gate").value):
            return None
        costmap_error = freshness_error(
            now_ns=now_ns,
            received_ns=self._costmap_received_ns,
            max_age_s=float(self.get_parameter("max_costmap_age_s").value),
            label="costmap",
        )
        if costmap_error:
            return costmap_error
        if self._latest_costmap is None:
            return "costmap not received"
        return frame_error(
            actual=self._latest_costmap.header.frame_id,
            expected=map_frame,
            label="costmap",
        )

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
        publication_gate_error = publication_error(
            dry_run=bool(goal_request.dry_run),
            allow_goal_publication=bool(
                self.get_parameter("allow_goal_publication").value
            ),
        )
        if publication_gate_error:
            self.get_logger().warn(f"goal rejected: {publication_gate_error}")
            return GoalResponse.REJECT
        runtime_error = self._runtime_gate_error()
        if runtime_error:
            self.get_logger().warn(f"goal rejected: {runtime_error}")
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
        start_ns = time.monotonic_ns()
        req: GroundAndNavigate.Goal = goal_handle.request
        if goal_handle.is_cancel_requested:
            return self._cancel_result(goal_handle, start_ns)
        publication_gate_error = publication_error(
            dry_run=bool(req.dry_run),
            allow_goal_publication=bool(
                self.get_parameter("allow_goal_publication").value
            ),
        )
        if publication_gate_error:
            return self._abort_result(
                goal_handle, "PUBLICATION_DISABLED", publication_gate_error, start_ns
            )
        runtime_error = self._runtime_gate_error()
        if runtime_error:
            return self._abort_result(
                goal_handle, "INPUTS_UNAVAILABLE", runtime_error, start_ns
            )
        graph = self._latest_scene_graph
        if graph is None or not graph.nodes:
            return self._abort_result(
                goal_handle, "GROUNDING_FAILED", "no scene graph available", start_ns
            )

        parsed = parse(req.text_query)
        if req.preferred_relation:
            parsed.relation = req.preferred_relation
        weights = self._current_weights()

        self._publish_feedback(goal_handle, "PARSING", [], len(graph.nodes))

        candidates = self._score_candidates(parsed, graph, weights)
        if not candidates:
            return self._abort_result(
                goal_handle, "GROUNDING_FAILED", "no candidates in scene graph", start_ns
            )

        candidates = self._apply_relation_filter(candidates, parsed, graph)
        self._publish_feedback(goal_handle, "SCORING", candidates, len(graph.nodes))
        if goal_handle.is_cancel_requested:
            return self._cancel_result(goal_handle, start_ns)
        if self._action_timed_out(start_ns, req.timeout_seconds):
            return self._abort_result(
                goal_handle, "TIMED_OUT", "grounding action exceeded its time limit", start_ns
            )

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
            return self._abort_result(
                goal_handle,
                "GROUNDING_FAILED",
                f"top candidate score {top.score:.3f} below absolute floor {absolute_floor:.2f}",
                start_ns,
                final_candidates=candidates[:5],
            )
        if top.score_label < label_floor and top.score_clip < clip_floor:
            return self._abort_result(
                goal_handle,
                "GROUNDING_FAILED",
                (f"top candidate weak on both label ({top.score_label:.2f}<{label_floor:.2f}) "
                 f"and CLIP ({top.score_clip:.2f}<{clip_floor:.2f}); refusing to guess"),
                start_ns,
                final_candidates=candidates[:5],
            )
        if margin_min > 0.0 and len(candidates) >= 2:
            margin = top.score - candidates[1].score
            if margin < margin_min:
                return self._abort_result(
                    goal_handle,
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
            return self._abort_result(
                goal_handle,
                "COSTMAP_UNREACHABLE",
                "no reachable pose in stand-off ring",
                start_ns,
                final_candidates=candidates[:5],
                chosen=top,
            )

        if goal_handle.is_cancel_requested:
            return self._cancel_result(goal_handle, start_ns)
        if self._action_timed_out(start_ns, req.timeout_seconds):
            return self._abort_result(
                goal_handle, "TIMED_OUT", "grounding action exceeded its time limit", start_ns
            )

        if not req.dry_run:
            return self._dispatch_navigation(
                goal_handle,
                goal_pose,
                candidates,
                len(graph.nodes),
                top,
                start_ns,
                float(req.timeout_seconds),
            )

        total_s = (time.monotonic_ns() - start_ns) / 1e9

        result = GroundAndNavigate.Result()
        result.success = True
        result.message = "dry-run: goal computed, not dispatched"
        result.final_goal = goal_pose
        result.chosen_object_id = top.object_id
        result.chosen_object_label = top.label
        result.grounding_score = float(top.score)
        result.terminal_state = "DRY_RUN_COMPLETE"
        result.total_latency_s = float(total_s)

        self._publish_feedback(goal_handle, result.terminal_state, candidates, len(graph.nodes))
        goal_handle.succeed()
        return result

    def _dispatch_navigation(
        self,
        goal_handle,
        goal_pose: PoseStamped,
        candidates: list[GroundingCandidate],
        graph_count: int,
        chosen: GroundingCandidate,
        start_ns: int,
        request_timeout_s: float,
    ) -> GroundAndNavigate.Result:
        """Dispatch a goal through the configured motion backend."""
        backend = str(self.get_parameter("navigation_backend").value)
        goal_pose.header.stamp = self.get_clock().now().to_msg()

        if backend == "goal_pose_topic":
            self._pub_goal.publish(goal_pose)
            result = self._success_result(
                goal_pose,
                chosen,
                "goal dispatched to legacy topic; completion is not tracked",
                "DISPATCHED",
                start_ns,
            )
            self._publish_feedback(goal_handle, "DISPATCHED", candidates, graph_count)
            goal_handle.succeed()
            return result

        if backend != "nav2_action":
            return self._abort_result(
                goal_handle,
                "NAV_UNAVAILABLE",
                f"unknown navigation_backend {backend!r}",
                start_ns,
                final_candidates=candidates[:5],
                chosen=chosen,
            )

        server_timeout = float(
            self.get_parameter("navigation_server_timeout_s").value
        )
        if not self._nav_client.wait_for_server(timeout_sec=server_timeout):
            return self._abort_result(
                goal_handle,
                "NAV_UNAVAILABLE",
                "NavigateToPose action server unavailable",
                start_ns,
                final_candidates=candidates[:5],
                chosen=chosen,
            )

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = goal_pose

        def on_nav_feedback(feedback_msg) -> None:
            distance = float(feedback_msg.feedback.distance_remaining)
            self._publish_feedback(
                goal_handle, "NAVIGATING", candidates, graph_count, distance
            )

        send_future = self._nav_client.send_goal_async(
            nav_goal, feedback_callback=on_nav_feedback
        )
        while rclpy.ok() and not send_future.done():
            if goal_handle.is_cancel_requested:
                return self._cancel_result(goal_handle, start_ns)
            if self._action_timed_out(start_ns, request_timeout_s):
                return self._abort_result(
                    goal_handle, "TIMED_OUT", "navigation dispatch timed out", start_ns
                )
            time.sleep(0.02)

        nav_handle = send_future.result() if send_future.done() else None
        if nav_handle is None or not nav_handle.accepted:
            return self._abort_result(
                goal_handle,
                "NAV_REJECTED",
                "NavigateToPose rejected the goal",
                start_ns,
                final_candidates=candidates[:5],
                chosen=chosen,
            )

        result_future = nav_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            if goal_handle.is_cancel_requested:
                cancel_note = self._cancel_navigation(nav_handle)
                return self._cancel_result(goal_handle, start_ns, cancel_note)
            runtime_error = self._runtime_gate_error()
            if runtime_error:
                cancel_note = self._cancel_navigation(nav_handle)
                return self._abort_result(
                    goal_handle,
                    "INPUTS_UNAVAILABLE",
                    f"navigation canceled: {runtime_error}; {cancel_note}",
                    start_ns,
                    final_candidates=candidates[:5],
                    chosen=chosen,
                )
            if self._action_timed_out(start_ns, request_timeout_s):
                cancel_note = self._cancel_navigation(nav_handle)
                return self._abort_result(
                    goal_handle,
                    "TIMED_OUT",
                    f"navigation exceeded its time limit; {cancel_note}",
                    start_ns,
                    final_candidates=candidates[:5],
                    chosen=chosen,
                )
            time.sleep(0.02)

        wrapped_result = result_future.result() if result_future.done() else None
        status = wrapped_result.status if wrapped_result is not None else GoalStatus.STATUS_UNKNOWN
        if status == GoalStatus.STATUS_SUCCEEDED:
            result = self._success_result(
                goal_pose, chosen, "navigation reached goal", "REACHED", start_ns
            )
            self._publish_feedback(goal_handle, "REACHED", candidates, graph_count, 0.0)
            goal_handle.succeed()
            return result
        if status == GoalStatus.STATUS_CANCELED:
            return self._cancel_result(goal_handle, start_ns)
        return self._abort_result(
            goal_handle,
            "NAV_ABORTED",
            f"NavigateToPose finished with status {status}",
            start_ns,
            final_candidates=candidates[:5],
            chosen=chosen,
        )

    def _success_result(
        self,
        goal_pose: PoseStamped,
        chosen: GroundingCandidate,
        message: str,
        terminal_state: str,
        start_ns: int,
    ) -> GroundAndNavigate.Result:
        result = GroundAndNavigate.Result()
        result.success = True
        result.message = message
        result.final_goal = goal_pose
        result.chosen_object_id = chosen.object_id
        result.chosen_object_label = chosen.label
        result.grounding_score = float(chosen.score)
        result.terminal_state = terminal_state
        result.total_latency_s = (time.monotonic_ns() - start_ns) / 1e9
        return result

    def _action_timed_out(self, start_ns: int, request_timeout_s: float = 0.0) -> bool:
        configured_limit = float(self.get_parameter("max_action_duration_s").value)
        limit_s = request_timeout_s if request_timeout_s > 0.0 else configured_limit
        if configured_limit > 0.0 and limit_s > 0.0:
            limit_s = min(limit_s, configured_limit)
        return limit_s > 0.0 and (time.monotonic_ns() - start_ns) / 1e9 > limit_s

    def _abort_result(
        self,
        goal_handle,
        terminal_state: str,
        message: str,
        start_ns: int,
        final_candidates: Optional[list[GroundingCandidate]] = None,
        chosen: Optional[GroundingCandidate] = None,
    ) -> GroundAndNavigate.Result:
        goal_handle.abort()
        return self._fail_result(
            terminal_state,
            message,
            start_ns,
            final_candidates=final_candidates,
            chosen=chosen,
        )

    def _cancel_navigation(self, nav_handle) -> str:
        """Cancel the Nav2 goal and wait, bounded, for the server to acknowledge.

        Reporting CANCELED before Nav2 confirms would let a caller believe the
        robot has stopped while the controller may still be driving.
        """
        timeout_s = float(self.get_parameter("navigation_cancel_timeout_s").value)
        cancel_future = nav_handle.cancel_goal_async()
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and not cancel_future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        response = cancel_future.result() if cancel_future.done() else None
        if response is not None and response.goals_canceling:
            return "Nav2 acknowledged cancel"
        note = (
            "Nav2 did NOT acknowledge cancel within "
            f"{timeout_s:.1f}s; motion may continue, use the base stop"
        )
        self.get_logger().error(note)
        return note

    def _cancel_result(
        self, goal_handle, start_ns: int, note: str = ""
    ) -> GroundAndNavigate.Result:
        goal_handle.canceled()
        message = f"goal canceled; {note}" if note else "goal canceled"
        return self._fail_result("CANCELED", message, start_ns)

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
        result.total_latency_s = (time.monotonic_ns() - start_ns) / 1e9
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
