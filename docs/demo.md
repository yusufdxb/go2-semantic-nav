# Demo procedure

A 10-minute live demo that makes the capability unambiguous. Designed for a recruiter, lab PI, or paper reviewer. Three takes are budgeted: *dry_run*, *commit*, *recovery*.

## Preconditions (check before the recorder is on)

- [ ] GO2 booted, crouched, motors off (Unitree app → motors disabled).
- [ ] Jetson on `192.168.123.15`, pings GO2 at `192.168.123.161`.
- [ ] RealSense streaming: `ros2 topic hz /camera/color/image_raw` shows ≥15 Hz.
- [ ] TF tree healthy: `ros2 run tf2_tools view_frames` shows `map → odom → base_link → camera_color_optical_frame`.
- [ ] Nav2 lifecycle nodes active: `ros2 lifecycle get /bt_navigator` → `active`.
- [ ] Semantic-nav overlay launched, `/semantic/scene_graph` publishing with ≥8 objects (for a small room after a 20-second scan).
- [ ] RViz preset `go2_semantic_nav.rviz` open with layers: map, costmap, object markers, grounding goal marker, camera image.

## Script

### Take 1 — Dry run ("which chair?")
Operator narrates:

> "The robot has been observing the room for 20 seconds. RViz shows the live scene graph — each colored sphere is an object with a language-queryable CLIP embedding. Let's ask it where the red chair is."

Operator runs:
```bash
ros2 action send_goal /semantic/ground_and_navigate \
    go2_semantic_msgs/action/GroundAndNavigate \
    "{text_query: 'go near the red chair', stand_off_m: 0.9, dry_run: true}" \
    --feedback
```

What the viewer sees:
- A magenta cylinder marker appears at the chosen object's location.
- A green arrow marker shows the computed goal pose (map frame).
- The action result prints `chosen_object_label: chair, grounding_score: 0.82`.

Narration:
> "The system found two chairs in the graph. The red one scored 0.82, the other 0.41. It picked the higher-scoring one and computed a reachable stand-off pose from the costmap."

### Take 2 — Commit ("now go there")
Operator runs:
```bash
ros2 action send_goal /semantic/ground_and_navigate \
    go2_semantic_msgs/action/GroundAndNavigate \
    "{text_query: 'go stand next to the red chair', stand_off_m: 0.9, dry_run: false}"
```

What happens:
- `/goal_pose` publishes once.
- Nav2 plans; green path appears in RViz.
- Robot walks via the existing `go2_gait_controller`.
- Action feedback shows state transitions: `GROUNDED → NAVIGATING → REACHED`.

### Take 3 — Recovery ("can you approach a window instead?")
```bash
ros2 action send_goal /semantic/ground_and_navigate \
    go2_semantic_msgs/action/GroundAndNavigate \
    "{text_query: 'move near the window', stand_off_m: 0.7, dry_run: false}"
```

If the scene graph does not contain a window (it typically will, via open-vocab prompts), the action returns `success=false, message='no reachable pose near matching object'`. The recovery narration:

> "The system is honest about failure. It returned no-reachable-pose, didn't make up a goal, and Nav2 was never given a bad command."

## Fallback demos (if robot can't move)

If motors must stay off:
- Use `dry_run: true` for all queries. The value proposition — open-vocab grounding, live scene graph, goal generation — is already complete in RViz.
- Publish a simulated base pose via `static_transform_publisher` to fake odometry.

If scene-graph is empty:
- Walk the camera handheld through the room for 30 seconds first.
- Verify via `ros2 topic echo /semantic/scene_graph --field nodes --once | wc -l`.

## What to emphasize on camera

- **Open vocabulary.** Try a class not in the prompt list mid-demo ("approach the backpack") — the system may or may not find it; use this to talk about the burst-mode OWLv2 fallback.
- **Failure honesty.** Show a query that returns `success=false` rather than hiding it.
- **Edge deployment.** In `tegrastats`, point out sustained GPU <90% at 25 W during steady-state detection.
- **Modularity.** Kill the detector node mid-demo: `ros2 lifecycle set /go2_open_vocab_detector deactivate`. The seeing-eye-dog stack keeps running.

## What NOT to say on camera

- Do not claim "3D Gaussian Splatting" — the representation is object-centric scene graph + CLIP.
- Do not claim "end-to-end learned" — grounding has a parser + CLIP scoring + geometric relation resolver, not a policy network.
- Do not claim latency numbers that were measured on RTX 5070. All reported latency is Jetson 25 W sustained.

## Recording setup

- Screen: RViz at 1080p, terminal split with a command palette and tegrastats stream.
- Audio: narration over; no robot motor whine unless actuating.
- Length target: 90–120 seconds for a resume-linked clip; 5–7 minutes for a PI-facing walkthrough with architecture intro.
