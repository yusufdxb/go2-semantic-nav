# Troubleshooting

Diagnostic commands first, fixes second.

## Diagnostics

```bash
# Is the overlay built?
ros2 pkg list | grep go2_ | sort

# Are our nodes running?
ros2 node list | grep /go2_

# Lifecycle state of each
for n in go2_open_vocab_detector go2_scene_graph go2_language_grounding; do
    echo "== $n =="
    ros2 lifecycle get /$n
done

# Topic health
ros2 topic hz /camera/color/image_raw
ros2 topic hz /camera/depth/image_rect_raw
ros2 topic hz /semantic/detections
ros2 topic hz /semantic/scene_graph

# TF chain
ros2 run tf2_ros tf2_echo map base_link
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame

# QoS mismatch hunt
ros2 topic info --verbose /camera/color/image_raw
```

## Common failures

### Detector subscribes but publishes nothing
- **Check 1:** camera topic hz > 0. If not, the RealSense driver isn't running.
- **Check 2:** `ros2 lifecycle get /go2_open_vocab_detector` is `active`, not `inactive` or `unconfigured`.
- **Check 3:** logs show `YOLO-World weights loaded`: if stuck, the weights download is probably blocked (no internet on Jetson? share via laptop).
- **Fix:** `ros2 lifecycle set /go2_open_vocab_detector configure` then `activate`.

### Detector CPU-bound, no GPU utilization
- **Check:** `nvidia-smi` (dev) or `tegrastats` (Jetson) shows GPU 0%.
- **Cause:** `device` param defaulted to `cpu`, or the weight file is a CPU variant.
- **Fix:** `ros2 param set /go2_open_vocab_detector device cuda:0`, then deactivate+activate.

### Scene graph has no objects in `map` frame
- **Symptom:** `ros2 topic echo /semantic/scene_graph` shows empty `nodes`.
- **Cause 1:** TF `camera_color_optical_frame → map` lookup failing (most common). Check `ros2 run tf2_ros tf2_echo map camera_color_optical_frame`: error means `/odom` is broken.
- **Cause 2:** detections topic not reaching the scene-graph node (QoS mismatch or remapping).
- **Cause 3:** `min_observations_to_publish` too high, objects observed fewer than N frames are filtered.
- **Fix for Cause 1:** launch with `use_slam_toolbox:=true`, or debug odometry in the sibling seeing-eye-dog stack (known: `/odom` stuck at (0,0,0) is a tracked blocker).

### Grounding returns wrong object
- **Check 1:** query `ros2 service call /semantic/query_objects go2_semantic_msgs/srv/QueryObjects "{text_query: 'red chair', top_k: 5}"`. Inspect top-5 with scores.
- **Check 2:** are the similar objects actually in the graph with distinguishable embeddings? `ros2 topic echo /semantic/scene_graph --field nodes`.
- **Fix 1:** raise attribute weight in `config/grounding.yaml` (`score_weights.clip` → 0.8).
- **Fix 2:** use the explicit attribute form `"the {color} {noun}"` consistently in the query.

### Goal pose is inside an obstacle
- **Cause:** costmap subscription not up yet, so the reachability filter is a no-op.
- **Check:** `ros2 topic hz /global_costmap/costmap`.
- **Fix:** wait for Nav2 lifecycle to be `active`, or add a startup synchronization in the launch file.

### Action returns success=false for a reasonable query
- **Cause 1:** `stand_off_m` too small, no free pose in the ring.
- **Cause 2:** object is close to a wall, all sampled stand-off poses are in-obstacle.
- **Fix:** increase `stand_off_m` or `goal_ring_samples`; or report honestly (this is a legitimate failure mode on tight scenes).

### On Jetson, sustained FPS degrades after a few minutes
- **Cause:** thermal throttling (expected on Orin NX 25 W with aggressive workload).
- **Check:** `tegrastats` shows GPU temp >85 °C with throttle flags.
- **Fix:** reduce `detection_rate_hz`, swap to smaller backbone (`yolo_world_v2_s` → `yoloe_11s`), or ensure `jetson_clocks` and `nvpmodel -m 0` are set.

### cv_bridge fails on import / `numpy 2.x` error
- **Cause:** `pip install` bumped numpy past 2.0 which is incompatible with cv_bridge Humble build.
- **Fix:** `pip install "numpy<2.0"` and rebuild affected C extensions.

### `ModuleNotFoundError: open_clip`
- **Cause:** Python venv not activated, or install into system Python.
- **Fix:** `source .venv/bin/activate && pip install open_clip_torch` from project root.

### `SIGSEGV` on MobileSAM import on Jetson
- **Cause:** the `mobile_sam` pip package bundles an x86 `.so`; it is not aarch64-compatible.
- **Fix:** On Jetson, use NanoSAM (`segmenter: nano_sam`). The `mobile_sam` package should be dev-only.

### `ros2 launch` fails: "Could not find go2_semantic_msgs"
- **Cause:** overlay workspace not sourced, or built out of order.
- **Fix:** `colcon build --packages-up-to go2_semantic_msgs` from `ros2_ws/`, then re-source `install/setup.bash`.

### Detector and scene-graph nodes are active but RViz shows no markers
- **Check:** Is the MarkerArray topic selected in RViz and the frame set to `map`?
- **Fix:** load the provided preset: `rviz2 -d $(ros2 pkg prefix go2_semantic_bringup)/share/go2_semantic_bringup/rviz/semantic_nav.rviz`.

### First detector run hangs on "Ultralytics requirement ['git+https://...CLIP.git'] not found, attempting AutoUpdate..."
- **Cause:** Ultralytics YOLO-World has a runtime dep on OpenAI's `clip`, declared via `uv pip install` inside the inference path, not via `requirements.txt`. First run hangs for minutes while `uv` fetches it, and in air-gapped environments it hangs forever.
- **Fix:** pre-install before the first inference:
  ```bash
  pip install git+https://github.com/openai/CLIP.git
  # Then immediately repin torch if the above pulled a non-cu128 wheel:
  pip install --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128" "torchvision==0.26.0+cu128"
  pip install --force-reinstall --no-deps "numpy<2.0"
  ```

### Grounding returns `success=True` with an obviously wrong object
- **Cause:** Raw CLIP cosine similarity between unrelated text and image crops sits in [0.15, 0.25]. A single `total_score >= 0.15` gate accepts any detection as "good enough" for any query.
- **Fix (already in code):** the two-layer rejection in `grounding_node._execute_action` combines an absolute score floor with a `(label_floor OR clip_floor)` secondary gate. Tune `label_floor` (default 0.40) and `clip_floor` (default 0.22) via params if you see the opposite problem, legitimate matches being refused.

### `torch` silently downgraded after `pip install --force-reinstall <anything>`
- **Cause:** pip's index resolution picks the default PyPI wheel (CUDA 13 suffix) over the cu128 one unless `--extra-index-url https://download.pytorch.org/whl/cu128` is passed on every reinstall.
- **Symptom:** `RuntimeError: CUDA initialization: The NVIDIA driver on your system is too old (found version 12080).` on any `torch.cuda` call.
- **Fix:** explicitly reinstall with the correct index:
  ```bash
  pip install --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu128 \
      "torch==2.11.0+cu128" "torchvision==0.26.0+cu128"
  ```

### `colcon build` fails with "option --editable not recognized"
- **Cause:** `setuptools>=80` dropped the `--editable` invocation path that ament_python's `colcon build --symlink-install` relies on.
- **Fix:** `pip install "setuptools<70"`.

## When to escalate

If you see symptoms outside this list, take three actions before pinging anyone:

1. Capture a `ros2 bag record -a` snapshot of a 30-second window around the failure.
2. Run `ros2 doctor --report > /tmp/doctor.txt`.
3. Attach the logs from `~/.ros/log/latest/` for the failing node.

Then open an issue with hardware spec, ROS distro, branch SHA, and the three artifacts above.
