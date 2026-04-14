#!/usr/bin/env bash
# One-shot diagnostic bundler. Captures everything a maintainer needs to triage
# a live or replayed semantic-nav session:
#   - ROS 2 env vars
#   - nodes, topics, services, actions
#   - TF tree
#   - per-topic rates for the required set
#   - lifecycle state of each semantic-nav node
#   - last 200 log lines per ~/.ros/log/latest/*
#
# Usage:
#   scripts/diagnose.sh [output_dir]
# Defaults output_dir = /tmp/go2-semantic-nav-diag-$(date).

set -euo pipefail

OUT="${1:-/tmp/go2-semantic-nav-diag-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
echo "[diagnose] writing bundle to $OUT"

# Ensure ROS is sourced — idempotent.
if [[ -z "${ROS_DISTRO:-}" ]]; then
  source /opt/ros/humble/setup.bash
fi

# --- Environment ---
{
  echo "=== timestamp ==="
  date -Iseconds
  echo "=== ros env ==="
  env | grep -E "^ROS_|^RMW_|^DOMAIN|^FASTRTPS_|^CYCLONEDDS_" | sort
  echo "=== user env ==="
  env | grep -E "^CUDA|^PATH|^LD_LIBRARY|^PYTHONPATH" | sort
} > "$OUT/00_env.txt" 2>&1

# --- nvidia-smi / tegrastats if available ---
(nvidia-smi > "$OUT/01_gpu.txt" 2>&1) || echo "no nvidia-smi" > "$OUT/01_gpu.txt"
(command -v tegrastats >/dev/null && timeout 2 tegrastats --interval 1000 > "$OUT/02_tegrastats.txt" 2>&1) || true

# --- ROS graph ---
timeout 5 ros2 node list > "$OUT/10_nodes.txt" 2>&1 || true
timeout 5 ros2 topic list -t > "$OUT/11_topics.txt" 2>&1 || true
timeout 5 ros2 service list > "$OUT/12_services.txt" 2>&1 || true
timeout 5 ros2 action list > "$OUT/13_actions.txt" 2>&1 || true

# --- Required topic health ---
{
  for topic in \
      /camera/color/image_raw \
      /camera/depth/image_rect_raw \
      /camera/color/camera_info \
      /semantic/detections \
      /semantic/scene_graph \
      /goal_pose; do
    echo "=== hz $topic (5s) ==="
    timeout 5 ros2 topic hz "$topic" 2>&1 | head -8 || true
    echo
  done
} > "$OUT/20_topic_rates.txt"

# --- TF tree (takes 5 s to listen) ---
timeout 8 ros2 run tf2_tools view_frames -o "$OUT/30_tf" > /dev/null 2>&1 || true
# The tool writes frames.pdf + frames.gv to cwd in older Humble; move if present.
for f in frames.pdf frames.gv; do
  [[ -f "$f" ]] && mv "$f" "$OUT/30_$f"
done

# --- Verbose QoS dump on semantic topics ---
for topic in /semantic/detections /semantic/scene_graph /goal_pose; do
  echo "=== topic info -v $topic ===" >> "$OUT/40_qos.txt"
  timeout 3 ros2 topic info --verbose "$topic" >> "$OUT/40_qos.txt" 2>&1 || true
  echo >> "$OUT/40_qos.txt"
done

# --- Lifecycle probe (works on nodes that implement the lifecycle interface) ---
for node in go2_open_vocab_detector go2_scene_graph go2_language_grounding; do
  echo "=== lifecycle $node ===" >> "$OUT/41_lifecycle.txt"
  timeout 3 ros2 lifecycle get "/$node" >> "$OUT/41_lifecycle.txt" 2>&1 || \
    echo "(not a lifecycle node / not present)" >> "$OUT/41_lifecycle.txt"
  echo >> "$OUT/41_lifecycle.txt"
done

# --- Logs from ~/.ros/log/latest/ ---
if [[ -d "$HOME/.ros/log/latest" ]]; then
  mkdir -p "$OUT/50_logs"
  for f in "$HOME/.ros/log/latest"/*.log; do
    [[ -f "$f" ]] || continue
    tail -200 "$f" > "$OUT/50_logs/$(basename "$f")"
  done
fi

# --- Summary ---
{
  echo "=== diagnostic bundle summary ==="
  echo "Path: $OUT"
  echo "Files:"
  ls -la "$OUT" | tail -n +2
} > "$OUT/99_summary.txt"

tar czf "$OUT.tgz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "[diagnose] bundled at $OUT.tgz"
echo
echo "Attach $OUT.tgz when filing an issue. Contains no secrets unless your"
echo "ROS_DOMAIN_ID or log lines do; inspect 00_env.txt + 50_logs/ before sharing."
