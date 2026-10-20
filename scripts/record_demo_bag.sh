#!/usr/bin/env bash
# Record a rosbag for offline eval + reproducible demos.
#
# Assumes you're on the laptop or Jetson with the sibling seeing-eye-dog stack
# running (RealSense driver + TF tree up). Captures exactly the topics
# go2-semantic-nav needs to replay a scene later.
#
# Usage:
#   scripts/record_demo_bag.sh                # writes data/rec_$(date)/
#   scripts/record_demo_bag.sh my_office      # writes data/my_office/

set -euo pipefail

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:-rec_$(date +%Y%m%d_%H%M%S)}"
OUT="$WS_ROOT/data/$NAME"
COLOR_TOPIC="${COLOR_TOPIC:-/camera/color/image_raw}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/camera/depth/image_rect_raw}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-/camera/color/camera_info}"

source /opt/ros/humble/setup.bash
if [[ -f "$HOME/ros2_ws/install/setup.bash" ]]; then
  source "$HOME/ros2_ws/install/setup.bash"
fi

mkdir -p "$WS_ROOT/data"
echo "[record_demo_bag] writing to $OUT"

TOPICS=(
  "$COLOR_TOPIC"
  "$DEPTH_TOPIC"
  "$CAMERA_INFO_TOPIC"
  /tf
  /tf_static
  /odom
  /scan
  /global_costmap/costmap
  /semantic/detections
  /semantic/scene_graph
  /semantic/object_markers
  /semantic/grounding_viz
  /goal_pose
  /parameter_events
)

echo "[record_demo_bag] recording ${#TOPICS[@]} topics. Ctrl-C to stop."
# Walk the camera slowly through the scene for 30+ seconds to populate the graph.
# Keep QoS overrides so BEST_EFFORT image topics don't get dropped.
QOS_FILE="$(mktemp /tmp/go2-semantic-qos.XXXXXX.yaml)"
trap 'rm -f "$QOS_FILE"' EXIT
cat > "$QOS_FILE" <<EOF
$COLOR_TOPIC:
  reliability: best_effort
  history: keep_last
  depth: 1
$DEPTH_TOPIC:
  reliability: best_effort
  history: keep_last
  depth: 1
EOF

ros2 bag record -o "$OUT" \
  --qos-profile-overrides-path "$QOS_FILE" \
  --compression-mode file --compression-format zstd \
  "${TOPICS[@]}"
