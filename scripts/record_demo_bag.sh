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

source /opt/ros/humble/setup.bash
if [[ -f "$HOME/ros2_ws/install/setup.bash" ]]; then
  source "$HOME/ros2_ws/install/setup.bash"
fi

mkdir -p "$WS_ROOT/data"
echo "[record_demo_bag] writing to $OUT"

TOPICS=(
  /camera/color/image_raw
  /camera/depth/image_rect_raw
  /camera/color/camera_info
  /camera/depth/camera_info
  /tf
  /tf_static
  /odom
  /scan
)

echo "[record_demo_bag] recording ${#TOPICS[@]} topics. Ctrl-C to stop."
# Walk the camera slowly through the scene for 30+ seconds to populate the graph.
# Keep QoS overrides so BEST_EFFORT image topics don't get dropped.
cat > /tmp/qos_override.yaml <<EOF
/camera/color/image_raw:
  reliability: best_effort
  history: keep_last
  depth: 1
/camera/depth/image_rect_raw:
  reliability: best_effort
  history: keep_last
  depth: 1
EOF

ros2 bag record -o "$OUT" \
  --qos-profile-overrides-path /tmp/qos_override.yaml \
  --compression-mode file --compression-format zstd \
  "${TOPICS[@]}"
