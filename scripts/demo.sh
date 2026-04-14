#!/usr/bin/env bash
# Quick local demo: brings up the semantic-nav overlay and fires a dry-run grounding query.
# Assumes RealSense or a rosbag is already publishing /camera/... and TF map→camera_color_optical_frame exists.

set -euo pipefail

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS_ROOT"

source /opt/ros/humble/setup.bash
if [[ -f "$HOME/ros2_ws/install/setup.bash" ]]; then
  source "$HOME/ros2_ws/install/setup.bash"
fi
source "$WS_ROOT/ros2_ws/install/setup.bash"

QUERY="${1:-go near the chair}"
echo "[demo.sh] launching semantic-nav bringup…"
ros2 launch go2_semantic_bringup semantic_nav.launch.py use_rviz:=true &
LAUNCH_PID=$!
trap 'kill $LAUNCH_PID 2>/dev/null || true' EXIT

echo "[demo.sh] waiting 15 s for nodes to warm up…"
sleep 15

echo "[demo.sh] sending grounding action: $QUERY (dry-run)"
ros2 action send_goal /semantic/ground_and_navigate \
  go2_semantic_msgs/action/GroundAndNavigate \
  "{text_query: '${QUERY}', stand_off_m: 0.9, dry_run: true}" \
  --feedback

echo "[demo.sh] done. Ctrl-C or kill $LAUNCH_PID to shut down bringup."
wait $LAUNCH_PID
