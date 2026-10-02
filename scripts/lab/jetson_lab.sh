#!/usr/bin/env bash
# Robot-computer side of the RGB + LiDAR lab run. Called by run_lab.sh over
# ssh; can also be run by hand on the robot computer.
#
#   jetson_lab.sh prepare            sysctl, dependency checks, colcon build (idempotent)
#   jetson_lab.sh up [avdec|nvv4l2]  start relay + C++ camera + lidar_depth in the background
#   jetson_lab.sh probe [seconds]    health/latency table (lab_probe.py)
#   jetson_lab.sh capture <segment> <seconds> [recorder args...]
#   jetson_lab.sh down               stop what `up` started (by process group)
#   jetson_lab.sh status
#
# Never commands robot motion: it starts sensor relays, the camera driver,
# the depth projector and recorders only.
set -uo pipefail

LAB=${GSN_LAB:-$HOME/gsn_lab}
WS=${GSN_WS:-$HOME/gsn_ws}
RUN=${GSN_RUN:-$LAB/runs/current}
PW=${JETSON_PW:-123}
IFACE=${GO2_IFACE:-enP8p1s0}
RMEM=33554432
# Rehearsal on a desktop (GSN_REHEARSAL=1): no sudo, loopback camera address,
# and a best-effort DDS buffer because rmem_max cannot be raised there.
REHEARSAL=${GSN_REHEARSAL:-0}
CAM_ADDRESS=${GSN_CAM_ADDRESS:-230.1.1.1}
CAM_PORT=${GSN_CAM_PORT:-1720}
mkdir -p "$RUN/logs" "$RUN/pids"

env_setup() {
  set +u
  source /opt/ros/humble/setup.bash
  [ -f "$HOME/unitree_ros2/setup.sh" ] && source "$HOME/unitree_ros2/setup.sh" >/dev/null
  [ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"
  set -u
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  # Large DDS receive buffers: a raw 720p frame is ~2.7 MB of UDP fragments
  # and the default buffer drops whole frames. min= makes CycloneDDS refuse to
  # start if the kernel limit was not raised, instead of silently losing frames.
  local buf='<CycloneDDS><Domain><Internal><SocketReceiveBufferSize min="16MB"/></Internal></Domain></CycloneDDS>'
  [ "$REHEARSAL" = 1 ] && buf='<CycloneDDS><Domain><Internal><SocketReceiveBufferSize min="default" max="16MB"/></Internal></Domain></CycloneDDS>'
  export CYCLONEDDS_URI="${CYCLONEDDS_URI:+$CYCLONEDDS_URI,}$buf"
}

ours() {  # PIDs of processes started from our workspace (matched on the program, not the command text)
  for d in /proc/[0-9]*; do
    local exe
    exe=$( { tr '\0' ' ' < "$d/cmdline"; } 2>/dev/null | awk '{print $1" "$2}')  # processes can exit mid-scan
    case "$exe" in *"$WS/install/"*|*"$LAB/bin/capture_recorder.py"*) echo "${d#/proc/}";; esac
  done
}

prepare() {
  local fail=0
  echo "== clock: $(date -u '+%Y-%m-%d %H:%M:%S') UTC"
  [ "$(date +%Y)" -ge 2026 ] || { echo "FAIL clock not set (boots at 1970): run_lab.sh sets it"; fail=1; }
  if [ "$REHEARSAL" = 1 ]; then
    echo "REHEARSAL: skipping sysctl and robot NIC checks"
  else
    echo "$PW" | sudo -S -p '' sysctl -q -w net.core.rmem_max=$RMEM && echo "ok   net.core.rmem_max=$(sysctl -n net.core.rmem_max)"
    [ "$(sysctl -n net.core.rmem_max)" -ge $RMEM ] || { echo "FAIL rmem_max not raised"; fail=1; }
    ip -br addr show "$IFACE" >/dev/null 2>&1 && echo "ok   robot NIC $IFACE" || { echo "FAIL no NIC $IFACE"; fail=1; }
  fi
  for e in udpsrc rtph264depay h264parse avdec_h264 videoconvert videoscale appsink nvv4l2decoder nvvidconv; do
    gst-inspect-1.0 --exists "$e" && echo "ok   gst $e" || { echo "MISS gst $e"; case $e in nv*) ;; *) fail=1;; esac; }
  done
  for p in libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev; do
    dpkg -s "$p" >/dev/null 2>&1 && echo "ok   $p" || { echo "FAIL $p missing"; fail=1; }
  done
  set +u; source /opt/ros/humble/setup.bash; set -u
  python3 -c "import numpy, cv2, yaml, sensor_msgs_py; print('ok   python numpy', numpy.__version__, 'cv2', cv2.__version__)" \
    || { echo "FAIL python deps"; fail=1; }
  ( cd "$WS" && colcon build --packages-select go2_front_camera_cpp go2_rgb_lidar go2_localization \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF > "$RUN/logs/colcon.txt" 2>&1 ) \
    && echo "ok   colcon build" || { echo "FAIL colcon build (see $RUN/logs/colcon.txt)"; tail -20 "$RUN/logs/colcon.txt"; fail=1; }
  env_setup
  echo "== CYCLONEDDS_URI: $CYCLONEDDS_URI"
  # Match the interface as a whole token (a bare substring test matched "lo" in "CycloneDDS").
  if printf '%s' "$CYCLONEDDS_URI" | grep -Eq "(\"|>|=)$IFACE(\"|<|,|$)"; then echo "ok   DDS bound to $IFACE"
  else echo "WARN DDS config does not name $IFACE (topics may list but carry no data)"; fi
  python3 -c "import rclpy; rclpy.init(); rclpy.create_node('gsn_dds_check'); print('ok   DDS node with 16MB receive buffer')" 2>&1 | tail -1 \
    | grep -q '^ok' && echo "ok   DDS buffer config accepted" || { echo "FAIL DDS refused the 16MB receive buffer (rmem_max?)"; fail=1; }
  echo "== other ROS nodes already running (left alone):"
  timeout 8 ros2 node list 2>/dev/null | head -20
  return $fail
}

start() {  # name, command...
  # Job control gives each background job its own process group whose id is
  # $!, so `down` can signal the whole group (ros2 run + the node). setsid
  # would fork when already a group leader, and $! would be the wrong PID.
  local name=$1; shift
  set -m
  nohup "$@" > "$RUN/logs/$name.log" 2>&1 < /dev/null &
  echo $! > "$RUN/pids/$name"
  set +m
}

camera_args() {  # decoder, latency: parameter overrides for the C++ camera node
  local a=(--params-file "$WS/install/go2_front_camera_cpp/share/go2_front_camera_cpp/config/front_camera.yaml"
           -p "decoder:=$1" -p "latency_s:=$2" -p "address:=$CAM_ADDRESS" -p "port:=$CAM_PORT")
  # An empty `-p name:=` is a ROS argument parse error: only pass a set interface.
  [ -n "$CAM_IFACE" ] && a+=(-p "multicast_iface:=$CAM_IFACE")
  printf '%s\n' "${a[@]}"
}

up() {
  local decoder=${1:-nvv4l2}
  env_setup
  [ -z "$(ours)" ] || { echo "already running: $(ours | tr '\n' ' ') (run down first)"; return 1; }
  local cfg="$WS/install/go2_rgb_lidar/share/go2_rgb_lidar/config"
  local intr=${GSN_INTRINSICS:-$cfg/front_camera_intrinsics_nominal.yaml}
  local ext=${GSN_EXTRINSICS:-$cfg/front_camera_extrinsics_nominal.yaml}
  local latency=${GSN_LATENCY_S:-0.0}
  CAM_IFACE=$IFACE
  [ "$REHEARSAL" = 1 ] && CAM_IFACE=""
  echo "$decoder" > "$RUN/decoder"; echo "$intr" > "$RUN/intrinsics"; echo "$ext" > "$RUN/extrinsics"; echo "$latency" > "$RUN/latency_s"
  start relay ros2 run go2_localization go2_state_relay_node --ros-args -p cloud_in_topic:=/utlidar/cloud_deskewed
  mapfile -t cam < <(camera_args "$decoder" "$latency")
  start camera ros2 run go2_front_camera_cpp front_camera_node --ros-args "${cam[@]}"
  start depth ros2 run go2_rgb_lidar lidar_depth_node --ros-args --params-file "$cfg/rgb_lidar.yaml" \
    -p intrinsics_file:="$intr" -p extrinsics_file:="$ext" -p allow_nominal_calibration:=true
  sleep 6
  # The hardware decoder path has never run against this camera: fall back to
  # software decode if it produces nothing, and record that it did.
  local fps
  fps=$(timeout 15 python3 "$LAB/bin/lab_probe.py" --seconds 4 --no-depth 2>/dev/null | python3 -c "
import json, sys
fps = 0.0
for line in sys.stdin:
    if line.startswith('{'):
        fps = json.loads(line).get('camera_fps', 0.0)
print(int(fps))")
  echo "camera fps with $decoder: ${fps:-0}"
  if [ "$decoder" = nvv4l2 ] && [ "${fps:-0}" -lt 1 ]; then
    echo "nvv4l2 produced no frames: restarting the camera with avdec"; tail -5 "$RUN/logs/camera.log"
    kill -INT -- "-$(cat "$RUN/pids/camera")" 2>/dev/null; sleep 2
    echo avdec > "$RUN/decoder"
    mapfile -t cam < <(camera_args avdec "$latency")
    start camera ros2 run go2_front_camera_cpp front_camera_node --ros-args "${cam[@]}"
    sleep 5
  fi
  status
}

down() {
  for f in "$RUN"/pids/*; do
    [ -f "$f" ] || continue
    kill -INT -- "-$(cat "$f")" 2>/dev/null
    rm -f "$f"
  done
  sleep 3
  local left; left=$(ours)
  [ -n "$left" ] && { kill -9 $left 2>/dev/null; sleep 1; }
  [ -z "$(ours)" ] && echo "down: nothing of ours running" || echo "down: STILL RUNNING $(ours | tr '\n' ' ')"
}

status() {
  echo "decoder $(cat "$RUN/decoder" 2>/dev/null)  run $RUN"
  for f in "$RUN"/pids/*; do
    [ -f "$f" ] || continue
    local n; n=$(basename "$f")
    if kill -0 "$(cat "$f")" 2>/dev/null; then echo "up   $n"; else echo "DEAD $n"; tail -5 "$RUN/logs/$n.log"; fi
  done
}

cmd=${1:-status}; shift || true
case "$cmd" in
  prepare) prepare ;;
  up) up "$@" ;;
  down) down ;;
  status) status ;;
  probe) env_setup; python3 "$LAB/bin/lab_probe.py" --seconds "${1:-15}" ;;
  capture)
    env_setup
    seg=$1; secs=$2; shift 2
    python3 "$LAB/bin/capture_recorder.py" --out "$RUN/capture" --segment "$seg" --seconds "$secs" \
      --intrinsics "$(cat "$RUN/intrinsics")" --extrinsics "$(cat "$RUN/extrinsics")" \
      --latency-s "$(cat "$RUN/latency_s")" "$@" ;;
  *) echo "usage: $0 prepare|up [nvv4l2|avdec]|probe [s]|capture <segment> <s> [args]|down|status"; exit 64 ;;
esac
