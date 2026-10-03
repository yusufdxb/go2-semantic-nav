#!/usr/bin/env bash
# Robot-computer side of the RGB + LiDAR lab run. Called by run_lab.sh over
# ssh; can also be run by hand on the robot computer.
#
#   jetson_lab.sh pycheck            exit 0 if numpy, cv2 and yaml import (with our private deps)
#   jetson_lab.sh pydeps             install them from $LAB/wheels into $LAB/pydeps (no network)
#   jetson_lab.sh prepare            sysctl, dependency checks, colcon build (idempotent)
#   jetson_lab.sh up [avdec|nvv4l2]  start relay + C++ camera + lidar_depth in the background
#   jetson_lab.sh probe [seconds]    health/latency table (lab_probe.py)
#   jetson_lab.sh capture <segment> <seconds> [recorder args...]
#   jetson_lab.sh down               stop what `up` started (by process group)
#   jetson_lab.sh calibrate          offline calibration of the current run, here
#   jetson_lab.sh mlinstall          detector wheels from $LAB/ml/wheels into $LAB/pydeps_ml (no network)
#   jetson_lab.sh mlcheck            CUDA torch + detector imports with the private deps
#   jetson_lab.sh slam [name]        2D scan + slam_toolbox (mapping, repo config) + map autosave every 60 s
#   jetson_lab.sh semantic [dir]     detector + scene graph (grounding off), JSON snapshots to dir
#   jetson_lab.sh status
#   jetson_lab.sh clockab [blocks]   NVDEC/VIC clock A/B in the background, after up
#                                    (docs/preregistration/nvdec-clock-latency.md)
#   jetson_lab.sh clockab_stop       end it early (the clocks are given back either way)
#
# Never commands robot motion: it starts sensor relays, the camera driver,
# the depth projector, mapping, perception and recorders only. clockab
# changes the NVDEC and VIC clock settings only, and restores them on exit.
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
  # Private Python deps (only present if the system lacked them): our processes only.
  [ -d "$LAB/pydeps" ] && export PYTHONPATH="$LAB/pydeps${PYTHONPATH:+:$PYTHONPATH}"
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  # numpy's OpenBLAS starts one spinning worker per core: lidar_depth_node sat
  # at ~260% CPU on the Orin NX for small 4x4 transforms, ~37% with one thread.
  export OPENBLAS_NUM_THREADS=1
  # Large DDS receive buffers: a raw 720p frame is ~2.7 MB of UDP fragments
  # and the default buffer drops whole frames. min= makes CycloneDDS refuse to
  # start if the kernel limit was not raised, instead of silently losing frames.
  local buf='<CycloneDDS><Domain><Internal><SocketReceiveBufferSize min="16MB"/></Internal></Domain></CycloneDDS>'
  [ "$REHEARSAL" = 1 ] && buf='<CycloneDDS><Domain><Internal><SocketReceiveBufferSize min="default" max="16MB"/></Internal></Domain></CycloneDDS>'
  # Multicast for discovery only. With two or more readers of the 720p image
  # CycloneDDS switched to multicast data on the robot NIC: ~41 MB/s onto the
  # GO2's internal network and camera arrival->publish p50 16 -> 53 ms.
  # Unicast keeps local readers on loopback.
  buf="$buf,<CycloneDDS><Domain><General><AllowMulticast>spdp</AllowMulticast></General></Domain></CycloneDDS>"
  export CYCLONEDDS_URI="${CYCLONEDDS_URI:+$CYCLONEDDS_URI,}$buf"
}

ml_env() {  # detector only: CUDA torch and model packages from the private folder, no network lookups
  local t=$LAB/pydeps_ml
  [ -d "$t/torch" ] || { echo "no $t (run_lab.sh mldeploy)"; return 1; }
  export PYTHONPATH="$t${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="$t/nvidia/cusparselt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export HF_HUB_OFFLINE=1 YOLO_OFFLINE=1
}

ours() {  # PIDs of processes started from our workspace (matched on the program, not the command text)
  for d in /proc/[0-9]*; do
    local exe
    exe=$( { tr '\0' ' ' < "$d/cmdline"; } 2>/dev/null | awk '{print $1" "$2" "$3}')  # processes can exit mid-scan
    case "$exe" in *"$WS/install/"*|*"$LAB/bin/capture_recorder.py"*|*" autosave_loop") echo "${d#/proc/}";; esac
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
  # Plain --exists also requires the plugin version to reach the core version
  # (1.20); JetPack's NVIDIA plugins report 1.14, so they would read as missing.
  has() { gst-inspect-1.0 --exists --atleast-version=1.0 "$1"; }
  for e in udpsrc rtph264depay h264parse videoconvert videoscale appsink; do
    has "$e" && echo "ok   gst $e" || { echo "FAIL gst $e missing"; fail=1; }
  done
  # One working H.264 decoder is enough: hardware (default) or software (fallback).
  local hw=0 sw=0
  has nvv4l2decoder && has nvvidconv && hw=1
  has avdec_h264 && sw=1
  echo "decoders: nvv4l2 $([ $hw = 1 ] && echo ok || echo MISSING), avdec $([ $sw = 1 ] && echo ok || echo MISSING)"
  [ $hw = 1 ] || [ $sw = 1 ] || { echo "FAIL no H.264 decoder"; fail=1; }
  [ $hw = 1 ] || echo "NOTE up will use avdec (pass 'avdec' to up to skip the nvv4l2 attempt)"
  [ $sw = 1 ] || echo "NOTE no software fallback if nvv4l2 yields no frames"
  for p in libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev; do
    dpkg -s "$p" >/dev/null 2>&1 && echo "ok   $p" || { echo "FAIL $p missing"; fail=1; }
  done
  env_setup
  python3 -c "import numpy, cv2, yaml, sensor_msgs_py; print('ok   python numpy', numpy.__version__, 'cv2', cv2.__version__, 'from', cv2.__file__)" \
    || { echo "FAIL python deps (run_lab.sh deploy installs them from wheels)"; fail=1; }
  ( cd "$WS" && colcon build --packages-select go2_front_camera_cpp go2_rgb_lidar go2_localization \
      go2_semantic_msgs go2_open_vocab_detector go2_scene_graph go2_language_grounding go2_semantic_bringup \
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

mlinstall() {
  ls "$LAB"/ml/wheels/*.whl >/dev/null 2>&1 || { echo "FAIL no wheels in $LAB/ml/wheels"; return 1; }
  rm -rf "$LAB/pydeps_ml"
  # --no-deps: everything else the wheels need is already on the robot computer
  # (checked on JetPack 6); a resolver run would try the network.
  python3 -m pip install -q --no-deps --no-index --target "$LAB/pydeps_ml" "$LAB"/ml/wheels/*.whl \
    > "$RUN/logs/mlinstall.txt" 2>&1 || { cat "$RUN/logs/mlinstall.txt"; return 1; }
  mkdir -p ~/.cache/ultralytics ~/.cache/mobile_sam ~/.cache/clip ~/.cache/huggingface/hub
  cp -n "$LAB"/ml/weights/yolov8s-worldv2.pt ~/.cache/ultralytics/
  cp -n "$LAB"/ml/weights/mobile_sam.pt ~/.cache/mobile_sam/
  cp -n "$LAB"/ml/weights/ViT-B-32.pt ~/.cache/clip/
  cp -rn "$LAB"/ml/weights/models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K ~/.cache/huggingface/hub/
  mlcheck
}

mlcheck() {
  env_setup; ml_env || return 1
  # Library warnings (timm deprecations) go to the log; the verdict line to the screen.
  python3 - 2> "$RUN/logs/mlcheck.txt" <<'PY' || { tail -5 "$RUN/logs/mlcheck.txt"; return 1; }
import torch, open_clip, timm, clip, mobile_sam, ultralytics  # noqa: F401
ok = torch.cuda.is_available()
print(("ok  " if ok else "FAIL") + f" torch {torch.__version__} cuda {ok} from {torch.__file__}")
raise SystemExit(0 if ok else 1)
PY
}

autosave_loop() {  # name, period: save the live slam_toolbox map, overwriting one name
  trap 'exit 0' INT TERM
  env_setup
  local name=$1 period=$2
  mkdir -p "$LAB/maps"
  while true; do
    sleep "$period" & wait $!
    timeout 40 ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "{name: {data: $LAB/maps/$name}}" >/dev/null 2>&1 \
      && timeout 60 ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: $LAB/maps/$name}" >/dev/null 2>&1 \
      && echo "$(date -u +%H:%M:%S) saved $LAB/maps/$name" || echo "$(date -u +%H:%M:%S) SAVE FAILED"
  done
}

slam() {
  # Repo slam_toolbox config: a scan joins the map only after 0.2 m / 0.2 rad
  # of motion. Forcing it to add scans while standing still (minimum travel 0)
  # drifted map->odom by 3.5 m and took 4+ cores within 45 min on the robot.
  env_setup
  local name=${1:-map_$(date -u +%Y%m%dT%H%M%SZ)}
  local cfg="$WS/install/go2_localization/share/go2_localization/config"
  start scan ros2 run pointcloud_to_laserscan pointcloud_to_laserscan_node --ros-args \
    --params-file "$cfg/pointcloud_to_laserscan.yaml" -r cloud_in:=/go2/lidar/points -r scan:=/scan
  start slam ros2 run slam_toolbox async_slam_toolbox_node --ros-args --params-file "$cfg/slam_toolbox.yaml" -p mode:=mapping
  start autosave bash "$0" autosave_loop "$name" 60
  echo "map autosave every 60 s to $LAB/maps/$name.{pgm,yaml,posegraph,data}"
}

semantic() {
  env_setup; ml_env || return 1
  local dir=${1:-$LAB/maps/semantic_$(date -u +%Y%m%dT%H%M%SZ)}
  local det="$WS/install/go2_open_vocab_detector/share/go2_open_vocab_detector/config/detector_rgb_lidar.yaml"
  start semantic ros2 launch go2_semantic_bringup semantic_nav.launch.py enable_grounding:=false use_rviz:=false \
    detector_params:="$det" image_topic:=/camera/front/image_raw depth_topic:=/camera/front/lidar_depth \
    camera_info_topic:=/camera/front/camera_info scene_graph_snapshot_dir:="$dir"
  echo "scene graph snapshots to $dir (detector loads in ~1 min)"
}

pycheck() {
  env_setup
  [ "${GSN_FORCE_PYDEPS:-0}" = 1 ] && [ ! -d "$LAB/pydeps" ] && { echo "pycheck: forced miss (rehearsal)"; return 1; }
  python3 -c "import numpy, cv2, yaml" 2>/dev/null && echo "pycheck: ok" || { echo "pycheck: missing"; return 1; }
}

pydeps() {
  python3 -m pip --version >/dev/null 2>&1 || { echo "FAIL pip missing on the robot computer"; return 1; }
  # pip prints a dependency-resolver "ERROR" about unrelated user packages even
  # when this isolated --target install succeeds: keep its output in a log.
  if ! python3 -m pip install -q --no-index --find-links "$LAB/wheels" --target "$LAB/pydeps" \
      "numpy==1.26.4" "opencv-python-headless==4.10.0.84" "pyyaml==6.0.2" > "$RUN/logs/pydeps.txt" 2>&1; then
    cat "$RUN/logs/pydeps.txt"; return 1
  fi
  echo "pydeps installed into $LAB/pydeps (our processes only)"
  pycheck
}

# NVDEC/VIC clock A/B, as registered in docs/preregistration/nvdec-clock-latency.md.
# Changes only the two engines' devfreq settings and gives them back on any exit.
NVDEC_DF=${GSN_NVDEC_DEVFREQ:-/sys/class/devfreq/15480000.nvdec}
VIC_DF=${GSN_VIC_DEVFREQ:-/sys/class/devfreq/15340000.vic}
CLK_SAVE=""
CLK_METHOD=""

sysw() {  # value, file: a root write here, a plain one in rehearsal (fake devfreq folders)
  if [ "$REHEARSAL" = 1 ]; then echo "$1" > "$2"; else echo "$PW" | sudo -S -p '' sh -c "echo '$1' > '$2'"; fi
}

clk() {  # devfreq folder, max|default: hold the engine's clock at max, or give it back its recorded default
  local d=$1 key; key=$(basename "$1")
  if [ "$2" = max ]; then
    if [ "$CLK_METHOD" = governor ]; then sysw performance "$d/governor"; else sysw "$(cat "$d/max_freq")" "$d/min_freq"; fi
  else
    sysw "$(cat "$CLK_SAVE/$key.governor")" "$d/governor"
    sysw "$(cat "$CLK_SAVE/$key.min_freq")" "$d/min_freq"
  fi
  if [ "$REHEARSAL" = 1 ]; then  # a fake devfreq folder has no clock behind it: mirror what the setting would do
    if [ "$2" = max ]; then cat "$d/max_freq" > "$d/cur_freq"; else cat "$d/min_freq" > "$d/cur_freq"; fi
  fi
}

clk_restore() {
  [ -n "$CLK_SAVE" ] || return 0
  clk "$NVDEC_DF" default; clk "$VIC_DF" default
  CLK_SAVE=""
  [ "$REHEARSAL" = 1 ] || { echo "$PW" | sudo -S -p '' tegrastats --stop; } > /dev/null 2>&1
  local d
  for d in "$NVDEC_DF" "$VIC_DF"; do
    echo "clockab: restored $(basename "$d"): $(cat "$d/governor"), min $(cat "$d/min_freq") Hz, now $(cat "$d/cur_freq") Hz"
  done
}

clockab_exit() {  # EXIT trap: the clocks never stay held, whatever ends the run
  clk_restore
  rm -f "$RUN/pids/clockab"
}

tegra_start() {  # tegrastats for one phase (power and temperature); none in rehearsal
  [ "$REHEARSAL" = 1 ] && return 0
  command -v tegrastats > /dev/null || return 0
  echo "$PW" | sudo -S -p '' tegrastats --interval 1000 --logfile "$1" --start
}

tegra_stop() {
  [ "$REHEARSAL" = 1 ] && return 0
  command -v tegrastats > /dev/null || return 0
  echo "$PW" | sudo -S -p '' tegrastats --stop
  echo "$PW" | sudo -S -p '' chown "$(id -u):$(id -g)" "$1" 2> /dev/null
  return 0
}

phase_record() {  # block, phase, order, tegrastats log: one {"kind": "phase"} JSON line
  python3 - "$@" "$CLK_METHOD" <<'PY'
import json, re, sys
block, phase, order, log, method = sys.argv[1:6]
vdd, tj = [], []
try:
    with open(log) as f:
        for line in f:
            m = re.search(r"VDD_IN (\d+)mW", line)
            if m:
                vdd.append(int(m.group(1)))
            tj += [float(x) for x in re.findall(r"tj@([0-9.]+)C", line)]
except OSError:
    pass
print(json.dumps({"kind": "phase", "block": int(block), "phase": phase, "order": order, "method": method,
                  "tegrastats_samples": len(vdd), "vdd_in_mw_avg": round(sum(vdd) / len(vdd)) if vdd else None,
                  "tj_c_max": max(tj) if tj else None}))
PY
}

clockab_run() {  # [blocks] [commit]: the A/B itself; normally started in the background by `clockab`
  env_setup
  local blocks=${1:-6} commit=${2:-unknown} root="$RUN/clockab" out data d f
  for d in "$NVDEC_DF" "$VIC_DF"; do
    for f in governor cur_freq min_freq max_freq available_governors; do
      [ -r "$d/$f" ] || { echo "clockab: FAIL cannot read $d/$f (set GSN_NVDEC_DEVFREQ / GSN_VIC_DEVFREQ)"; return 1; }
    done
  done
  if [ "$REHEARSAL" != 1 ] && [ "$(cat "$RUN/decoder" 2> /dev/null)" != nvv4l2 ]; then
    echo "clockab: FAIL the camera is not on nvv4l2 (decoder: $(cat "$RUN/decoder" 2> /dev/null)); the A/B is about the hardware decoder"
    return 1
  fi
  # Camera first, clocks second: nothing is changed unless the stats topic is live.
  python3 "$LAB/bin/latency_window.py" --windows 1 --settle 0 --timeout 10 > /dev/null \
    || { echo "clockab: FAIL no /camera/front/latency within 10 s (run up first)"; return 1; }
  out="$root/$(date -u +%Y%m%dT%H%M%SZ)"
  data="$out/windows.jsonl"
  mkdir -p "$out/defaults"
  ln -sfn "$out" "$root/latest"
  for d in "$NVDEC_DF" "$VIC_DF"; do
    cat "$d/governor" > "$out/defaults/$(basename "$d").governor"
    cat "$d/min_freq" > "$out/defaults/$(basename "$d").min_freq"
  done
  CLK_METHOD=minfreq
  grep -qw performance "$NVDEC_DF/available_governors" && grep -qw performance "$VIC_DF/available_governors" \
    && CLK_METHOD=governor
  CLK_SAVE="$out/defaults"
  trap clockab_exit EXIT
  trap 'exit 130' INT TERM HUP
  local nvp=""
  [ "$REHEARSAL" = 1 ] || nvp=$( { echo "$PW" | sudo -S -p '' nvpmodel -q; } 2> /dev/null | tr '\n' ' ')
  python3 - "$out" "$NVDEC_DF" "$VIC_DF" "$CLK_METHOD" "$blocks" "$commit" "$REHEARSAL" \
    "$(cat "$RUN/decoder" 2> /dev/null)" "$nvp" "$(timeout 8 ros2 node list 2> /dev/null | tr '\n' ' ')" > "$data" <<'PY'
import datetime, json, pathlib, sys
out, nv, vic, method, blocks, commit, rehearsal, decoder, nvp, nodes = sys.argv[1:11]

def rd(d, f):
    try:
        return pathlib.Path(d, f).read_text().strip()
    except OSError:
        return None

cfg = {"kind": "config", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
       "method": method, "blocks": int(blocks), "settle_s": 5, "windows": 25, "commit": commit,
       "rehearsal": rehearsal == "1", "decoder": decoder or None, "nvpmodel": nvp.strip() or None,
       "ros_nodes": nodes.split()}
for eng, d in (("nvdec", nv), ("vic", vic)):
    key = pathlib.Path(d).name
    cfg[f"{eng}_devfreq"] = d
    cfg[f"{eng}_default_governor"] = rd(f"{out}/defaults", f"{key}.governor")
    cfg[f"{eng}_default_min_hz"] = int(rd(f"{out}/defaults", f"{key}.min_freq"))
    cfg[f"{eng}_max_hz"] = int(rd(d, "max_freq"))
    cfg[f"{eng}_available_governors"] = (rd(d, "available_governors") or "").split()
print(json.dumps(cfg))
PY
  echo "clockab: $blocks blocks, clock method $CLK_METHOD, results in $out"
  local b p order sum
  local -a orders
  mapfile -t orders < <(python3 "$LAB/bin/clockab_analyze.py" --schedule "$blocks")
  for b in $(seq 1 "$blocks"); do
    order=${orders[$((b - 1))]}
    for p in $(echo "$order" | fold -w1); do
      case $p in  # every phase writes both engines' settings, so the switching itself is the same in every arm
        A) clk "$NVDEC_DF" default; clk "$VIC_DF" default ;;
        B) clk "$NVDEC_DF" max; clk "$VIC_DF" default ;;
        C) clk "$NVDEC_DF" max; clk "$VIC_DF" max ;;
      esac
      tegra_start "$out/tegrastats_b$b$p.log"
      sum=$(python3 "$LAB/bin/latency_window.py" --block "$b" --phase "$p" --order "$order" --settle 5 --windows 25 \
        --timeout 60 --nvdec-devfreq "$NVDEC_DF" --vic-devfreq "$VIC_DF" --out "$data" 2>> "$out/collector.err")
      tegra_stop "$out/tegrastats_b$b$p.log"
      phase_record "$b" "$p" "$order" "$out/tegrastats_b$b$p.log" >> "$data"
      echo "clockab: block $b/$blocks ($order) $p: $sum"
    done
  done
  clk_restore
  python3 "$LAB/bin/clockab_analyze.py" "$data" > "$out/analysis.txt"
  local rc=$?
  sed '/^{/d' "$out/analysis.txt"
  echo "clockab: done (analysis exit $rc; $out)"
}

clockab() {  # [blocks] [commit]: start clockab_run in the background; status shows it, clockab_stop ends it
  if [ -f "$RUN/pids/clockab" ] && kill -0 "$(cat "$RUN/pids/clockab")" 2> /dev/null; then
    echo "clockab already running (clockab_stop ends it)"
    return 1
  fi
  start clockab bash "$0" clockab_run "$@"
  echo "clockab started, about $(( ${1:-6} * 100 / 60 + 1 )) min; log $RUN/logs/clockab.log"
}

clockab_stop() {  # end a running clockab early; its exit trap gives the clocks back
  [ -f "$RUN/pids/clockab" ] || { echo "clockab not running"; return 0; }
  kill -INT -- "-$(cat "$RUN/pids/clockab")" 2> /dev/null
  sleep 3
  tail -4 "$RUN/logs/clockab.log"
}

cmd=${1:-status}; shift || true
case "$cmd" in
  pycheck) pycheck ;;
  pydeps) pydeps ;;
  mlinstall) mlinstall ;;
  mlcheck) mlcheck ;;
  slam) slam "$@" ;;
  semantic) semantic "$@" ;;
  autosave_loop) autosave_loop "$@" ;;
  prepare) prepare ;;
  up) up "$@" ;;
  down) down ;;
  status) status ;;
  clockab) clockab "$@" ;;
  clockab_run) clockab_run "$@" ;;
  clockab_stop) clockab_stop ;;
  probe) env_setup; python3 "$LAB/bin/lab_probe.py" --seconds "${1:-15}" ;;
  calibrate) env_setup; bash "$LAB/bin/offline_calibrate.sh" "$RUN" "$LAB/repo/scripts/calib" ;;
  capture)
    env_setup
    seg=$1; secs=$2; shift 2
    python3 "$LAB/bin/capture_recorder.py" --out "$RUN/capture" --segment "$seg" --seconds "$secs" \
      --intrinsics "$(cat "$RUN/intrinsics")" --extrinsics "$(cat "$RUN/extrinsics")" \
      --latency-s "$(cat "$RUN/latency_s")" "$@" ;;
  *) echo "usage: $0 prepare|up [nvv4l2|avdec]|probe [s]|capture <segment> <s> [args]|slam [name]|semantic [dir]|mlinstall|mlcheck|clockab [blocks]|clockab_stop|down|status"; exit 64 ;;
esac
