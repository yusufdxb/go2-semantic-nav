#!/usr/bin/env bash
# Control-PC side of the RGB + LiDAR lab run (target: under 10 minutes at the robot).
#
#   run_lab.sh deploy                 set robot-computer clock, copy sources, prepare + build
#   run_lab.sh up [nvv4l2|avdec]      start relay + C++ camera + lidar_depth (nominal calibration)
#   run_lab.sh probe [seconds]        health + latency table
#   run_lab.sh checkerboard           <= 120 s, stops itself at 30 boards / 75 % coverage
#   run_lab.sh scene label:x:y ...    15 s, robot still, taped objects in base_link metres
#   run_lab.sh yaw                    30 s, operator yaws the robot slowly with the remote
#   run_lab.sh down                   stop everything `up` started
#   run_lab.sh pull                   copy the capture here
#   run_lab.sh offline                intrinsics -> latency -> extrinsics -> acceptance (no robot)
#   run_lab.sh all label:x:y ...      every step above in order, with operator prompts
#
# Robot computer: the first answering address in JETSON_HOSTS (default the
# GO2 payload's cable address 192.168.123.18; add the lab wifi address), or
# JETSON_HOST to force one. BASE_STACK: checkout of the base navigation stack
# that provides go2_localization. Nothing here commands robot motion.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PW=${JETSON_PW:-123}
BASE_STACK=${BASE_STACK:-}
LOCAL_RUNS=${LOCAL_RUNS:-$HOME/rgb_lidar_runs}
JETSON_HOSTS=${JETSON_HOSTS:-192.168.123.18}
T0=$(date +%s)

host() {
  [ -n "${JETSON_HOST:-}" ] && { echo "$JETSON_HOST"; return; }
  for h in $JETSON_HOSTS; do
    ping -c1 -W1 "$h" >/dev/null 2>&1 && { echo "$h"; return; }
  done
  echo "no route to the robot computer (tried: $JETSON_HOSTS; set JETSON_HOSTS or JETSON_HOST)" >&2; exit 1
}
H=""
r() { sshpass -p "$PW" ssh -o StrictHostKeyChecking=accept-new "unitree@$H" "$@"; }
step() { printf '\n== [%3ss] %s\n' "$(( $(date +%s) - T0 ))" "$*"; }
lab() { r "bash ~/gsn_lab/bin/jetson_lab.sh $*"; }

deploy() {
  [ -d "$BASE_STACK/go2_localization" ] || { echo "set BASE_STACK to the base stack checkout (needs go2_localization)"; exit 64; }
  step "clock + copy"
  r "echo $PW | sudo -S -p '' date -u -s '$(date -u '+%Y-%m-%d %H:%M:%S')' >/dev/null && date -u"
  r "mkdir -p ~/gsn_ws/src ~/gsn_lab/bin"
  local RS="sshpass -p $PW rsync -az --delete -e ssh"
  $RS --exclude build --exclude __pycache__ "$REPO/ros2_ws/src/go2_front_camera_cpp" "$REPO/ros2_ws/src/go2_rgb_lidar" \
      "unitree@$H:gsn_ws/src/"
  $RS --exclude __pycache__ --exclude test "$BASE_STACK/go2_localization" "unitree@$H:gsn_ws/src/"
  $RS "$REPO/scripts/lab/jetson_lab.sh" "$REPO/scripts/lab/lab_probe.py" "$REPO/scripts/lab/capture_recorder.py" \
      "unitree@$H:gsn_lab/bin/"
  echo "base stack go2_localization from $BASE_STACK @ $(git -C "$BASE_STACK" rev-parse --short HEAD)"
  step "prepare + build on the robot computer"
  lab prepare
}

pull() {
  step "pull capture"
  local dst="$LOCAL_RUNS/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$dst"
  sshpass -p "$PW" rsync -az -e ssh "unitree@$H:gsn_lab/runs/current/" "$dst/"
  ln -sfn "$dst" "$LOCAL_RUNS/latest"
  du -sh "$dst"
}

offline() {
  local run=${1:-$LOCAL_RUNS/latest}
  local cap="$run/capture" out="$run/calibration" c="$REPO/scripts/calib"
  mkdir -p "$out"
  # rc_e/rc_a start as "skipped" so a step that never ran cannot read as 0 (PASS).
  local intr="$out/front_camera_intrinsics.yaml" rc_i=0 rc_l=0 rc_e=skipped rc_a=skipped
  step "intrinsics"
  python3 "$c/intrinsics_from_capture.py" --capture "$cap" --board 7x6 --square "${SQUARE_M:-0.025}" \
    --out "$intr" > >(tee "$out/intrinsics.txt") 2>&1 || rc_i=$?
  step "latency"
  # Latency needs only fx; it can use the capture's own (possibly nominal) intrinsics.
  if [ $rc_i -eq 0 ]; then
    python3 "$c/latency_from_capture.py" --capture "$cap" --intrinsics "$intr" > >(tee "$out/latency.txt") 2>&1 || rc_l=$?
  else
    python3 "$c/latency_from_capture.py" --capture "$cap" > >(tee "$out/latency.txt") 2>&1 || rc_l=$?
  fi
  if [ $rc_i -ne 0 ]; then
    echo "intrinsics FAILED (exit $rc_i): extrinsics and acceptance need them; skipped"
  else
    step "extrinsics"
    rc_e=0
    python3 "$c/extrinsics_from_capture.py" --capture "$cap" --intrinsics "$intr" \
      --out "$out/front_camera_extrinsics.yaml" --overlay-dir "$out/overlay" > >(tee "$out/extrinsics.txt") 2>&1 || rc_e=$?
    if [ -f "$out/front_camera_extrinsics.yaml" ]; then
      step "acceptance"
      rc_a=0
      python3 "$c/acceptance_from_capture.py" --capture "$cap" --intrinsics "$intr" \
        --extrinsics "$out/front_camera_extrinsics.yaml" > >(tee "$out/acceptance.txt") 2>&1 || rc_a=$?
    else
      echo "no extrinsics written (exit $rc_e): acceptance skipped"
    fi
  fi
  sleep 0.2
  step "offline summary"
  printf 'intrinsics %s | latency %s | extrinsics %s (3 = initial guess already optimal) | acceptance %s   (0 = PASS)\n' \
    "$rc_i" "$rc_l" "$rc_e" "$rc_a" | tee "$out/summary.txt"
  echo "files: $out"
}

prompt() {  # message; waits for Enter when interactive, else gives 10 s
  echo; echo ">>> $*"
  if [ -t 0 ]; then read -r -p ">>> press Enter to start " _; else sleep 10; fi
}

cmd=${1:-}; shift || true
case "$cmd" in
  offline) offline "$@"; exit 0 ;;
esac
H=$(host)
case "$cmd" in
  deploy) deploy ;;
  up) step "up"; lab up "${1:-nvv4l2}" ;;
  probe) step "probe"; lab probe "${1:-15}" ;;
  checkerboard) step "checkerboard"; lab capture checkerboard 120 --image-rate 3 ;;
  scene) step "scene"; lab capture scene 15 --image-rate 2 --taped "$@" ;;
  yaw) step "yaw"; lab capture yaw 30 ;;
  down) step "down"; lab down ;;
  status) lab status ;;
  pull) pull ;;
  all)
    [ $# -ge 1 ] || { echo "all: give the taped objects, e.g. chair:1.5:0 chair:3.0:0 chair:2.0:0.8"; exit 64; }
    deploy
    step "up"; lab up nvv4l2
    step "probe"; lab probe 15 || echo "probe reported FAILs: read the table before continuing"
    prompt "CHECKERBOARD: hold the board 0.5-1.5 m in front of the camera; sweep it over all four corners and the centre, tilt it. Stops itself when coverage is enough."
    step "checkerboard"; lab capture checkerboard 120 --image-rate 3 || true
    prompt "SCENE: robot still, board out of view, objects placed at: $*"
    step "scene"; lab capture scene 15 --image-rate 2 --taped "$@" || true
    prompt "YAW: robot standing, yaw it slowly left and right with the remote (about 0.5 rad/s) for 30 s."
    step "yaw"; lab capture yaw 30 || true
    step "down"; lab down
    pull
    offline
    step "done"
    ;;
  *) sed -n '2,20p' "$0"; exit 64 ;;
esac
