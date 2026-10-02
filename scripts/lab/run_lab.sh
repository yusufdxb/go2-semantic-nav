#!/usr/bin/env bash
# Control-PC side of the RGB + LiDAR lab run (target: under 10 minutes at the robot).
#
#   run_lab.sh wheels                 BEFORE the session (needs internet): fetch aarch64 Python wheels
#   run_lab.sh deploy                 set robot-computer clock, copy sources, prepare + build
#   run_lab.sh up [nvv4l2|avdec]      start relay + C++ camera + lidar_depth (nominal calibration)
#   run_lab.sh probe [seconds]        health + latency table
#   run_lab.sh checkerboard           <= 120 s, stops itself at 30 boards / 75 % coverage
#   run_lab.sh scene label:x:y ...    15 s, robot still, taped objects in base_link metres
#   run_lab.sh yaw                    30 s, operator yaws the robot slowly with the remote
#   run_lab.sh down                   stop everything `up` started
#   run_lab.sh calibrate              offline calibration ON the robot computer (no bulk copy)
#   run_lab.sh pull [--full]          copy results + logs here (--full: raw capture, use the cable)
#   run_lab.sh offline [run_dir]      offline calibration on this PC from a --full pull
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
WHEELS=${WHEELS:-$LOCAL_RUNS/wheels}  # cp310 numpy/opencv-headless/pyyaml wheels for offline install
WHEEL_GLOB=${WHEEL_GLOB:-*aarch64*.whl}  # only the robot computer's architecture goes over the link
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
      "$REPO/scripts/lab/offline_calibrate.sh" "unitree@$H:gsn_lab/bin/"
  # Calibration tools keep their repo-relative layout (they import go2_rgb_lidar from it).
  r "mkdir -p ~/gsn_lab/repo/scripts ~/gsn_lab/repo/ros2_ws/src"
  $RS --exclude __pycache__ --exclude 'test_*' "$REPO/scripts/calib" "unitree@$H:gsn_lab/repo/scripts/"
  $RS --exclude __pycache__ --exclude test "$REPO/ros2_ws/src/go2_rgb_lidar" "unitree@$H:gsn_lab/repo/ros2_ws/src/"
  echo "base stack go2_localization from $BASE_STACK @ $(git -C "$BASE_STACK" rev-parse --short HEAD)"
  # The lab network has no internet: if numpy/cv2/yaml are missing there,
  # install pre-downloaded wheels into a private folder (only our processes use it).
  if ! lab pycheck; then
    step "python deps from wheels"
    ls "$WHEELS"/*.whl >/dev/null 2>&1 || { echo "no wheels in $WHEELS (see docs/lab_run.md)"; exit 1; }
    r "mkdir -p ~/gsn_lab/wheels"
    sshpass -p "$PW" rsync -az -e ssh --include "$WHEEL_GLOB" --exclude '*' "$WHEELS"/ "unitree@$H:gsn_lab/wheels/"
    lab pydeps
  fi
  step "prepare + build on the robot computer"
  lab prepare
}

pull() {  # default: calibration results, logs and capture metadata (KB); --full: raw capture too
  step "pull ${1:-results}"
  local dst="$LOCAL_RUNS/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$dst"
  local filt=()
  # The robot computer's 2.4 GHz wifi measured 0.2-1.4 Mbit/s: raw frames stay
  # there unless asked for (use the cable for --full).
  [ "${1:-}" = "--full" ] || filt=(--include '*/' --include '*.yaml' --include '*.txt' --include '*.csv' \
                                   --include '*.log' --include 'overlay/*' --exclude 'frames/*' --exclude 'clouds/*')
  sshpass -p "$PW" rsync -az -e ssh "${filt[@]}" "unitree@$H:gsn_lab/runs/current/" "$dst/"
  ln -sfn "$dst" "$LOCAL_RUNS/latest"
  du -sh "$dst"
}

offline() {  # on this PC, from a pulled run (pull --full first)
  bash "$REPO/scripts/lab/offline_calibrate.sh" "${1:-$LOCAL_RUNS/latest}" "$REPO/scripts/calib"
}

prompt() {  # message; waits for Enter when interactive, else gives 10 s
  echo; echo ">>> $*"
  if [ -t 0 ]; then read -r -p ">>> press Enter to start " _; else sleep 10; fi
}

wheels() {  # numpy/cv2/yaml for the robot computer's Python 3.10 (aarch64), installed only if missing there
  mkdir -p "$WHEELS"
  for plat in manylinux2014_aarch64 manylinux_2_17_aarch64; do
    python3 -m pip download -q --dest "$WHEELS" --only-binary=:all: --python-version 3.10 --implementation cp \
      --platform "$plat" "numpy==1.26.4" "opencv-python-headless==4.10.0.84" "pyyaml==6.0.2"
  done
  ls -la "$WHEELS"/*aarch64*.whl
}

cmd=${1:-}; shift || true
case "$cmd" in
  offline) offline "$@"; exit 0 ;;
  wheels) wheels; exit 0 ;;
esac
H=$(host)
case "$cmd" in
  deploy) deploy ;;
  up) step "up"; lab up "${1:-nvv4l2}" ;;
  probe) step "probe"; lab probe "${1:-15}" ;;
  checkerboard) step "checkerboard"; lab capture checkerboard 120 --image-rate 3 --no-clouds ;;
  scene) step "scene"; lab capture scene 15 --image-rate 2 --taped "$@" ;;
  yaw) step "yaw"; lab capture yaw 30 --no-clouds ;;
  down) step "down"; lab down ;;
  status) lab status ;;
  pull) pull "$@" ;;
  calibrate) step "calibrate on the robot computer"; lab calibrate ;;
  all)
    [ $# -ge 1 ] || { echo "all: give the taped objects, e.g. chair:1.5:0 chair:3.0:0 chair:2.0:0.8"; exit 64; }
    deploy
    step "up"; lab up nvv4l2
    step "probe"
    if ! lab probe 15; then
      echo "probe reported FAILs (table above)"
      if [ -t 0 ]; then
        read -r -p ">>> continue anyway? [y/N] " ans
        [ "$ans" = y ] || { lab down; exit 2; }
      fi
    fi
    prompt "CHECKERBOARD: hold the board 0.5-1.5 m in front of the camera; sweep it over all four corners and the centre, tilt it. Stops itself when coverage is enough."
    step "checkerboard"; lab capture checkerboard 120 --image-rate 3 --no-clouds || true
    prompt "SCENE: robot still, board out of view, objects placed at: $*"
    step "scene"; lab capture scene 15 --image-rate 2 --taped "$@" || true
    prompt "YAW: robot standing, yaw it slowly left and right with the remote (about 0.5 rad/s) for 30 s."
    step "yaw"; lab capture yaw 30 --no-clouds || true
    step "down"; lab down
    step "calibrate on the robot computer"; lab calibrate || true
    pull
    step "done (robot time ends here; raw capture stays on the robot computer: run_lab.sh pull --full)"
    ;;
  *) sed -n '2,20p' "$0"; exit 64 ;;
esac
