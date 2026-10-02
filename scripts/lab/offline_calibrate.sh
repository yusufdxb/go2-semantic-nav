#!/usr/bin/env bash
# Offline calibration of one lab run: intrinsics -> latency -> extrinsics -> acceptance.
#   offline_calibrate.sh <run_dir> [calib_tools_dir]
# Reads <run_dir>/capture, writes <run_dir>/calibration (summary.txt last).
# Runs on the control PC or on the robot computer (needs numpy, cv2, yaml).
set -uo pipefail
run=$1
c=${2:-$(cd "$(dirname "$0")/../calib" && pwd)}
cap="$run/capture" out="$run/calibration"
mkdir -p "$out"
T0=$(date +%s)
step() { printf '\n== [%3ss] %s\n' "$(( $(date +%s) - T0 ))" "$*"; }
# rc_e/rc_a start as "skipped" so a step that never ran cannot read as 0 (PASS).
intr="$out/front_camera_intrinsics.yaml" rc_i=0 rc_l=0 rc_e=skipped rc_a=skipped
step "intrinsics"
python3 "$c/intrinsics_from_capture.py" --capture "$cap" --board 7x6 --square "${SQUARE_M:-0.025}" \
  --out "$intr" 2>&1 | tee "$out/intrinsics.txt"; rc_i=${PIPESTATUS[0]}
step "latency"
# Latency needs only fx; it can use the capture's own (possibly nominal) intrinsics.
if [ "$rc_i" -eq 0 ]; then
  python3 "$c/latency_from_capture.py" --capture "$cap" --intrinsics "$intr" 2>&1 | tee "$out/latency.txt"; rc_l=${PIPESTATUS[0]}
else
  python3 "$c/latency_from_capture.py" --capture "$cap" 2>&1 | tee "$out/latency.txt"; rc_l=${PIPESTATUS[0]}
fi
if [ "$rc_i" -ne 0 ]; then
  echo "intrinsics FAILED (exit $rc_i): extrinsics and acceptance need them; skipped"
else
  step "extrinsics"
  python3 "$c/extrinsics_from_capture.py" --capture "$cap" --intrinsics "$intr" \
    --out "$out/front_camera_extrinsics.yaml" --overlay-dir "$out/overlay" 2>&1 | tee "$out/extrinsics.txt"
  rc_e=${PIPESTATUS[0]}
  if [ -f "$out/front_camera_extrinsics.yaml" ]; then
    step "acceptance"
    python3 "$c/acceptance_from_capture.py" --capture "$cap" --intrinsics "$intr" \
      --extrinsics "$out/front_camera_extrinsics.yaml" 2>&1 | tee "$out/acceptance.txt"; rc_a=${PIPESTATUS[0]}
  else
    echo "no extrinsics written (exit $rc_e): acceptance skipped"
  fi
fi
step "offline summary"
printf 'intrinsics %s | latency %s | extrinsics %s (3 = initial guess already optimal) | acceptance %s   (0 = PASS)\n' \
  "$rc_i" "$rc_l" "$rc_e" "$rc_a" | tee "$out/summary.txt"
echo "files: $out"
