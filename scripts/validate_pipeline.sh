#!/usr/bin/env bash
# One-command operator-facing pipeline validator.
#
# Runs the smoke + record-replay integration tests in sequence and emits a
# green/red PASS/FAIL bar per stage. Intended to be run after any code change
# that touches the detector / scene_graph / grounding packages before a demo.
#
# Exits 0 only if every stage passed.

set -euo pipefail

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS_ROOT"

source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash

green() { printf "\033[32m%s\033[0m\n" "$1"; }
red()   { printf "\033[31m%s\033[0m\n" "$1"; }

log_dir="/tmp/go2-semantic-nav-validate-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
echo "[validate] logs → $log_dir"
overall=0

# 1) Colcon build
echo
echo "=== 1/5  colcon build ==="
if colcon build --symlink-install > "$log_dir/01_build.log" 2>&1; then
    green "  PASS  build"
else
    red   "  FAIL  build — see $log_dir/01_build.log"
    exit 1
fi
source ros2_ws/install/setup.bash

# 2) Unit-logic sweep
echo
echo "=== 2/5  unit logic ==="
if python3 - <<'PY' > "$log_dir/02_units.log" 2>&1
import sys, numpy as np
sys.path.insert(0, 'ros2_ws/src/go2_open_vocab_detector')
from go2_open_vocab_detector.depth_backproject import (
    CameraIntrinsics, backproject_pixel, mask_rle_encode, mask_rle_decode)
intr = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
assert np.allclose(backproject_pixel(intr.cx, intr.cy, 1.5, intr), [0,0,1.5])
rng = np.random.default_rng(0)
for _ in range(20):
    h, w = int(rng.integers(3,50)), int(rng.integers(3,50))
    m = rng.integers(0,2,size=(h,w)).astype(bool)
    assert np.array_equal(mask_rle_decode(mask_rle_encode(m), h, w), m)
sys.path.insert(0, 'ros2_ws/src/go2_scene_graph')
from go2_scene_graph.graph_store import SceneGraphStore, AssociationParams
s = SceneGraphStore(AssociationParams(assoc_max_dist_m=2.0, pose_ema_alpha=0.4))
e = np.array([1.,0,0], dtype=np.float32)
s.ingest('chair', 0.9, np.array([0.,0,0]), np.array([.4,.4,1.]), e, 2., 1)
oid, _ = s.ingest('chair', 0.9, np.array([1.,0,0]), np.array([.4,.4,1.]), e, 2., 2)
assert abs(s.get(oid).position_map[0] - 0.4) < 1e-5
sys.path.insert(0, 'ros2_ws/src/go2_language_grounding')
from go2_language_grounding.query_parser import parse
for q, (n, r) in [('chair', ('chair', None)), ('go near the window', ('window', 'near'))]:
    p = parse(q); assert p.target_noun == n and p.relation == r
print('unit logic: OK')
PY
then
    green "  PASS  unit logic (depth + scene graph + parser)"
else
    red   "  FAIL  unit logic — see $log_dir/02_units.log"
    overall=1
fi

# 3) Live smoke test
echo
echo "=== 3/5  live smoke test (publisher + 3 nodes + grounding) ==="
if timeout 180 python3 scripts/smoke_test_integration.py --text person > "$log_dir/03_smoke.log" 2>&1; then
    green "  PASS  smoke test (record/play grounding)"
else
    red   "  FAIL  smoke test — see $log_dir/03_smoke.log"
    overall=1
fi

# 4) Record-replay integration
echo
echo "=== 4/5  record → replay → ground ==="
if timeout 180 python3 scripts/record_replay_integration_test.py --record-duration-s 6 --text-query person > "$log_dir/04_replay.log" 2>&1; then
    green "  PASS  record-replay integration"
else
    red   "  FAIL  record-replay — see $log_dir/04_replay.log"
    overall=1
fi

# 5) Eval harness (synthetic)
echo
echo "=== 5/5  20-query grounding eval (synthetic) ==="
if timeout 240 python3 eval/run_eval.py --mode synthetic --warmup-s 12 \
        --out "$log_dir/eval_results" --config-name validate_run > "$log_dir/05_eval.log" 2>&1; then
    green "  PASS  20-query eval"
    echo "  → summary: $log_dir/eval_results/summary.json"
else
    red   "  FAIL  eval — see $log_dir/05_eval.log"
    overall=1
fi

echo
if [[ $overall -eq 0 ]]; then
    green "====================="
    green "  ALL STAGES PASSED"
    green "====================="
else
    red   "====================="
    red   "  FAILURES DETECTED — inspect $log_dir"
    red   "====================="
fi
exit $overall
