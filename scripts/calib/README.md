# Offline calibration from a GO2 capture

These tools turn a short capture recorded on the robot into the files the
RGB + LiDAR depth path needs, so the time spent at the robot is only the
recording. They read a capture folder and never talk to ROS or the robot.
Geometry (projection, depth rendering, visibility filter, depth limits) is
imported from `ros2_ws/src/go2_rgb_lidar`, so the results match what
`lidar_depth_node` computes on the robot.

## Capture format

```
<capture_root>/<segment>/        segment: checkerboard | scene | yaw
  meta.yaml       segment, started_utc, camera_frame, latency_s (used at capture),
                  intrinsics (ROS ost.yaml dict, may be nominal),
                  extrinsics (parent_frame, child_frame, xyz, rpy, nominal),
                  optional fixed_frame (default odom), optional lidar_xyz
                  (LiDAR origin in the base frame, default the base stack mount
                  0.28945, 0, -0.046825), optional taped_objects
  frames.csv      stamp_ns,file,width,height
  frames/<stamp_ns>.jpg
  clouds.csv      stamp_ns,file,frame_id,n_points
  clouds/<stamp_ns>.npy   Nx3 float32 in frame_id (odom, or the base frame)
  odom.csv        stamp_ns,x,y,z,qx,qy,qz,qw,wz   (base pose in odom, base yaw rate)
```

`taped_objects`: `[{label, x_m, y_m, z_m}]` in the base frame at capture start
(`z_m` optional, default 0). Tape the point on the object's surface that faces
the robot: the LiDAR measures the surface, not the centre.

## Order and criteria

| step | tool | PASS | other exits |
|---|---|---|---|
| 1 | `intrinsics_from_capture.py --capture C --board 7x6 --square 0.025 --out intr.yaml` | RMS <= 0.5 px, >= 15 views, corners in >= 75 % of a 4x4 grid | 2: FAIL, file not written (`--force` writes anyway) |
| 2 | `latency_from_capture.py --capture C [--intrinsics intr.yaml]` | odom yaw-rate RMS >= 0.2 rad/s, peak correlation >= 0.6, lag not on the +/-0.5 s edge | 2: FAIL. Apply the printed `latency_s` to `go2_front_camera` |
| 3 | `extrinsics_from_capture.py --capture C --intrinsics intr.yaml [--init-extrinsics tape.yaml] --out ext.yaml [--overlay-dir D]` | optimum inside +/-5 deg / +/-5 cm of the initial guess, every rotation axis >= 0.5 px score contrast at +/-2 deg, >= 50 LiDAR silhouette returns, and >= 0.25 px better than the initial guess | 3: initial guess already optimal within 0.25 px (file written); 2: FAIL, file not written |
| 4 | `acceptance_from_capture.py --capture C --intrinsics intr.yaml --extrinsics ext.yaml` | every taped object in view, with LiDAR returns in a 40x40 px window, within 0.15 m; AND LiDAR silhouettes within 2.0 px (mean) of image edges | 2: FAIL |

How each works, briefly:

* **Intrinsics**: OpenCV chessboard detection with sub-pixel refinement on up
  to 400 frames, greedy farthest-point choice of up to 40 views over board
  position, size and tilt, plumb_bob `calibrateCamera`.
* **Latency**: image yaw rate (median change of feature azimuth between frames,
  after undistortion, so it holds for every image row) cross-correlated with
  odometry yaw rate; positive lag means image stamps are late.
* **Extrinsics**: LiDAR silhouette returns are found in the LiDAR's own view
  (azimuth/elevation range image around `lidar_xyz`: a neighbouring direction
  ranges > 0.3 m farther). In the camera's view the LiDAR, mounted below the
  camera, sees background the camera cannot, which flags false silhouettes.
  The score is the mean distance from projected silhouettes to Canny edges
  (truncated at 15 px). Grid search over rpy then xyz, then a joint 6-D
  Nelder-Mead (yaw and a sideways shift are coupled). A translation axis with
  < 0.1 px contrast at +/-2 cm keeps its taped value.
* **Acceptance**: rebuilds the depth image `lidar_depth_node` would publish and
  compares it with the taped ranges. That range check cannot see camera
  calibration errors (taped point and LiDAR are both base-frame geometry), so
  acceptance also requires the silhouette-to-edge alignment. The open-vocab
  detector is not exercised.

## Tests

```bash
python3 -m pytest -q scripts/calib -p no:launch_testing -p no:launch_ros --import-mode=importlib
```

Synthetic captures only: rendered checkerboards with known K and D, a yaw
panorama stamped a known lag late, and a ray-cast box scene with a known
extrinsic. They show the tools recover known answers; they say nothing about
the real GO2 camera, its LiDAR density, or real-image edge clutter.

## Pole-based extrinsics (when the scene has no clean edges)

`extrinsics_from_poles.py` fits the camera pose in the LiDAR base frame from
single-pole captures (an orange pole, robot still, one capture per spot; scene
folders named `scene_<name>`):

```sh
python3 scripts/calib/extrinsics_from_poles.py --capture <root> --intrinsics intr.yaml \
    --poles C1:4:0,C2:4:2,... --out ext.yaml --leave-one-out --overlay-dir overlays/
python3 scripts/calib/extrinsics_from_poles.py --capture <new_root> --intrinsics intr.yaml \
    --poles ... --evaluate ext.yaml     # score a pose on captures that were not in the fit
```

`--poles` is `<scene>:<ahead_ft>:<left_ft>` from the tape; the tape only picks
which LiDAR cluster is the pole. The objective is the distance from projected
pole returns to the orange mask plus the pole's floor contact. Why it exists:
on cluttered lab scenes the silhouette-edge score of `extrinsics_from_capture.py`
and `acceptance_from_capture.py` does not separate a good pose from one that is
5-8 degrees wrong (11.3 px against 11.8-12.5 px on the 2026-10-15 S1 scene;
see `docs/preregistration/pole-extrinsics.md`). Pole captures do not have that
problem. `--evaluate` exits 0 only when every pole is within 2 px of its mask
and 4 cm at the floor contact.

