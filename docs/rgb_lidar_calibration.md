# RGB + LiDAR depth on the GO2: bring-up and calibration

The GO2 has a front RGB camera and no depth camera. `go2_rgb_lidar` gives the
open-vocab detector an aligned depth stream by projecting the GO2 LiDAR into
the camera:

| node | in | out |
|---|---|---|
| `front_camera_node` (`go2_front_camera_cpp`, default; Python fallback in `go2_rgb_lidar`) | RTP H.264 multicast `230.1.1.1:1720` | `/camera/front/image_raw` (bgr8, stamped at packet arrival on the local clock, `front_camera_optical_frame`) |
| `lidar_depth_node` | the image, `/go2/lidar/points` (base stack relay), TF `odom -> base_link` | `/camera/front/lidar_depth` (16UC1 mm, 0 = no return), `/camera/front/camera_info`, static TF `base_link -> front_camera_optical_frame`, optional `/camera/front/lidar_overlay` |

Depth and camera_info carry the colour frame's exact header, so the detector's
RGB-D synchroniser and alignment gate work unchanged. Clouds are accumulated
for 0.5 s in `odom` to densify the rings, then a visibility filter drops
returns that sit behind a nearer return (the LiDAR is below the camera and sees
background through the gaps). The detector runs with
`config/detector_rgb_lidar.yaml`, which requires at least 50 valid depth pixels
(two LiDAR returns) on an object's mask.

`deploy.launch.py` uses this by default (`camera:=rgb_lidar`).

## Fail-closed behaviour

No depth and no camera_info are published, so the detector reports nothing and
grounding refuses every query, when:

* either calibration file is `nominal: true` (the shipped files are) and
  `allow_nominal_calibration` is false;
* no LiDAR cloud is within 0.3 s of the image;
* the image size or frame differs from the calibration;
* TF `odom -> base_link` is missing at the image stamp.

The overlay is still published in all of these cases: it is the calibration tool.

## What has and has not been verified

| check | status |
|---|---|
| projection math vs OpenCV `projectPoints`, depth rendering, visibility filter | unit tests |
| `lidar_depth_node` with real TF and messages, every fail-closed path, the detector's QoS | in-process ROS tests |
| `front_camera_node` decoding an RTP H.264 stream, recovering after the stream stops and returns at a new resolution | loopback test (unicast; the multicast join is not tested off the robot) |
| real GO2 frame -> `lidar_depth_node` -> real detector -> correct 3D depth and lateral position | `scripts/smoke_rgb_lidar.py`, with a SYNTHETIC LiDAR scene |
| `deploy.launch.py` brings the nodes up with the RGB-LiDAR topics | dry run on the desktop, no robot |
| the GO2 camera stream format (1280x720, 15 fps), the multicast join, real LiDAR density in the camera view, calibration accuracy, capture latency | **NOT verified: needs the robot** |

## Lab procedure

The fast path is [`lab_run.md`](lab_run.md): one script records everything in
under 10 minutes at the robot and the calibration below is computed offline by
`scripts/calib`. The manual procedure below remains the fallback.

Prerequisites: Jetson clock set (it boots at 1970), base stack built and its
localization running (`/go2/lidar/points` and `odom -> base_link` live),
CycloneDDS bound to the robot NIC (`enP8p1s0`). Robot standing still unless a
step says otherwise.

### 1. Camera stream

```bash
ros2 launch go2_rgb_lidar rgb_lidar.launch.py multicast_iface:=enP8p1s0 \
    allow_nominal_calibration:=true publish_overlay:=true
ros2 topic hz /camera/front/image_raw          # expect ~15 Hz
ros2 topic echo --once /camera/front/image_raw --field width   # and height
```

If the size is not 1280x720, the nominal intrinsics do not apply: calibrate at
the real size (step 2) before anything else. If no frames arrive, check the
interface name and that `230.1.1.1` reaches the Jetson (`sudo tcpdump -i enP8p1s0 -c 5 udp port 1720`).

### 2. Intrinsics (checkerboard)

Board: a MATLAB-style A4 checkerboard with 25 mm squares and 7x6 inner
corners. Print at 100 % and measure a square with a ruler: use the measured
size for `--square`.

```bash
ros2 run camera_calibration cameracalibrator --size 7x6 --square 0.025 \
    --no-service-check image:=/camera/front/image_raw
```

Cover the whole image, especially the corners (a 100 deg lens distorts most
there), until X, Y, Size and Skew bars are green; CALIBRATE, then SAVE. Take
`ost.yaml` from `/tmp/calibrationdata.tar.gz` and save it as
`front_camera_intrinsics.yaml` (no `nominal` key). Accept a reprojection error
under 0.5 px. If straight lines near the image edge still bend after
rectification, plumb_bob is not enough for this lens; lower
`max_normalized_radius` so edge pixels are not used.

### 3. Extrinsics (LiDAR overlay)

Measure the camera's position from `base_link` with a tape and enter it in a
copy of `config/front_camera_extrinsics_nominal.yaml`. Then face the robot at
something with sharp vertical and horizontal edges 1 to 3 m away (a door frame,
a box on the floor, a wall corner) and view the overlay:

```bash
ros2 launch go2_rgb_lidar rgb_lidar.launch.py multicast_iface:=enP8p1s0 \
    intrinsics_file:=<front_camera_intrinsics.yaml> extrinsics_file:=<your copy> \
    allow_nominal_calibration:=true publish_overlay:=true
ros2 run rqt_image_view rqt_image_view /camera/front/lidar_overlay
ros2 param set /go2_lidar_depth extrinsic_rpy "[-1.5708, 0.0, -1.5708]"
ros2 param set /go2_lidar_depth extrinsic_xyz "[0.32, 0.0, 0.03]"
```

Red = near, blue = far. Adjust yaw until vertical edges line up, pitch until
the floor/wall boundary lines up, then roll. Write the final values into the
file and set `nominal: false`.

### 4. Latency

The camera is stamped at receipt. Yaw the robot slowly in place (remote in
hand) and watch the overlay: if LiDAR edges lead or lag the image during the
turn but line up when still, set `latency_s` on `go2_front_camera` to the delay
that removes the slip (start at 0.1 s).

### 5. Acceptance before any motion

Put a chair at a tape-measured 1.5 m and 3.0 m straight ahead and at 2.0 m
off to one side. Run the full deploy with the calibrated files
(`hardware_adapter:=dry_run`) and read `/semantic/scene_graph`: every chair
must be within 0.15 m of the measured position. Record the result, including
failures, before enabling motion.

```bash
ros2 launch go2_semantic_bringup deploy.launch.py hardware_adapter:=dry_run \
    camera_intrinsics_file:=<front_camera_intrinsics.yaml> \
    camera_extrinsics_file:=<front_camera_extrinsics.yaml> multicast_iface:=enP8p1s0
```
