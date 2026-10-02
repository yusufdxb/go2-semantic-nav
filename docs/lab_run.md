# RGB + LiDAR lab run (target: under 10 minutes at the robot)

One script, run on the control PC, deploys to the robot computer, checks
health and latency, records three short captures, computes the calibration
on the robot computer and copies back only the results (a few MB). The raw
capture stays on the robot computer: the GO2 payload's 2.4 GHz wifi has been
measured at 0.2-1.4 Mbit/s, far too slow for hundreds of MB inside the time
budget. Pull it later with `run_lab.sh pull --full`, preferably over the cable. Nothing in these scripts commands robot motion; the operator
moves the robot with the remote in the yaw step.

```bash
scripts/lab/run_lab.sh all chair:1.5:0 chair:3.0:0 chair:2.0:0.8
```

`label:x:y` are the taped object positions in `base_link` (x forward, y left,
metres) to the point on each object's surface facing the robot. Each step can
also be run alone (`run_lab.sh deploy|up|probe|checkerboard|scene ...|yaw|down|calibrate|pull [--full]|offline`).
Set `BASE_STACK` (checkout providing `go2_localization`) and `JETSON_HOSTS`
(addresses to try, default the cable address `192.168.123.18`).

## Before the session (no robot)

* Print the 7x6-inner-corner, 25 mm checkerboard at 100 %, measure a square;
  if it is not 25.0 mm set `SQUARE_M` for the offline step.
* Tape marks for the objects; objects with flat faces toward the robot.
* Robot charged; remote in hand; cable or lab wifi to the robot computer.
* `run_lab.sh wheels` once (needs internet): aarch64 wheels for numpy, OpenCV
  and PyYAML, installed on the robot computer into a private folder only if
  its Python lacks them (the lab network has no internet).

## Timeline

| step | robot time | what happens | operator |
|---|---|---|---|
| deploy | ~2-3 min (first build), ~20 s after | robot-computer clock set from the PC, sources copied, Python deps from wheels if missing, `sysctl net.core.rmem_max`, GStreamer/Python/DDS checks, `colcon build` | nothing |
| up | ~20 s | base-stack LiDAR relay, C++ camera (Jetson hardware decoder, automatic fallback to software decode if it yields no frames), `lidar_depth_node` | nothing |
| probe | 15 s | PASS/FAIL table: camera fps, arrival-to-publish p95, DDS delivery, LiDAR and odometry rates, depth density; asks before continuing on a FAIL | read the table |
| checkerboard | <= 2 min, stops early | records frames at 3 Hz (no LiDAR), prints a 4x4 coverage map | sweep the board over corners and centre, 0.5-1.5 m, tilted |
| scene | 15 s | robot still, objects at the taped marks | board out of view |
| yaw | 30 s | full-rate frames + odometry (no LiDAR) | yaw slowly left/right with the remote (~0.5 rad/s) |
| down + calibrate | ~1-2 min | stop; intrinsics, latency, extrinsics, acceptance on the robot computer | robot can sit |
| pull | ~15 s | results, logs, capture metadata (~3 MB) to `~/rgb_lidar_runs/<UTC>`; `calibration/summary.txt` | read the summary |

## Pass criteria

Probe (live): camera >= 12 fps, arrival-to-publish p95 <= 25 ms, DDS delivery
>= 0.95, LiDAR >= 10 Hz, odometry >= 50 Hz, depth >= 3 Hz with >= 2000 valid
pixels. Offline: see `scripts/calib/README.md` (intrinsics RMS <= 0.5 px,
latency correlation >= 0.6, extrinsics alignment, every taped object within
0.15 m).

## Using the result

```bash
GSN_INTRINSICS=<run>/calibration/front_camera_intrinsics.yaml \
GSN_EXTRINSICS=<run>/calibration/front_camera_extrinsics.yaml \
GSN_LATENCY_S=<from latency.txt> \
  bash ~/gsn_lab/bin/jetson_lab.sh up      # on the robot computer, after copying the files there
```

or pass them to `deploy.launch.py` as `camera_intrinsics_file`,
`camera_extrinsics_file` (and `latency_s` on `go2_front_camera`). Until both
files exist, `lidar_depth_node` withholds depth outside these lab runs.

## If something fails

| symptom | likely cause | action |
|---|---|---|
| deploy: `FAIL DDS refused the 16MB receive buffer` | `rmem_max` not raised | rerun deploy; check sudo on the robot computer |
| deploy: `WARN DDS config does not name enP8p1s0` | DDS bound to the wrong NIC: topics list but carry no data | fix `CYCLONEDDS_URI` in the unitree setup before continuing |
| probe: camera fps 0 | multicast not reaching the NIC | `sudo tcpdump -i enP8p1s0 -c 5 udp port 1720` on the robot computer |
| probe: DDS delivery < 0.95 | receive buffers, CPU load | check `sysctl net.core.rmem_max`, `tegrastats` |
| probe: lidar/odom 0 | relay inputs missing | `ros2 topic hz /utlidar/robot_odom /utlidar/cloud_deskewed` |
| probe: arrival->publish p95 > 25 ms with several image readers | DDS sending frames as multicast on the robot NIC | the lab env sets `AllowMulticast=spdp`; check `CYCLONEDDS_URI` of the camera process and `enP8p1s0` tx bytes (should be kB/s, not MB/s) |
| probe: latency high, load average > 8 | CPU contention (another job, SLAM forced to add scans while still) | `top`; keep the repo slam config |
| checkerboard: coverage stays low | board too far / one region | bring it closer, cover the empty cells of the map |
| offline latency: yaw-rate RMS < 0.2 | yawed too slowly | redo `run_lab.sh yaw` (30 s) |

## Mapping and semantic perception (robot standing still)

After `up`, with no robot motion:

```bash
scripts/lab/run_lab.sh mlwheels        # once, on a PC with internet: detector wheels + weights (~2 GB)
scripts/lab/run_lab.sh mldeploy        # copy, install into ~/gsn_lab/pydeps_ml, CUDA check
scripts/lab/run_lab.sh slam            # /scan + slam_toolbox mapping, map saved every 60 s to ~/gsn_lab/maps/
scripts/lab/run_lab.sh semantic        # detector + scene graph, grounding off, JSON snapshots every 10 s
scripts/lab/run_lab.sh maps            # copy maps + snapshots to ~/rgb_lidar_runs/maps
```

`down` stops these too. The detector torch lives in a private folder that only
the `semantic` processes put on `PYTHONPATH`; the robot computer's own Python
packages are not changed. With the repo slam config a scan joins the map only
after 0.2 m or 0.2 rad of motion, so a robot that stands still keeps the first
scan; object positions use the camera calibration, so they are only as good
as it is (nominal until the calibration above has run).

Measured on the robot (2026-10-02, standing still, nominal calibration, MAXN):
the detector loads in about a minute and runs near 1 Hz, not the configured
5 Hz: YOLO-World v2-s 34 ms, MobileSAM 573 ms, OpenCLIP ViT-B/16 511 ms per
frame with 11 objects. Camera arrival->publish with the depth node and the
detector reading the image: p50 about 18 ms; p95 per one-second window
18.6-25.4 ms, so marginal against the 25 ms probe limit (the probe itself, as
a third reader, measured 25.2-27.6 ms). The lab wifi to the robot computer measured 46 Mbit/s that day.

## Camera latency: where the time goes

Per-element GStreamer latency tracer on the robot, same pipeline as the node
(`GST_TRACERS="latency(flags=element)"`), p50 / p95:

| stage | ms |
|---|---|
| RTP depay + H.264 parse | 0.2 / 0.4 |
| `nvv4l2decoder` (hardware decode) | 16.0 / 22.6 |
| `nvvidconv` NV12 -> BGRx | 3.8 / 4.6 |
| `videoconvert` BGRx -> BGR (CPU) | 2.1 / 3.5 |

The traced pipeline ran next to the live one, so two streams shared the
decoder and its 16 ms is likely inflated. In the node itself (p50 about 18 ms
with two readers), subtracting the conversions (~6 ms) and the copy plus DDS
publish (about 1-2 ms per local reader) leaves roughly 10 ms for decode: still
the largest stage. The NVDEC clock read 115 MHz (its floor) in 18
of 20 samples while decoding, against a maximum of 858 MHz;
`enable-max-performance=true` does not raise it, and `enable-full-frame=true`
made no difference. Whether pinning the NVDEC clock shortens decode has not
been tested.

## Rehearsal without the robot

`jetson_lab.sh` runs on a desktop with `GSN_REHEARSAL=1` (no sudo, loopback
camera, best-effort DDS buffer), `scripts/lab/fake_go2_sensors.py` (robot-clock
odometry and LiDAR) and `scripts/bench/rtp_counter_sender.py` (RTP H.264 camera
stream). `run_lab.sh all` itself has been rehearsed end to end on one desktop
with stand-ins for `ssh`, `sshpass` and `sudo` that run the "remote" side in a
scratch home folder (forced wheel install, nvv4l2-to-avdec fallback, all
captures, calibration, results-only pull: exit 0). The calibration tools'
PASS path has only been exercised on synthetic captures. On the robot,
`deploy`, `up`, `probe`, `slam` and `semantic` have run (2026-10-02, no motion);
the calibration captures have not.
